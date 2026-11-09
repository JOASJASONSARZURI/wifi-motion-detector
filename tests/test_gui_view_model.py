"""Pruebas SYNTHETIC a mano para el adaptador visual."""

from __future__ import annotations

import math
import re

from wifi_motion.detector import DetectionResult
from wifi_motion.gui import theme
from wifi_motion.gui.view_model import (
    build_monitor_view,
    format_stats,
    log_line,
)


def _result(
    state: str,
    *,
    scores: dict[str, float] | None = None,
    confidence: float = 0.857,
    changed: bool = True,
    reason: str | None = None,
) -> DetectionResult:
    return DetectionResult(
        node_id="synthetic-node",
        timestamp=1_700_000_000.0,
        state=state,
        scores=scores
        or {
            "absent": 0.1,
            "present_still": 0.2,
            "present_moving": 0.7,
        },
        confidence=confidence,
        changed=changed,
        input_quality={
            "window_quality": "ok",
            "n_samples": 24,
            "estimated_rate_hz": 12.5,
            "rejected_samples": 0,
        },
        reason=reason,
    )


def test_none_shows_waiting_and_not_available_fields() -> None:
    view = build_monitor_view(None)

    assert view.headline == "ESPERANDO DATOS"
    assert view.subline == "N/D"
    assert view.color == theme.TEXT
    assert view.symbol == "none"
    assert view.confidence_text == "N/D"
    assert view.confidence_ratio is None
    assert all(text == "N/D" for _, text in view.score_rows)
    assert view.quality_text == "N/D"


def test_absent_mapping_is_gray() -> None:
    view = build_monitor_view(_result("absent"))

    assert view.headline == "[ PERSONA AUSENTE ]"
    assert view.color == theme.ABSENT
    assert view.symbol == "none"
    assert "AUSENTE" in view.headline


def test_present_still_mapping_is_red() -> None:
    view = build_monitor_view(_result("present_still"))

    assert view.headline == "[ PERSONA PRESENTE ]"
    assert view.subline == "[ ESTADO: QUIETA ]"
    assert view.color == theme.PRESENT
    assert view.symbol == "heart"


def test_present_moving_mapping_is_yellow() -> None:
    view = build_monitor_view(_result("present_moving"))

    assert view.headline == "[ PERSONA PRESENTE ]"
    assert view.subline == "[ ESTADO: EN MOVIMIENTO ]"
    assert view.color == theme.WARNING
    assert view.symbol == "heart_moving"


def test_unknown_is_not_presented_as_absent() -> None:
    view = build_monitor_view(_result("unknown", reason="gap"))
    visible_text = " ".join(
        (
            view.headline,
            view.subline,
            view.flavor,
            view.quality_text,
            *(label for label, _value in view.score_rows),
        )
    )

    assert view.headline == "[ SIN DATOS VÁLIDOS ]"
    assert view.subline == "gap"
    assert view.color == theme.WARNING
    assert view.symbol == "warning"
    assert "AUSENTE" not in visible_text


def test_score_percentages_and_confidence_follow_result_values() -> None:
    result = _result(
        "present_moving",
        scores={
            "absent": 0.143,
            "present_still": 0.286,
            "present_moving": 0.571,
        },
        confidence=0.857,
    )

    view = build_monitor_view(result)

    assert view.score_rows == (
        ("Ausente", "14.3%"),
        ("Presente quieto", "28.6%"),
        ("Presente en movimiento", "57.1%"),
    )
    assert view.confidence_text == "85.7%"
    assert view.confidence_ratio == 0.857


def test_nan_scores_confidence_and_stats_are_not_available() -> None:
    result = _result(
        "unknown",
        scores={
            "absent": math.nan,
            "present_still": 0.25,
            "present_moving": 0.75,
        },
        confidence=math.nan,
        reason="bad feature",
    )
    view = build_monitor_view(result)
    stats = format_stats(
        {
            "samples_received": math.nan,
            "samples_rejected": 1,
            "windows_processed": None,
            "state_changes": 2,
        },
        result,
    )

    assert view.confidence_text == "N/D"
    assert view.confidence_ratio is None
    assert view.score_rows[0] == ("Ausente", "N/D")
    assert stats == {
        "muestras_recibidas": "N/D",
        "muestras_rechazadas": "1",
        "ventanas_procesadas": "N/D",
        "cambios_de_estado": "2",
        "frecuencia_de_muestreo": "12.50 Hz",
        "calidad_de_datos": (
            "VENTANA: ok | MUESTRAS: 24 | FRECUENCIA: 12.50 Hz | RECHAZADAS: 0"
        ),
    }


def test_format_stats_missing_values_and_result() -> None:
    assert format_stats(None, None) == {
        "muestras_recibidas": "N/D",
        "muestras_rechazadas": "N/D",
        "ventanas_procesadas": "N/D",
        "cambios_de_estado": "N/D",
        "frecuencia_de_muestreo": "N/D",
        "calidad_de_datos": "N/D",
    }


def test_log_line_only_emits_for_changes_or_unknown() -> None:
    assert log_line(_result("absent", changed=False)) is None
    changed_line = log_line(_result("present_still", changed=True))
    assert changed_line is not None
    assert re.fullmatch(
        r"\d{2}:\d{2}:\d{2} - synthetic-node: present_still \(present_still\)",
        changed_line,
    )
    assert log_line(
        _result("unknown", changed=False, reason="gap")
    ).endswith("unknown (gap)")
