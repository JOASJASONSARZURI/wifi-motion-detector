"""Adaptadores puros entre resultados del detector y datos de presentación."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from typing import Mapping

from wifi_motion.detector import DetectionResult
from wifi_motion.gui import theme

__all__ = ["MonitorView", "build_monitor_view", "format_stats", "log_line"]

_CLASS_SCORE_ROWS = (
    ("Ausente", "absent"),
    ("Presente quieto", "present_still"),
    ("Presente en movimiento", "present_moving"),
)


@dataclass(frozen=True, slots=True)
class MonitorView:
    headline: str
    subline: str
    color: str
    symbol: str
    confidence_text: str
    confidence_ratio: float | None
    score_rows: tuple[tuple[str, str], ...]
    flavor: str
    quality_text: str


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _percentage(value: object) -> str:
    numeric = _finite_number(value)
    return "N/D" if numeric is None else f"{numeric * 100:.1f}%"


def _quality_summary(result: DetectionResult) -> str:
    quality = result.input_quality
    window_quality = quality.get("window_quality")
    samples = quality.get("n_samples")
    rate = quality.get("estimated_rate_hz")
    rejected = quality.get("rejected_samples")
    details = (
        f"VENTANA: {window_quality if window_quality is not None else 'N/D'}",
        f"MUESTRAS: {samples if samples is not None else 'N/D'}",
        f"FRECUENCIA: {_format_hz(rate)}",
        f"RECHAZADAS: {rejected if rejected is not None else 'N/D'}",
    )
    return " | ".join(details)


def _format_hz(value: object) -> str:
    numeric = _finite_number(value)
    return "N/D" if numeric is None else f"{numeric:.2f} Hz"


def build_monitor_view(result: DetectionResult | None) -> MonitorView:
    """Construye el contenido del monitor sin acceder a Tkinter."""
    if result is None:
        return MonitorView(
            headline="ESPERANDO DATOS",
            subline="N/D",
            color=theme.TEXT,
            symbol="none",
            confidence_text="N/D",
            confidence_ratio=None,
            score_rows=tuple((label, "N/D") for label, _key in _CLASS_SCORE_ROWS),
            flavor="El silencio espera su turno.",
            quality_text="N/D",
        )

    score_rows = tuple(
        (label, _percentage(result.scores.get(key)))
        for label, key in _CLASS_SCORE_ROWS
    )
    confidence = _finite_number(result.confidence)
    confidence_ratio = (
        min(1.0, max(0.0, confidence)) if confidence is not None else None
    )
    confidence_text = _percentage(confidence)
    quality_text = _quality_summary(result)

    if result.state == "absent":
        headline = "[ PERSONA AUSENTE ]"
        subline = "N/D"
        color = theme.ABSENT
        symbol = "none"
        flavor = "La habitación respira tranquila."
    elif result.state == "present_still":
        headline = "[ PERSONA PRESENTE ]"
        subline = "[ ESTADO: QUIETA ]"
        color = theme.PRESENT
        symbol = "heart"
        flavor = "Una presencia guarda silencio."
    elif result.state == "present_moving":
        headline = "[ PERSONA PRESENTE ]"
        subline = "[ ESTADO: EN MOVIMIENTO ]"
        color = theme.WARNING
        symbol = "heart_moving"
        flavor = "Algo se mueve entre las señales."
    else:
        reason = result.reason or "N/D"
        headline = "[ SIN DATOS VÁLIDOS ]"
        subline = reason
        color = theme.WARNING
        symbol = "warning"
        flavor = "La señal se perdió; el estado no se adivina."

    return MonitorView(
        headline=headline,
        subline=subline,
        color=color,
        symbol=symbol,
        confidence_text=confidence_text,
        confidence_ratio=confidence_ratio,
        score_rows=score_rows,
        flavor=flavor,
        quality_text=quality_text,
    )


def format_stats(
    stats: dict[str, object] | None, last_result: DetectionResult | None
) -> dict[str, str]:
    """Formatea las seis métricas del panel; los datos ausentes son N/D."""
    source: Mapping[str, object] = {} if stats is None else stats
    quality = "N/D" if last_result is None else _quality_summary(last_result)
    rate = (
        "N/D"
        if last_result is None
        else _format_hz(last_result.input_quality.get("estimated_rate_hz"))
    )
    fields = (
        ("muestras_recibidas", "samples_received"),
        ("muestras_rechazadas", "samples_rejected"),
        ("ventanas_procesadas", "windows_processed"),
        ("cambios_de_estado", "state_changes"),
    )
    formatted = {
        key: _format_stat(source.get(stat_key))
        for key, stat_key in fields
    }
    formatted["frecuencia_de_muestreo"] = rate
    formatted["calidad_de_datos"] = quality
    return formatted


def _format_stat(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/D"
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return "N/D"
    if not math.isfinite(numeric):
        return "N/D"
    return str(int(value)) if isinstance(value, int) else f"{numeric:g}"


def log_line(result: DetectionResult) -> str | None:
    """Devuelve una línea solo ante cambio de estado o salida unknown."""
    if not result.changed and result.state != "unknown":
        return None
    try:
        timestamp = datetime.fromtimestamp(result.timestamp).strftime("%H:%M:%S")
    except (OverflowError, OSError, ValueError):
        timestamp = "N/D"
    details = result.reason or result.state
    return f"{timestamp} - {result.node_id}: {result.state} ({details})"
