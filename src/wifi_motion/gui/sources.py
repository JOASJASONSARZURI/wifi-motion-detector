"""Fuentes de resultados para la interfaz visual."""

from __future__ import annotations

import time
from typing import Protocol

from wifi_motion.detector import DetectionResult

__all__ = ["DemoSource", "ResultSource"]


class ResultSource(Protocol):
    """Interfaz para obtener resultados y contadores del monitor."""

    def poll(self) -> list[DetectionResult]:
        """Devuelve los resultados disponibles desde la última consulta."""
        ...

    def stats(self) -> dict[str, object] | None:
        """Devuelve estadísticas de la fuente si están disponibles."""
        ...


class DemoSource:
    """Secuencia SYNTHETIC de demostración; no representa mediciones reales."""

    evidence = "SYNTHETIC"
    _states = ("absent", "present_still", "present_moving", "unknown")
    _scores = {
        "absent": {"absent": 0.86, "present_still": 0.09, "present_moving": 0.05},
        "present_still": {
            "absent": 0.08,
            "present_still": 0.84,
            "present_moving": 0.08,
        },
        "present_moving": {
            "absent": 0.05,
            "present_still": 0.10,
            "present_moving": 0.85,
        },
        "unknown": {"absent": 0.0, "present_still": 0.0, "present_moving": 0.0},
    }

    def __init__(self) -> None:
        self._index = 0
        self._samples_received = 0
        self._windows_processed = 0
        self._unknown_results = 0
        self._state_changes = 0

    def poll(self) -> list[DetectionResult]:
        """Produce un resultado sintético por consulta."""
        state = self._states[self._index % len(self._states)]
        scores = dict(self._scores[state])
        self._index += 1
        self._samples_received += 1
        self._windows_processed += 1
        self._state_changes += 1
        if state == "unknown":
            self._unknown_results += 1
        return [
            DetectionResult(
                node_id="synthetic-node",
                timestamp=time.time(),
                state=state,
                scores=scores,
                confidence=max(scores.values()),
                changed=True,
                input_quality={
                    "window_quality": "synthetic",
                    "n_samples": 0,
                    "estimated_rate_hz": 0.0,
                    "rejected_samples": 0,
                },
                reason="SYNTHETIC: resultado de demostración",
            )
        ]

    def stats(self) -> dict[str, object]:
        """Devuelve contadores sintéticos claramente marcados."""
        return {
            "evidence": "SYNTHETIC",
            "samples_received": self._samples_received,
            "samples_rejected": 0,
            "rejected_by_issue": {},
            "windows_processed": self._windows_processed,
            "unknown_results": self._unknown_results,
            "state_changes": self._state_changes,
        }
