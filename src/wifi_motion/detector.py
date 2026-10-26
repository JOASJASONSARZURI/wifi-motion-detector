"""Pipeline en línea de detección: muestra, ventana, características y estado."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from wifi_motion.classifiers.base import (
    UNKNOWN,
    Classifier,
    ClassifierError,
    Prediction,
)
from wifi_motion.features import FeatureConfig, FeatureError, extract_features
from wifi_motion.schema import CsiSample
from wifi_motion.smoothing import SmoothedResult, StateSmoother
from wifi_motion.validation import BLOCKING_CODES, IssueCode
from wifi_motion.windowing import SlidingWindower, Window, WindowQuality

__all__ = ["DetectionResult", "Detector", "run_samples"]


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Resultado inmutable; confidence no es una probabilidad calibrada salvo que se mida y calibre."""

    node_id: str
    timestamp: float
    state: str
    scores: Mapping[str, float]
    confidence: float
    changed: bool
    input_quality: dict[str, object]
    reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))
        object.__setattr__(self, "input_quality", dict(self.input_quality))


class Detector:
    """Orquesta validación, ventanas, clasificación y suavizado por nodo."""

    def __init__(
        self,
        windower: SlidingWindower,
        feature_config: FeatureConfig,
        classifier: Classifier,
        smoother: StateSmoother,
        stale_timeout_s: float,
    ) -> None:
        if stale_timeout_s <= 0 or not np.isfinite(stale_timeout_s):
            raise ValueError("stale_timeout_s debe ser finito y mayor que cero")
        self.windower = windower
        self.feature_config = feature_config
        self.classifier = classifier
        self.smoother = smoother
        self.stale_timeout_s = float(stale_timeout_s)
        # TODO: adaptar baseline solo durante periodos quietos confirmados.
        self._received = 0
        self._rejected_by_issue: Counter[IssueCode] = Counter()
        self._windows_processed = 0
        self._unknown_results = 0
        self._state_changes = 0
        self._stale_reported: set[str] = set()

    def process(self, sample: CsiSample) -> list[DetectionResult]:
        """Procesa una muestra; las inválidas se cuentan pero no generan estado."""
        self._received += 1
        dropped_before = self.windower.dropped
        issue_counts_before = self.windower.issue_counts.copy()
        windows = self.windower.push(sample)
        issue_delta = self.windower.issue_counts - issue_counts_before
        rejected = self.windower.dropped - dropped_before
        if rejected:
            for code, count in issue_delta.items():
                if code in BLOCKING_CODES:
                    self._rejected_by_issue[code] += count
        if not rejected:
            self._stale_reported.discard(sample.node_id)

        results: list[DetectionResult] = []
        for window in windows:
            self._windows_processed += 1
            quality = self._quality(window, rejected_samples=rejected)
            if window.quality is WindowQuality.HAS_GAP:
                smoothed = self.smoother.mark_unknown(
                    window.node_id, window.t_end, "gap"
                )
                results.append(self._result(smoothed, quality, 0.0))
                continue
            if window.quality is WindowQuality.INSUFFICIENT_SAMPLES:
                smoothed = self.smoother.mark_unknown(
                    window.node_id, window.t_end, "insufficient_samples"
                )
                results.append(self._result(smoothed, quality, 0.0))
                continue

            try:
                features = extract_features(window, self.feature_config)
                prediction = self.classifier.predict(features)
            except (FeatureError, ClassifierError) as exc:
                smoothed = self.smoother.mark_unknown(
                    window.node_id, window.t_end, str(exc)
                )
                results.append(self._result(smoothed, quality, 0.0))
                continue

            smoothed = self.smoother.update(
                window.node_id, window.t_end, prediction
            )
            results.append(self._result(smoothed, quality, prediction.confidence))
        return results

    def check_stale(self, now: float) -> list[DetectionResult]:
        """Emite UNKNOWN una vez para cada nodo cuyo flujo superó el timeout."""
        results = []
        for node_id in self.windower.nodes_stale(now, self.stale_timeout_s):
            if node_id in self._stale_reported:
                continue
            smoothed = self.smoother.mark_unknown(node_id, now, "stale_node")
            results.append(
                self._result(
                    smoothed,
                    {
                        "window_quality": None,
                        "n_samples": 0,
                        "estimated_rate_hz": 0.0,
                        "rejected_samples": 0,
                    },
                    0.0,
                )
            )
            self._stale_reported.add(node_id)
        return results

    def stats(self) -> dict[str, object]:
        """Devuelve contadores acumulados del flujo y los resultados emitidos."""
        return {
            "samples_received": self._received,
            "samples_rejected": self.windower.dropped,
            "rejected_by_issue": dict(self._rejected_by_issue),
            "windows_processed": self._windows_processed,
            "unknown_results": self._unknown_results,
            "state_changes": self._state_changes,
        }

    def reset(self) -> None:
        """Reinicia el estado del detector y los componentes con estado."""
        self.windower.reset()
        self.smoother.reset()
        self._received = 0
        self._rejected_by_issue.clear()
        self._windows_processed = 0
        self._unknown_results = 0
        self._state_changes = 0
        self._stale_reported.clear()

    @staticmethod
    def _quality(window: Window, *, rejected_samples: int) -> dict[str, object]:
        return {
            "window_quality": window.quality.value,
            "n_samples": len(window.timestamps),
            "estimated_rate_hz": window.estimated_rate_hz,
            "rejected_samples": rejected_samples,
        }

    def _result(
        self,
        smoothed: SmoothedResult,
        input_quality: dict[str, object],
        confidence: float,
    ) -> DetectionResult:
        if smoothed.state == UNKNOWN:
            self._unknown_results += 1
        if smoothed.changed:
            self._state_changes += 1
        return DetectionResult(
            node_id=smoothed.node_id,
            timestamp=smoothed.timestamp,
            state=smoothed.state,
            scores=smoothed.scores,
            confidence=float(confidence),
            changed=smoothed.changed,
            input_quality=input_quality,
            reason=smoothed.reason,
        )


def run_samples(
    detector: Detector, samples: Iterable[CsiSample]
) -> Iterator[DetectionResult]:
    """Procesa una grabación y produce los resultados en orden de emisión."""
    for sample in samples:
        yield from detector.process(sample)