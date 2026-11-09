"""Reproduce grabaciones en segundo plano para la interfaz visual."""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Literal

from wifi_motion.calibration import load_baseline
from wifi_motion.classifiers.softmax import SoftmaxClassifier
from wifi_motion.classifiers.threshold import ThresholdClassifier, ThresholdConfig
from wifi_motion.detector import DetectionResult, Detector, run_samples
from wifi_motion.features import FeatureConfig
from wifi_motion.gui.sources import ResultSource
from wifi_motion.io_jsonl import iter_recording
from wifi_motion.smoothing import SmoothingConfig, StateSmoother
from wifi_motion.windowing import SlidingWindower

__all__ = ["ReplaySource", "ReplaySourceError"]

_END = object()


class ReplaySourceError(RuntimeError):
    """Error legible al abrir o procesar una grabación reproducida."""


class ReplaySource:
    """Procesa un JSONL etiquetado en un hilo y entrega resultados con ``poll``."""

    def __init__(
        self,
        input_path: str | Path,
        *,
        classifier: Literal["threshold", "softmax"],
        baseline_path: str | Path | None = None,
        model_path: str | Path | None = None,
        window_s: float = 4.0,
        hop_s: float = 1.0,
        min_samples: int = 50,
        max_gap_s: float = 0.15,
        ema_alpha: float = 0.4,
        confirm_n: int = 3,
        confirm_n_to_absent: int = 5,
        margin: float = 0.05,
        stale_timeout_s: float = 5.0,
        motion_z: float = 3.0,
        presence_z: float = 120.0,
    ) -> None:
        self.input_path = Path(input_path)
        self.classifier_type = classifier
        self.baseline_path = None if baseline_path is None else Path(baseline_path)
        self.model_path = None if model_path is None else Path(model_path)
        self.window_config = {
            "window_s": window_s,
            "hop_s": hop_s,
            "min_samples": min_samples,
            "max_gap_s": max_gap_s,
        }
        self.smoothing_config = {
            "ema_alpha": ema_alpha,
            "confirm_n": confirm_n,
            "confirm_n_to_absent": confirm_n_to_absent,
            "margin": margin,
        }
        self.stale_timeout_s = stale_timeout_s
        self.motion_z = motion_z
        self.presence_z = presence_z
        self._results: Queue[DetectionResult | Exception | object] = Queue()
        self._stats_lock = Lock()
        self._detector: Detector | None = None
        self._pending_error: Exception | None = None
        self._finished = False
        self.evidence: Literal["SYNTHETIC", "MEASURED"] | None = None

        if classifier not in ("threshold", "softmax"):
            raise ValueError("classifier debe ser 'threshold' o 'softmax'")
        if classifier == "threshold" and self.baseline_path is None:
            raise ValueError("baseline_path es obligatorio para classifier='threshold'")
        if classifier == "softmax" and self.model_path is None:
            raise ValueError("model_path es obligatorio para classifier='softmax'")

        self._thread = Thread(
            target=self._run,
            name="wifi-motion-replay",
            daemon=True,
        )
        self._thread.start()

    def poll(self) -> list[DetectionResult]:
        """Retira resultados listos sin esperar al hilo de reproducción."""
        if self._pending_error is not None:
            error, self._pending_error = self._pending_error, None
            raise error

        available: list[DetectionResult] = []
        while True:
            try:
                item = self._results.get_nowait()
            except Empty:
                break
            if item is _END:
                self._finished = True
                break
            if isinstance(item, Exception):
                if available:
                    self._pending_error = item
                    break
                raise item
            if isinstance(item, DetectionResult):
                available.append(item)
        return available

    def stats(self) -> dict[str, object] | None:
        """Devuelve literalmente los contadores actuales del detector."""
        with self._stats_lock:
            detector = self._detector
        return None if detector is None else detector.stats()

    def _build_detector(self) -> Detector:
        if self.classifier_type == "threshold":
            assert self.baseline_path is not None
            classifier = ThresholdClassifier(
                load_baseline(self.baseline_path),
                ThresholdConfig(
                    motion_features=("motion_band_power",),
                    presence_features=("breathing_band_power",),
                    motion_z=self.motion_z,
                    presence_z=self.presence_z,
                ),
            )
        else:
            assert self.model_path is not None
            classifier = SoftmaxClassifier.load(self.model_path)

        return Detector(
            SlidingWindower(**self.window_config),
            FeatureConfig(),
            classifier,
            StateSmoother(SmoothingConfig(**self.smoothing_config)),
            stale_timeout_s=self.stale_timeout_s,
        )

    def _run(self) -> None:
        try:
            metadata, iterator = iter_recording(self.input_path)
            self.evidence = "SYNTHETIC" if metadata.synthetic else "MEASURED"
            with iterator:
                detector = self._build_detector()
                with self._stats_lock:
                    self._detector = detector
                for result in run_samples(detector, iterator):
                    self._results.put(result)
        except Exception as exc:
            self._results.put(
                ReplaySourceError(
                    f"No se pudo reproducir la grabación {self.input_path}: {exc}"
                )
            )
        finally:
            self._results.put(_END)
