"""Pruebas SYNTHETIC del pipeline del detector."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from wifi_motion.classifiers.base import (
    ABSENT,
    CLASS_STATES,
    PRESENT_MOVING,
    UNKNOWN,
    ClassifierError,
    Prediction,
)
from wifi_motion.detector import Detector, run_samples
from wifi_motion.features import FeatureConfig
from wifi_motion.schema import CsiSample
from wifi_motion.smoothing import SmoothingConfig, StateSmoother
from wifi_motion.synthetic import generate_session, inject_faults
from wifi_motion.validation import IssueCode
from wifi_motion.windowing import SlidingWindower

RATE = 16.0
WINDOW_CONFIG = {
    "window_s": 2.0,
    "hop_s": 1.0,
    "min_samples": 24,
    "max_gap_s": 0.2,
}
FEATURE_CONFIG = FeatureConfig(
    motion_band_hz=(1.0, 4.0),
    breathing_band_hz=(0.5, 1.5),
)


class VarianceClassifier:
    name = "synthetic-variance"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def predict(self, features: np.ndarray) -> Prediction:
        if self.fail:
            raise ClassifierError("stub failure")
        state = PRESENT_MOVING if features[0] > 0.04 else ABSENT
        scores = {label: 0.02 for label in CLASS_STATES}
        scores[state] = 0.96
        return Prediction(state, scores, scores[state], self.name)

    def predict_batch(self, X: np.ndarray) -> list[Prediction]:
        return [self.predict(row) for row in X]


def detector(*, classifier=None, confirm_n: int = 2, max_gap_s: float = 0.2) -> Detector:
    windower = SlidingWindower(
        **{**WINDOW_CONFIG, "max_gap_s": max_gap_s}
    )
    smoother = StateSmoother(
        SmoothingConfig(ema_alpha=1.0, confirm_n=confirm_n, margin=0.0)
    )
    return Detector(
        windower,
        FEATURE_CONFIG,
        VarianceClassifier() if classifier is None else classifier,
        smoother,
        stale_timeout_s=1.0,
    )


def synthetic_stream(
    segments: list[tuple[str, float]], *, seed: int = 0, node_id: str = "node-1"
) -> list[CsiSample]:
    _, samples = generate_session(
        segments,
        session_id="syn-detector",
        node_id=node_id,
        rate_hz=RATE,
        n_subcarriers=8,
        seed=seed,
    )
    return samples


def test_absent_to_moving_changes_after_debounce() -> None:
    system = detector(confirm_n=2)
    results = list(
        run_samples(
            system,
            synthetic_stream([("absent", 4.0), ("present_moving", 6.0)], seed=4),
        )
    )

    assert any(result.state == ABSENT for result in results)
    assert any(result.state == PRESENT_MOVING and result.changed for result in results)
    assert all(
        set(result.input_quality)
        == {"window_quality", "n_samples", "estimated_rate_hz", "rejected_samples"}
        for result in results
    )


def test_long_gap_emits_unknown_with_gap_reason() -> None:
    system = detector()
    before = synthetic_stream([("absent", 2.0)], seed=6)
    after = synthetic_stream([("absent", 2.0)], seed=7)
    resumed = [
        replace(sample, timestamp=sample.timestamp + 3.0)
        for sample in after
    ]
    results = list(run_samples(system, before + resumed))

    assert any(result.state == UNKNOWN and result.reason == "gap" for result in results)


def test_nan_sample_is_rejected_counted_and_does_not_predict_absent() -> None:
    system = detector()
    samples = synthetic_stream([("absent", 3.0)])
    faulty = inject_faults(samples, nan_rate=1.0, seed=22)
    outputs = list(run_samples(system, faulty))

    assert outputs == []
    stats = system.stats()
    assert stats["samples_received"] == len(samples)
    assert stats["samples_rejected"] == len(samples)
    assert stats["rejected_by_issue"][IssueCode.NON_FINITE_VALUE] == len(samples)
    assert stats["windows_processed"] == 0
    assert stats["unknown_results"] == 0


def test_stale_node_emits_unknown_with_reason() -> None:
    system = detector()
    samples = synthetic_stream([("absent", 0.5)])
    list(run_samples(system, samples))

    stale = system.check_stale(samples[-1].timestamp + 2.0)

    assert len(stale) == 1
    assert stale[0].state == UNKNOWN
    assert stale[0].reason == "stale_node"
    assert stale[0].input_quality["window_quality"] is None
    assert system.check_stale(samples[-1].timestamp + 3.0) == []


def test_nodes_keep_independent_window_and_smoothing_state() -> None:
    system = detector()
    absent = synthetic_stream([("absent", 4.0)], node_id="node-a")
    moving = synthetic_stream([("present_moving", 4.0)], node_id="node-b")
    interleaved = []
    for left, right in zip(absent, moving):
        interleaved.extend((left, right))
    results = list(run_samples(system, interleaved))

    by_node = {
        node_id: [result.state for result in results if result.node_id == node_id]
        for node_id in ("node-a", "node-b")
    }
    assert ABSENT in by_node["node-a"]
    assert PRESENT_MOVING in by_node["node-b"]


def test_classifier_error_marks_unknown() -> None:
    system = detector(classifier=VarianceClassifier(fail=True))
    results = list(
        run_samples(system, synthetic_stream([("absent", 4.0)]))
    )

    assert results
    assert all(result.state == UNKNOWN for result in results)
    assert all(result.reason == "stub failure" for result in results)


def test_stats_count_windows_unknowns_and_changes() -> None:
    system = detector(classifier=VarianceClassifier(fail=True))
    results = list(
        run_samples(system, synthetic_stream([("absent", 5.0)]))
    )
    stats = system.stats()

    assert stats["samples_received"] == 80
    assert stats["samples_rejected"] == 0
    assert stats["windows_processed"] == len(results) > 0
    assert stats["unknown_results"] == len(results)
    assert stats["state_changes"] == sum(result.changed for result in results)


def test_reset_clears_detector_and_windower_state() -> None:
    system = detector()
    list(run_samples(system, synthetic_stream([("absent", 4.0)])))

    system.reset()

    assert system.stats() == {
        "samples_received": 0,
        "samples_rejected": 0,
        "rejected_by_issue": {},
        "windows_processed": 0,
        "unknown_results": 0,
        "state_changes": 0,
    }
    assert system.windower.nodes_stale(1000.0, 1.0) == []
