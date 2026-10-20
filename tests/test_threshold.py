"""Pruebas SYNTHETIC del clasificador por umbrales."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_motion.calibration import MAD_TO_SIGMA, Baseline, fit_baseline
from wifi_motion.classifiers.base import (
    ABSENT,
    PRESENT_MOVING,
    PRESENT_STILL,
    ClassifierError,
)
from wifi_motion.classifiers.threshold import (
    ThresholdClassifier,
    ThresholdConfig,
    tune_thresholds,
)
from wifi_motion.features import FEATURE_NAMES


def synthetic_baseline() -> Baseline:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, len(FEATURE_NAMES)))
    return fit_baseline("synthetic-node", X, "synthetic empty room")


def shifted(baseline: Baseline, name: str, z: float) -> np.ndarray:
    features = baseline.median.copy()
    index = FEATURE_NAMES.index(name)
    features[index] += z * MAD_TO_SIGMA * baseline.mad[index]
    return features


def test_empty_room_is_absent() -> None:
    baseline = synthetic_baseline()
    prediction = ThresholdClassifier(baseline, ThresholdConfig()).predict(baseline.median)
    assert prediction.state == ABSENT
    assert prediction.confidence == prediction.scores[prediction.state]


def test_large_motion_deviation_is_present_moving() -> None:
    baseline = synthetic_baseline()
    classifier = ThresholdClassifier(baseline, ThresholdConfig())
    prediction = classifier.predict(shifted(baseline, "motion_band_power", 8.0))
    assert prediction.state == PRESENT_MOVING
    assert prediction.confidence == prediction.scores[PRESENT_MOVING]


def test_breathing_deviation_alone_is_present_still() -> None:
    baseline = synthetic_baseline()
    classifier = ThresholdClassifier(baseline, ThresholdConfig())
    prediction = classifier.predict(shifted(baseline, "breathing_band_power", 8.0))
    assert prediction.state == PRESENT_STILL
    assert prediction.confidence == prediction.scores[PRESENT_STILL]


def test_empty_presence_features_never_emits_present_still() -> None:
    baseline = synthetic_baseline()
    config = ThresholdConfig(presence_features=())
    classifier = ThresholdClassifier(baseline, config)
    assert classifier.predict(shifted(baseline, "breathing_band_power", 8.0)).state == ABSENT
    assert classifier.predict(shifted(baseline, "motion_band_power", 8.0)).state == PRESENT_MOVING


def test_non_finite_feature_is_rejected() -> None:
    baseline = synthetic_baseline()
    features = baseline.median.copy()
    features[FEATURE_NAMES.index("variance")] = np.nan
    with pytest.raises(ClassifierError):
        ThresholdClassifier(baseline, ThresholdConfig()).predict(features)


def test_tuning_improves_macro_f1_on_separable_validation_data() -> None:
    baseline = synthetic_baseline()
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for _ in range(5):
        rows.append(baseline.median.copy())
        labels.append(ABSENT)
        rows.append(shifted(baseline, "breathing_band_power", 2.0))
        labels.append(PRESENT_STILL)
        rows.append(shifted(baseline, "motion_band_power", 2.0))
        labels.append(PRESENT_MOVING)

    X_val = np.vstack(rows)
    default_predictions = ThresholdClassifier(baseline, ThresholdConfig()).predict_batch(X_val)
    default_f1 = _macro_f1(labels, [prediction.state for prediction in default_predictions])
    tuned = tune_thresholds(
        baseline,
        X_val,
        labels,
        motion_grid=(0.5, 3.0),
        presence_grid=(0.5, 3.0),
    )
    tuned_predictions = ThresholdClassifier(baseline, tuned).predict_batch(X_val)
    tuned_f1 = _macro_f1(labels, [prediction.state for prediction in tuned_predictions])
    assert tuned_f1 > default_f1
    assert tuned_f1 == pytest.approx(1.0)


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    classes = (ABSENT, PRESENT_STILL, PRESENT_MOVING)
    class_scores = []
    for state in classes:
        true_positive = sum(
            actual == state and predicted == state
            for actual, predicted in zip(y_true, y_pred)
        )
        false_positive = sum(
            actual != state and predicted == state
            for actual, predicted in zip(y_true, y_pred)
        )
        false_negative = sum(
            actual == state and predicted != state
            for actual, predicted in zip(y_true, y_pred)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(class_scores))