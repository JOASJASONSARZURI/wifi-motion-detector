"""Pruebas del contrato base de clasificadores."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from wifi_motion.classifiers.base import (
    ABSENT,
    CLASS_STATES,
    ClassifierError,
    Prediction,
    check_feature_vector,
)
from wifi_motion.features import FEATURE_NAMES


def test_prediction_is_immutable() -> None:
    prediction = Prediction(
        state=ABSENT,
        scores={state: 0.0 for state in CLASS_STATES},
        confidence=0.0,
        source="test",
    )
    with pytest.raises(FrozenInstanceError):
        prediction.state = "unknown"  # type: ignore[misc]


def test_check_feature_vector_rejects_wrong_length() -> None:
    with pytest.raises(ClassifierError, match="forma"):
        check_feature_vector(np.zeros(len(FEATURE_NAMES) - 1))


def test_check_feature_vector_rejects_nan_outside_mean_rssi() -> None:
    features = np.zeros(len(FEATURE_NAMES))
    features[0] = np.nan
    with pytest.raises(ClassifierError, match="variance"):
        check_feature_vector(features)


def test_check_feature_vector_allows_nan_in_mean_rssi() -> None:
    features = np.zeros(len(FEATURE_NAMES))
    features[FEATURE_NAMES.index("mean_rssi")] = np.nan
    check_feature_vector(features)