"""Shared prediction and input-validation contracts for classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from wifi_motion.features import FEATURE_NAMES

ABSENT = "absent"
PRESENT_STILL = "present_still"
PRESENT_MOVING = "present_moving"
UNKNOWN = "unknown"

CLASS_STATES: tuple[str, ...] = (ABSENT, PRESENT_STILL, PRESENT_MOVING)

__all__ = [
    "ABSENT",
    "CLASS_STATES",
    "PRESENT_MOVING",
    "PRESENT_STILL",
    "UNKNOWN",
    "Classifier",
    "ClassifierError",
    "Prediction",
    "check_feature_vector",
]


@dataclass(frozen=True, slots=True)
class Prediction:
    """Resultado de clasificación.

    ``scores`` y ``confidence`` NO son probabilidades calibradas salvo que se
    haya medido y calibrado explícitamente.
    """

    state: str
    scores: dict[str, float]
    confidence: float
    source: str


class ClassifierError(ValueError):
    """Una entrada no cumple el contrato de un clasificador."""


class Classifier(Protocol):
    """Contrato común: los vectores de entrada son características crudas."""

    name: str

    def predict(self, features: np.ndarray) -> Prediction:
        """Clasifica un vector crudo en el orden de ``FEATURE_NAMES``."""
        ...

    def predict_batch(self, X: np.ndarray) -> list[Prediction]:
        """Clasifica una matriz de vectores crudos."""
        ...


def check_feature_vector(features: object) -> None:
    """Valida un vector de 15 características; solo ``mean_rssi`` admite NaN."""
    try:
        vector = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ClassifierError(f"features no se pudo convertir a números: {exc}") from exc

    if vector.shape != (len(FEATURE_NAMES),):
        raise ClassifierError(
            f"features debe tener forma ({len(FEATURE_NAMES)},) en el orden de "
            f"FEATURE_NAMES, no {vector.shape}"
        )

    bad = ~np.isfinite(vector)
    rssi_index = FEATURE_NAMES.index("mean_rssi")
    if np.isnan(vector[rssi_index]):
        bad[rssi_index] = False
    if bad.any():
        invalid = [FEATURE_NAMES[i] for i in np.flatnonzero(bad)]
        raise ClassifierError(f"features tiene valores no finitos en: {invalid}")