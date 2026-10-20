"""Clasificador por umbrales de desvío respecto a una línea base robusta.

Los umbrales iniciales son valores a ajustar con datos medidos, no umbrales
universales. ``present_still`` por umbral es evidencia débil; la ausencia de
movimiento por sí sola NO implica presencia quieta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence

import numpy as np

from wifi_motion.calibration import Baseline, CalibrationError, deviation
from wifi_motion.classifiers.base import (
	ABSENT,
	CLASS_STATES,
	PRESENT_MOVING,
	PRESENT_STILL,
	ClassifierError,
	Prediction,
	check_feature_vector,
)
from wifi_motion.features import FEATURE_NAMES

__all__ = ["ThresholdClassifier", "ThresholdConfig", "tune_thresholds"]


def _checked_feature_names(value: object, label: str, allow_empty: bool) -> tuple[str, ...]:
	if isinstance(value, str) or not isinstance(value, (tuple, list)):
		raise ValueError(f"{label} debe ser una tupla de nombres de FEATURE_NAMES")
	names = tuple(value)
	if not names and not allow_empty:
		raise ValueError(f"{label} no puede estar vacío")
	if any(not isinstance(name, str) or name not in FEATURE_NAMES for name in names):
		raise ValueError(f"{label} solo admite nombres de FEATURE_NAMES")
	if len(set(names)) != len(names):
		raise ValueError(f"{label} no puede contener nombres repetidos")
	return names


def _checked_threshold(value: object, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		raise ValueError(f"{label} debe ser un número finito > 0, no {value!r}")
	threshold = float(value)
	if not math.isfinite(threshold) or threshold <= 0.0:
		raise ValueError(f"{label} debe ser un número finito > 0, no {value!r}")
	return threshold


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
	"""Características y umbrales iniciales a ajustar con datos medidos.

	``motion_z`` y ``presence_z`` son valores iniciales a ajustar con datos
	medidos, no umbrales universales. ``presence_features=()`` activa el modo
	"solo movimiento": el clasificador nunca emite ``present_still``.
	"""

	motion_features: tuple[str, ...] = (
		"motion_band_power",
		"variance",
		"change_points",
	)
	presence_features: tuple[str, ...] = ("breathing_band_power",)
	motion_z: float = 3.0
	presence_z: float = 3.0

	def __post_init__(self) -> None:
		object.__setattr__(
			self,
			"motion_features",
			_checked_feature_names(self.motion_features, "motion_features", allow_empty=False),
		)
		object.__setattr__(
			self,
			"presence_features",
			_checked_feature_names(self.presence_features, "presence_features", allow_empty=True),
		)
		object.__setattr__(self, "motion_z", _checked_threshold(self.motion_z, "motion_z"))
		object.__setattr__(self, "presence_z", _checked_threshold(self.presence_z, "presence_z"))


class ThresholdClassifier:
	"""Clasifica vectores crudos con el desvío robusto de su línea base.

	Para cada grupo, su score es ``clip(max(|z|) / (2 * umbral), 0, 1)``;
	``absent`` recibe ``1 - max(score_motion, score_presence)``. Son márgenes
	normalizados, no probabilidades. ``confidence`` copia el score de la clase
	elegida.

	``present_still`` por umbral es evidencia débil; la ausencia de movimiento
	por sí sola NO implica presencia quieta.
	"""

	name = "threshold"

	def __init__(self, baseline: Baseline, config: ThresholdConfig) -> None:
		if not isinstance(baseline, Baseline):
			raise ClassifierError("baseline debe ser Baseline")
		if not isinstance(config, ThresholdConfig):
			raise ClassifierError("config debe ser ThresholdConfig")
		self.baseline = baseline
		self.config = config

	def _max_abs_deviation(self, scores: np.ndarray, names: tuple[str, ...]) -> float:
		if not names:
			return 0.0
		indices = [FEATURE_NAMES.index(name) for name in names]
		selected = scores[indices]
		if np.isnan(selected).any():
			missing = [name for name, value in zip(names, selected) if np.isnan(value)]
			raise ClassifierError(f"no se puede calcular el desvío de: {missing}")
		return float(np.max(np.abs(selected)))

	@staticmethod
	def _margin_score(max_abs_z: float, threshold: float) -> float:
		return float(np.clip(max_abs_z / (2.0 * threshold), 0.0, 1.0))

	def predict(self, features: np.ndarray) -> Prediction:
		"""Clasifica un vector crudo de 15 valores en el orden de FEATURE_NAMES."""
		check_feature_vector(features)
		try:
			z_scores = deviation(self.baseline, features)
		except CalibrationError as exc:
			raise ClassifierError(f"no se pudo calcular el desvío: {exc}") from exc

		motion_deviation = self._max_abs_deviation(z_scores, self.config.motion_features)
		presence_deviation = self._max_abs_deviation(z_scores, self.config.presence_features)
		motion_score = self._margin_score(motion_deviation, self.config.motion_z)
		presence_score = self._margin_score(presence_deviation, self.config.presence_z)
		scores = {
			ABSENT: 1.0 - max(motion_score, presence_score),
			PRESENT_STILL: presence_score,
			PRESENT_MOVING: motion_score,
		}

		if motion_deviation > self.config.motion_z:
			state = PRESENT_MOVING
		elif self.config.presence_features and presence_deviation > self.config.presence_z:
			state = PRESENT_STILL
		else:
			state = ABSENT

		return Prediction(
			state=state,
			scores=scores,
			confidence=scores[state],
			source=self.name,
		)

	def predict_batch(self, X: np.ndarray) -> list[Prediction]:
		"""Clasifica cada fila de una matriz de vectores crudos."""
		try:
			matrix = np.asarray(X, dtype=np.float64)
		except (TypeError, ValueError, OverflowError) as exc:
			raise ClassifierError(f"X no se pudo convertir a números: {exc}") from exc
		if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
			raise ClassifierError(
				f"X debe tener forma (n, {len(FEATURE_NAMES)}), no {matrix.shape}"
			)
		return [self.predict(row) for row in matrix]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Calcula F1 macro de las tres clases; F1 de clase vacía vale cero."""
	state_index = {state: index for index, state in enumerate(CLASS_STATES)}
	true_indices = np.asarray([state_index[state] for state in y_true], dtype=np.intp)
	pred_indices = np.asarray([state_index[state] for state in y_pred], dtype=np.intp)
	confusion = np.bincount(
		true_indices * len(CLASS_STATES) + pred_indices,
		minlength=len(CLASS_STATES) ** 2,
	).reshape((len(CLASS_STATES), len(CLASS_STATES)))
	true_positive = np.diag(confusion)
	predicted_count = confusion.sum(axis=0)
	actual_count = confusion.sum(axis=1)
	precision = np.divide(
		true_positive,
		predicted_count,
		out=np.zeros(len(CLASS_STATES), dtype=np.float64),
		where=predicted_count != 0,
	)
	recall = np.divide(
		true_positive,
		actual_count,
		out=np.zeros(len(CLASS_STATES), dtype=np.float64),
		where=actual_count != 0,
	)
	f1 = np.divide(
		2.0 * precision * recall,
		precision + recall,
		out=np.zeros(len(CLASS_STATES), dtype=np.float64),
		where=(precision + recall) != 0,
	)
	return float(np.mean(f1))


def tune_thresholds(
	baseline: Baseline,
	X_val: np.ndarray,
	y_val: Sequence[str],
	motion_grid: Iterable[float],
	presence_grid: Iterable[float],
) -> ThresholdConfig:
	"""Elige umbrales por F1 macro usando exclusivamente los datos recibidos.

	``X_val`` y ``y_val`` deben ser datos de validación; nunca se debe pasar el
	conjunto de prueba final, que se reserva para una evaluación única.
	"""
	if not isinstance(baseline, Baseline):
		raise ClassifierError("baseline debe ser Baseline")
	try:
		matrix = np.asarray(X_val, dtype=np.float64)
	except (TypeError, ValueError, OverflowError) as exc:
		raise ClassifierError(f"X_val no se pudo convertir a números: {exc}") from exc
	if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
		raise ClassifierError(
			f"X_val debe tener forma (n, {len(FEATURE_NAMES)}), no {matrix.shape}"
		)
	if matrix.shape[0] == 0:
		raise ClassifierError("X_val no puede estar vacío")

	labels = np.asarray(y_val, dtype=object)
	if labels.ndim != 1 or len(labels) != len(matrix):
		raise ClassifierError("y_val debe tener una etiqueta por fila de X_val")
	if any(not isinstance(label, str) or label not in CLASS_STATES for label in labels):
		raise ClassifierError(f"y_val solo admite etiquetas de {CLASS_STATES}")
	for row in matrix:
		check_feature_vector(row)

	try:
		motion_values = tuple(motion_grid)
		presence_values = tuple(presence_grid)
	except TypeError as exc:
		raise ValueError("motion_grid y presence_grid deben ser iterables") from exc
	if not motion_values or not presence_values:
		raise ValueError("motion_grid y presence_grid no pueden estar vacías")

	best_config: ThresholdConfig | None = None
	best_f1 = -1.0
	for motion_z in motion_values:
		for presence_z in presence_values:
			config = ThresholdConfig(motion_z=motion_z, presence_z=presence_z)
			classifier = ThresholdClassifier(baseline, config)
			predictions = classifier.predict_batch(matrix)
			predicted_states = np.asarray([prediction.state for prediction in predictions])
			score = _macro_f1(labels, predicted_states)
			if score > best_f1:
				best_config = config
				best_f1 = score

	assert best_config is not None
	return best_config
