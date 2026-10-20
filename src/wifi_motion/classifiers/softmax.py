"""Regresión logística softmax multiclase con numpy y persistencia JSON."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Sequence

import numpy as np

from wifi_motion.classifiers.base import (
	CLASS_STATES,
	ClassifierError,
	Prediction,
	check_feature_vector,
)
from wifi_motion.features import FEATURE_NAMES
from wifi_motion.normalization import NORMALIZER_VERSION, Normalizer

SOFTMAX_VERSION = 1
_RSSI_INDEX = FEATURE_NAMES.index("mean_rssi")
_MODEL_FIELDS = {
	"softmax_version",
	"weights",
	"bias",
	"classes",
	"normalizer",
	"rssi_median",
	"feature_names",
}

__all__ = [
	"SOFTMAX_VERSION",
	"SoftmaxClassifier",
	"SoftmaxConfig",
	"confusion_matrix",
	"macro_f1",
]


def _softmax(logits: np.ndarray) -> np.ndarray:
	"""Softmax estable al restar el máximo de cada fila."""
	values = np.asarray(logits, dtype=np.float64)
	with np.errstate(over="ignore", under="ignore", invalid="ignore"):
		shifted = values - np.max(values, axis=-1, keepdims=True)
		exponentials = np.exp(shifted)
		return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _is_finite_real(value: object) -> bool:
	if isinstance(value, bool) or not isinstance(value, Real):
		return False
	try:
		return math.isfinite(float(value))
	except (OverflowError, ValueError):
		return False


def _positive_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 1:
		raise ValueError(f"{name} debe ser un entero >= 1")
	return value


@dataclass(frozen=True, slots=True)
class SoftmaxConfig:
	"""Hiperparámetros del descenso de gradiente minibatch."""

	lr: float = 0.05
	epochs: int = 300
	batch_size: int = 32
	l2: float = 1e-4
	seed: int = 0
	class_weight: str | None = None
	patience: int = 30

	def __post_init__(self) -> None:
		if not _is_finite_real(self.lr) or self.lr <= 0:
			raise ValueError("lr debe ser un número finito > 0")
		if not _is_finite_real(self.l2) or self.l2 < 0:
			raise ValueError("l2 debe ser un número finito >= 0")
		_positive_int(self.epochs, "epochs")
		_positive_int(self.batch_size, "batch_size")
		_positive_int(self.patience, "patience")
		if isinstance(self.seed, bool) or not isinstance(self.seed, int):
			raise ValueError("seed debe ser un entero")
		if self.class_weight not in (None, "balanced"):
			raise ValueError("class_weight debe ser 'balanced' o None")
		object.__setattr__(self, "lr", float(self.lr))
		object.__setattr__(self, "l2", float(self.l2))


def _labels(values: Sequence[str], name: str, count: int) -> np.ndarray:
	if isinstance(values, str):
		raise ValueError(f"{name} debe ser una lista de etiquetas")
	try:
		labels = list(values)
	except TypeError as exc:
		raise ValueError(f"{name} debe ser una lista de etiquetas") from exc
	if len(labels) != count:
		raise ValueError(f"{name} debe tener {count} etiqueta(s), no {len(labels)}")
	unknown = [label for label in labels if not isinstance(label, str) or label not in CLASS_STATES]
	if unknown:
		raise ValueError(f"{name} contiene etiqueta(s) desconocida(s): {unknown!r}")
	return np.asarray(labels, dtype=str)


def _prepare_matrix(
	X: object,
	name: str,
	rssi_median: float | None = None,
) -> tuple[np.ndarray, float | None]:
	try:
		matrix = np.array(X, dtype=np.float64, copy=True)
	except (TypeError, ValueError, OverflowError) as exc:
		raise ClassifierError(f"{name} no se pudo convertir a números: {exc}") from exc
	if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
		raise ClassifierError(
			f"{name} debe tener forma (n, {len(FEATURE_NAMES)}) en el orden de "
			f"FEATURE_NAMES, no {matrix.shape}"
		)

	invalid = ~np.isfinite(matrix)
	invalid_rssi = invalid[:, _RSSI_INDEX] & ~np.isnan(matrix[:, _RSSI_INDEX])
	invalid[:, _RSSI_INDEX] = invalid_rssi
	if invalid.any():
		columns = np.flatnonzero(invalid.any(axis=0))
		bad_names = [FEATURE_NAMES[index] for index in columns]
		raise ClassifierError(f"{name} tiene valores no finitos en: {bad_names}")

	missing_rssi = np.isnan(matrix[:, _RSSI_INDEX])
	if rssi_median is None:
		finite_rssi = matrix[~missing_rssi, _RSSI_INDEX]
		rssi_median = float(np.median(finite_rssi)) if finite_rssi.size else 0.0
	matrix[missing_rssi, _RSSI_INDEX] = rssi_median
	return matrix, rssi_median


def _checked_metrics_inputs(
	y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
	class_names = tuple(classes)
	if not class_names or any(not isinstance(value, str) for value in class_names):
		raise ValueError("classes debe contener textos y no estar vacío")
	if len(set(class_names)) != len(class_names):
		raise ValueError("classes no puede contener duplicados")
	true_labels = list(y_true)
	predicted_labels = list(y_pred)
	if len(true_labels) != len(predicted_labels):
		raise ValueError("y_true e y_pred deben tener la misma longitud")
	if not true_labels:
		raise ValueError("y_true e y_pred no pueden estar vacíos")
	allowed = set(class_names)
	if any(label not in allowed for label in true_labels + predicted_labels):
		raise ValueError("y_true e y_pred deben contener solo clases indicadas en classes")
	return np.asarray(true_labels), np.asarray(predicted_labels), class_names


def confusion_matrix(
	y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> np.ndarray:
	"""Cuenta etiquetas; filas son clases reales y columnas clases predichas."""
	true_labels, predicted_labels, class_names = _checked_metrics_inputs(
		y_true, y_pred, classes
	)
	indices = {label: index for index, label in enumerate(class_names)}
	true_indices = np.asarray([indices[label] for label in true_labels], dtype=np.intp)
	predicted_indices = np.asarray(
		[indices[label] for label in predicted_labels], dtype=np.intp
	)
	size = len(class_names)
	return np.bincount(
		true_indices * size + predicted_indices,
		minlength=size * size,
	).reshape((size, size))


def macro_f1(
	y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> float:
	"""F1 macro calculado a mano; una clase sin predicciones tiene F1 cero."""
	matrix = confusion_matrix(y_true, y_pred, classes)
	true_positive = np.diag(matrix).astype(np.float64)
	predicted = matrix.sum(axis=0)
	actual = matrix.sum(axis=1)
	denominator = predicted + actual
	class_f1 = np.divide(
		2.0 * true_positive,
		denominator,
		out=np.zeros_like(true_positive),
		where=denominator != 0,
	)
	return float(np.mean(class_f1))


class SoftmaxClassifier:
	"""Modelo softmax lineal; sus scores no son probabilidades calibradas."""

	name = "softmax"

	def __init__(self) -> None:
		self.weights: np.ndarray | None = None
		self.bias: np.ndarray | None = None
		self.classes: tuple[str, ...] = CLASS_STATES
		self.normalizer: Normalizer | None = None
		self.rssi_median: float | None = None
		self.history: list[dict[str, float | int]] = []
		self.config: SoftmaxConfig | None = None

	@staticmethod
	def _loss(
		X: np.ndarray,
		y: np.ndarray,
		weights: np.ndarray,
		bias: np.ndarray,
		sample_weights: np.ndarray,
		l2: float,
	) -> float:
		logits = X @ weights + bias
		max_logits = np.max(logits, axis=1, keepdims=True)
		with np.errstate(over="ignore", under="ignore", invalid="ignore"):
			log_partition = max_logits[:, 0] + np.log(
				np.exp(logits - max_logits).sum(axis=1)
			)
		log_probability = logits[np.arange(len(y)), y] - log_partition
		cross_entropy = -np.dot(sample_weights, log_probability) / sample_weights.sum()
		return float(cross_entropy + 0.5 * l2 * np.sum(weights * weights))

	def fit(
		self,
		X_train: np.ndarray,
		y_train: Sequence[str],
		X_val: np.ndarray,
		y_val: Sequence[str],
		config: SoftmaxConfig,
	) -> SoftmaxClassifier:
		"""Entrena con train y usa val solo para early stopping y métricas.

		Nunca pase aquí el conjunto de prueba final: se reserva para una
		evaluación única después de fijar el modelo y los hiperparámetros.
		"""
		if not isinstance(config, SoftmaxConfig):
			raise ValueError("config debe ser SoftmaxConfig")
		train_matrix, rssi_median = _prepare_matrix(X_train, "X_train")
		val_matrix, _ = _prepare_matrix(X_val, "X_val", rssi_median)
		if train_matrix.shape[0] == 0:
			raise ValueError("X_train no puede estar vacío")
		if val_matrix.shape[0] == 0:
			raise ValueError("X_val no puede estar vacío")
		train_labels = _labels(y_train, "y_train", len(train_matrix))
		val_labels = _labels(y_val, "y_val", len(val_matrix))
		missing = [state for state in CLASS_STATES if state not in train_labels]
		if missing:
			raise ValueError(f"falta la clase requerida en train: {missing[0]}")

		class_indices = {state: index for index, state in enumerate(CLASS_STATES)}
		encoded_train = np.asarray([class_indices[label] for label in train_labels])
		encoded_val = np.asarray([class_indices[label] for label in val_labels])

		normalizer = Normalizer()
		try:
			normalized_train = normalizer.fit_transform(train_matrix, FEATURE_NAMES)
			normalized_val = normalizer.transform(val_matrix)
		except ValueError as exc:
			raise ClassifierError(f"no se pudieron normalizar los datos: {exc}") from exc

		class_counts = np.bincount(encoded_train, minlength=len(CLASS_STATES))
		if config.class_weight == "balanced":
			class_weights = len(encoded_train) / (len(CLASS_STATES) * class_counts)
			train_sample_weights = class_weights[encoded_train]
			val_sample_weights = class_weights[encoded_val]
		else:
			train_sample_weights = np.ones(len(encoded_train), dtype=np.float64)
			val_sample_weights = np.ones(len(encoded_val), dtype=np.float64)

		rng = np.random.default_rng(config.seed)
		weights = rng.normal(0.0, 0.01, size=(len(FEATURE_NAMES), len(CLASS_STATES)))
		bias = np.zeros(len(CLASS_STATES), dtype=np.float64)
		best_weights = weights.copy()
		best_bias = bias.copy()
		best_val_loss = math.inf
		epochs_without_improvement = 0
		history: list[dict[str, float | int]] = []
		one_hot = np.eye(len(CLASS_STATES), dtype=np.float64)[encoded_train]

		for epoch in range(1, config.epochs + 1):
			order = rng.permutation(len(normalized_train))
			for start in range(0, len(order), config.batch_size):
				batch_indices = order[start : start + config.batch_size]
				batch_X = normalized_train[batch_indices]
				batch_weights = train_sample_weights[batch_indices]
				logits = batch_X @ weights + bias
				probabilities = _softmax(logits)
				delta = (probabilities - one_hot[batch_indices]) * batch_weights[:, None]
				denominator = batch_weights.sum()
				gradient_weights = batch_X.T @ delta / denominator + config.l2 * weights
				gradient_bias = delta.sum(axis=0) / denominator
				weights -= config.lr * gradient_weights
				bias -= config.lr * gradient_bias

			train_loss = self._loss(
				normalized_train,
				encoded_train,
				weights,
				bias,
				train_sample_weights,
				config.l2,
			)
			val_loss = self._loss(
				normalized_val,
				encoded_val,
				weights,
				bias,
				val_sample_weights,
				config.l2,
			)
			val_predictions = np.argmax(normalized_val @ weights + bias, axis=1)
			val_macro_f1 = macro_f1(
				val_labels.tolist(),
				[CLASS_STATES[index] for index in val_predictions],
				CLASS_STATES,
			)
			history.append(
				{
					"epoch": epoch,
					"train_loss": train_loss,
					"val_loss": val_loss,
					"val_macro_f1": val_macro_f1,
				}
			)

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				best_weights = weights.copy()
				best_bias = bias.copy()
				epochs_without_improvement = 0
			else:
				epochs_without_improvement += 1
				if epochs_without_improvement >= config.patience:
					break

		self.weights = best_weights
		self.bias = best_bias
		self.classes = CLASS_STATES
		self.normalizer = normalizer
		self.rssi_median = rssi_median
		self.history = history
		self.config = config
		return self

	def _transformed(self, X: object) -> np.ndarray:
		if self.weights is None or self.bias is None or self.normalizer is None:
			raise ClassifierError("SoftmaxClassifier no está entrenado ni cargado")
		matrix, _ = _prepare_matrix(X, "features", self.rssi_median)
		try:
			return self.normalizer.transform(matrix)
		except ValueError as exc:
			raise ClassifierError(f"no se pudieron normalizar las características: {exc}") from exc

	def predict(self, features: np.ndarray) -> Prediction:
		"""Predice un vector crudo; los scores softmax no están calibrados."""
		check_feature_vector(features)
		transformed = self._transformed(np.asarray(features, dtype=np.float64).reshape(1, -1))
		logits = transformed @ self.weights + self.bias
		scores = _softmax(logits)[0]
		if not np.all(np.isfinite(scores)):
			raise ClassifierError("softmax produjo scores no finitos")
		index = int(np.argmax(scores))
		state = self.classes[index]
		score_map = {label: float(scores[i]) for i, label in enumerate(self.classes)}
		return Prediction(state, score_map, score_map[state], self.name)

	def predict_batch(self, X: np.ndarray) -> list[Prediction]:
		"""Predice una matriz de vectores crudos en el orden de FEATURE_NAMES."""
		transformed = self._transformed(X)
		logits = transformed @ self.weights + self.bias
		scores = _softmax(logits)
		if not np.all(np.isfinite(scores)):
			raise ClassifierError("softmax produjo scores no finitos")
		indices = np.argmax(scores, axis=1)
		predictions: list[Prediction] = []
		for row, index in zip(scores, indices):
			state = self.classes[int(index)]
			score_map = {label: float(row[i]) for i, label in enumerate(self.classes)}
			predictions.append(Prediction(state, score_map, score_map[state], self.name))
		return predictions

	def save(self, path: str | Path) -> None:
		"""Guarda pesos y preprocesamiento en un archivo JSON versionado."""
		if (
			self.weights is None
			or self.bias is None
			or self.normalizer is None
			or self.rssi_median is None
			or self.normalizer.mean is None
			or self.normalizer.std is None
			or self.normalizer.feature_names is None
		):
			raise ClassifierError("SoftmaxClassifier no está entrenado ni cargado")
		record = {
			"softmax_version": SOFTMAX_VERSION,
			"weights": self.weights.tolist(),
			"bias": self.bias.tolist(),
			"classes": list(self.classes),
			"normalizer": {
				"normalizer_version": NORMALIZER_VERSION,
				"feature_names": list(self.normalizer.feature_names),
				"mean": self.normalizer.mean.tolist(),
				"std": self.normalizer.std.tolist(),
				"constant_features": list(self.normalizer.constant_features),
			},
			"rssi_median": self.rssi_median,
			"feature_names": list(FEATURE_NAMES),
		}
		Path(path).write_text(
			json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
			encoding="utf-8",
		)

	@classmethod
	def load(cls, path: str | Path) -> SoftmaxClassifier:
		"""Carga el JSON del modelo y valida versión, campos y dimensiones."""
		try:
			record = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
		except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
			raise ValueError(f"modelo JSON inválido: {exc}") from exc
		if not isinstance(record, dict):
			raise ValueError("el modelo debe ser un objeto JSON")
		missing = _MODEL_FIELDS - record.keys()
		unknown = record.keys() - _MODEL_FIELDS
		if missing or unknown:
			details = []
			if missing:
				details.append(f"faltan campos: {sorted(missing)}")
			if unknown:
				details.append(f"campos desconocidos: {sorted(unknown)}")
			raise ValueError("; ".join(details))
		version = record["softmax_version"]
		if isinstance(version, bool) or not isinstance(version, int) or version != SOFTMAX_VERSION:
			raise ValueError(f"softmax_version no soportada: {version!r}")
		if record["feature_names"] != list(FEATURE_NAMES):
			raise ValueError("feature_names no coincide con FEATURE_NAMES")
		if record["classes"] != list(CLASS_STATES):
			raise ValueError("classes debe coincidir con CLASS_STATES y su orden")
		weights = _json_array(record["weights"], (len(FEATURE_NAMES), len(CLASS_STATES)), "weights")
		bias = _json_array(record["bias"], (len(CLASS_STATES),), "bias")
		rssi_median = record["rssi_median"]
		if not _is_finite_real(rssi_median):
			raise ValueError("rssi_median debe ser un número finito")
		normalizer_record = record["normalizer"]
		if not isinstance(normalizer_record, dict):
			raise ValueError("normalizer debe ser un objeto JSON")
		try:
			normalizer = Normalizer._from_record(normalizer_record, "normalizer")
		except (KeyError, TypeError, ValueError) as exc:
			raise ValueError(f"normalizer inválido: {exc}") from exc
		if normalizer.feature_names != FEATURE_NAMES:
			raise ValueError("normalizer.feature_names no coincide con FEATURE_NAMES")

		model = cls()
		model.weights = weights
		model.bias = bias
		model.classes = CLASS_STATES
		model.normalizer = normalizer
		model.rssi_median = float(rssi_median)
		return model


def _json_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
	def values_are_numeric(item: object) -> bool:
		if isinstance(item, list):
			return all(values_are_numeric(child) for child in item)
		return _is_finite_real(item)

	if not isinstance(value, list) or not values_are_numeric(value):
		raise ValueError(f"{name} debe contener solo números finitos")
	array = np.asarray(value, dtype=np.float64)
	if array.shape != shape:
		raise ValueError(f"{name} debe tener forma {shape}, no {array.shape}")
	return array


def _reject_constant(name: str) -> None:
	raise ValueError(f"constante JSON no permitida: {name}")
