"""EMA and per-node debounce for classifier predictions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping

import numpy as np

from wifi_motion.classifiers.base import CLASS_STATES, UNKNOWN, Prediction

__all__ = [
	"SmoothedResult",
	"SmoothingConfig",
	"StateSmoother",
	"worst_case_latency_updates",
]


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
	"""Parámetros de EMA y confirmación consecutiva de estados."""

	ema_alpha: float = 0.4
	confirm_n: int = 3
	confirm_n_to_absent: int = 5
	margin: float = 0.05

	def __post_init__(self) -> None:
		if (
			isinstance(self.ema_alpha, bool)
			or not isinstance(self.ema_alpha, Real)
			or not isfinite(float(self.ema_alpha))
			or not 0.0 < self.ema_alpha <= 1.0
		):
			raise ValueError("ema_alpha debe estar en (0, 1]")
		for name in ("confirm_n", "confirm_n_to_absent"):
			value = getattr(self, name)
			if isinstance(value, bool) or not isinstance(value, int) or value < 1:
				raise ValueError(f"{name} debe ser un entero >= 1")
		if (
			isinstance(self.margin, bool)
			or not isinstance(self.margin, Real)
			or not isfinite(float(self.margin))
			or self.margin < 0.0
		):
			raise ValueError("margin debe ser un número finito >= 0")
		object.__setattr__(self, "ema_alpha", float(self.ema_alpha))
		object.__setattr__(self, "margin", float(self.margin))


@dataclass(frozen=True, slots=True)
class SmoothedResult:
	"""Resultado inmutable de una actualización del suavizador."""

	node_id: str
	timestamp: float
	state: str
	scores: Mapping[str, float]
	changed: bool
	pending_state: str | None
	pending_count: int
	reason: str | None

	def __post_init__(self) -> None:
		object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


@dataclass(slots=True)
class _NodeState:
	scores: np.ndarray
	state: str
	pending_state: str | None = None
	pending_count: int = 0


def _checked_timestamp(timestamp: object) -> float:
	if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
		raise ValueError("timestamp debe ser un número finito")
	value = float(timestamp)
	if not isfinite(value):
		raise ValueError("timestamp debe ser un número finito")
	return value


def _checked_node_id(node_id: object) -> str:
	if not isinstance(node_id, str) or not node_id.strip():
		raise ValueError("node_id debe ser un texto no vacío")
	return node_id


class StateSmoother:
	"""Suaviza scores y aplica histéresis independiente para cada nodo."""

	def __init__(self, config: SmoothingConfig) -> None:
		if not isinstance(config, SmoothingConfig):
			raise TypeError("config debe ser SmoothingConfig")
		self.config = config
		self._nodes: dict[str, _NodeState] = {}

	@staticmethod
	def _checked_scores(prediction: Prediction) -> np.ndarray:
		if not isinstance(prediction, Prediction):
			raise TypeError("prediction debe ser Prediction")
		if prediction.state not in CLASS_STATES:
			raise ValueError(f"prediction.state debe pertenecer a CLASS_STATES, no {prediction.state!r}")
		if set(prediction.scores) != set(CLASS_STATES):
			raise ValueError("prediction.scores debe contener exactamente CLASS_STATES")
		try:
			scores = np.asarray([prediction.scores[state] for state in CLASS_STATES], dtype=np.float64)
		except (TypeError, ValueError, OverflowError) as exc:
			raise ValueError(f"prediction.scores debe contener números: {exc}") from exc
		if not np.all(np.isfinite(scores)):
			raise ValueError("prediction.scores debe contener solo valores finitos")
		return scores

	@staticmethod
	def _result(
		node_id: str,
		timestamp: float,
		state: str,
		scores: np.ndarray,
		changed: bool,
		pending_state: str | None,
		pending_count: int,
		reason: str | None = None,
	) -> SmoothedResult:
		score_map = {name: float(scores[index]) for index, name in enumerate(CLASS_STATES)}
		return SmoothedResult(
			node_id=node_id,
			timestamp=timestamp,
			state=state,
			scores=score_map,
			changed=changed,
			pending_state=pending_state,
			pending_count=pending_count,
			reason=reason,
		)

	def update(self, node_id: str, timestamp: float, prediction: Prediction) -> SmoothedResult:
		"""Actualiza EMA y confirma el estado candidato tras su debounce."""
		checked_node = _checked_node_id(node_id)
		checked_time = _checked_timestamp(timestamp)
		incoming = self._checked_scores(prediction)
		current = self._nodes.get(checked_node)
		if current is None:
			state = CLASS_STATES[int(np.argmax(incoming))]
			current = _NodeState(scores=incoming.copy(), state=state)
			self._nodes[checked_node] = current
			return self._result(checked_node, checked_time, state, current.scores, True, None, 0)

		alpha = self.config.ema_alpha
		current.scores = alpha * incoming + (1.0 - alpha) * current.scores
		candidate = CLASS_STATES[int(np.argmax(current.scores))]
		changed = False

		if candidate == current.state:
			current.pending_state = None
			current.pending_count = 0
		else:
			if candidate == current.pending_state:
				current.pending_count += 1
			else:
				current.pending_state = candidate
				current.pending_count = 1

			required = (
				self.config.confirm_n_to_absent
				if candidate == CLASS_STATES[0]
				else self.config.confirm_n
			)
			candidate_index = CLASS_STATES.index(candidate)
			state_index = CLASS_STATES.index(current.state)
			margin_cleared = (
				current.scores[candidate_index] - current.scores[state_index]
				>= self.config.margin
			)
			if current.pending_count >= required and margin_cleared:
				current.state = candidate
				current.pending_state = None
				current.pending_count = 0
				changed = True

		return self._result(
			checked_node,
			checked_time,
			current.state,
			current.scores,
			changed,
			current.pending_state,
			current.pending_count,
		)

	def mark_unknown(self, node_id: str, timestamp: float, reason: str) -> SmoothedResult:
		"""Marca UNKNOWN de inmediato y elimina cualquier EMA/debounce del nodo."""
		checked_node = _checked_node_id(node_id)
		checked_time = _checked_timestamp(timestamp)
		if not isinstance(reason, str):
			raise ValueError("reason debe ser texto")
		previous = self._nodes.pop(checked_node, None)
		empty_scores = np.zeros(len(CLASS_STATES), dtype=np.float64)
		return self._result(
			checked_node,
			checked_time,
			UNKNOWN,
			empty_scores,
			previous is None or previous.state != UNKNOWN,
			None,
			0,
			reason,
		)

	def reset(self, node_id: str | None = None) -> None:
		"""Borra el historial de un nodo o el de todos los nodos."""
		if node_id is None:
			self._nodes.clear()
			return
		self._nodes.pop(_checked_node_id(node_id), None)


def worst_case_latency_updates(config: SmoothingConfig, destination: str) -> int:
	"""Actualizaciones consecutivas necesarias si el margen ya está superado."""
	if not isinstance(config, SmoothingConfig):
		raise TypeError("config debe ser SmoothingConfig")
	if destination not in CLASS_STATES:
		raise ValueError(f"destination debe pertenecer a CLASS_STATES, no {destination!r}")
	if destination == CLASS_STATES[0]:
		return config.confirm_n_to_absent
	return config.confirm_n
