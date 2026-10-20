"""Pruebas SYNTHETIC del suavizado de estado, sin datos de sensores."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from wifi_motion.classifiers.base import ABSENT, CLASS_STATES, PRESENT_MOVING, PRESENT_STILL, UNKNOWN, Prediction
from wifi_motion.smoothing import (
	SmoothedResult,
	SmoothingConfig,
	StateSmoother,
	worst_case_latency_updates,
)


def prediction(state: str, **scores: float) -> Prediction:
	values = {label: 0.05 for label in CLASS_STATES}
	values.update(scores)
	if not scores:
		values[state] = 0.9
	return Prediction(state, values, values[state], "synthetic")


def test_isolated_peak_does_not_change_state() -> None:
	smoother = StateSmoother(SmoothingConfig(ema_alpha=1.0, confirm_n=2, margin=0.0))
	assert smoother.update("node", 0.0, prediction(ABSENT)).state == ABSENT
	peak = smoother.update("node", 1.0, prediction(PRESENT_MOVING))
	recovery = smoother.update("node", 2.0, prediction(ABSENT))
	assert peak.state == recovery.state == ABSENT
	assert peak.pending_count == 1
	assert recovery.pending_state is None


def test_sustained_change_commits_exactly_at_confirm_n() -> None:
	smoother = StateSmoother(SmoothingConfig(ema_alpha=1.0, confirm_n=3, margin=0.0))
	smoother.update("node", 0.0, prediction(ABSENT))
	results = [
		smoother.update("node", float(index), prediction(PRESENT_MOVING))
		for index in range(1, 4)
	]
	assert [item.state for item in results] == [ABSENT, ABSENT, PRESENT_MOVING]
	assert [item.pending_count for item in results] == [1, 2, 0]
	assert results[-1].changed is True


def test_return_to_absent_uses_stricter_confirmation() -> None:
	smoother = StateSmoother(
		SmoothingConfig(ema_alpha=1.0, confirm_n=2, confirm_n_to_absent=4, margin=0.0)
	)
	smoother.update("node", 0.0, prediction(PRESENT_STILL))
	results = [
		smoother.update("node", float(index), prediction(ABSENT))
		for index in range(1, 5)
	]
	assert [item.state for item in results] == [PRESENT_STILL] * 3 + [ABSENT]


def test_margin_blocks_changes_with_near_equal_scores() -> None:
	smoother = StateSmoother(
		SmoothingConfig(ema_alpha=1.0, confirm_n=1, margin=0.1)
	)
	smoother.update("node", 0.0, prediction(ABSENT))
	result = smoother.update(
		"node",
		1.0,
		prediction(PRESENT_STILL, absent=0.48, present_still=0.52, present_moving=0.0),
	)
	assert result.state == ABSENT
	assert result.pending_state == PRESENT_STILL
	assert result.pending_count == 1


def test_ema_alpha_one_uses_incoming_scores_without_smoothing() -> None:
	smoother = StateSmoother(SmoothingConfig(ema_alpha=1.0, confirm_n=5, margin=0.0))
	first = smoother.update("node", 0.0, prediction(ABSENT))
	second = smoother.update(
		"node",
		1.0,
		prediction(ABSENT, absent=0.2, present_still=0.7, present_moving=0.1),
	)
	assert first.scores[ABSENT] == pytest.approx(0.9)
	assert second.scores[PRESENT_STILL] == pytest.approx(0.7)
	assert second.scores[ABSENT] == pytest.approx(0.2)


def test_mark_unknown_is_immediate_and_next_update_starts_clean() -> None:
	smoother = StateSmoother(SmoothingConfig(ema_alpha=0.2, confirm_n=5))
	smoother.update("node", 0.0, prediction(PRESENT_MOVING))
	unknown = smoother.mark_unknown("node", 1.0, "stale")
	restarted = smoother.update("node", 2.0, prediction(ABSENT))
	assert unknown.state == UNKNOWN
	assert unknown.reason == "stale"
	assert unknown.changed is True
	assert restarted.state == ABSENT
	assert restarted.changed is True
	assert restarted.pending_count == 0
	assert restarted.scores[ABSENT] == pytest.approx(0.9)


def test_interleaved_nodes_keep_independent_state() -> None:
	smoother = StateSmoother(SmoothingConfig(ema_alpha=1.0, confirm_n=2, margin=0.0))
	assert smoother.update("a", 0.0, prediction(ABSENT)).state == ABSENT
	assert smoother.update("b", 0.0, prediction(PRESENT_STILL)).state == PRESENT_STILL
	a_pending = smoother.update("a", 1.0, prediction(PRESENT_MOVING))
	b_steady = smoother.update("b", 1.0, prediction(PRESENT_STILL))
	a_changed = smoother.update("a", 2.0, prediction(PRESENT_MOVING))
	assert a_pending.state == ABSENT and a_pending.pending_count == 1
	assert b_steady.state == PRESENT_STILL and b_steady.pending_count == 0
	assert a_changed.state == PRESENT_MOVING


@pytest.mark.parametrize(
	"kwargs",
	[
		{"ema_alpha": 0.0},
		{"ema_alpha": 1.1},
		{"ema_alpha": float("nan")},
		{"confirm_n": 0},
		{"confirm_n": True},
		{"confirm_n_to_absent": 0},
		{"margin": -0.1},
		{"margin": float("inf")},
	],
)
def test_invalid_config_raises_value_error(kwargs: dict[str, object]) -> None:
	with pytest.raises(ValueError):
		SmoothingConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
	("destination", "expected"),
	[(PRESENT_STILL, 2), (PRESENT_MOVING, 2), (ABSENT, 4)],
)
def test_worst_case_latency_matches_observed_updates(destination: str, expected: int) -> None:
	config = SmoothingConfig(
		ema_alpha=1.0,
		confirm_n=2,
		confirm_n_to_absent=4,
		margin=0.0,
	)
	start = PRESENT_STILL if destination == ABSENT else ABSENT
	smoother = StateSmoother(config)
	smoother.update("node", 0.0, prediction(start))
	result = None
	for update_index in range(1, expected + 1):
		result = smoother.update("node", float(update_index), prediction(destination))
	assert worst_case_latency_updates(config, destination) == expected
	assert result is not None and result.state == destination


def test_result_and_scores_are_immutable() -> None:
	smoother = StateSmoother(SmoothingConfig(ema_alpha=1.0))
	result = smoother.update("node", 0.0, prediction(ABSENT))
	assert isinstance(result, SmoothedResult)
	with pytest.raises(FrozenInstanceError):
		result.state = PRESENT_STILL  # type: ignore[misc]
	with pytest.raises(TypeError):
		result.scores[ABSENT] = 0.0  # type: ignore[index]


def test_reset_clears_one_or_all_nodes() -> None:
	smoother = StateSmoother(SmoothingConfig())
	smoother.update("a", 0.0, prediction(ABSENT))
	smoother.update("b", 0.0, prediction(PRESENT_STILL))
	smoother.reset("a")
	assert smoother.update("a", 1.0, prediction(PRESENT_MOVING)).state == PRESENT_MOVING
	smoother.reset()
	assert smoother.update("b", 2.0, prediction(ABSENT)).state == ABSENT
