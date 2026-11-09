"""Pruebas SYNTHETIC de métricas y evaluación de flujo."""

from __future__ import annotations

import pytest

from wifi_motion.classifiers.base import UNKNOWN
from wifi_motion.detector import DetectionResult
from wifi_motion.evaluation import (
    check_disjoint_sessions,
    majority_baseline,
    report_label,
    stream_evaluation,
    window_metrics,
)
from wifi_motion.schema import CsiSample, SessionMetadata


def _metadata(session_id: str = "syn-1", *, synthetic: bool = True) -> SessionMetadata:
    return SessionMetadata(
        session_id=session_id,
        environment="test",
        sensor_layout="test",
        nominal_rate_hz=1.0,
        synthetic=synthetic,
    )


def _samples(labels: list[str], node_id: str = "n1") -> list[CsiSample]:
    return [
        CsiSample(float(index), node_id, (1.0,), label=label)
        for index, label in enumerate(labels)
    ]


def test_window_metrics_hand_calculation_and_unknown_is_error() -> None:
    report = window_metrics(
        ["absent", "absent", "present_still", "present_moving"],
        ["absent", UNKNOWN, "present_still", "absent"],
    )

    assert report.confusion == (
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (1, 0, 0, 0),
    )
    assert report.per_class[0].precision == pytest.approx(0.5)
    assert report.per_class[0].recall == pytest.approx(0.5)
    assert report.per_class[0].f1 == pytest.approx(0.5)
    assert report.per_class[1].f1 == pytest.approx(1.0)
    assert report.per_class[2].f1 == pytest.approx(0.0)
    assert report.macro_f1 == pytest.approx(0.5)
    assert report.unknown_rate == pytest.approx(0.25)
    assert report.false_alarm_rate == pytest.approx(0.0)
    assert report.false_negative_rate == pytest.approx(0.5)


def test_class_without_support_has_none_metrics_and_is_reported() -> None:
    report = window_metrics(["absent"], ["absent"])

    assert report.missing_classes == ("present_still", "present_moving")
    assert report.per_class[1].precision is None
    assert report.per_class[1].recall is None
    assert report.per_class[1].f1 is None
    assert report.macro_f1 == 1.0
    assert report.macro_f1_classes == ("absent",)
    assert report.false_negative_rate is None


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [([], []), (["absent"], []), (["bad"], ["absent"]), (["absent"], ["bad"])],
)
def test_window_metrics_rejects_empty_mismatched_or_invalid_labels(
    actual: list[str], predicted: list[str]
) -> None:
    with pytest.raises(ValueError):
        window_metrics(actual, predicted)


def test_majority_baseline_breaks_ties_by_class_state_order() -> None:
    assert majority_baseline(
        ["present_moving", "absent", "present_still", "absent"],
        ["absent", "present_still"],
    ) == ["absent", "absent"]
    assert majority_baseline(["present_still"], []) == []


def test_report_label_rejects_mixed_evidence_and_markdown_warns() -> None:
    assert report_label([_metadata()]) == "SYNTHETIC"
    assert report_label([_metadata(synthetic=False)]) == "MEASURED"
    with pytest.raises(ValueError, match="mezclar"):
        report_label([_metadata(), _metadata("measured", synthetic=False)])
    assert "Evidence: SYNTHETIC" in stream_evaluation(
        lambda: _ScriptedDetector([("absent", 1.0)]),
        [(_metadata(), _samples(["absent", "absent"]))],
        transition_tolerance_s=0,
    ).to_markdown()


class _ScriptedDetector:
    def __init__(self, outputs: list[tuple[str, float]]) -> None:
        self.outputs = outputs
        self.index = 0

    def process(self, sample: CsiSample) -> list[DetectionResult]:
        if self.index >= len(self.outputs):
            return []
        state, timestamp = self.outputs[self.index]
        self.index += 1
        return [
            DetectionResult(
                node_id=sample.node_id,
                timestamp=timestamp,
                state=state,
                scores={},
                confidence=0.0,
                changed=False,
                input_quality={},
                reason=None,
            )
        ]


def _factory(states: list[str], timestamps: list[float]):
    outputs = list(zip(states, timestamps))

    def create() -> _ScriptedDetector:
        return _ScriptedDetector(outputs)

    return create


def test_stream_latency_exact_missed_and_transition_exclusion() -> None:
    recording = (_metadata(), _samples(["absent", "absent", "present_moving", "present_moving", "absent", "absent"]))
    report = stream_evaluation(
        _factory(
            ["absent", "absent", "absent", "present_moving", "present_moving", "present_moving"],
            [0.0, 0.5, 2.0, 3.0, 4.0, 5.0],
        ),
        [recording],
        transition_tolerance_s=0.5,
    )

    moving = report.latencies["absent->present_moving"]
    absent = report.latencies["present_moving->absent"]
    assert moving.latencies_s == (1.0,)
    assert moving.mean_s == 1.0
    assert moving.median_s == 1.0
    assert moving.max_s == 1.0
    assert moving.missed == 0
    assert absent.latencies_s == ()
    assert absent.missed == 1
    assert report.window_report.excluded_transition_windows == 2
    assert report.evidence == "SYNTHETIC"
    assert report.session_ids == ("syn-1",)
    assert "La separabilidad es por construcción; esto no valida el sistema con hardware real" in report.to_markdown()


def test_stream_false_alarm_and_unknown_rates() -> None:
    report = stream_evaluation(
        _factory(
            ["present_still", UNKNOWN, "present_moving"],
            [0.0, 1.0, 2.0],
        ),
        [(_metadata(), _samples(["absent", "absent", "present_still"]))],
        transition_tolerance_s=0,
    )
    assert report.flow_false_alarm_count == 1
    assert report.flow_false_alarm_rate == 0.5
    assert report.flow_unknown_rate == pytest.approx(1 / 3)


def test_stream_rejects_mixed_evidence_and_session_overlap() -> None:
    with pytest.raises(ValueError, match="mezclar"):
        stream_evaluation(
            _factory([], []),
            [
                (_metadata(), _samples(["absent"])),
                (_metadata("real", synthetic=False), _samples(["absent"])),
            ],
            transition_tolerance_s=0,
        )
    with pytest.raises(ValueError, match="solapadas"):
        check_disjoint_sessions(["one"], ["two", "one"], ["three"])


def test_stream_uses_new_detector_for_every_session() -> None:
    calls = 0

    def factory() -> _ScriptedDetector:
        nonlocal calls
        calls += 1
        return _ScriptedDetector([("absent", 0.0)])

    stream_evaluation(
        factory,
        [
            (_metadata("syn-1"), _samples(["absent"])),
            (_metadata("syn-2"), _samples(["absent"])),
        ],
        transition_tolerance_s=0,
    )
    assert calls == 2
