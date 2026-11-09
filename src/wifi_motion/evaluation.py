"""Métricas reproducibles para clasificación por ventana y flujos etiquetados."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from wifi_motion.classifiers.base import CLASS_STATES, UNKNOWN
from wifi_motion.classifiers.softmax import confusion_matrix
from wifi_motion.detector import DetectionResult, Detector, run_samples
from wifi_motion.schema import CsiSample, SessionMetadata

__all__ = [
    "ClassMetrics",
    "ClassificationReport",
    "StreamReport",
    "TransitionLatency",
    "check_disjoint_sessions",
    "majority_baseline",
    "report_label",
    "stream_evaluation",
    "window_metrics",
]

_PREDICTION_STATES = (*CLASS_STATES, UNKNOWN)
_SYNTHETIC_WARNING = (
    "La separabilidad es por construcción; esto no valida el sistema con hardware real"
)


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    precision: float | None
    recall: float | None
    f1: float | None
    support: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    classes: tuple[str, ...]
    confusion: tuple[tuple[int, ...], ...]
    per_class: tuple[ClassMetrics, ...]
    macro_f1: float | None
    macro_f1_classes: tuple[str, ...]
    missing_classes: tuple[str, ...]
    false_alarm_rate: float | None
    false_negative_rate: float | None
    unknown_rate: float | None
    n_windows: int
    evidence: str = "UNSPECIFIED"
    n_sessions: int = 0
    n_windows_by_class: tuple[tuple[str, int], ...] = ()
    session_ids: tuple[str, ...] = ()
    excluded_transition_windows: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence": self.evidence,
            "n_sessions": self.n_sessions,
            "session_ids": list(self.session_ids),
            "n_windows": self.n_windows,
            "n_windows_by_class": dict(self.n_windows_by_class),
            "classes": list(self.classes),
            "confusion": [list(row) for row in self.confusion],
            "per_class": {
                label: metrics.to_dict()
                for label, metrics in zip(self.classes, self.per_class)
            },
            "macro_f1": self.macro_f1,
            "macro_f1_classes": list(self.macro_f1_classes),
            "missing_classes": list(self.missing_classes),
            "false_alarm_rate": self.false_alarm_rate,
            "false_negative_rate": self.false_negative_rate,
            "unknown_rate": self.unknown_rate,
            "excluded_transition_windows": self.excluded_transition_windows,
        }

    def to_markdown(self) -> str:
        lines = [_evidence_line(self.evidence)]
        lines.extend(
            [
                "",
                "| Clase | Precision | Recall | F1 | Soporte |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, metrics in zip(self.classes, self.per_class):
            lines.append(
                f"| {label} | {_fmt(metrics.precision)} | {_fmt(metrics.recall)} "
                f"| {_fmt(metrics.f1)} | {metrics.support} |"
            )
        lines.extend(
            [
                "",
                f"- Macro F1: {_fmt(self.macro_f1)} "
                f"(clases: {', '.join(self.macro_f1_classes) or 'ninguna'})",
                f"- False alarm rate: {_fmt(self.false_alarm_rate)}",
                f"- False negative rate: {_fmt(self.false_negative_rate)}",
                f"- Unknown rate: {_fmt(self.unknown_rate)}",
                f"- Ventanas: {self.n_windows}; excluidas por transición: "
                f"{self.excluded_transition_windows}",
                f"- Sesiones: {', '.join(self.session_ids) or 'ninguna'}",
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TransitionLatency:
    transition: str
    latencies_s: tuple[float, ...]
    mean_s: float | None
    median_s: float | None
    max_s: float | None
    missed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "latencies_s": list(self.latencies_s),
            "mean_s": self.mean_s,
            "median_s": self.median_s,
            "max_s": self.max_s,
            "missed": self.missed,
        }


@dataclass(frozen=True, slots=True)
class StreamReport:
    window_report: ClassificationReport
    latency_by_transition: tuple[TransitionLatency, ...]
    flow_false_alarm_count: int
    flow_false_alarm_rate: float | None
    flow_unknown_rate: float | None
    n_flow_results: int
    n_absent_flow_results: int

    @property
    def evidence(self) -> str:
        return self.window_report.evidence

    @property
    def n_sessions(self) -> int:
        return self.window_report.n_sessions

    @property
    def n_windows_by_class(self) -> tuple[tuple[str, int], ...]:
        return self.window_report.n_windows_by_class

    @property
    def session_ids(self) -> tuple[str, ...]:
        return self.window_report.session_ids

    @property
    def latencies(self) -> Mapping[str, TransitionLatency]:
        return {item.transition: item for item in self.latency_by_transition}

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence": self.evidence,
            "n_sessions": self.n_sessions,
            "session_ids": list(self.session_ids),
            "n_windows_by_class": dict(self.n_windows_by_class),
            "window_metrics": self.window_report.to_dict(),
            "latency_by_transition": {
                item.transition: item.to_dict()
                for item in self.latency_by_transition
            },
            "flow_false_alarm_count": self.flow_false_alarm_count,
            "flow_false_alarm_rate": self.flow_false_alarm_rate,
            "flow_unknown_rate": self.flow_unknown_rate,
            "n_flow_results": self.n_flow_results,
            "n_absent_flow_results": self.n_absent_flow_results,
        }

    def to_markdown(self) -> str:
        lines = [_evidence_line(self.evidence), "", "## Métricas por ventana", ""]
        lines.extend(self.window_report.to_markdown().splitlines()[1:])
        lines.extend(
            [
                "",
                "## Latencia por transición",
                "",
                "| Transición | Latencias (s) | Media (s) | Mediana (s) | Máximo (s) | Missed |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in self.latency_by_transition:
            values = ", ".join(_fmt(value) for value in item.latencies_s)
            lines.append(
                f"| {item.transition} | {values} | {_fmt(item.mean_s)} "
                f"| {_fmt(item.median_s)} | {_fmt(item.max_s)} | {item.missed} |"
            )
        lines.extend(
            [
                "",
                f"- Falsas alarmas del flujo: {self.flow_false_alarm_count} "
                f"(tasa: {_fmt(self.flow_false_alarm_rate)})",
                f"- Unknown rate del flujo: {_fmt(self.flow_unknown_rate)}",
            ]
        )
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def _evidence_line(evidence: str) -> str:
    line = f"Evidence: {evidence}"
    if evidence == "SYNTHETIC":
        return f"{line} — {_SYNTHETIC_WARNING}"
    return line


def _sequence(values: Sequence[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} debe ser una secuencia de etiquetas")
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{name} debe ser una secuencia de etiquetas") from exc


def _classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    evidence: str = "UNSPECIFIED",
    n_sessions: int = 0,
    session_ids: tuple[str, ...] = (),
    excluded_transition_windows: int = 0,
    allow_empty: bool = False,
) -> ClassificationReport:
    actual = _sequence(y_true, "y_true")
    predicted = _sequence(y_pred, "y_pred")
    if len(actual) != len(predicted):
        raise ValueError("y_true e y_pred deben tener la misma longitud")
    if not actual and not allow_empty:
        raise ValueError("y_true e y_pred no pueden estar vacíos")
    invalid_actual = [label for label in actual if label not in CLASS_STATES]
    invalid_predicted = [label for label in predicted if label not in _PREDICTION_STATES]
    if invalid_actual or invalid_predicted:
        invalid = invalid_actual + invalid_predicted
        raise ValueError(f"etiquetas desconocidas: {invalid!r}")

    matrix = (
        confusion_matrix(actual, predicted, _PREDICTION_STATES)[: len(CLASS_STATES), :]
        if actual
        else np.zeros((len(CLASS_STATES), len(_PREDICTION_STATES)), dtype=np.int64)
    )
    supports = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    class_metrics = []
    present_f1 = []
    missing = []
    for index, label in enumerate(CLASS_STATES):
        support = int(supports[index])
        if support == 0:
            class_metrics.append(ClassMetrics(None, None, None, 0))
            missing.append(label)
            continue
        true_positive = int(matrix[index, index])
        precision = (
            true_positive / int(predicted_counts[index])
            if predicted_counts[index]
            else None
        )
        recall = true_positive / support
        f1_denominator = 2 * true_positive + int(predicted_counts[index]) - true_positive + support - true_positive
        f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
        class_metrics.append(ClassMetrics(precision, recall, f1, support))
        present_f1.append(f1)

    absent_support = int(supports[0])
    present_support = int(supports[1] + supports[2])
    false_alarm_rate = (
        int(matrix[0, 1] + matrix[0, 2]) / absent_support
        if absent_support
        else None
    )
    false_negative_rate = (
        int(matrix[1, 0] + matrix[2, 0]) / present_support
        if present_support
        else None
    )
    counts_by_class = tuple(
        (label, int(supports[index])) for index, label in enumerate(CLASS_STATES)
    )
    confusion = tuple(tuple(int(value) for value in row) for row in matrix)
    return ClassificationReport(
        classes=CLASS_STATES,
        confusion=confusion,
        per_class=tuple(class_metrics),
        macro_f1=float(np.mean(present_f1)) if present_f1 else None,
        macro_f1_classes=tuple(label for label in CLASS_STATES if label not in missing),
        missing_classes=tuple(missing),
        false_alarm_rate=false_alarm_rate,
        false_negative_rate=false_negative_rate,
        unknown_rate=(int(matrix[:, 3].sum()) / len(actual)) if actual else None,
        n_windows=len(actual),
        evidence=evidence,
        n_sessions=n_sessions,
        n_windows_by_class=counts_by_class,
        session_ids=session_ids,
        excluded_transition_windows=excluded_transition_windows,
    )


def window_metrics(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> ClassificationReport:
    """Calcula métricas por ventana; UNKNOWN cuenta como error de su clase real."""
    return _classification_report(y_true, y_pred)


def majority_baseline(y_train: Sequence[str], y_true: Sequence[str]) -> list[str]:
    """Predice la clase más frecuente; los empates siguen ``CLASS_STATES``."""
    training = _sequence(y_train, "y_train")
    actual = _sequence(y_true, "y_true")
    if not training:
        raise ValueError("y_train no puede estar vacío")
    invalid = [label for label in training + actual if label not in CLASS_STATES]
    if invalid:
        raise ValueError(f"etiquetas desconocidas: {invalid!r}")
    counts = Counter(training)
    majority = max(CLASS_STATES, key=lambda label: counts[label])
    return [majority] * len(actual)


def report_label(metadatas: Sequence[SessionMetadata]) -> str:
    """Asigna la evidencia sin permitir mezclar datos sintéticos y medidos."""
    if not metadatas:
        raise ValueError("metadatas no puede estar vacío")
    synthetic = [metadata.synthetic for metadata in metadatas]
    if any(synthetic) and not all(synthetic):
        raise ValueError("no se pueden mezclar sesiones sintéticas y medidas")
    return "SYNTHETIC" if all(synthetic) else "MEASURED"


def check_disjoint_sessions(
    train_ids: Sequence[str], val_ids: Sequence[str], test_ids: Sequence[str]
) -> None:
    """Lanza ValueError si alguna sesión está asignada a más de un split."""
    splits = (set(train_ids), set(val_ids), set(test_ids))
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            overlap = left & right
            if overlap:
                raise ValueError(
                    f"sesiones solapadas entre conjuntos: {', '.join(sorted(overlap))}"
                )


def _validate_tolerance(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError("transition_tolerance_s debe ser finito y >= 0")
    return float(value)


def _transitions(
    samples_by_node: Mapping[str, list[CsiSample]],
) -> dict[str, list[tuple[float, str, float | None]]]:
    transitions: dict[str, list[tuple[float, str, float | None]]] = {}
    for node_id, samples in samples_by_node.items():
        ordered = sorted(samples, key=lambda sample: sample.timestamp)
        changes: list[tuple[float, str]] = []
        for index in range(1, len(ordered)):
            previous, current = ordered[index - 1], ordered[index]
            if current.label != previous.label:
                changes.append((current.timestamp, current.label or ""))
        transitions[node_id] = [
            (
                timestamp,
                label,
                changes[index + 1][0] if index + 1 < len(changes) else None,
            )
            for index, (timestamp, label) in enumerate(changes)
        ]
    return transitions


def stream_evaluation(
    detector_factory: Callable[[], Detector],
    recordings: Sequence[tuple[SessionMetadata, list[CsiSample]]],
    *,
    transition_tolerance_s: float,
) -> StreamReport:
    """Evalúa un detector nuevo por sesión, alineando cada salida con su etiqueta."""
    tolerance = _validate_tolerance(transition_tolerance_s)
    if not recordings:
        raise ValueError("recordings no puede estar vacío")
    metadata = [recording[0] for recording in recordings]
    evidence = report_label(metadata)
    session_ids = tuple(item.session_id for item in metadata)
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("session_id debe ser único en recordings")

    all_actual: list[str] = []
    all_predicted: list[str] = []
    total_excluded = 0
    flow_results = 0
    absent_flow_results = 0
    false_alarm_count = 0
    unknown_count = 0
    transition_values: dict[str, list[float]] = {}
    transition_missed: Counter[str] = Counter()

    for _metadata, samples in recordings:
        samples_by_node: dict[str, list[CsiSample]] = {}
        for sample in samples:
            if sample.label not in CLASS_STATES:
                raise ValueError(
                    f"grabación contiene etiqueta ausente o desconocida: {sample.label!r}"
                )
            samples_by_node.setdefault(sample.node_id, []).append(sample)

        detector = detector_factory()
        results = list(run_samples(detector, samples))
        ordered_by_node = {
            node_id: sorted(node_samples, key=lambda sample: sample.timestamp)
            for node_id, node_samples in samples_by_node.items()
        }
        timestamps_by_node = {
            node_id: np.asarray(
                [sample.timestamp for sample in node_samples], dtype=np.float64
            )
            for node_id, node_samples in ordered_by_node.items()
        }
        transitions = _transitions(samples_by_node)
        transition_times = {
            node_id: np.asarray(
                [change[0] for change in changes], dtype=np.float64
            )
            for node_id, changes in transitions.items()
        }
        results_by_node: dict[str, list[DetectionResult]] = {}
        for result in results:
            if result.state not in _PREDICTION_STATES:
                raise ValueError(f"el detector devolvió un estado desconocido: {result.state!r}")
            results_by_node.setdefault(result.node_id, []).append(result)
            node_samples = ordered_by_node.get(result.node_id, [])
            timestamps = timestamps_by_node.get(result.node_id, np.empty(0))
            position = int(np.searchsorted(timestamps, result.timestamp, side="right")) - 1
            if position < 0:
                continue
            true_state = node_samples[position].label
            assert true_state is not None
            flow_results += 1
            if result.state == UNKNOWN:
                unknown_count += 1
            if true_state == "absent":
                absent_flow_results += 1
                if result.state in ("present_still", "present_moving"):
                    false_alarm_count += 1
            change_times = transition_times.get(result.node_id, np.empty(0))
            insertion = int(np.searchsorted(change_times, result.timestamp))
            near_transition = tolerance > 0 and any(
                abs(result.timestamp - change_times[index]) < tolerance
                for index in (insertion - 1, insertion)
                if 0 <= index < len(change_times)
            )
            if near_transition:
                total_excluded += 1
                continue
            all_actual.append(true_state)
            all_predicted.append(result.state)

        for node_id, changes in transitions.items():
            node_results = results_by_node.get(node_id, [])
            for index, (timestamp, new_state, next_change) in enumerate(changes):
                previous_state = (
                    changes[index - 1][1]
                    if index
                    else _previous_label(samples_by_node[node_id], timestamp)
                )
                transition = f"{previous_state}->{new_state}"
                transition_values.setdefault(transition, [])
                match = next(
                    (
                        result
                        for result in node_results
                        if timestamp <= result.timestamp
                        and (next_change is None or result.timestamp < next_change)
                        and result.state == new_state
                    ),
                    None,
                )
                if match is None:
                    transition_missed[transition] += 1
                else:
                    transition_values[transition].append(
                        float(match.timestamp - timestamp)
                    )

    window_report = _classification_report(
        all_actual,
        all_predicted,
        evidence=evidence,
        n_sessions=len(metadata),
        session_ids=session_ids,
        excluded_transition_windows=total_excluded,
        allow_empty=True,
    )
    latency_reports = []
    for transition in transition_values:
        values = transition_values[transition]
        latency_reports.append(
            TransitionLatency(
                transition=transition,
                latencies_s=tuple(values),
                mean_s=float(np.mean(values)) if values else None,
                median_s=float(np.median(values)) if values else None,
                max_s=max(values) if values else None,
                missed=transition_missed[transition],
            )
        )
    return StreamReport(
        window_report=window_report,
        latency_by_transition=tuple(latency_reports),
        flow_false_alarm_count=false_alarm_count,
        flow_false_alarm_rate=(
            false_alarm_count / absent_flow_results if absent_flow_results else None
        ),
        flow_unknown_rate=unknown_count / flow_results if flow_results else None,
        n_flow_results=flow_results,
        n_absent_flow_results=absent_flow_results,
    )


def _previous_label(samples: Sequence[CsiSample], timestamp: float) -> str:
    ordered = sorted(samples, key=lambda sample: sample.timestamp)
    previous = [sample for sample in ordered if sample.timestamp < timestamp]
    return previous[-1].label or "" if previous else ""
