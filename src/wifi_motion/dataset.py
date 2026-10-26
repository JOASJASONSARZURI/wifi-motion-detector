"""Ajusta normalización y umbrales solo con train/validación; el conjunto de prueba se usa una sola vez al final."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from wifi_motion.features import (
    FEATURE_NAMES,
    FeatureConfig,
    FeatureError,
    extract_features,
)
from wifi_motion.schema import LABELS, CsiSample, SessionMetadata
from wifi_motion.windowing import WindowQuality, make_windows

__all__ = ["Dataset", "build_dataset", "split_by_session"]


@dataclass(slots=True)
class Dataset:
    """Características etiquetadas por ventana y procedencia de cada fila."""

    X: np.ndarray
    y: list[str]
    session_ids: list[str]
    node_ids: list[str]
    t_end: np.ndarray
    skipped: dict[str, int]


def _window_label(
    samples: Sequence[CsiSample],
    timestamps: np.ndarray,
    start: float,
    end: float,
) -> tuple[str | None, float, str | None]:
    left = int(np.searchsorted(timestamps, start, side="left"))
    right = int(np.searchsorted(timestamps, end, side="right"))
    labels = [sample.label for sample in samples[left:right]]
    if not labels or any(label is None for label in labels):
        return None, 0.0, "missing_label"
    if any(label not in LABELS for label in labels):
        return None, 0.0, "invalid_label"

    counts = Counter(labels)
    majority_label, majority_count = max(
        ((label, counts[label]) for label in LABELS if counts[label]),
        key=lambda item: item[1],
    )
    return majority_label, majority_count / len(labels), None


def build_dataset(
    recordings: list[tuple[SessionMetadata, list[CsiSample]]],
    window_cfg: Mapping[str, object],
    feature_cfg: FeatureConfig,
    min_purity: float = 0.9,
) -> Dataset:
    """Construye una matriz de características y asigna una etiqueta por ventana."""
    if (
        isinstance(min_purity, bool)
        or not isinstance(min_purity, Real)
        or not math.isfinite(float(min_purity))
        or not 0 <= min_purity <= 1
    ):
        raise ValueError("min_purity debe ser un número finito entre 0 y 1")

    rows: list[np.ndarray] = []
    labels: list[str] = []
    sessions: list[str] = []
    nodes: list[str] = []
    ends: list[float] = []
    skipped: Counter[str] = Counter()

    for metadata, samples in recordings:
        samples_by_node: dict[str, list[CsiSample]] = {}
        for sample in samples:
            samples_by_node.setdefault(sample.node_id, []).append(sample)

        for node_id, node_samples in samples_by_node.items():
            ordered = sorted(node_samples, key=lambda sample: sample.timestamp)
            timestamps = np.asarray([sample.timestamp for sample in ordered], dtype=np.float64)
            windows = make_windows(node_samples, **dict(window_cfg))
            for window in windows:
                if window.quality is not WindowQuality.OK:
                    skipped[f"quality_{window.quality.name.lower()}"] += 1
                    continue

                label, purity, reason = _window_label(
                    ordered, timestamps, window.t_start, window.t_end
                )
                if reason is not None:
                    skipped[reason] += 1
                    continue

                if purity < min_purity:
                    skipped["transition"] += 1
                    continue

                try:
                    features = extract_features(window, feature_cfg)
                except FeatureError:
                    skipped["feature_error"] += 1
                    continue
                rows.append(features)
                assert label is not None
                labels.append(label)
                sessions.append(metadata.session_id)
                nodes.append(node_id)
                ends.append(window.t_end)

    matrix = (
        np.stack(rows).astype(np.float64, copy=False)
        if rows
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    )
    return Dataset(
        X=matrix,
        y=labels,
        session_ids=sessions,
        node_ids=nodes,
        t_end=np.asarray(ends, dtype=np.float64),
        skipped=dict(skipped),
    )


def split_by_session(
    dataset: Dataset,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> tuple[Dataset, Dataset, Dataset]:
    """Separa filas por sesión y rechaza asignaciones repetidas o inexistentes."""
    assignments = (train_ids, val_ids, test_ids)
    assigned: dict[str, int] = {}
    for split_index, session_ids in enumerate(assignments):
        for session_id in session_ids:
            if session_id in assigned and assigned[session_id] != split_index:
                raise ValueError(f"session_id aparece en más de un conjunto: {session_id!r}")
            assigned[session_id] = split_index

    available = set(dataset.session_ids)
    missing = set(assigned) - available
    if missing:
        raise ValueError(f"session_id inexistente en el dataset: {', '.join(sorted(missing))}")

    session_array = np.asarray(dataset.session_ids, dtype=object)
    splits = []
    for session_ids in assignments:
        mask = np.isin(session_array, list(session_ids))
        splits.append(
            Dataset(
                X=dataset.X[mask].copy(),
                y=[label for label, keep in zip(dataset.y, mask) if keep],
                session_ids=[
                    session_id
                    for session_id, keep in zip(dataset.session_ids, mask)
                    if keep
                ],
                node_ids=[
                    node_id for node_id, keep in zip(dataset.node_ids, mask) if keep
                ],
                t_end=dataset.t_end[mask].copy(),
                skipped=dict(dataset.skipped),
            )
        )
    return splits[0], splits[1], splits[2]
