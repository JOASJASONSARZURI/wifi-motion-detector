"""Pruebas de grabaciones generadas; todos los datos son SYNTHETIC."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_motion.schema import CsiSample
from wifi_motion.synthetic import generate_dataset, generate_session, inject_faults


def _samples() -> list[CsiSample]:
    return [
        CsiSample(float(index), "node-1", (1.0, 2.0), label="absent")
        for index in range(8)
    ]


def test_generate_session_is_deterministic_and_has_regular_increasing_timestamps() -> None:
    args = ([("absent", 1.0), ("present_still", 1.5)],)
    first = generate_session(*args, session_id="syn-test", seed=12)
    second = generate_session(*args, session_id="syn-test", seed=12)

    assert first == second
    timestamps = [sample.timestamp for sample in first[1]]
    assert all(right > left for left, right in zip(timestamps, timestamps[1:]))
    assert len(timestamps) == pytest.approx(2.5 * 100.0, abs=1)


def test_generate_session_assigns_labels_and_synthetic_metadata() -> None:
    metadata, samples = generate_session(
        [("absent", 0.2), ("present_still", 0.3), ("present_moving", 0.4)],
        session_id="syn-labels",
        rate_hz=20.0,
        n_subcarriers=5,
    )

    assert metadata.synthetic is True
    assert metadata.environment == metadata.sensor_layout == "SYNTHETIC"
    assert [sample.label for sample in samples] == (
        ["absent"] * 4 + ["present_still"] * 6 + ["present_moving"] * 8
    )
    assert all(len(sample.amplitudes) == 5 for sample in samples)


def test_signal_variance_increases_with_activity() -> None:
    _, samples = generate_session(
        [("absent", 12.0), ("present_still", 12.0), ("present_moving", 12.0)],
        session_id="syn-variance",
        seed=31,
        n_subcarriers=16,
    )
    variances = {}
    for label in ("absent", "present_still", "present_moving"):
        values = np.asarray(
            [sample.amplitudes for sample in samples if sample.label == label]
        )
        variances[label] = float(np.var(values, axis=0).mean())

    assert variances["present_moving"] > variances["present_still"] > variances["absent"]


def test_dataset_sessions_have_distinct_ids_and_signals() -> None:
    dataset = generate_dataset(3, seed=7, segments=[("absent", 1.0)])

    assert [metadata.session_id for metadata, _ in dataset] == [
        "syn-000",
        "syn-001",
        "syn-002",
    ]
    assert len({tuple(samples[0].amplitudes) for _, samples in dataset}) == 3
    assert len({metadata.synthetic for metadata, _ in dataset}) == 1
    assert all(metadata.synthetic for metadata, _ in dataset)


def test_inject_faults_with_zero_rates_returns_equal_copy() -> None:
    samples = _samples()
    result = inject_faults(samples)

    assert result == samples
    assert result is not samples


@pytest.mark.parametrize(
    ("rate_name", "expected"),
    [
        ("nan_rate", lambda result, original: any(
            np.isnan(sample.amplitudes).any() for sample in result
        )),
        ("drop_rate", lambda result, original: len(result) < len(original)),
        ("duplicate_rate", lambda result, original: len(result) > len(original)),
        ("swap_rate", lambda result, original: any(
            later.timestamp < earlier.timestamp
            for earlier, later in zip(result, result[1:])
        )),
    ],
)
def test_inject_faults_creates_requested_defect(rate_name, expected) -> None:
    original = _samples()
    result = inject_faults(original, seed=4, **{rate_name: 1.0})

    assert expected(result, original)
