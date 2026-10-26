"""Pruebas SYNTHETIC del armado de datasets y partición por sesión."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_motion.dataset import Dataset, build_dataset, split_by_session
from wifi_motion.features import FEATURE_NAMES, FeatureConfig
from wifi_motion.schema import CsiSample, SessionMetadata

RATE = 16.0
WINDOW_CFG = {
    "window_s": 2.0,
    "hop_s": 2.0,
    "min_samples": 24,
    "max_gap_s": 0.2,
}
FEATURE_CFG = FeatureConfig(
    motion_band_hz=(1.0, 4.0),
    breathing_band_hz=(0.25, 1.0),
)


def recording(
    session_id: str,
    labels: list[str | None],
    *,
    gap: bool = False,
) -> tuple[SessionMetadata, list[CsiSample]]:
    metadata = SessionMetadata(
        session_id=session_id,
        environment="SYNTHETIC",
        sensor_layout="SYNTHETIC",
        nominal_rate_hz=RATE,
        synthetic=True,
    )
    samples = []
    for index, label in enumerate(labels):
        if gap and index == 10:
            continue
        timestamp = 100.0 + index / RATE
        signal = tuple(
            float(2.0 + np.sin(2 * np.pi * 1.5 * timestamp + carrier))
            for carrier in (0.0, 0.5, 1.0)
        )
        samples.append(
            CsiSample(timestamp, "node-a", signal, label=label)
        )
    samples.append(
        CsiSample(100.0 + len(labels) / RATE, "node-a", (2.0, 2.5, 3.0), label=labels[-1])
    )
    return metadata, samples


def build(*recordings):
    return build_dataset(recordings, WINDOW_CFG, FEATURE_CFG)


def test_majority_label_is_assigned_and_feature_shape_matches_names() -> None:
    _, samples = recording(
        "session-a",
        ["absent"] * 30 + ["present_still"] * 2 + ["absent"] * 32,
    )
    dataset = build((metadata("session-a"), samples))

    assert dataset.y[0] == "absent"
    assert dataset.X.shape == (len(dataset.y), len(FEATURE_NAMES)) == (2, 15)
    assert dataset.session_ids == ["session-a", "session-a"]
    assert dataset.t_end.shape == (2,)


def metadata(session_id: str) -> SessionMetadata:
    return SessionMetadata(
        session_id=session_id,
        environment="SYNTHETIC",
        sensor_layout="SYNTHETIC",
        nominal_rate_hz=RATE,
        synthetic=True,
    )


def test_transition_window_is_skipped() -> None:
    dataset = build(
        (
            metadata("transition"),
            recording(
                "transition",
                ["absent"] * 20 + ["present_moving"] * 48,
            )[1],
        )
    )

    assert dataset.skipped["transition"] >= 1


def test_missing_labels_are_counted_as_skipped() -> None:
    dataset = build(
        (
            metadata("missing-label"),
            recording("missing-label", ["absent"] * 32 + [None] * 32)[1],
        )
    )

    assert dataset.skipped["missing_label"] >= 1


def test_windows_with_gaps_are_counted_in_skipped() -> None:
    metadata_value, samples = recording("gap-session", ["absent"] * 64, gap=True)
    dataset = build_dataset(
        [(metadata_value, samples)],
        {**WINDOW_CFG, "max_gap_s": 0.08},
        FEATURE_CFG,
    )

    assert dataset.skipped["quality_has_gap"] >= 1


def test_split_by_session_keeps_sessions_disjoint() -> None:
    dataset = build(
        (metadata("train"), recording("train", ["absent"] * 64)[1]),
        (metadata("val"), recording("val", ["present_still"] * 64)[1]),
        (metadata("test"), recording("test", ["present_moving"] * 64)[1]),
    )

    train, val, test = split_by_session(dataset, ["train"], ["val"], ["test"])

    assert set(train.session_ids).isdisjoint(val.session_ids)
    assert set(train.session_ids).isdisjoint(test.session_ids)
    assert set(val.session_ids).isdisjoint(test.session_ids)
    assert set(train.session_ids) == {"train"}
    assert set(val.session_ids) == {"val"}
    assert set(test.session_ids) == {"test"}


def test_split_rejects_session_assigned_to_multiple_sets() -> None:
    dataset = Dataset(
        np.zeros((1, 15)),
        ["absent"],
        ["session"],
        ["node-a"],
        np.array([102.0]),
        {},
    )

    with pytest.raises(ValueError, match="más de un conjunto"):
        split_by_session(dataset, ["session"], ["session"], [])


def test_split_rejects_unknown_session_id() -> None:
    dataset = Dataset(
        np.zeros((1, 15)),
        ["absent"],
        ["session"],
        ["node-a"],
        np.array([102.0]),
        {},
    )

    with pytest.raises(ValueError, match="inexistente"):
        split_by_session(dataset, ["not-in-data"], [], [])
