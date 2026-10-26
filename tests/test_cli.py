"""Pruebas de integración SYNTHETIC de la CLI."""

from __future__ import annotations

import json

import numpy as np

from wifi_motion.calibration import load_baseline
from wifi_motion.classifiers.softmax import SoftmaxClassifier, SoftmaxConfig
from wifi_motion.cli import main
from wifi_motion.features import FEATURE_NAMES
from wifi_motion.io_jsonl import iter_recording, write_recording
from wifi_motion.synthetic import generate_session


def empty_recording(session_id: str = "quiet-session"):
    metadata, samples = generate_session(
        [("absent", 80.0)],
        session_id=session_id,
        seed=17,
        n_subcarriers=8,
    )
    return metadata, samples


def test_generate_synthetic_writes_one_file_per_session(tmp_path, capsys) -> None:
    output = tmp_path / "generated"

    assert main(
        ["generate-synthetic", "--out", str(output), "--sessions", "2", "--seed", "9"]
    ) == 0
    assert sorted(path.name for path in output.glob("*.jsonl")) == [
        "syn-000.jsonl",
        "syn-001.jsonl",
        "syn-calibration-absent.jsonl",
    ]
    _metadata, samples = iter_recording(output / "syn-calibration-absent.jsonl")
    with samples:
        assert {sample.label for sample in samples} == {"absent"}
    assert "SYNTHETIC" in capsys.readouterr().out


def test_generate_calibrate_and_run_show_synthetic_class_transitions(
    tmp_path, capsys
) -> None:
    output = tmp_path / "synthetic"
    baseline = tmp_path / "baseline.json"
    assert main(
        [
            "generate-synthetic",
            "--out",
            str(output),
            "--sessions",
            "1",
            "--seed",
            "1",
        ]
    ) == 0
    assert main(
        [
            "calibrate",
            "--input",
            str(output / "syn-calibration-absent.jsonl"),
            "--node-id",
            "node-1",
            "--out",
            str(baseline),
        ]
    ) == 0
    capsys.readouterr()

    assert main(
        [
            "run",
            "--input",
            str(output / "syn-000.jsonl"),
            "--classifier",
            "threshold",
            "--baseline",
            str(baseline),
        ]
    ) == 0

    captured = capsys.readouterr()
    states = [json.loads(line)["state"] for line in captured.out.splitlines()]
    unique_states = list(dict.fromkeys(states))
    assert unique_states == ["absent", "present_still", "present_moving"]
    stats = json.loads(captured.err)
    assert stats["unknown_results"] == 0


def test_calibrate_saves_loadable_baseline(tmp_path, capsys) -> None:
    recording = tmp_path / "empty.jsonl"
    baseline_path = tmp_path / "baseline.json"
    metadata, samples = empty_recording()
    write_recording(recording, metadata, samples)

    code = main(
        [
            "calibrate",
            "--input",
            str(recording),
            "--node-id",
            "node-1",
            "--out",
            str(baseline_path),
        ]
    )

    assert code == 0
    assert load_baseline(baseline_path).n_windows >= 30
    assert "ventanas" in capsys.readouterr().out


def test_calibrate_rejects_recording_with_movement(tmp_path, capsys) -> None:
    path = tmp_path / "moving.jsonl"
    metadata, samples = generate_session(
        [("absent", 2.0), ("present_moving", 2.0)],
        session_id="not-empty",
        n_subcarriers=8,
    )
    write_recording(path, metadata, samples)

    code = main(
        [
            "calibrate",
            "--input",
            str(path),
            "--node-id",
            "node-1",
            "--out",
            str(tmp_path / "baseline.json"),
        ]
    )

    assert code != 0
    assert "solo como absent" in capsys.readouterr().err


def test_train_saves_loadable_model_and_reports_synthetic_validation(
    tmp_path, capsys
) -> None:
    paths = []
    for index in range(3):
        metadata, samples = generate_session(
            [
                ("absent", 6.0),
                ("present_still", 6.0),
                ("present_moving", 6.0),
            ],
            session_id=f"syn-train-{index}",
            seed=index + 3,
            n_subcarriers=8,
        )
        path = tmp_path / f"session-{index}.jsonl"
        write_recording(path, metadata, samples)
        paths.append(path)
    model_path = tmp_path / "model.json"

    code = main(
        [
            "train",
            "--train",
            f"{paths[0]},{paths[1]}",
            "--val",
            str(paths[2]),
            "--out",
            str(model_path),
            "--epochs",
            "20",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "(validación, SYNTHETIC)" in output
    assert isinstance(SoftmaxClassifier.load(model_path), SoftmaxClassifier)


def test_run_prints_valid_json_states_and_stats(tmp_path, capsys) -> None:
    data_path = tmp_path / "run.jsonl"
    model_path = tmp_path / "model.json"
    metadata, samples = generate_session(
        [("absent", 6.0), ("present_moving", 6.0)],
        session_id="run-synthetic",
        seed=4,
        n_subcarriers=8,
    )
    write_recording(data_path, metadata, samples)
    train_x, train_y = [], []
    for class_index, label in enumerate(
        ("absent", "present_still", "present_moving")
    ):
        for repeat in range(5):
            row = np.full(len(FEATURE_NAMES), class_index * 2.0 + repeat * 0.01)
            row[FEATURE_NAMES.index("mean_rssi")] = -50.0
            train_x.append(row)
            train_y.append(label)
    training = np.asarray(train_x)
    model = SoftmaxClassifier().fit(
        training,
        train_y,
        training,
        train_y,
        SoftmaxConfig(epochs=5),
    )
    model.save(model_path)

    code = main(
        [
            "run",
            "--input",
            str(data_path),
            "--classifier",
            "softmax",
            "--model",
            str(model_path),
        ]
    )

    captured = capsys.readouterr()
    output_lines = captured.out.splitlines()
    assert code == 0
    assert output_lines
    assert all(
        json.loads(line)["state"]
        in {"absent", "present_still", "present_moving", "unknown"}
        for line in output_lines
    )
    stats = json.loads(captured.err)
    assert "samples_rejected" in stats
    assert "unknown_results" in stats


def test_run_missing_file_returns_clean_error(tmp_path, capsys) -> None:
    model_path = tmp_path / "model.json"
    features = np.vstack(
        [
            np.full(len(FEATURE_NAMES), index + 1.0)
            for index in range(3)
        ]
    )
    model = SoftmaxClassifier().fit(
        features,
        ["absent", "present_still", "present_moving"],
        features,
        ["absent", "present_still", "present_moving"],
        SoftmaxConfig(epochs=2),
    )
    model.save(model_path)
    code = main(
        [
            "run",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--classifier",
            "softmax",
            "--model",
            str(model_path),
        ]
    )

    error = capsys.readouterr().err
    assert code != 0
    assert "error:" in error
    assert "Traceback" not in error


def test_record_is_not_implemented_with_status_two(capsys) -> None:
    assert main(["record"]) == 2
    assert capsys.readouterr().err.strip() == (
        "record requiere hardware y el formato real de la fuente CSI; aún no implementado"
    )
