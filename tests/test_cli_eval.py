"""Integración SYNTHETIC del subcomando eval."""

from __future__ import annotations

import json

import numpy as np

from wifi_motion.calibration import fit_baseline, save_baseline
from wifi_motion.classifiers.softmax import SoftmaxClassifier, SoftmaxConfig
from wifi_motion.cli import main
from wifi_motion.features import FEATURE_NAMES
from wifi_motion.io_jsonl import write_recording
from wifi_motion.synthetic import generate_session


def _recording(path, session_id: str) -> None:
    metadata, samples = generate_session(
        [
            ("absent", 8.0),
            ("present_still", 8.0),
            ("present_moving", 8.0),
        ],
        session_id=session_id,
        rate_hz=20.0,
        n_subcarriers=8,
        seed=17,
    )
    write_recording(path, metadata, samples)


def _artifacts(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    test = tmp_path / "test.jsonl"
    _recording(train, "syn-train")
    _recording(val, "syn-val")
    _recording(test, "syn-test")

    rng = np.random.default_rng(22)
    baseline = fit_baseline(
        "node-1",
        rng.normal(size=(40, len(FEATURE_NAMES))),
        "SYNTHETIC test fixture",
        created_at=1.0,
    )
    baseline_path = tmp_path / "baseline.json"
    save_baseline(baseline, baseline_path)

    vectors = []
    labels = []
    for index, state in enumerate(("absent", "present_still", "present_moving")):
        for _ in range(8):
            vector = rng.normal(0.0, 0.1, len(FEATURE_NAMES))
            vector[0] += index * 3.0
            vector[1] += index * 2.0
            vector[FEATURE_NAMES.index("mean_rssi")] = -50.0 + index
            vectors.append(vector)
            labels.append(state)
    model = SoftmaxClassifier().fit(
        np.asarray(vectors),
        labels,
        np.asarray(vectors),
        labels,
        SoftmaxConfig(epochs=3),
    )
    model_path = tmp_path / "model.json"
    model.save(model_path)
    return train, val, test, baseline_path, model_path


def _args(train, val, test, baseline, model, output) -> list[str]:
    return [
        "eval",
        "--train",
        str(train),
        "--val",
        str(val),
        "--test",
        str(test),
        "--baseline",
        str(baseline),
        "--model",
        str(model),
        "--out-dir",
        str(output),
        "--window-s",
        "4",
        "--hop-s",
        "2",
        "--min-samples",
        "40",
        "--max-gap-s",
        "0.2",
        "--transition-tolerance-s",
        "0",
    ]


def test_eval_writes_comparison_reports_and_synthetic_evidence(
    tmp_path, capsys
) -> None:
    train, val, test, baseline, model = _artifacts(tmp_path)
    output = tmp_path / "evaluation"

    assert main(_args(train, val, test, baseline, model, output)) == 0

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    markdown = (output / "report.md").read_text(encoding="utf-8")
    log_lines = (output / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert report["evidence"] == "SYNTHETIC"
    assert report["session_ids"] == ["syn-test"]
    assert report["n_windows"] > 0
    assert set(report["comparison"]) == {"baseline", "threshold", "softmax"}
    assert report["parameters"]["threshold_config"]["tuned_on_validation"] is True
    assert all(
        report["comparison"][name]["window"]["n_windows"] > 0
        for name in ("baseline", "threshold", "softmax")
    )
    assert all(f"| {name} |" in markdown for name in ("baseline", "threshold", "softmax"))
    assert len(log_lines) == 1
    assert capsys.readouterr().err == ""


def test_eval_rejects_overlapping_session_ids(tmp_path, capsys) -> None:
    train, val, test, baseline, model = _artifacts(tmp_path)
    _recording(val, "syn-train")

    code = main(_args(train, val, test, baseline, model, tmp_path / "out"))

    assert code != 0
    assert "solapadas" in capsys.readouterr().err


def test_eval_warns_when_same_test_and_models_are_reused(
    tmp_path, capsys
) -> None:
    train, val, test, baseline, model = _artifacts(tmp_path)
    output = tmp_path / "evaluation"
    arguments = _args(train, val, test, baseline, model, output)

    assert main(arguments) == 0
    capsys.readouterr()
    assert main(arguments) == 0

    assert "este conjunto de prueba ya se usó; el resultado deja de ser una evaluación única" in capsys.readouterr().err
    assert len((output / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_eval_missing_file_has_clean_error(tmp_path, capsys) -> None:
    train, val, _test, baseline, model = _artifacts(tmp_path)
    arguments = _args(
        train,
        val,
        tmp_path / "missing.jsonl",
        baseline,
        model,
        tmp_path / "out",
    )

    code = main(arguments)
    error = capsys.readouterr().err

    assert code != 0
    assert "error:" in error
    assert "missing.jsonl" in error
    assert "Traceback" not in error
