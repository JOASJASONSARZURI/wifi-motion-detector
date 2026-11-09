"""Pruebas SYNTHETIC a mano de reproducción JSONL para la GUI."""

from __future__ import annotations

import time

import numpy as np

from wifi_motion.calibration import fit_baseline, save_baseline
from wifi_motion.gui.replay_source import ReplaySource, ReplaySourceError
from wifi_motion.io_jsonl import write_recording
from wifi_motion.synthetic import generate_session


def _baseline(path) -> None:
    rng = np.random.default_rng(42)
    baseline = fit_baseline(
        "node-1",
        rng.normal(size=(40, 15)),
        "SYNTHETIC fixture",
        created_at=1_700_000_000.0,
    )
    save_baseline(baseline, path)


def _wait_for_replay(source: ReplaySource):
    results = []
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        results.extend(source.poll())
        if not source._thread.is_alive():
            results.extend(source.poll())
            return results
        time.sleep(0.01)
    raise AssertionError("la reproducción SYNTHETIC no terminó a tiempo")


def test_replay_source_processes_synthetic_jsonl_in_worker(tmp_path) -> None:
    recording_path = tmp_path / "synthetic.jsonl"
    baseline_path = tmp_path / "baseline.json"
    metadata, samples = generate_session(
        [("absent", 8.0), ("present_still", 8.0), ("present_moving", 8.0)],
        session_id="synthetic-replay",
        node_id="node-1",
        rate_hz=20.0,
        n_subcarriers=8,
        seed=15,
    )
    write_recording(recording_path, metadata, samples)
    _baseline(baseline_path)

    source = ReplaySource(
        recording_path,
        classifier="threshold",
        baseline_path=baseline_path,
    )
    results = _wait_for_replay(source)

    assert results
    assert source.evidence == "SYNTHETIC"
    assert all(result.node_id == "node-1" for result in results)
    assert all(
        set(result.scores) == {"absent", "present_still", "present_moving"}
        for result in results
    )
    stats = source.stats()
    assert stats is not None
    assert stats["samples_received"] == len(samples)
    assert stats["windows_processed"] == len(results)
    assert source.poll() == []


def test_replay_source_reports_missing_recording_as_clear_error(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    _baseline(baseline_path)
    source = ReplaySource(
        tmp_path / "missing.jsonl",
        classifier="threshold",
        baseline_path=baseline_path,
    )

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            source.poll()
        except ReplaySourceError as exc:
            assert "missing.jsonl" in str(exc)
            assert "No se pudo reproducir" in str(exc)
            break
        if not source._thread.is_alive():
            raise AssertionError("se esperaba un error de archivo")
        time.sleep(0.01)
    else:
        raise AssertionError("el error de archivo no llegó a la cola")
