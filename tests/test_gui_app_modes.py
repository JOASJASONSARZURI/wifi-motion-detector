"""Pruebas SYNTHETIC del selector de fuentes de la GUI."""

from __future__ import annotations

import tkinter as tk

import pytest

from wifi_motion.gui import app as app_module
from wifi_motion.gui.app import App
from wifi_motion.gui.sources import DemoSource


def _root_or_skip() -> tk.Tk:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no hay pantalla Tk disponible: {exc}")
    root.withdraw()
    return root


def test_gui_can_select_jsonl_replay_and_demo(monkeypatch) -> None:
    root = _root_or_skip()
    chosen_paths = iter(("recording.jsonl", "baseline.json"))
    captured: dict[str, object] = {}

    class FakeReplaySource:
        evidence = "SYNTHETIC"

        def __init__(self, input_path, **kwargs) -> None:
            captured["input_path"] = input_path
            captured.update(kwargs)

        def poll(self):
            return []

        def stats(self):
            return None

    monkeypatch.setattr(
        app_module.filedialog,
        "askopenfilename",
        lambda **_kwargs: next(chosen_paths),
    )
    monkeypatch.setattr(
        app_module.messagebox,
        "askyesnocancel",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(app_module, "ReplaySource", FakeReplaySource)

    application = App(root)
    try:
        application.open_replay()
        assert captured == {
            "input_path": "recording.jsonl",
            "classifier": "threshold",
            "baseline_path": "baseline.json",
        }
        assert application.demo_banner.cget("text") == (
            "REPRODUCCIÓN JSONL - SYNTHETIC"
        )

        application.use_demo()
        assert isinstance(application.source, DemoSource)
        assert application.demo_banner.cget("text") == "MODO DEMO - SYNTHETIC"
    finally:
        application.destroy()
        root.destroy()
