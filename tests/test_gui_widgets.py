"""Pruebas SYNTHETIC de humo para widgets Tkinter."""

from __future__ import annotations

import tkinter as tk

import pytest

from wifi_motion.gui import theme
from wifi_motion.gui.widgets import PixelSymbol, RetroBar, RetroButton, RetroPanel


def _root_or_skip() -> tk.Tk:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no hay pantalla Tk disponible: {exc}")
    root.withdraw()
    return root


def test_retro_widgets_can_be_created_configured_and_drawn() -> None:
    root = _root_or_skip()
    invoked: list[bool] = []
    try:
        panel = RetroPanel(root, title="SYNTHETIC")
        panel.pack()
        panel.configure(bg=theme.PANEL)

        button = RetroButton(
            panel,
            "INICIAR",
            command=lambda: invoked.append(True),
        )
        button.pack()
        button._on_enter(None)
        assert button.cget("text") == "[ INICIAR ]"
        assert button.cget("fg") == theme.HOVER
        button._on_leave(None)
        button.configure(state=tk.DISABLED)
        root.update_idletasks()
        assert button.cget("disabledforeground") == theme.ABSENT
        button.configure(state=tk.NORMAL)
        button.invoke()
        assert invoked == [True]

        bar = RetroBar(panel, color=theme.PRESENT)
        bar.pack()
        bar.set_ratio(0.5)
        root.update_idletasks()
        assert len(bar.find_all()) >= 2
        bar.set_ratio(None)
        root.update_idletasks()
        assert len(bar.find_all()) == 1

        symbol = PixelSymbol(panel, cell_size=4)
        symbol.pack()
        symbol.set_symbol("heart", theme.PRESENT)
        root.update_idletasks()
        assert len(symbol.find_all()) > 0
        symbol.set_symbol("heart_moving", theme.WARNING)
        root.update_idletasks()
        assert len(symbol.find_all()) > 0
        symbol.set_symbol("warning", theme.WARNING)
        root.update_idletasks()
        assert len(symbol.find_all()) > 0
        symbol.set_symbol("none", theme.TEXT)
        root.update_idletasks()
        assert len(symbol.find_all()) == 0
    finally:
        root.destroy()
