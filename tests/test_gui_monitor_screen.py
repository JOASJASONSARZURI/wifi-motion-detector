"""Prueba SYNTHETIC de humo para la pantalla Tkinter."""

from __future__ import annotations

import tkinter as tk

import pytest

from wifi_motion.gui.monitor_screen import MonitorScreen
from wifi_motion.gui.view_model import build_monitor_view


def _root_or_skip() -> tk.Tk:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no hay pantalla Tk disponible: {exc}")
    root.withdraw()
    return root


def test_monitor_screen_renders_stats_controls_and_caps_log() -> None:
    root = _root_or_skip()
    try:
        screen = MonitorScreen(root)
        screen.pack(fill=tk.BOTH, expand=True)
        root.update_idletasks()
        screen.render(
            build_monitor_view(None),
            {
                "muestras_recibidas": "N/D",
                "muestras_rechazadas": "0",
                "ventanas_procesadas": "N/D",
                "cambios_de_estado": "N/D",
                "frecuencia_de_muestreo": "N/D",
                "calidad_de_datos": "N/D",
                "connection_status": "OFFLINE",
                "node_id": "N/D",
                "activity_indicator": "N/D",
            },
        )
        root.update_idletasks()

        assert screen.headline_label.cget("text") == "ESPERANDO DATOS"
        assert screen.status_label.cget("text") == "OFFLINE"
        assert screen.node_label.cget("text") == "N/D"
        assert screen.stat_values["muestras_rechazadas"].cget("text") == "0"
        assert screen.start_button.cget("text") == "[ INICIAR ]"
        assert screen.stop_button.cget("text") == "[ DETENER ]"
        assert screen.calibrate_button.cget("text") == "[ CALIBRAR ]"
        assert screen.export_button.cget("text") == "[ EXPORTAR ]"

        for index in range(55):
            screen.add_log(f"SYNTHETIC línea {index}")
        root.update_idletasks()
        displayed_lines = screen.log_text.get("1.0", "end-1c").splitlines()
        assert len(displayed_lines) == 50
        assert displayed_lines[0] == "SYNTHETIC línea 5"
        assert displayed_lines[-1] == "SYNTHETIC línea 54"
    finally:
        root.destroy()
