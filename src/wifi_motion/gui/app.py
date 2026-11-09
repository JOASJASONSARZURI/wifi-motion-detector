"""Aplicación visual Tkinter que presenta resultados de una fuente."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

from wifi_motion.detector import DetectionResult
from wifi_motion.gui import theme
from wifi_motion.gui.monitor_screen import MonitorScreen
from wifi_motion.gui.replay_source import ReplaySource
from wifi_motion.gui.sources import DemoSource, ResultSource
from wifi_motion.gui.view_model import build_monitor_view, format_stats, log_line
from wifi_motion.gui.widgets import RetroButton

__all__ = ["App"]


class App(tk.Frame):
    """Conecta una fuente de resultados a la pantalla sin ejecutar detección."""

    def __init__(
        self,
        root: tk.Tk,
        source: ResultSource | None = None,
    ) -> None:
        super().__init__(root, bg=theme.BACKGROUND)
        self.root = root
        self.source = DemoSource() if source is None else source
        self._is_demo = isinstance(self.source, DemoSource)
        self._running = False
        self._after_id: str | None = None
        self._typewriter_id: str | None = None
        self._log_lines: list[str] = []
        self._last_result: DetectionResult | None = None
        self._last_state: str | None = None

        root.title("WIFI MOTION DETECTOR")
        root.configure(bg=theme.BACKGROUND)
        self.pack(fill=tk.BOTH, expand=True)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        toolbar = tk.Frame(self, bg=theme.BACKGROUND)
        toolbar.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 0))
        toolbar.grid_columnconfigure(0, weight=1)
        self.demo_banner = tk.Label(
            toolbar,
            text="MODO DEMO - SYNTHETIC" if self._is_demo else "",
            bg=theme.BACKGROUND,
            fg=theme.WARNING,
            font=(theme.FONT_FAMILY, theme.FONT_SIZE_NORMAL, "bold"),
            anchor="w",
        )
        self.demo_banner.grid(row=0, column=0, sticky="ew")
        self.demo_button = RetroButton(
            toolbar, "DEMO SYNTHETIC", command=self.use_demo
        )
        self.demo_button.grid(row=0, column=1, padx=(8, 0))
        self.replay_button = RetroButton(
            toolbar, "REPRODUCIR JSONL", command=self.open_replay
        )
        self.replay_button.grid(row=0, column=2, padx=(8, 0))

        self.screen = MonitorScreen(self)
        self.screen.grid(row=1, column=0, sticky="nsew")
        self.screen.start_button.configure(command=self.start)
        self.screen.stop_button.configure(command=self.stop)
        self.screen.calibrate_button.configure(
            text="[ N/D ]",
            state=tk.DISABLED,
        )
        self.screen.export_button.configure(command=self.export_log)
        self.message_label = tk.Label(
            self,
            text="",
            bg=theme.BACKGROUND,
            fg=theme.TEXT,
            font=(theme.FONT_FAMILY, theme.FONT_SIZE_SMALL),
            anchor="w",
        )
        self.message_label.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        self._update_source_banner()
        self._render_status("OFFLINE", None, None)

    def start(self) -> None:
        """Inicia el sondeo periódico sin bloquear el bucle Tkinter."""
        if self._running:
            return
        self._running = True
        self._type_message("SISTEMA INICIADO")
        self._poll()

    def stop(self) -> None:
        """Detiene el sondeo pendiente."""
        self._running = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._render_status("OFFLINE", self._last_result, None)

    def use_demo(self) -> None:
        """Cambia a la fuente de demostración SYNTHETIC."""
        self._switch_source(DemoSource())

    def open_replay(self) -> None:
        """Selecciona una grabación JSONL y el clasificador a reproducir."""
        input_path = filedialog.askopenfilename(
            title="Seleccionar grabación JSONL",
            filetypes=(("Grabaciones JSONL", "*.jsonl"), ("Todos los archivos", "*.*")),
        )
        if not input_path:
            return

        use_threshold = messagebox.askyesnocancel(
            "Seleccionar clasificador",
            "¿Usar el clasificador por umbral?\n\n"
            "Sí: umbral\nNo: softmax\nCancelar: volver",
            parent=self.root,
        )
        if use_threshold is None:
            return

        title = "Seleccionar línea base" if use_threshold else "Seleccionar modelo softmax"
        filetypes = (
            ("Archivos JSON", "*.json"),
            ("Todos los archivos", "*.*"),
        )
        artifact_path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
        )
        if not artifact_path:
            return

        try:
            if use_threshold:
                source = ReplaySource(
                    input_path,
                    classifier="threshold",
                    baseline_path=artifact_path,
                )
            else:
                source = ReplaySource(
                    input_path,
                    classifier="softmax",
                    model_path=artifact_path,
                )
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "No se pudo abrir la reproducción",
                str(exc) or "Error desconocido",
                parent=self.root,
            )
            return
        self._switch_source(source)

    def _switch_source(self, source: ResultSource) -> None:
        self.stop()
        self.source = source
        self._last_result = None
        self._last_state = None
        self._update_source_banner()
        self._render_status("OFFLINE", None, None)
        self._append_log(f"Fuente seleccionada: {self.demo_banner.cget('text')}")
        self.start()

    def _update_source_banner(self) -> None:
        if isinstance(self.source, DemoSource):
            text = "MODO DEMO - SYNTHETIC"
        elif isinstance(self.source, ReplaySource):
            if self.source.evidence == "SYNTHETIC":
                text = "REPRODUCCIÓN JSONL - SYNTHETIC"
            else:
                text = "REPRODUCCIÓN JSONL - ARCHIVO (NO EN VIVO)"
        else:
            text = ""
        self.demo_banner.configure(text=text)

    def _poll(self) -> None:
        self._after_id = None
        if not self._running:
            return
        try:
            results = self.source.poll()
            source_stats = self.source.stats()
        except Exception as exc:
            self._append_log(f"ERROR: fuente offline: {str(exc) or 'N/D'}")
            self._render_status("OFFLINE", self._last_result, None)
            self._after_id = self.after(200, self._poll)
            return

        self._update_source_banner()
        for result in results:
            self._last_result = result
            if result.state != self._last_state:
                self._type_message(result.state)
                self._last_state = result.state
            event_line = log_line(result)
            if event_line is not None:
                self._append_log(event_line)

        self._render_status("ONLINE", self._last_result, source_stats)
        self._after_id = self.after(200, self._poll)

    def _render_status(
        self,
        connection: str,
        result: DetectionResult | None,
        source_stats: dict[str, object] | None,
    ) -> None:
        formatted = format_stats(source_stats, result)
        formatted["connection_status"] = connection
        formatted["node_id"] = "N/D" if result is None else result.node_id or "N/D"
        formatted["activity_indicator"] = "●" if connection == "ONLINE" else "N/D"
        self.screen.render(build_monitor_view(result), formatted)

    def _type_message(self, message: str) -> None:
        if self._typewriter_id is not None:
            self.after_cancel(self._typewriter_id)
            self._typewriter_id = None
        self.message_label.configure(text="")

        def reveal(position: int) -> None:
            self.message_label.configure(text=message[:position])
            if position < len(message):
                self._typewriter_id = self.after(25, reveal, position + 1)
            else:
                self._typewriter_id = None

        reveal(1 if message else 0)

    def _append_log(self, line: str) -> None:
        self._log_lines.append(line)
        self._log_lines = self._log_lines[-50:]
        self.screen.add_log(line)

    def export_log(self) -> None:
        """Exporta el log visible a un archivo de texto elegido por el usuario."""
        destination = filedialog.asksaveasfilename(
            title="Exportar system log",
            defaultextension=".txt",
            filetypes=(("Archivo de texto", "*.txt"),),
        )
        if not destination:
            return
        try:
            with open(destination, "w", encoding="utf-8") as output:
                output.write("\n".join(self._log_lines))
                if self._log_lines:
                    output.write("\n")
        except OSError as exc:
            self._append_log(f"ERROR: no se pudo exportar el log: {str(exc) or 'N/D'}")

    def destroy(self) -> None:
        """Cancela callbacks pendientes antes de destruir la interfaz."""
        self.stop()
        if self._typewriter_id is not None:
            self.after_cancel(self._typewriter_id)
            self._typewriter_id = None
        super().destroy()
