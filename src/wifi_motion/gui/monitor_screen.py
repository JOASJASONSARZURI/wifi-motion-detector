"""Pantalla de monitor Tkinter; presenta valores ya preparados por la aplicación."""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont

from wifi_motion.gui import theme
from wifi_motion.gui.view_model import MonitorView
from wifi_motion.gui.widgets import PixelSymbol, RetroBar, RetroButton, RetroPanel

__all__ = ["MonitorScreen"]

_STAT_ITEMS = (
    ("MUESTRAS RECIBIDAS", "muestras_recibidas"),
    ("MUESTRAS RECHAZADAS", "muestras_rechazadas"),
    ("VENTANAS PROCESADAS", "ventanas_procesadas"),
    ("CAMBIOS DE ESTADO", "cambios_de_estado"),
    ("FRECUENCIA DE MUESTREO", "frecuencia_de_muestreo"),
    ("CALIDAD DE DATOS", "calidad_de_datos"),
)


def _font_family(master: tk.Misc) -> str:
    try:
        available = {name.casefold(): name for name in tkfont.families(master)}
    except tk.TclError:
        return theme.FONT_FAMILY
    for candidate in theme.FONT_FAMILIES:
        if candidate.casefold() in available:
            return available[candidate.casefold()]
    return theme.FONT_FAMILIES[-1]


class MonitorScreen(tk.Frame):
    """Composición visual de estado, métricas, log y controles."""

    def __init__(self, master: tk.Misc, **kwargs: object) -> None:
        super().__init__(master, bg=theme.BACKGROUND, **kwargs)
        self._font = _font_family(self)
        self._configure_grid()
        self._build_header()
        self._build_detection_area()
        self._build_stats()
        self._build_log()
        self._build_controls()

    def _configure_grid(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        for row, weight in enumerate((0, 5, 2, 2, 0)):
            self.grid_rowconfigure(row, weight=weight)

    def _label(
        self,
        master: tk.Misc,
        *,
        text: str = "",
        size: int = theme.FONT_SIZE_NORMAL,
        color: str = theme.TEXT,
        **kwargs: object,
    ) -> tk.Label:
        return tk.Label(
            master,
            text=text,
            bg=theme.PANEL,
            fg=color,
            font=(self._font, size),
            **kwargs,
        )

    def _content(self, panel: RetroPanel) -> tk.Frame:
        content = tk.Frame(panel, bg=theme.PANEL)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        return content

    def _build_header(self) -> None:
        header = tk.Frame(self, bg=theme.BACKGROUND)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 4))
        header.grid_columnconfigure(0, weight=1)
        self.title_label = tk.Label(
            header,
            text="WIFI MOTION DETECTOR",
            bg=theme.BACKGROUND,
            fg=theme.TEXT,
            font=(self._font, theme.FONT_SIZE_LARGE, "bold"),
            anchor="w",
        )
        self.title_label.grid(row=0, column=0, sticky="w")
        self.subtitle_label = tk.Label(
            header,
            text="SYSTEM MONITORING v1.0",
            bg=theme.BACKGROUND,
            fg=theme.TEXT,
            font=(self._font, theme.FONT_SIZE_SMALL),
            anchor="w",
        )
        self.subtitle_label.grid(row=1, column=0, sticky="w")

        self.status_label = tk.Label(
            header,
            text="N/D",
            bg=theme.BACKGROUND,
            fg=theme.WARNING,
            font=(self._font, theme.FONT_SIZE_NORMAL, "bold"),
        )
        self.status_label.grid(row=0, column=1, rowspan=2, padx=10)
        self.node_label = tk.Label(
            header,
            text="N/D",
            bg=theme.BACKGROUND,
            fg=theme.TEXT,
            font=(self._font, theme.FONT_SIZE_NORMAL),
        )
        self.node_label.grid(row=0, column=2, rowspan=2, padx=10)
        self.activity_indicator = tk.Label(
            header,
            text="N/D",
            bg=theme.BACKGROUND,
            fg=theme.WARNING,
            font=(self._font, theme.FONT_SIZE_NORMAL, "bold"),
        )
        self.activity_indicator.grid(row=0, column=3, rowspan=2, padx=(4, 0))

    def _build_detection_area(self) -> None:
        area = tk.Frame(self, bg=theme.BACKGROUND)
        area.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        area.grid_columnconfigure(0, weight=3)
        area.grid_columnconfigure(1, weight=2)
        area.grid_rowconfigure(0, weight=1)

        self.detection_panel = RetroPanel(area, title="[ DETECTION ]")
        self.detection_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        detection = self._content(self.detection_panel)
        detection.grid_columnconfigure(0, weight=1)
        detection.grid_rowconfigure(0, weight=1)

        text_area = tk.Frame(detection, bg=theme.PANEL)
        text_area.grid(row=0, column=0, sticky="nsew")
        text_area.grid_columnconfigure(0, weight=1)
        self.headline_label = self._label(
            text_area,
            size=theme.FONT_SIZE_HEADLINE,
            color=theme.TEXT,
            anchor="w",
            justify=tk.LEFT,
            wraplength=600,
        )
        self.headline_label.grid(row=0, column=0, sticky="w", pady=4)
        self.subline_label = self._label(
            text_area,
            size=theme.FONT_SIZE_LARGE,
            anchor="w",
            justify=tk.LEFT,
            wraplength=600,
        )
        self.subline_label.grid(row=1, column=0, sticky="w", pady=4)
        self.flavor_label = self._label(
            text_area,
            size=theme.FONT_SIZE_NORMAL,
            color=theme.TEXT,
            anchor="w",
            justify=tk.LEFT,
            wraplength=600,
        )
        self.flavor_label.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.quality_label = self._label(
            text_area,
            size=theme.FONT_SIZE_SMALL,
            color=theme.ABSENT,
            anchor="w",
            justify=tk.LEFT,
            wraplength=600,
        )
        self.quality_label.grid(row=3, column=0, sticky="w", pady=(8, 0))

        self.pixel_symbol = PixelSymbol(detection, width=150, height=150)
        self.pixel_symbol.grid(row=0, column=1, sticky="nsew", padx=(10, 4))

        self.confidence_panel = RetroPanel(area, title="[ CLASSIFIER SCORES ]")
        self.confidence_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        confidence = self._content(self.confidence_panel)
        confidence.grid_columnconfigure(0, weight=1)
        self.confidence_label = self._label(
            confidence, size=theme.FONT_SIZE_HEADLINE, color=theme.PRESENT
        )
        self.confidence_label.grid(row=0, column=0, sticky="w", pady=(4, 2))
        self.confidence_bar = RetroBar(confidence, color=theme.PRESENT)
        self.confidence_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.score_labels: list[tk.Label] = []
        self.score_values: list[tk.Label] = []
        for index in range(3):
            name = self._label(confidence, anchor="w")
            value = self._label(confidence, anchor="e")
            name.grid(row=2 + index, column=0, sticky="w")
            value.grid(row=2 + index, column=1, sticky="e")
            self.score_labels.append(name)
            self.score_values.append(value)
        self.score_note = self._label(
            confidence,
            text="puntuación del clasificador, no probabilidad",
            size=theme.FONT_SIZE_SMALL,
            color=theme.ABSENT,
            anchor="w",
            justify=tk.LEFT,
            wraplength=320,
        )
        self.score_note.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_stats(self) -> None:
        self.stats_panel = RetroPanel(self, title="[ STATISTICS ]")
        self.stats_panel.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)
        stats = self._content(self.stats_panel)
        for column in range(3):
            stats.grid_columnconfigure(column, weight=1)
        self.stat_values: dict[str, tk.Label] = {}
        for index, (label, key) in enumerate(_STAT_ITEMS):
            card = RetroPanel(stats)
            card.grid(
                row=index // 3,
                column=index % 3,
                sticky="nsew",
                padx=3,
                pady=3,
            )
            card.grid_columnconfigure(0, weight=1)
            self._label(
                card,
                text=label,
                size=theme.FONT_SIZE_SMALL,
                color=theme.ABSENT,
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
            value = self._label(
                card,
                text="N/D",
                size=theme.FONT_SIZE_NORMAL,
                anchor="w",
                justify=tk.LEFT,
                wraplength=320,
            )
            value.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 6))
            self.stat_values[key] = value

    def _build_log(self) -> None:
        self.log_panel = RetroPanel(self, title="[ SYSTEM LOG ]")
        self.log_panel.grid(row=3, column=0, sticky="nsew", padx=10, pady=4)
        content = self._content(self.log_panel)
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            content,
            height=5,
            bg=theme.BACKGROUND,
            fg=theme.TEXT,
            insertbackground=theme.TEXT,
            font=(self._font, theme.FONT_SIZE_SMALL),
            relief=tk.FLAT,
            bd=0,
            wrap=tk.NONE,
            state=tk.DISABLED,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(content, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _build_controls(self) -> None:
        controls = tk.Frame(self, bg=theme.BACKGROUND)
        controls.grid(row=4, column=0, sticky="ew", padx=10, pady=(4, 10))
        for column in range(4):
            controls.grid_columnconfigure(column, weight=1)
        self.start_button = RetroButton(controls, "INICIAR")
        self.stop_button = RetroButton(controls, "DETENER")
        self.calibrate_button = RetroButton(controls, "CALIBRAR")
        self.export_button = RetroButton(controls, "EXPORTAR")
        for column, button in enumerate(
            (
                self.start_button,
                self.stop_button,
                self.calibrate_button,
                self.export_button,
            )
        ):
            button.grid(row=0, column=column, sticky="ew", padx=4)

    def render(self, view: MonitorView, stats: dict[str, str]) -> None:
        """Coloca los textos preparados y actualiza barra y símbolo."""
        self.headline_label.configure(text=view.headline, fg=view.color)
        self.subline_label.configure(text=view.subline, fg=view.color)
        self.flavor_label.configure(text=view.flavor)
        self.quality_label.configure(text=view.quality_text)
        self.confidence_label.configure(text=view.confidence_text, fg=view.color)
        self.confidence_bar._color = view.color
        self.confidence_bar.set_ratio(view.confidence_ratio)
        self.pixel_symbol.set_symbol(view.symbol, view.color)
        for index, (label, value) in enumerate(view.score_rows):
            self.score_labels[index].configure(text=label)
            self.score_values[index].configure(text=value)
        for _label, key in _STAT_ITEMS:
            self.stat_values[key].configure(text=stats.get(key, "N/D"))
        self.status_label.configure(text=stats.get("connection_status", "N/D"))
        self.node_label.configure(text=stats.get("node_id", "N/D"))
        self.activity_indicator.configure(
            text=stats.get("activity_indicator", "N/D")
        )

    def add_log(self, line: str) -> None:
        """Agrega una línea y conserva como máximo las cincuenta más recientes."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{line}\n")
        line_count = len(self.log_text.get("1.0", "end-1c").splitlines())
        excess = line_count - 50
        if excess > 0:
            self.log_text.delete("1.0", f"{excess + 1}.0")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
