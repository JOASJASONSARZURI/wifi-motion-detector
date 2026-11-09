"""Widgets Tkinter rectangulares con el tema retro del proyecto."""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable

from wifi_motion.gui import theme

__all__ = ["PixelSymbol", "RetroBar", "RetroButton", "RetroPanel"]

_PIXEL_MAPS = {
    "none": (),
    "heart": (
        "  XX   XX  ",
        " XXXX XXXX ",
        "XXXXXXXXXXX",
        "XXXXXXXXXXX",
        " XXXXXXXXX ",
        "  XXXXXXX  ",
        "   XXXXX   ",
        "    XXX    ",
        "     X     ",
    ),
    "heart_moving": (
        "  XX   XX     ",
        " XXXX XXXX  X ",
        "XXXXXXXXXXX XX",
        "XXXXXXXXXXX XX",
        " XXXXXXXXX  X ",
        "  XXXXXXX     ",
        "   XXXXX      ",
        "    XXX       ",
        "     X        ",
    ),
    "warning": (
        "      X      ",
        "     XXX     ",
        "     XXX     ",
        "    XXXXX    ",
        "    XXXXX    ",
        "   XXXXXXX   ",
        "   XXXXXXX   ",
        "  XXXXXXXXX  ",
        "  XXXXXXXXX  ",
        " XXXXXXXXXXX ",
        " XXXXXXXXXXX ",
        "XXXXXXXXXXXXX",
        "     XXX     ",
        "     XXX     ",
        "             ",
    ),
}


def _font_family(master: tk.Misc) -> str:
    """Escoge la primera fuente instalada según el orden del tema."""
    try:
        available = {name.casefold(): name for name in tkfont.families(master)}
    except tk.TclError:
        return theme.FONT_FAMILY
    for candidate in theme.FONT_FAMILIES:
        if candidate.casefold() in available:
            return available[candidate.casefold()]
    return theme.FONT_FAMILIES[-1]


class RetroPanel(tk.Frame):
    """Panel negro rectangular con borde blanco y título opcional."""

    def __init__(
        self,
        master: tk.Misc,
        title: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            master,
            bg=theme.PANEL,
            bd=0,
            relief=tk.FLAT,
            highlightthickness=2,
            highlightbackground=theme.BORDER,
            highlightcolor=theme.BORDER,
            **kwargs,
        )
        self.title_label: tk.Label | None = None
        if title:
            self.title_label = tk.Label(
                self,
                text=title,
                bg=theme.PANEL,
                fg=theme.TEXT,
                font=(_font_family(self), theme.FONT_SIZE_NORMAL),
                anchor="w",
            )
            self.title_label.pack(fill=tk.X, padx=8, pady=(6, 2))


class RetroButton(tk.Button):
    """Botón entre corchetes con foco visual amarillo al pasar el cursor."""

    def __init__(
        self,
        master: tk.Misc,
        text: str,
        command: Callable[[], object] | None = None,
        **kwargs: object,
    ) -> None:
        self._label = text
        self._enabled_foreground = str(kwargs.pop("fg", theme.TEXT))
        super().__init__(
            master,
            text=f"[ {text} ]",
            command=command,
            bg=theme.BACKGROUND,
            fg=self._enabled_foreground,
            activebackground=theme.BACKGROUND,
            activeforeground=theme.HOVER,
            disabledforeground=theme.ABSENT,
            font=(_font_family(master), theme.FONT_SIZE_NORMAL),
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=theme.BORDER,
            highlightcolor=theme.BORDER,
            takefocus=True,
            **kwargs,
        )
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _event: tk.Event[tk.Misc]) -> None:
        if str(self.cget("state")) != tk.DISABLED:
            self.configure(fg=theme.HOVER, highlightbackground=theme.HOVER)

    def _on_leave(self, _event: tk.Event[tk.Misc]) -> None:
        disabled = str(self.cget("state")) == tk.DISABLED
        self.configure(
            fg=theme.ABSENT if disabled else self._enabled_foreground,
            highlightbackground=theme.BORDER,
        )


class RetroBar(tk.Canvas):
    """Barra de progreso rectangular; ``None`` limpia el indicador."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        color: str = theme.PRESENT,
        **kwargs: object,
    ) -> None:
        kwargs.setdefault("height", 18)
        kwargs.setdefault("width", 220)
        super().__init__(
            master,
            bg=theme.PANEL,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._color = color
        self._ratio: float | None = None
        self.bind("<Configure>", self._redraw)
        self._redraw()

    def set_ratio(self, ratio: float | None) -> None:
        """Actualiza el relleno; el valor se limita al intervalo visible."""
        if ratio is not None:
            value = float(ratio)
            if not 0.0 <= value <= 1.0:
                value = min(1.0, max(0.0, value))
            self._ratio = value
        else:
            self._ratio = None
        self._redraw()

    def _redraw(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        if self._ratio is not None and self._ratio > 0.0:
            fill_right = max(1, round((width - 2) * self._ratio))
            self.create_rectangle(
                1,
                1,
                min(width - 1, fill_right),
                height - 1,
                fill=self._color,
                outline="",
            )
        self.create_rectangle(
            0,
            0,
            width - 1,
            height - 1,
            outline=theme.BORDER,
            width=1,
        )


class PixelSymbol(tk.Canvas):
    """Canvas que representa símbolos con celdas rectangulares."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        cell_size: int = 8,
        **kwargs: object,
    ) -> None:
        kwargs.setdefault("width", 120)
        kwargs.setdefault("height", 120)
        super().__init__(
            master,
            bg=theme.BACKGROUND,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._cell_size = max(1, int(cell_size))
        self._symbol = "none"
        self._color = theme.TEXT
        self.bind("<Configure>", self._redraw)
        self._redraw()

    def set_symbol(self, name: str, color: str) -> None:
        """Dibuja ``none``, ``heart``, ``heart_moving`` o ``warning``."""
        if name not in _PIXEL_MAPS:
            raise ValueError(f"símbolo desconocido: {name!r}")
        self._symbol = name
        self._color = color
        self._redraw()

    def _redraw(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self.delete("all")
        rows = _PIXEL_MAPS[self._symbol]
        if not rows:
            return
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        columns = max(len(row) for row in rows)
        cell = min(
            self._cell_size,
            max(1, width // columns),
            max(1, height // len(rows)),
        )
        left = (width - columns * cell) // 2
        top = (height - len(rows) * cell) // 2
        for row_index, row in enumerate(rows):
            for column_index, pixel in enumerate(row):
                if pixel != "X":
                    continue
                x1 = left + column_index * cell
                y1 = top + row_index * cell
                self.create_rectangle(
                    x1,
                    y1,
                    x1 + cell,
                    y1 + cell,
                    fill=self._color,
                    outline=self._color,
                )
