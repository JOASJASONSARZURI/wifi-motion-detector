"""Punto de entrada para ``python -m wifi_motion.gui``."""

import tkinter as tk

from wifi_motion.gui.app import App


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
