
"""Ventanas temporales por nodo a partir de un flujo de ``CsiSample``.

Reglas (ver también las decisiones de diseño en ``SlidingWindower``):

- Los límites de las ventanas salen SOLO de las marcas de tiempo, nunca del
  número de muestras. Cada nodo tiene su propia rejilla de ventanas, anclada en
  su primera muestra utilizable: ``[t0 + k*hop_s, t0 + k*hop_s + window_s)``.
- Nunca se interpola ni se rellena: una ventana contiene exactamente las
  muestras reales que cayeron en ella, y los huecos se marcan con ``HAS_GAP``.
- Las muestras que ``validation.is_usable`` rechaza se descartan y se cuentan
  en ``SlidingWindower.dropped``; nunca entran en una ventana.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any

import numpy as np

from wifi_motion.schema import CsiSample
from wifi_motion.validation import IssueCode, StreamValidator, is_usable

__all__ = ["SlidingWindower", "Window", "WindowQuality", "make_windows"]


class WindowQuality(str, Enum):
    """Calidad de una ventana. Solo ``OK`` es apta sin reservas."""

    OK = "ok"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    HAS_GAP = "has_gap"


@dataclass(frozen=True, slots=True, eq=False)
class Window:
    """Ventana temporal de un nodo, con sus muestras reales (sin rellenar).

    ``t_start``/``t_end`` son los límites nominales de la ventana (intervalo
    semiabierto ``[t_start, t_end)``), no la primera/última muestra. Los arrays
    que construye ``SlidingWindower`` son de solo lectura. ``eq=False``: la
    igualdad es por identidad, porque comparar arrays con ``==`` es ambiguo.
    """

    node_id: str
    t_start: float
    t_end: float
    timestamps: np.ndarray  # (n,)
    amplitudes: np.ndarray  # (n, n_subcarriers)
    rssi: np.ndarray | None  # (n,), solo si TODAS las muestras traen un RSSI válido
    estimated_rate_hz: float  # 1 / mediana de diferencias de timestamps; 0.0 si no se puede
    quality: WindowQuality


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _estimate_rate_hz(timestamps: np.ndarray) -> float:
    """1 / mediana de las diferencias entre timestamps; 0.0 si no hay forma."""
    if len(timestamps) < 2:
        return 0.0
    median_dt = float(np.median(np.diff(timestamps)))
    if not math.isfinite(median_dt) or median_dt <= 0.0:
        return 0.0
    return 1.0 / median_dt


@dataclass(slots=True)
class _NodeState:
    """Estado mutable de un nodo."""

    t0: float | None = None  # ancla de la rejilla: primera muestra utilizable
    next_k: int = 0  # índice de la próxima ventana por emitir
    # (timestamp, amplitudes (n_subcarriers,), rssi válido o None)
    buffer: deque[tuple[float, np.ndarray, float | None]] = field(default_factory=deque)
    last_ts: float | None = None  # timestamp de la última muestra utilizable


class SlidingWindower:
    """Ventanas deslizantes por ``node_id`` a partir de muestras en orden.

    Decisiones de diseño (la opción más simple y la que no oculta datos malos):

    - Una ventana se emite cuando llega una muestra de ese nodo con
      ``timestamp >= t_end``; una muestra justo en ``t_end`` pertenece a la
      ventana siguiente. Las últimas ventanas incompletas de un flujo no se
      emiten (el fin de los datos no es un hueco).
    - Hueco: la mayor distancia sin muestras dentro de la ventana, medida sobre
      ``[t_start, ts_1, ..., ts_n, t_end]``, es estrictamente mayor que
      ``max_gap_s``. Así también se ven los huecos al borde de la ventana, no
      solo entre muestras. Si coinciden hueco y pocas muestras, gana HAS_GAP.
    - Las ventanas sin ninguna muestra no se emiten (una caída larga no
      genera miles de ventanas vacías); el hueco queda visible en las ventanas
      vecinas con HAS_GAP, en el salto de ``t_start`` entre ventanas
      consecutivas y en ``nodes_stale``.
    - ``rssi`` de la ventana es un array solo si todas sus muestras traen un
      RSSI finito; si falta alguno o no es finito, es ``None`` (no se rellena).
    - El orden y la integridad del flujo los garantiza un ``StreamValidator``
      interno (con este ``max_gap_s``): duplicados, atrasadas y cambios en el
      número de subportadoras se descartan y se cuentan en ``dropped``.
    """

    def __init__(
        self, window_s: float, hop_s: float, min_samples: int, max_gap_s: float
    ) -> None:
        if not _is_finite_number(window_s) or window_s <= 0:
            raise ValueError(f"window_s debe ser un número finito > 0, no {window_s!r}")
        if not _is_finite_number(hop_s) or not 0 < hop_s <= window_s:
            raise ValueError(
                f"hop_s debe cumplir 0 < hop_s <= window_s ({window_s!r}), no {hop_s!r}"
            )
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
            raise ValueError(f"min_samples debe ser un entero >= 2, no {min_samples!r}")
        if not _is_finite_number(max_gap_s) or max_gap_s <= 0:
            raise ValueError(f"max_gap_s debe ser un número finito > 0, no {max_gap_s!r}")
        self._window_s = float(window_s)
        self._hop_s = float(hop_s)
        self._min_samples = min_samples
        self._max_gap_s = float(max_gap_s)
        self._validator = StreamValidator(max_gap_s=self._max_gap_s)
        self._nodes: dict[str, _NodeState] = {}
        # Contador público: muestras descartadas por no ser utilizables.
        self.dropped: int = 0
        self.issue_counts: Counter[IssueCode] = Counter()

    # ------------------------------------------------------------------ push

    def push(self, sample: CsiSample) -> list[Window]:
        """Procesa una muestra y devuelve las ventanas que ella completa."""
        issues = self._validator.check(sample)
        self.issue_counts.update(issue.code for issue in issues)
        node_known = all(issue.code is not IssueCode.MISSING_NODE_ID for issue in issues)
        if node_known:
            # Un nodo que solo manda basura queda registrado (y se verá obsoleto).
            state = self._nodes.setdefault(sample.node_id, _NodeState())
        if not is_usable(issues):
            self.dropped += 1
            return []

        ts = float(sample.timestamp)
        amplitudes = np.asarray(sample.amplitudes, dtype=np.float64)
        rssi = float(sample.rssi) if _is_finite_number(sample.rssi) else None  # type: ignore[arg-type]
        if state.t0 is None:
            state.t0 = ts  # la primera muestra utilizable ancla la rejilla del nodo
        windows = self._complete_windows(sample.node_id, state, ts)
        state.buffer.append((ts, amplitudes, rssi))
        state.last_ts = ts
        return windows

    def reset(self) -> None:
        """Borra el estado de ventanas y validación para iniciar un flujo nuevo."""
        self._validator = StreamValidator(max_gap_s=self._max_gap_s)
        self._nodes.clear()
        self.dropped = 0
        self.issue_counts.clear()

    def _complete_windows(self, node_id: str, state: _NodeState, ts: float) -> list[Window]:
        """Emite las ventanas del nodo que termina de completar un timestamp ``ts``."""
        assert state.t0 is not None
        t0, hop, length = state.t0, self._hop_s, self._window_s
        windows: list[Window] = []
        k = state.next_k
        while True:
            t_start = t0 + k * hop
            t_end = t_start + length
            if t_end > ts:
                break  # esta ventana aún no se completa
            buffer = state.buffer
            while buffer and buffer[0][0] < t_start:
                buffer.popleft()  # ya no hacen falta para ventanas futuras
            if not buffer:
                # Sin datos hasta ts: saltar las ventanas vacías sin emitirlas.
                k = self._first_open_index(t0, ts, k + 1)
                continue
            members = [item for item in buffer if item[0] < t_end]
            if members:
                windows.append(self._build(node_id, t_start, t_end, members))
            k += 1
        state.next_k = k
        return windows

    def _first_open_index(self, t0: float, ts: float, k_min: int) -> int:
        """Menor k >= k_min cuya ventana aún no se completó en ``ts`` (t_end > ts)."""
        hop, length = self._hop_s, self._window_s
        k = max(k_min, math.floor((ts - length - t0) / hop) + 1)
        while k > k_min and t0 + (k - 1) * hop + length > ts:  # corrige redondeo
            k -= 1
        while t0 + k * hop + length <= ts:
            k += 1
        return k

    def _build(
        self,
        node_id: str,
        t_start: float,
        t_end: float,
        members: list[tuple[float, np.ndarray, float | None]],
    ) -> Window:
        timestamps = np.array([m[0] for m in members], dtype=np.float64)
        amplitudes = np.stack([m[1] for m in members])
        rssi_values = [m[2] for m in members]
        rssi = (
            np.array(rssi_values, dtype=np.float64)
            if all(v is not None for v in rssi_values)
            else None
        )
        edges = np.concatenate(([t_start], timestamps, [t_end]))
        longest_gap = float(np.max(np.diff(edges)))
        if longest_gap > self._max_gap_s:
            quality = WindowQuality.HAS_GAP
        elif len(members) < self._min_samples:
            quality = WindowQuality.INSUFFICIENT_SAMPLES
        else:
            quality = WindowQuality.OK
        for array in (timestamps, amplitudes, rssi):
            if array is not None:
                array.flags.writeable = False
        return Window(
            node_id=node_id,
            t_start=t_start,
            t_end=t_end,
            timestamps=timestamps,
            amplitudes=amplitudes,
            rssi=rssi,
            estimated_rate_hz=_estimate_rate_hz(timestamps),
            quality=quality,
        )

    # ----------------------------------------------------------- nodes_stale

    def nodes_stale(self, now: float, timeout_s: float) -> list[str]:
        """Nodos sin muestras utilizables recientes: ``now - última > timeout_s``.

        ``now`` está en la misma escala que los timestamps de las muestras. Un
        nodo que solo ha mandado muestras descartadas cuenta como obsoleto.
        Devuelve los ids ordenados, para que quien consuma reporte "degradado".
        """
        if not _is_finite_number(now):
            raise ValueError(f"now debe ser un número finito, no {now!r}")
        if not _is_finite_number(timeout_s) or timeout_s <= 0:
            raise ValueError(f"timeout_s debe ser un número finito > 0, no {timeout_s!r}")
        return sorted(
            node_id
            for node_id, state in self._nodes.items()
            if state.last_ts is None or now - state.last_ts > timeout_s
        )


def make_windows(samples: Iterable[CsiSample], **config: Any) -> list[Window]:
    """Versión por lotes: ventanas de una grabación, en orden de emisión.

    ``config`` son los argumentos de ``SlidingWindower``. Las muestras
    descartadas no se pierden en silencio: si hay alguna se emite un
    ``RuntimeWarning`` con el total (para el contador usa ``SlidingWindower``).
    """
    windower = SlidingWindower(**config)
    windows: list[Window] = []
    for sample in samples:
        windows.extend(windower.push(sample))
    if windower.dropped:
        warnings.warn(
            f"make_windows descartó {windower.dropped} muestra(s) inutilizable(s)",
            RuntimeWarning,
            stacklevel=2,
        )
    return windows