"""Pruebas de wifi_motion.windowing.

SYNTHETIC: todos los datos de este archivo están escritos a mano; no son
mediciones reales. Los timestamps usan fracciones binarias exactas (0.125,
0.25, 0.5...) para que los límites de ventana no dependan del redondeo. Cada
muestra lleva ``amplitudes[0] == timestamp``: así se comprueba que las
amplitudes de una ventana son las de las muestras reales, sin rellenar.
"""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from wifi_motion.schema import CsiSample
from wifi_motion.windowing import SlidingWindower, Window, WindowQuality, make_windows

NAN = float("nan")
INF = float("inf")
CONFIG: dict[str, Any] = {"window_s": 1.0, "hop_s": 0.5, "min_samples": 3, "max_gap_s": 0.5}


def s(ts: float, node_id: str = "n1", **overrides: Any) -> CsiSample:
    """SYNTHETIC: muestra de 3 subportadoras con amplitudes[0] == ts."""
    base: dict[str, Any] = {
        "timestamp": ts,
        "node_id": node_id,
        "amplitudes": (ts, ts + 1.0, ts + 2.0),
    }
    base.update(overrides)
    return CsiSample(**base)


def regular(start: float, count: int, step: float = 0.25, node_id: str = "n1") -> list[CsiSample]:
    return [s(start + i * step, node_id) for i in range(count)]


def make_windower(**overrides: Any) -> SlidingWindower:
    return SlidingWindower(**{**CONFIG, **overrides})


def push_all(windower: SlidingWindower, samples: list[CsiSample]) -> list[Window]:
    windows: list[Window] = []
    for sample in samples:
        windows.extend(windower.push(sample))
    return windows


def same_window(a: Window, b: Window) -> bool:
    return (
        a.node_id == b.node_id
        and a.t_start == b.t_start
        and a.t_end == b.t_end
        and a.quality == b.quality
        and a.estimated_rate_hz == b.estimated_rate_hz
        and np.array_equal(a.timestamps, b.timestamps)
        and np.array_equal(a.amplitudes, b.amplitudes)
    )


# ------------------------------------------------------------- ventana normal


def test_window_quality_members() -> None:
    assert {q.name for q in WindowQuality} == {"OK", "INSUFFICIENT_SAMPLES", "HAS_GAP"}


def test_normal_window_completes_when_a_sample_reaches_t_end() -> None:
    w = make_windower()
    for ts in (100.0, 100.25, 100.5, 100.75):
        assert w.push(s(ts)) == []  # aún no se completa
    windows = w.push(s(101.0))  # t_end = 101.0
    assert len(windows) == 1
    win = windows[0]
    assert (win.node_id, win.t_start, win.t_end) == ("n1", 100.0, 101.0)
    assert win.quality is WindowQuality.OK
    assert win.estimated_rate_hz == 4.0
    assert win.rssi is None
    assert win.timestamps.tolist() == [100.0, 100.25, 100.5, 100.75]
    assert win.amplitudes.shape == (4, 3)
    assert win.timestamps.dtype == np.float64 and win.amplitudes.dtype == np.float64
    assert np.array_equal(win.amplitudes[:, 0], win.timestamps)  # sin rellenar
    assert w.dropped == 0


def test_window_interval_is_half_open() -> None:
    w = make_windower()
    first = push_all(w, regular(100.0, 5))  # la muestra de 101.0 cierra la ventana
    assert 101.0 not in first[0].timestamps.tolist()
    second = push_all(w, regular(101.25, 2))  # 101.25 y 101.5; 101.5 cierra [100.5, 101.5)
    assert 101.0 in second[0].timestamps.tolist()


def test_first_window_is_anchored_at_first_usable_sample() -> None:
    w = make_windower()
    assert w.push(s(99.0, amplitudes=(1.0, NAN, 3.0))) == []  # descartada: no ancla
    windows = push_all(w, regular(250.5, 5))
    assert windows[0].t_start == 250.5 and windows[0].t_end == 251.5


def test_window_boundaries_depend_on_timestamps_not_on_sample_count() -> None:
    four_hz = make_windows(regular(100.0, 9, step=0.25), **CONFIG)
    eight_hz = make_windows(regular(100.0, 17, step=0.125), **CONFIG)
    bounds = [(100.0, 101.0), (100.5, 101.5), (101.0, 102.0)]
    assert [(x.t_start, x.t_end) for x in four_hz] == bounds
    assert [(x.t_start, x.t_end) for x in eight_hz] == bounds
    assert [len(x.timestamps) for x in four_hz] == [4, 4, 4]
    assert [len(x.timestamps) for x in eight_hz] == [8, 8, 8]


def test_overlap_by_hop() -> None:
    w = make_windower()
    emitted_at: list[float] = []
    windows: list[Window] = []
    for sample in regular(100.0, 9):  # 100.0 ... 102.0
        new = w.push(sample)
        windows.extend(new)
        emitted_at.extend([sample.timestamp] * len(new))
    assert emitted_at == [101.0, 101.5, 102.0]
    assert [x.t_start for x in windows] == [100.0, 100.5, 101.0]
    shared = set(windows[0].timestamps.tolist()) & set(windows[1].timestamps.tolist())
    assert shared == {100.5, 100.75}  # solapamiento = window_s - hop_s
    assert all(len(x.timestamps) == 4 for x in windows)


# --------------------------------------------------------------------- huecos


def test_gap_in_the_middle_is_marked_and_nothing_is_filled() -> None:
    samples = [s(100.0), s(100.125), s(100.75), s(100.875), s(101.0)]
    w = make_windower()
    windows = push_all(w, samples)
    assert len(windows) == 1
    win = windows[0]
    assert win.quality is WindowQuality.HAS_GAP
    assert win.timestamps.tolist() == [100.0, 100.125, 100.75, 100.875]  # solo reales
    assert np.array_equal(win.amplitudes[:, 0], win.timestamps)
    assert w.dropped == 0  # el hueco no descarta muestras


def test_gap_exactly_equal_to_max_gap_is_not_a_gap() -> None:
    windows = make_windows([s(100.0), s(100.5), s(101.0)], **{**CONFIG, "min_samples": 2})
    assert windows[0].timestamps.tolist() == [100.0, 100.5]
    assert windows[0].quality is WindowQuality.OK


def test_gap_at_the_window_edge_is_not_hidden() -> None:
    # Datos solo al principio de la ventana: entre muestras no hay hueco,
    # pero 100.375 -> 101.0 sin datos supera max_gap_s.
    samples = [s(100.0), s(100.125), s(100.25), s(100.375), s(101.0)]
    windows = make_windows(samples, **CONFIG)
    assert windows[0].timestamps.tolist() == [100.0, 100.125, 100.25, 100.375]
    assert windows[0].quality is WindowQuality.HAS_GAP


def test_long_outage_emits_no_empty_windows_and_flags_the_neighbour() -> None:
    w = make_windower()
    before = push_all(w, regular(100.0, 4))  # 100.0 ... 100.75
    assert before == []
    at_resume = w.push(s(110.0))  # tras 9.25 s sin datos
    assert [x.t_start for x in at_resume] == [100.0, 100.5]  # sin ventanas vacías
    assert [x.quality for x in at_resume] == [WindowQuality.OK, WindowQuality.HAS_GAP]
    assert all(len(x.timestamps) > 0 for x in at_resume)
    after = push_all(w, regular(110.25, 3))  # 110.25, 110.5, 110.75 (+ 111.0 abajo)
    after += w.push(s(111.0))
    assert [x.t_start for x in after] == [109.5, 110.0]  # la rejilla sigue alineada
    assert after[0].timestamps.tolist() == [110.0, 110.25]
    assert after[0].quality is WindowQuality.INSUFFICIENT_SAMPLES
    assert after[1].timestamps.tolist() == [110.0, 110.25, 110.5, 110.75]
    assert after[1].quality is WindowQuality.OK


# ------------------------------------------------------ pocas muestras / tasa


def test_too_few_samples_is_insufficient() -> None:
    windows = make_windows([s(100.0), s(100.5), s(101.0)], **CONFIG)  # min_samples=3
    assert windows[0].timestamps.tolist() == [100.0, 100.5]
    assert windows[0].quality is WindowQuality.INSUFFICIENT_SAMPLES


def test_min_samples_boundary() -> None:
    samples = regular(100.0, 5)  # la ventana [100, 101) tiene 4 muestras
    assert make_windows(samples, **{**CONFIG, "min_samples": 4})[0].quality is WindowQuality.OK
    low = make_windows(samples, **{**CONFIG, "min_samples": 5})[0]
    assert low.quality is WindowQuality.INSUFFICIENT_SAMPLES


def test_gap_takes_precedence_over_insufficient_samples() -> None:
    win = make_windows([s(100.0), s(101.0)], **CONFIG)[0]
    assert len(win.timestamps) == 1
    assert win.quality is WindowQuality.HAS_GAP


def test_estimated_rate_uses_the_median_of_timestamp_differences() -> None:
    # Diferencias: 0.25, 0.25, 0.125, 0.25 -> mediana 0.25 -> 4.0 Hz (la media daría ~4.57).
    samples = [s(100.0), s(100.25), s(100.5), s(100.625), s(100.875), s(101.0)]
    win = make_windows(samples, **CONFIG)[0]
    assert win.timestamps.tolist() == [100.0, 100.25, 100.5, 100.625, 100.875]
    assert win.estimated_rate_hz == 4.0


def test_estimated_rate_is_zero_with_a_single_sample() -> None:
    win = make_windows([s(100.0), s(101.0)], **CONFIG)[0]
    assert win.estimated_rate_hz == 0.0


# --------------------------------------------------------------- varios nodos


def test_two_interleaved_nodes_do_not_mix_or_interfere() -> None:
    n1 = regular(100.0, 13, node_id="n1")
    n2 = regular(5000.0, 25, step=0.125, node_id="n2")  # reloj y tasa distintos
    interleaved: list[CsiSample] = []
    for i in range(25):
        interleaved.append(n2[i])
        if i < 13:
            interleaved.append(n1[i])
    both = make_windows(interleaved, **CONFIG)
    for node_id, alone_samples in (("n1", n1), ("n2", n2)):
        mine = [x for x in both if x.node_id == node_id]
        alone = make_windows(alone_samples, **CONFIG)
        assert len(mine) == len(alone) > 0
        assert all(same_window(a, b) for a, b in zip(mine, alone))
    assert {x.node_id for x in both} == {"n1", "n2"}


def test_a_silent_node_does_not_block_another() -> None:
    w = make_windower()
    windows = push_all(w, [s(100.0, "n2"), s(100.25, "n2")])
    windows += push_all(w, regular(100.0, 9, node_id="n1"))  # n1 sigue y n2 calla
    assert [x.node_id for x in windows] == ["n1", "n1", "n1"]
    assert w.nodes_stale(now=102.0, timeout_s=1.0) == ["n2"]


# ---------------------------------------------------- muestras inutilizables

BAD_SAMPLES = [
    s(100.75, amplitudes=(1.0, NAN, 3.0)),  # no finita
    s(100.75, amplitudes=()),  # vacía
    s(100.75, node_id=""),  # sin node_id
    s(100.75, amplitudes=(1.0, 2.0)),  # distinto número de subportadoras
    s(100.5),  # timestamp duplicado
    s(100.25),  # atrasada
]


@pytest.mark.parametrize("bad", BAD_SAMPLES)
def test_unusable_sample_is_dropped_counted_and_never_windowed(bad: CsiSample) -> None:
    w = make_windower()
    assert push_all(w, regular(100.0, 3)) == []  # 100.0, 100.25, 100.5
    assert w.dropped == 0
    assert w.push(bad) == []
    assert w.dropped == 1
    windows = w.push(s(101.0))
    assert len(windows) == 1
    assert windows[0].timestamps.tolist() == [100.0, 100.25, 100.5]  # la mala no entra
    assert w.dropped == 1


def test_dropped_counter_accumulates() -> None:
    w = make_windower()
    w.push(s(100.0))
    for _ in range(3):
        w.push(s(100.0))  # duplicados
    assert w.dropped == 3


def test_non_blocking_issues_do_not_drop_the_sample() -> None:
    samples = [
        s(100.0),
        s(100.25, amplitudes=(-1.0, 2.0, 3.0)),  # NEGATIVE_AMPLITUDE
        s(100.5, label="zzz"),  # INVALID_LABEL
        s(100.75),
        s(101.0),
    ]
    w = make_windower()
    windows = push_all(w, samples)
    assert w.dropped == 0
    assert windows[0].timestamps.tolist() == [100.0, 100.25, 100.5, 100.75]


# ----------------------------------------------------------------------- rssi


def test_rssi_array_only_when_every_sample_has_a_valid_rssi() -> None:
    full = [s(100.0 + 0.25 * i, rssi=-50.0 - i) for i in range(5)]
    assert make_windows(full, **CONFIG)[0].rssi.tolist() == [-50.0, -51.0, -52.0, -53.0]  # type: ignore[union-attr]
    missing = [s(100.0, rssi=-50.0), s(100.25), s(100.5, rssi=-52.0), s(100.75, rssi=-53.0), s(101.0)]
    assert make_windows(missing, **CONFIG)[0].rssi is None
    not_finite = [s(100.0 + 0.25 * i, rssi=NAN if i == 1 else -50.0) for i in range(5)]
    assert make_windows(not_finite, **CONFIG)[0].rssi is None


# ----------------------------------------------------------------- nodes_stale


def test_nodes_stale() -> None:
    w = make_windower()
    assert w.nodes_stale(now=100.0, timeout_s=1.0) == []  # sin nodos conocidos
    w.push(s(100.0, "n1"))
    w.push(s(100.25, "n1"))
    w.push(s(100.0, "n2"))
    w.push(s(104.0, "n1"))
    assert w.nodes_stale(now=104.0, timeout_s=2.0) == ["n2"]
    assert w.nodes_stale(now=104.0, timeout_s=4.0) == []  # justo en el límite: no obsoleto
    assert w.nodes_stale(now=106.5, timeout_s=2.0) == ["n1", "n2"]
    assert w.nodes_stale(now=100.0, timeout_s=1.0) == []  # now anterior a las muestras


def test_node_that_only_sent_unusable_samples_is_stale() -> None:
    w = make_windower()
    w.push(s(100.0, "n1"))
    w.push(s(100.0, "nx", amplitudes=()))
    assert w.dropped == 1
    assert w.nodes_stale(now=100.0, timeout_s=5.0) == ["nx"]


def test_sample_without_node_id_never_creates_a_node() -> None:
    w = make_windower()
    w.push(s(100.0, ""))
    assert w.nodes_stale(now=1000.0, timeout_s=1.0) == []


@pytest.mark.parametrize("bad_timeout", [0, -1.0, NAN, INF, "1"])
def test_nodes_stale_rejects_bad_timeout(bad_timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        make_windower().nodes_stale(now=100.0, timeout_s=bad_timeout)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_now", [NAN, INF, None])
def test_nodes_stale_rejects_bad_now(bad_now: object) -> None:
    with pytest.raises(ValueError, match="now"):
        make_windower().nodes_stale(now=bad_now, timeout_s=1.0)  # type: ignore[arg-type]


# --------------------------------------------------------------- configuración


@pytest.mark.parametrize(
    "override, match",
    [
        ({"window_s": 0}, "window_s"),
        ({"window_s": -1.0}, "window_s"),
        ({"window_s": NAN}, "window_s"),
        ({"window_s": INF}, "window_s"),
        ({"hop_s": 0}, "hop_s"),
        ({"hop_s": -0.5}, "hop_s"),
        ({"hop_s": 1.5}, "hop_s"),  # hop_s > window_s
        ({"hop_s": NAN}, "hop_s"),
        ({"min_samples": 1}, "min_samples"),
        ({"min_samples": 0}, "min_samples"),
        ({"min_samples": -3}, "min_samples"),
        ({"min_samples": 2.5}, "min_samples"),
        ({"min_samples": True}, "min_samples"),
        ({"max_gap_s": 0}, "max_gap_s"),
        ({"max_gap_s": -1.0}, "max_gap_s"),
        ({"max_gap_s": NAN}, "max_gap_s"),
    ],
)
def test_invalid_configuration_raises_value_error(override: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_windower(**override)


def test_boundary_configurations_are_valid() -> None:
    make_windower(hop_s=1.0)  # hop_s == window_s: sin solapamiento
    make_windower(min_samples=2)
    make_windower(window_s=0.125, hop_s=0.0625)


def test_no_overlap_when_hop_equals_window() -> None:
    windows = make_windows(regular(100.0, 9), **{**CONFIG, "hop_s": 1.0})
    assert [(x.t_start, x.t_end) for x in windows] == [(100.0, 101.0), (101.0, 102.0)]
    assert not set(windows[0].timestamps.tolist()) & set(windows[1].timestamps.tolist())


# ------------------------------------------------------------------- inmutable


def test_window_is_immutable() -> None:
    win = make_windows([*regular(100.0, 4, node_id="n1")] + [s(101.0, rssi=-50.0)], **CONFIG)[0]
    with pytest.raises(FrozenInstanceError):
        win.node_id = "otro"  # type: ignore[misc]
    with pytest.raises(ValueError):
        win.timestamps[0] = 0.0
    with pytest.raises(ValueError):
        win.amplitudes[0, 0] = 0.0
    assert win == win  # igualdad por identidad: no lanza por comparar arrays


def test_window_rssi_array_is_read_only() -> None:
    samples = [s(100.0 + 0.25 * i, rssi=-50.0) for i in range(5)]
    win = make_windows(samples, **CONFIG)[0]
    assert win.rssi is not None
    with pytest.raises(ValueError):
        win.rssi[0] = 0.0


def test_overlapping_windows_do_not_share_memory() -> None:
    a, b = make_windows(regular(100.0, 7), **CONFIG)[:2]
    assert not np.shares_memory(a.timestamps, b.timestamps)


# ----------------------------------------------------------------- make_windows


def test_make_windows_matches_push_by_push() -> None:
    samples = regular(100.0, 13)
    w = make_windower()
    expected = push_all(w, samples)
    got = make_windows(samples, **CONFIG)
    assert len(got) == len(expected) > 0
    assert all(same_window(a, b) for a, b in zip(got, expected))


def test_make_windows_empty_and_short_inputs() -> None:
    assert make_windows([], **CONFIG) == []
    assert make_windows(regular(100.0, 3), **CONFIG) == []  # ninguna ventana se completa


def test_make_windows_accepts_any_iterable() -> None:
    assert len(make_windows(iter(regular(100.0, 5)), **CONFIG)) == 1


def test_make_windows_warns_when_it_drops_samples() -> None:
    samples = [*regular(100.0, 4), s(100.5), s(101.0)]  # una duplicada
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        windows = make_windows(samples, **CONFIG)
    assert len(windows) == 1
    assert [w.category for w in caught] == [RuntimeWarning]
    assert "1 muestra" in str(caught[0].message)


def test_make_windows_does_not_warn_when_nothing_is_dropped() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_windows(regular(100.0, 5), **CONFIG)
    assert caught == []


def test_make_windows_validates_configuration() -> None:
    with pytest.raises(ValueError, match="hop_s"):
        make_windows([], window_s=1.0, hop_s=2.0, min_samples=3, max_gap_s=0.5)
    with pytest.raises(TypeError):
        make_windows([], **CONFIG, extra=1)