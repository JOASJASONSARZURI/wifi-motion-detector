"""Pruebas de wifi_motion.features.

SYNTHETIC: todas las señales de este archivo se generan aquí con numpy (senos,
ruido con semilla fija, constantes); no son mediciones reales y no demuestran
exactitud de campo. Las tasas y duraciones usan fracciones binarias exactas
(16 Hz, 32 Hz) para que las marcas de tiempo y los bins de la FFT no dependan
del redondeo.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from wifi_motion.features import (
    FEATURE_NAMES,
    FeatureConfig,
    FeatureError,
    extract_features,
    extract_features_dict,
)
from wifi_motion.schema import CsiSample
from wifi_motion.windowing import Window, WindowQuality, make_windows

RATE = 16.0
N = 128  # 8 s a 16 Hz -> resolución de 0.125 Hz
OFFSETS = (1.0, 2.0, 3.0)
# Para ventanas diminutas (n = 2..4) hacen falta bandas que contengan algún bin.
TINY = FeatureConfig(motion_band_hz=(1.0, 2.0), breathing_band_hz=(0.5, 1.5))


def build_window(
    amplitudes: Any,
    *,
    rate_hz: float = RATE,
    rssi: Any = None,
    quality: WindowQuality = WindowQuality.OK,
) -> Window:
    """SYNTHETIC: ventana armada a mano con timestamps uniformes a rate_hz."""
    amps = np.asarray(amplitudes, dtype=np.float64)
    n = amps.shape[0] if amps.ndim else 0
    timestamps = 100.0 + np.arange(n) / rate_hz
    return Window(
        node_id="n1",
        t_start=100.0,
        t_end=100.0 + n / rate_hz,
        timestamps=timestamps,
        amplitudes=amps,
        rssi=None if rssi is None else np.asarray(rssi, dtype=np.float64),
        estimated_rate_hz=rate_hz,
        quality=quality,
    )


def tone(freq_hz: float, amp: float = 2.0, n: int = N, rate_hz: float = RATE) -> np.ndarray:
    """SYNTHETIC: seno de amplitud ``amp`` en cada subportadora, con distinto offset."""
    t = np.arange(n) / rate_hz
    return np.stack([off + amp * np.sin(2 * np.pi * freq_hz * t) for off in OFFSETS], axis=1)


def noisy(sigma: float, seed: int = 0, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 5.0 + sigma * rng.standard_normal((n, len(OFFSETS)))


def feats(amplitudes: Any, config: FeatureConfig = FeatureConfig(), **kw: Any) -> dict[str, float]:
    return extract_features_dict(build_window(amplitudes, **kw), config)


# ------------------------------------------------------------- FEATURE_NAMES


def test_feature_names_order_and_length() -> None:
    assert FEATURE_NAMES == (
        "variance",
        "motion_band_power",
        "breathing_band_power",
        "spectral_power",
        "dominant_freq_hz",
        "change_points",
        "mean_rssi",
        "amp_mean",
        "amp_std",
        "amp_skew",
        "amp_kurtosis",
        "amp_iqr",
        "amp_entropy",
        "amp_max",
        "amp_range",
    )
    assert len(FEATURE_NAMES) == 15 == len(set(FEATURE_NAMES))


def test_vector_and_dict_agree_and_follow_the_order() -> None:
    win = build_window(tone(1.25), rssi=np.full(N, -50.0))
    vector = extract_features(win)
    as_dict = extract_features_dict(win)
    assert vector.shape == (15,) and vector.dtype == np.float64
    assert list(as_dict) == list(FEATURE_NAMES)
    assert vector.tolist() == [as_dict[name] for name in FEATURE_NAMES]


# ------------------------------------------------------------ señal constante


@pytest.mark.parametrize("level", [2.0, 0.1])  # 0.1 no es exacto en binario
def test_constant_signal_has_defined_values_without_division_by_zero(level: float) -> None:
    amps = np.full((N, 3), level)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # cualquier RuntimeWarning haría fallar
        f = feats(amps)
    assert f["variance"] == 0.0
    assert f["motion_band_power"] == 0.0
    assert f["breathing_band_power"] == 0.0
    assert f["spectral_power"] == 0.0
    assert f["dominant_freq_hz"] == 0.0
    assert f["change_points"] == 0.0
    assert f["amp_std"] == 0.0
    assert f["amp_skew"] == 0.0
    assert f["amp_kurtosis"] == 0.0
    assert f["amp_iqr"] == 0.0
    assert f["amp_entropy"] == 0.0
    assert f["amp_range"] == 0.0
    assert f["amp_mean"] == pytest.approx(level)
    assert f["amp_max"] == level


def test_constant_in_time_but_different_per_subcarrier() -> None:
    amps = np.tile([[1.0, 2.0, 3.5]], (N, 1))
    f = feats(amps)
    assert f["variance"] == 0.0 and f["spectral_power"] == 0.0
    assert f["change_points"] == 0.0 and f["dominant_freq_hz"] == 0.0
    assert f["amp_std"] > 0.0  # la dispersión entre subportadoras sí cuenta
    assert math.isfinite(f["amp_skew"]) and math.isfinite(f["amp_kurtosis"])


# ------------------------------------------------------------------ frecuencia


def test_dominant_frequency_of_an_exact_bin_sine() -> None:
    f = feats(tone(1.25))  # 10 ciclos exactos en la ventana
    assert f["dominant_freq_hz"] == pytest.approx(1.25, abs=1e-9)
    # Sin fuga: toda la potencia (amp**2 / 2 = 2.0) cae en la banda de movimiento.
    assert f["spectral_power"] == pytest.approx(2.0, abs=1e-9)
    assert f["motion_band_power"] == pytest.approx(2.0, abs=1e-9)
    assert f["breathing_band_power"] == pytest.approx(0.0, abs=1e-9)
    assert f["variance"] == pytest.approx(2.0, abs=1e-9)


def test_breathing_band_sine_goes_to_the_breathing_band() -> None:
    f = feats(tone(0.25))  # 2 ciclos exactos
    assert f["dominant_freq_hz"] == pytest.approx(0.25, abs=1e-9)
    assert f["breathing_band_power"] == pytest.approx(2.0, abs=1e-9)
    assert f["motion_band_power"] == pytest.approx(0.0, abs=1e-9)


def test_dominant_frequency_between_bins_is_within_half_a_bin() -> None:
    f = feats(tone(1.3))
    assert abs(f["dominant_freq_hz"] - 1.3) <= 0.125 / 2 + 1e-9


def test_dominant_frequency_uses_the_window_sampling_rate() -> None:
    # 32 Hz, 256 muestras (8 s): mismo n/rate, otra escala de frecuencias.
    f = feats(tone(3.0, n=256, rate_hz=32.0), rate_hz=32.0)
    assert f["dominant_freq_hz"] == pytest.approx(3.0, abs=1e-9)
    assert f["motion_band_power"] == pytest.approx(2.0, abs=1e-9)


def test_dominant_frequency_ignores_dc_offset() -> None:
    f = feats(tone(2.0, amp=0.01) + 1000.0)  # DC enorme, oscilación mínima
    assert f["dominant_freq_hz"] == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("n", [64, 63])  # par e impar: pesos de Nyquist/espejo
def test_spectral_power_is_the_variance_of_the_mean_series(n: int) -> None:
    rng = np.random.default_rng(3)
    amps = 4.0 + rng.standard_normal((n, 3))
    f = feats(amps)
    series = amps.mean(axis=1)
    assert f["spectral_power"] == pytest.approx(np.var(series), rel=1e-9)
    assert f["motion_band_power"] + f["breathing_band_power"] <= f["spectral_power"] + 1e-12
    assert f["variance"] >= f["spectral_power"] - 1e-12  # var media >= var de la media


# ---------------------------------------------------------------------- ruido


def test_more_noise_gives_more_variance() -> None:
    clean = feats(tone(1.25))["variance"]
    low = feats(tone(1.25) + noisy(0.1, seed=1) - 5.0)["variance"]
    high = feats(tone(1.25) + noisy(1.0, seed=1) - 5.0)["variance"]
    assert clean < low < high
    assert feats(noisy(2.0))["variance"] > feats(noisy(0.2))["variance"]


# ------------------------------------------------------------- cambios bruscos


def test_constant_signal_with_one_step_has_exactly_one_change_point() -> None:
    amps = np.full((N, 3), 2.0)
    assert feats(amps)["change_points"] == 0.0
    amps[64:] += 5.0
    assert feats(amps)["change_points"] == 1.0


def test_step_on_top_of_noise_increases_change_points() -> None:
    base = noisy(0.05, seed=7)
    stepped = base.copy()
    stepped[64:] += 5.0
    before = feats(base)["change_points"]
    after = feats(stepped)["change_points"]
    assert after >= 1.0
    assert after > before


def test_change_k_controls_the_threshold() -> None:
    amps = noisy(0.05, seed=7)
    strict = feats(amps, FeatureConfig(change_k=50.0))["change_points"]
    loose = feats(amps, FeatureConfig(change_k=1.0))["change_points"]
    assert loose > strict


# -------------------------------------------------------- estadísticas de amp_*


def test_amplitude_statistics_on_a_two_point_distribution() -> None:
    # Valores {0, 0, 0, 1}: Bernoulli(p = 0.25), calculable a mano.
    p = 0.25
    f = feats([[0.0], [0.0], [0.0], [1.0]], TINY, rate_hz=4.0)
    assert f["amp_mean"] == pytest.approx(0.25)
    assert f["amp_std"] == pytest.approx(math.sqrt(p * (1 - p)))
    assert f["amp_skew"] == pytest.approx((1 - 2 * p) / math.sqrt(p * (1 - p)))
    assert f["amp_kurtosis"] == pytest.approx((1 - 6 * p * (1 - p)) / (p * (1 - p)))  # exceso
    assert f["amp_iqr"] == pytest.approx(0.25)  # percentiles 0.0 y 0.25
    assert f["amp_max"] == 1.0 and f["amp_range"] == 1.0
    h = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert f["amp_entropy"] == pytest.approx(h)


def test_symmetric_distribution_has_zero_skew_and_negative_excess_kurtosis() -> None:
    f = feats([[0.0], [1.0], [0.0], [1.0]], TINY, rate_hz=4.0)  # moneda 50/50
    assert f["amp_skew"] == pytest.approx(0.0, abs=1e-12)
    assert f["amp_kurtosis"] == pytest.approx(-2.0)
    assert f["amp_entropy"] == pytest.approx(1.0)  # 1 bit


def test_amp_statistics_use_all_amplitudes_of_the_window() -> None:
    amps = np.array([[1.0, 5.0], [3.0, 9.0], [2.0, 7.0], [4.0, 3.0]])
    f = feats(amps, TINY, rate_hz=4.0)
    assert f["amp_mean"] == pytest.approx(amps.mean())
    assert f["amp_std"] == pytest.approx(amps.std())
    assert f["amp_max"] == 9.0
    assert f["amp_range"] == 8.0


def test_entropy_is_bounded_by_log2_of_bins() -> None:
    rng = np.random.default_rng(5)
    f = feats(rng.uniform(0, 10, size=(N, 3)), FeatureConfig(entropy_bins=8))
    assert 0.0 < f["amp_entropy"] <= 3.0 + 1e-12


# ------------------------------------------------------------------------ RSSI


def test_mean_rssi_with_rssi() -> None:
    rssi = -60.0 + np.arange(N) % 4  # media -58.5
    assert feats(tone(1.25), rssi=rssi)["mean_rssi"] == pytest.approx(-58.5)


def test_missing_rssi_gives_nan_and_is_not_an_error() -> None:
    f = feats(tone(1.25))
    assert math.isnan(f["mean_rssi"])
    assert math.isnan(extract_features(build_window(tone(1.25)))[FEATURE_NAMES.index("mean_rssi")])


@pytest.mark.parametrize("with_rssi", [False, True])
def test_no_nan_except_mean_rssi_without_rssi(with_rssi: bool) -> None:
    rssi = np.full(N, -55.0) if with_rssi else None
    signals = [tone(1.25), tone(0.25), noisy(0.5), np.full((N, 3), 2.0)]
    for amps in signals:
        f = feats(amps, rssi=rssi)
        nan_names = {name for name, v in f.items() if math.isnan(v)}
        assert nan_names == (set() if with_rssi else {"mean_rssi"})
        assert all(math.isfinite(v) for name, v in f.items() if name != "mean_rssi")


# ---------------------------------------------------------------- ventana mínima


def test_minimal_window_of_two_samples() -> None:
    # s = [1.5, 3.5]; bandas hasta Nyquist (1 Hz): el extremo se incluye.
    config = FeatureConfig(motion_band_hz=(0.5, 1.0), breathing_band_hz=(0.5, 1.0))
    f = feats([[1.0, 2.0], [3.0, 4.0]], config, rate_hz=2.0)
    assert f["variance"] == pytest.approx(1.0)
    assert f["spectral_power"] == pytest.approx(1.0)
    assert f["motion_band_power"] == pytest.approx(1.0)
    assert f["dominant_freq_hz"] == pytest.approx(1.0)
    assert f["change_points"] == 0.0  # con k >= 1 una única diferencia no supera k * ella misma
    assert all(math.isfinite(v) for name, v in f.items() if name != "mean_rssi")


def test_minimal_window_of_three_samples() -> None:
    f = feats([[1.0], [2.0], [4.0]], TINY, rate_hz=4.0)
    assert all(math.isfinite(v) for name, v in f.items() if name != "mean_rssi")
    series = np.array([1.0, 2.0, 4.0])
    assert f["spectral_power"] == pytest.approx(np.var(series))


def test_window_with_fewer_than_two_samples_is_an_error() -> None:
    with pytest.raises(FeatureError, match="al menos 2"):
        extract_features(build_window([[1.0, 2.0]]), TINY)


# ------------------------------------------------------------------- Nyquist


def test_motion_band_above_nyquist_is_an_error() -> None:
    config = FeatureConfig(motion_band_hz=(5.0, 9.0))  # Nyquist = 8 Hz
    with pytest.raises(FeatureError, match="Nyquist"):
        extract_features(build_window(tone(1.25)), config)


def test_breathing_band_above_nyquist_is_an_error() -> None:
    config = FeatureConfig(breathing_band_hz=(0.1, 8.5))
    with pytest.raises(FeatureError, match="breathing_band_hz.*Nyquist"):
        extract_features(build_window(tone(1.25)), config)


def test_band_up_to_exactly_nyquist_is_allowed_and_includes_that_bin() -> None:
    config = FeatureConfig(motion_band_hz=(5.0, 8.0))
    f = feats(tone(8.0 - 0.125), config)  # justo bajo Nyquist
    assert f["motion_band_power"] == pytest.approx(2.0, abs=1e-9)
    # Una oscilación en Nyquist (+a, -a, ...) cae en el último bin, que se incluye.
    alternating = np.tile([[2.0], [4.0]], (N // 2, 3))
    g = feats(alternating, config)
    assert g["dominant_freq_hz"] == pytest.approx(8.0)
    assert g["motion_band_power"] == pytest.approx(1.0)  # amplitud 1 en Nyquist: potencia 1


def test_nyquist_follows_the_window_rate_not_a_constant() -> None:
    config = FeatureConfig(motion_band_hz=(1.0, 5.0))
    extract_features(build_window(tone(1.25), rate_hz=16.0), config)  # Nyquist 8: bien
    with pytest.raises(FeatureError, match="Nyquist"):
        extract_features(build_window(tone(1.25, rate_hz=8.0), rate_hz=8.0), config)  # Nyquist 4


def test_band_without_fft_bins_is_an_error_not_a_zero() -> None:
    # 4 muestras a 16 Hz: resolución 4 Hz; la banda de respiración no tiene bins.
    with pytest.raises(FeatureError, match="breathing_band_hz.*ningún bin"):
        extract_features(build_window(noisy(1.0, n=4)))


# ------------------------------------------------------------- ventanas malas


@pytest.mark.parametrize("quality", [WindowQuality.HAS_GAP, WindowQuality.INSUFFICIENT_SAMPLES])
def test_window_with_non_ok_quality_is_an_error(quality: WindowQuality) -> None:
    win = build_window(tone(1.25), quality=quality)
    with pytest.raises(FeatureError, match=quality.value):
        extract_features(win)
    with pytest.raises(FeatureError):
        extract_features_dict(win)


def test_window_with_a_real_gap_from_the_windower_is_an_error() -> None:
    # Falta el tramo [102.0, 105.0): el windower marca HAS_GAP.
    samples = [
        CsiSample(100.0 + i / RATE, "n1", (1.0, 2.0, 3.0)) for i in range(0, 129) if not 32 <= i < 80
    ]
    windows = make_windows(samples, window_s=8.0, hop_s=8.0, min_samples=3, max_gap_s=0.5)
    assert windows[0].quality is WindowQuality.HAS_GAP
    with pytest.raises(FeatureError):
        extract_features(windows[0])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_amplitudes_are_an_error(bad: float) -> None:
    amps = tone(1.25)
    amps[10, 1] = bad
    with pytest.raises(FeatureError, match="amplitudes no finitas"):
        extract_features(build_window(amps))


def test_non_finite_timestamps_and_rssi_are_an_error() -> None:
    win = build_window(tone(1.25))
    bad_ts = win.timestamps.copy()
    bad_ts[3] = float("nan")
    with pytest.raises(FeatureError, match="timestamps"):
        extract_features(replace(win, timestamps=bad_ts))
    rssi = np.full(N, -50.0)
    rssi[5] = float("inf")
    with pytest.raises(FeatureError, match="RSSI"):
        extract_features(replace(win, rssi=rssi))


@pytest.mark.parametrize("bad_rate", [0.0, -16.0, float("nan"), float("inf")])
def test_unusable_estimated_rate_is_an_error(bad_rate: float) -> None:
    with pytest.raises(FeatureError, match="estimated_rate_hz"):
        extract_features(replace(build_window(tone(1.25)), estimated_rate_hz=bad_rate))


def test_inconsistent_shapes_are_an_error() -> None:
    win = build_window(tone(1.25))
    with pytest.raises(FeatureError, match="timestamps"):
        extract_features(replace(win, timestamps=win.timestamps[:-1]))
    with pytest.raises(FeatureError, match="amplitudes"):
        extract_features(replace(win, amplitudes=win.amplitudes[:, 0]))
    with pytest.raises(FeatureError, match="rssi"):
        extract_features(replace(win, rssi=np.zeros(N - 1)))


def test_values_that_overflow_are_reported_not_returned() -> None:
    amps = np.full((N, 3), 1e200)
    amps[::2] = -1e200  # los momentos de orden 4 desbordan a inf
    with pytest.raises(FeatureError, match="no es finita"):
        extract_features(build_window(amps))


def test_feature_error_is_a_value_error() -> None:
    assert issubclass(FeatureError, ValueError)


def test_wrong_argument_types_raise_type_error() -> None:
    with pytest.raises(TypeError, match="Window"):
        extract_features(np.zeros((N, 3)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FeatureConfig"):
        extract_features(build_window(tone(1.25)), {"change_k": 5.0})  # type: ignore[arg-type]


# --------------------------------------------------------------- configuración


def test_default_config_values_are_documented_initial_values() -> None:
    config = FeatureConfig()
    assert config.motion_band_hz == (1.0, 5.0)
    assert config.breathing_band_hz == (0.1, 0.5)
    assert config.change_k == 5.0 and config.entropy_bins == 16
    assert "NO UMBRALES UNIVERSALES" in (FeatureConfig.__doc__ or "")


def test_config_is_immutable_and_normalizes_bands_to_float_tuples() -> None:
    config = FeatureConfig(motion_band_hz=[1, 4])  # type: ignore[arg-type]
    assert config.motion_band_hz == (1.0, 4.0)
    assert all(isinstance(v, float) for v in config.motion_band_hz)
    with pytest.raises(FrozenInstanceError):
        config.change_k = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "override, match",
    [
        ({"motion_band_hz": (5.0, 1.0)}, "motion_band_hz"),
        ({"motion_band_hz": (1.0, 1.0)}, "motion_band_hz"),
        ({"motion_band_hz": (-1.0, 2.0)}, "motion_band_hz"),
        ({"motion_band_hz": (1.0, float("nan"))}, "motion_band_hz"),
        ({"motion_band_hz": (1.0, float("inf"))}, "motion_band_hz"),
        ({"motion_band_hz": (1.0,)}, "motion_band_hz"),
        ({"motion_band_hz": "ab"}, "motion_band_hz"),
        ({"breathing_band_hz": (0.5, 0.1)}, "breathing_band_hz"),
        ({"breathing_band_hz": (True, 2.0)}, "breathing_band_hz"),
        ({"change_k": 0}, "change_k"),
        ({"change_k": -1.0}, "change_k"),
        ({"change_k": float("nan")}, "change_k"),
        ({"entropy_bins": 1}, "entropy_bins"),
        ({"entropy_bins": 2.5}, "entropy_bins"),
        ({"entropy_bins": True}, "entropy_bins"),
    ],
)
def test_invalid_config_raises_value_error(override: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FeatureConfig(**override)


def test_custom_bands_move_the_power() -> None:
    amps = tone(1.25)
    inside = feats(amps, FeatureConfig(motion_band_hz=(1.0, 2.0)))["motion_band_power"]
    outside = feats(amps, FeatureConfig(motion_band_hz=(2.0, 3.0)))["motion_band_power"]
    assert inside == pytest.approx(2.0, abs=1e-9)
    assert outside == pytest.approx(0.0, abs=1e-9)


def test_adjacent_bands_do_not_double_count_the_edge_bin() -> None:
    amps = tone(1.0)  # justo en el borde (1.0 Hz = bin 8)
    low = feats(amps, FeatureConfig(breathing_band_hz=(0.5, 1.0), motion_band_hz=(1.0, 2.0)))
    assert low["breathing_band_power"] == pytest.approx(0.0, abs=1e-9)
    assert low["motion_band_power"] == pytest.approx(2.0, abs=1e-9)


# ----------------------------------------------------------------- integración


def test_works_on_a_real_window_from_the_windower_and_does_not_mutate_it() -> None:
    t = np.arange(129) / RATE
    samples = [
        CsiSample(
            100.0 + ti,
            "n1",
            tuple(float(off + 2.0 * np.sin(2 * np.pi * 1.25 * ti)) for off in OFFSETS),
            rssi=-50.0,
        )
        for ti in t
    ]
    windows = make_windows(samples, window_s=8.0, hop_s=8.0, min_samples=3, max_gap_s=0.5)
    win = windows[0]
    assert win.quality is WindowQuality.OK and len(win.timestamps) == N
    before = win.amplitudes.copy()
    f = extract_features_dict(win)
    assert f["dominant_freq_hz"] == pytest.approx(1.25, abs=1e-9)
    assert f["mean_rssi"] == pytest.approx(-50.0)
    assert all(math.isfinite(v) for v in f.values())
    assert np.array_equal(win.amplitudes, before)