"""Pruebas de wifi_motion.calibration.

SYNTHETIC: todos los datos de este archivo se generan aquí con numpy (semilla
fija) o se escriben a mano; no son mediciones reales y no demuestran exactitud
de campo. Los umbrales (z_max, alpha...) que aparecen en las pruebas son
arbitrarios y solo sirven para ejercitar el código. Las pruebas de archivos
usan ``tmp_path`` de pytest.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wifi_motion.calibration import (
    BASELINE_VERSION,
    MAD_EPSILON,
    MAD_TO_SIGMA,
    Baseline,
    CalibrationError,
    deviation,
    fit_baseline,
    is_quiet,
    load_baseline,
    save_baseline,
    update_baseline,
)
from wifi_motion.features import FEATURE_NAMES

NAN = float("nan")
INF = float("inf")
RSSI = FEATURE_NAMES.index("mean_rssi")
CHANGE = FEATURE_NAMES.index("change_points")
CHECK = ("variance", "motion_band_power", "spectral_power")

# Centro y dispersión SYNTHETIC por característica, en el orden de FEATURE_NAMES.
BASE = np.array([2.0, 0.5, 0.8, 3.0, 1.0, 2.0, -55.0, 20.0, 1.5, 0.1, 0.2, 2.0, 2.5, 30.0, 6.0])
SD = np.array([0.2, 0.05, 0.08, 0.3, 0.1, 0.5, 1.5, 0.5, 0.1, 0.05, 0.05, 0.2, 0.1, 1.0, 0.3])


def quiet_features(n: int = 60, seed: int = 0) -> np.ndarray:
    """SYNTHETIC: n vectores de características de una sala vacía (ruido gaussiano)."""
    return np.random.default_rng(seed).normal(BASE, SD, size=(n, len(FEATURE_NAMES)))


def make_baseline(**overrides: Any) -> Baseline:
    kwargs: dict[str, Any] = {
        "node_id": "n1",
        "features": quiet_features(),
        "environment_note": "sala-101 vacía",
        "created_at": 1_700_000_000.5,
    }
    kwargs.update(overrides)
    return fit_baseline(**kwargs)


def moved(baseline: Baseline, names: tuple[str, ...], z: float) -> np.ndarray:
    """SYNTHETIC: vector en la mediana, con ``names`` desplazadas z puntuaciones robustas."""
    vector = baseline.median.copy()
    for name in names:
        i = FEATURE_NAMES.index(name)
        vector[i] += z * MAD_TO_SIGMA * baseline.mad[i]
    return vector


# ------------------------------------------------------------------ calibración


def test_test_data_matches_the_feature_names() -> None:
    assert len(FEATURE_NAMES) == len(BASE) == len(SD) == 15


def test_fit_baseline_computes_median_and_raw_mad() -> None:
    X = quiet_features()
    base = fit_baseline("n1", X, "sala-101 vacía", created_at=1_700_000_000.5)
    expected_median = np.median(X, axis=0)
    assert base.node_id == "n1"
    assert base.feature_names == FEATURE_NAMES
    assert np.array_equal(base.median, expected_median)
    assert np.array_equal(base.mad, np.median(np.abs(X - expected_median), axis=0))
    assert base.median.shape == base.mad.shape == (15,)
    assert base.n_windows == 60
    assert base.created_at == 1_700_000_000.5
    assert base.environment_note == "sala-101 vacía"
    assert np.all(base.mad > 0)
    # El centro recuperado está cerca del real (datos SYNTHETIC gaussianos).
    assert np.all(np.abs(base.median - BASE) < 0.6 * SD)
    assert base.floored_features == ()


def test_created_at_defaults_to_now() -> None:
    base = fit_baseline("n1", quiet_features(), "nota")
    assert abs(base.created_at - time.time()) < 60.0


def test_baseline_is_immutable_and_arrays_are_read_only() -> None:
    base = make_baseline()
    with pytest.raises(FrozenInstanceError):
        base.node_id = "otro"  # type: ignore[misc]
    with pytest.raises(ValueError):
        base.median[0] = 0.0
    with pytest.raises(ValueError):
        base.mad[0] = 1.0
    with pytest.raises((AttributeError, TypeError)):
        base.extra = 1  # type: ignore[attr-defined]


def test_baseline_does_not_share_memory_with_its_inputs() -> None:
    X = quiet_features()
    base = fit_baseline("n1", X, "nota")
    median_before = base.median.copy()
    X[:] = 0.0
    assert np.array_equal(base.median, median_before)
    source = np.array(median_before)
    rebuilt = Baseline("n1", FEATURE_NAMES, source, base.mad, 5, 1.0, "")
    source[:] = 0.0
    assert np.array_equal(rebuilt.median, median_before)


def test_accepts_a_list_of_lists() -> None:
    base = fit_baseline("n1", quiet_features(40).tolist(), "nota", min_windows=40)
    assert base.n_windows == 40


# ------------------------------------------------------------- pocas ventanas


def test_too_few_windows_is_an_error_with_the_default_minimum() -> None:
    with pytest.raises(CalibrationError, match="29.*30"):
        fit_baseline("n1", quiet_features(29), "nota")


def test_exactly_min_windows_is_enough_and_min_windows_is_configurable() -> None:
    assert fit_baseline("n1", quiet_features(30), "nota").n_windows == 30
    with pytest.raises(CalibrationError, match="9.*10"):
        fit_baseline("n1", quiet_features(9), "nota", min_windows=10)
    assert fit_baseline("n1", quiet_features(5), "nota", min_windows=5).n_windows == 5


def test_empty_calibration_is_an_error() -> None:
    with pytest.raises(CalibrationError):
        fit_baseline("n1", np.zeros((0, 15)), "nota")


@pytest.mark.parametrize("bad", [0, 1, -5, 2.5, True, "30", None])
def test_bad_min_windows_is_a_value_error(bad: object) -> None:
    with pytest.raises(ValueError, match="min_windows"):
        fit_baseline("n1", quiet_features(), "nota", min_windows=bad)  # type: ignore[arg-type]


# ----------------------------------------------------------- valores no finitos


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
@pytest.mark.parametrize("name", ["variance", "change_points", "amp_range"])
def test_non_finite_in_any_other_feature_is_rejected(bad: float, name: str) -> None:
    X = quiet_features()
    X[7, FEATURE_NAMES.index(name)] = bad
    with pytest.raises(CalibrationError, match=name):
        fit_baseline("n1", X, "nota")


def test_infinite_rssi_is_rejected_but_nan_rssi_is_not() -> None:
    X = quiet_features()
    X[3, RSSI] = INF
    with pytest.raises(CalibrationError, match="mean_rssi"):
        fit_baseline("n1", X, "nota")
    X[3, RSSI] = -INF
    with pytest.raises(CalibrationError, match="mean_rssi"):
        fit_baseline("n1", X, "nota")


def test_nan_in_mean_rssi_is_ignored_with_nanmedian() -> None:
    X = quiet_features()
    X[[2, 9, 21, 33, 50], RSSI] = NAN
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        base = fit_baseline("n1", X, "nota")
    valid = X[~np.isnan(X[:, RSSI]), RSSI]
    assert base.median[RSSI] == np.median(valid)
    assert base.mad[RSSI] == np.median(np.abs(valid - np.median(valid)))
    assert base.n_windows == 60  # cuenta todas las ventanas, con o sin RSSI
    assert np.all(np.isfinite(base.median)) and np.all(np.isfinite(base.mad))
    # Las demás columnas no se ven afectadas por los NaN del RSSI.
    clean = fit_baseline("n1", quiet_features(), "nota")
    others = [i for i in range(15) if i != RSSI]
    assert np.array_equal(base.median[others], clean.median[others])


def test_all_nan_rssi_gives_no_rssi_baseline_without_inventing_one() -> None:
    X = quiet_features()
    X[:, RSSI] = NAN
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nada de RuntimeWarning "All-NaN slice"
        base = fit_baseline("n1", X, "nota")
        z = deviation(base, quiet_features(1, seed=5)[0])
    assert np.isnan(base.median[RSSI]) and np.isnan(base.mad[RSSI])
    others = [i for i in range(15) if i != RSSI]
    assert np.all(np.isfinite(base.median[others])) and np.all(np.isfinite(base.mad[others]))
    assert np.isnan(z[RSSI]) and np.all(np.isfinite(z[others]))
    assert base.floored_features == ()


def test_wrong_shape_or_type_is_a_calibration_error() -> None:
    with pytest.raises(CalibrationError, match="forma"):
        fit_baseline("n1", np.zeros((40, 14)), "nota")
    with pytest.raises(CalibrationError, match="forma"):
        fit_baseline("n1", np.zeros(15), "nota")
    with pytest.raises(CalibrationError, match="forma"):
        fit_baseline("n1", np.zeros((2, 40, 15)), "nota")
    with pytest.raises(CalibrationError):
        fit_baseline("n1", [["x"] * 15] * 40, "nota")


@pytest.mark.parametrize("node_id", ["", "   ", None, 7])
def test_bad_node_id_is_an_error(node_id: Any) -> None:
    with pytest.raises(CalibrationError, match="node_id"):
        fit_baseline(node_id, quiet_features(), "nota")


def test_calibration_error_is_a_value_error() -> None:
    assert issubclass(CalibrationError, ValueError)


# ------------------------------------------------------ robustez ante outliers


def test_outliers_during_calibration_do_not_drag_the_median() -> None:
    clean = quiet_features(60)
    outliers = BASE + 50.0 * SD  # 5 ventanas con "movimiento" muy fuerte
    dirty = np.vstack([clean, np.tile(outliers, (5, 1))])
    clean_base = fit_baseline("n1", clean, "nota")
    dirty_base = fit_baseline("n1", dirty, "nota")
    median_shift = np.abs(dirty_base.median - clean_base.median)
    mean_shift = np.abs(dirty.mean(axis=0) - clean.mean(axis=0))
    assert np.all(median_shift < 0.5 * SD)  # la mediana casi no se mueve...
    assert np.all(mean_shift > 3.0 * SD)  # ...la media sí lo haría
    ratio = dirty_base.mad / clean_base.mad
    assert np.all((ratio > 0.7) & (ratio < 1.4))  # el MAD tampoco explota


# ------------------------------------------------------------------- MAD cero


def test_zero_mad_is_replaced_by_a_visible_epsilon() -> None:
    X = quiet_features()
    X[:, CHANGE] = 0.0  # característica sin variación durante la calibración
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # sin división por cero
        base = fit_baseline("n1", X, "nota")
        flat = base.median.copy()
        z_flat = deviation(base, flat)
        bumped = flat.copy()
        bumped[CHANGE] = 1.0
        z_bumped = deviation(base, bumped)
    assert MAD_EPSILON > 0
    assert base.mad[CHANGE] == MAD_EPSILON
    assert base.floored_features == ("change_points",)
    assert z_flat[CHANGE] == 0.0
    assert np.isfinite(z_bumped[CHANGE])
    assert z_bumped[CHANGE] == pytest.approx(1.0 / (MAD_TO_SIGMA * MAD_EPSILON))  # muy alto


def test_a_small_nonzero_mad_is_not_replaced() -> None:
    X = quiet_features()
    X[:, 0] = 1.0 + 1e-9 * np.arange(60)  # escala diminuta pero real
    base = fit_baseline("n1", X, "nota")
    assert 0 < base.mad[0] < MAD_EPSILON
    assert "variance" not in base.floored_features


# -------------------------------------------------------------------- deviation


def test_deviation_formula_and_zero_at_the_median() -> None:
    base = make_baseline()
    assert np.all(deviation(base, base.median) == 0.0)
    x = base.median + 3.0 * MAD_TO_SIGMA * base.mad
    assert np.allclose(deviation(base, x), 3.0)
    assert np.allclose(deviation(base, base.median - 2.0 * MAD_TO_SIGMA * base.mad), -2.0)
    y = quiet_features(1, seed=11)[0]
    assert np.allclose(deviation(base, y), (y - base.median) / (1.4826 * base.mad))
    assert MAD_TO_SIGMA == 1.4826


def test_deviation_returns_a_float_vector_and_does_not_modify_the_input() -> None:
    base = make_baseline()
    x = quiet_features(1, seed=2)[0]
    before = x.copy()
    z = deviation(base, x)
    assert z.shape == (15,) and z.dtype == np.float64
    assert np.array_equal(x, before)
    assert deviation(base, list(x)).tolist() == z.tolist()


def test_deviation_is_high_with_movement_and_low_when_quiet() -> None:
    base = make_baseline()
    moving = moved(base, ("variance", "motion_band_power", "spectral_power", "change_points"), 20.0)
    z_moving = deviation(base, moving)
    for name in ("variance", "motion_band_power", "spectral_power", "change_points"):
        assert z_moving[FEATURE_NAMES.index(name)] == pytest.approx(20.0)
    quiet = quiet_features(1, seed=99)[0]  # otra muestra de la misma sala vacía
    assert np.max(np.abs(deviation(base, quiet))) < 5.0


def test_deviation_nan_rssi_gives_nan_only_there() -> None:
    base = make_baseline()
    x = quiet_features(1, seed=2)[0]
    x[RSSI] = NAN
    z = deviation(base, x)
    assert np.isnan(z[RSSI])
    assert np.all(np.isfinite(np.delete(z, RSSI)))


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_deviation_rejects_non_finite_in_other_features(bad: float) -> None:
    base = make_baseline()
    x = quiet_features(1, seed=2)[0]
    x[4] = bad
    with pytest.raises(CalibrationError, match="dominant_freq_hz"):
        deviation(base, x)


def test_deviation_rejects_infinite_rssi_and_wrong_shapes() -> None:
    base = make_baseline()
    x = quiet_features(1, seed=2)[0]
    x[RSSI] = INF
    with pytest.raises(CalibrationError, match="mean_rssi"):
        deviation(base, x)
    with pytest.raises(CalibrationError, match="forma"):
        deviation(base, np.zeros(14))
    with pytest.raises(CalibrationError, match="forma"):
        deviation(base, np.zeros((1, 15)))
    with pytest.raises(CalibrationError):
        deviation(base, ["x"] * 15)


def test_deviation_overflow_is_reported() -> None:
    base = make_baseline()
    x = base.median.copy()
    x[0] = 1e308
    with pytest.raises(CalibrationError, match="desborda"):
        deviation(base, x)


def test_wrong_baseline_type_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="Baseline"):
        deviation({"median": 1}, np.zeros(15))  # type: ignore[arg-type]


# -------------------------------------------------------------------- is_quiet


def test_is_quiet_true_for_a_quiet_vector_and_false_with_movement() -> None:
    base = make_baseline()
    quiet = quiet_features(1, seed=99)[0]
    result = is_quiet(base, quiet, z_max=4.0, features_to_check=CHECK)
    assert result is True
    moving = moved(base, ("motion_band_power",), 15.0)
    assert is_quiet(base, moving, z_max=4.0, features_to_check=CHECK) is False


def test_is_quiet_threshold_is_configurable_and_compares_absolute_deviation() -> None:
    base = make_baseline()
    just_inside = moved(base, ("variance",), 1.9)
    just_outside = moved(base, ("variance",), 2.1)
    assert is_quiet(base, just_inside, z_max=2.0, features_to_check=("variance",)) is True
    assert is_quiet(base, just_outside, z_max=2.0, features_to_check=("variance",)) is False
    assert is_quiet(base, just_outside, z_max=3.0, features_to_check=("variance",)) is True
    below = moved(base, ("variance",), -2.1)  # desvío negativo también cuenta
    assert is_quiet(base, below, z_max=2.0, features_to_check=("variance",)) is False


def test_is_quiet_only_looks_at_the_requested_features() -> None:
    base = make_baseline()
    moving_other = moved(base, ("amp_max", "amp_range"), 30.0)
    assert is_quiet(base, moving_other, z_max=3.0, features_to_check=CHECK) is True
    assert is_quiet(base, moving_other, z_max=3.0, features_to_check=("amp_max",)) is False
    # Basta una característica comprobada fuera de umbral para no estar quieto.
    one_bad = moved(base, ("spectral_power",), 10.0)
    assert is_quiet(base, one_bad, z_max=3.0, features_to_check=CHECK) is False
    assert is_quiet(base, one_bad, z_max=3.0, features_to_check=list(CHECK)) is False  # type: ignore[arg-type]


def test_is_quiet_ignores_missing_rssi_if_it_is_not_checked() -> None:
    base = make_baseline()
    x = quiet_features(1, seed=99)[0]
    x[RSSI] = NAN
    assert is_quiet(base, x, z_max=4.0, features_to_check=CHECK) is True


def test_is_quiet_cannot_judge_a_missing_checked_feature() -> None:
    base = make_baseline()
    x = quiet_features(1, seed=99)[0]
    x[RSSI] = NAN
    with pytest.raises(CalibrationError, match="mean_rssi"):
        is_quiet(base, x, z_max=4.0, features_to_check=("variance", "mean_rssi"))
    no_rssi = make_baseline(features=np.where(np.arange(15) == RSSI, NAN, quiet_features()))
    with pytest.raises(CalibrationError, match="mean_rssi"):
        is_quiet(no_rssi, quiet_features(1, seed=99)[0], z_max=4.0, features_to_check=("mean_rssi",))


@pytest.mark.parametrize("bad", [0, -1.0, NAN, INF, "3", None, True])
def test_is_quiet_rejects_bad_z_max(bad: object) -> None:
    base = make_baseline()
    with pytest.raises(ValueError, match="z_max"):
        is_quiet(base, base.median, z_max=bad, features_to_check=CHECK)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [(), [], "variance", None, ("variance", "no_existe"), 5])
def test_is_quiet_rejects_bad_features_to_check(bad: Any) -> None:
    base = make_baseline()
    with pytest.raises(ValueError, match="features_to_check"):
        is_quiet(base, base.median, z_max=3.0, features_to_check=bad)


def test_is_quiet_propagates_bad_vectors() -> None:
    base = make_baseline()
    x = base.median.copy()
    x[0] = NAN
    with pytest.raises(CalibrationError):
        is_quiet(base, x, z_max=3.0, features_to_check=CHECK)


# -------------------------------------------------------------- update_baseline


def test_update_with_quiet_false_returns_the_same_baseline_unchanged() -> None:
    base = make_baseline()
    median_before = base.median.copy()
    far = moved(base, FEATURE_NAMES, 50.0)
    assert update_baseline(base, far, alpha=0.5, quiet=False) is base
    assert np.array_equal(base.median, median_before)


def test_update_with_quiet_false_does_not_even_read_the_vector() -> None:
    base = make_baseline()
    assert update_baseline(base, np.full(15, NAN), alpha=0.5, quiet=False) is base
    assert update_baseline(base, "basura", alpha=0.5, quiet=False) is base


def test_update_with_quiet_true_moves_the_median_slightly_by_ema() -> None:
    base = make_baseline()
    x = moved(base, FEATURE_NAMES, 2.0)  # 2 puntuaciones robustas por encima
    new = update_baseline(base, x, alpha=0.05, quiet=True)
    assert new is not base
    assert np.allclose(new.median, 0.95 * base.median + 0.05 * x)
    shift = new.median - base.median
    assert np.allclose(shift, 0.05 * (x - base.median))
    assert np.all(np.abs(shift) < 0.2 * MAD_TO_SIGMA * base.mad)  # mueve poco: 0.1 sigma
    assert np.all(shift > 0)
    # La original no se toca; el resto de campos se conserva.
    assert np.array_equal(base.median, make_baseline().median)
    assert np.array_equal(new.mad, base.mad)
    assert (new.node_id, new.n_windows, new.created_at, new.environment_note) == (
        base.node_id,
        base.n_windows,
        base.created_at,
        base.environment_note,
    )
    assert new.feature_names == FEATURE_NAMES


def test_update_extremes_of_alpha() -> None:
    base = make_baseline()
    x = moved(base, FEATURE_NAMES, 2.0)
    assert np.allclose(update_baseline(base, x, alpha=1.0, quiet=True).median, x)
    tiny = update_baseline(base, x, alpha=1e-6, quiet=True)
    assert np.allclose(tiny.median, base.median, rtol=0, atol=1e-5 * np.abs(x - base.median).max())


def test_repeated_quiet_updates_converge_to_the_new_level() -> None:
    base = make_baseline()
    target = moved(base, FEATURE_NAMES, 2.0)
    current = base
    for _ in range(200):
        current = update_baseline(current, target, alpha=0.1, quiet=True)
    assert np.allclose(current.median, target, rtol=1e-6, atol=1e-6)
    assert np.array_equal(base.median, make_baseline().median)  # sin efectos laterales


def test_update_leaves_rssi_alone_when_either_side_lacks_it() -> None:
    base = make_baseline()
    x = moved(base, FEATURE_NAMES, 2.0)
    x[RSSI] = NAN
    new = update_baseline(base, x, alpha=0.5, quiet=True)
    assert new.median[RSSI] == base.median[RSSI]  # el vector no trae RSSI
    assert new.median[0] != base.median[0]
    no_rssi = make_baseline(features=np.where(np.arange(15) == RSSI, NAN, quiet_features()))
    y = quiet_features(1, seed=3)[0]
    new2 = update_baseline(no_rssi, y, alpha=0.5, quiet=True)
    assert np.isnan(new2.median[RSSI]) and np.isnan(new2.mad[RSSI])  # no se inventa
    assert new2.median[0] != no_rssi.median[0]


@pytest.mark.parametrize("alpha", [0, 0.0, -0.1, 1.0001, 2, NAN, INF, "0.5", None, True])
def test_update_rejects_alpha_outside_zero_one(alpha: object) -> None:
    base = make_baseline()
    for quiet in (True, False):  # se valida siempre, también con quiet=False
        with pytest.raises(ValueError, match="alpha"):
            update_baseline(base, base.median, alpha=alpha, quiet=quiet)  # type: ignore[arg-type]


@pytest.mark.parametrize("quiet", [1, 0, "False", None, "yes"])
def test_update_rejects_non_boolean_quiet(quiet: object) -> None:
    base = make_baseline()
    with pytest.raises(TypeError, match="quiet"):
        update_baseline(base, base.median, alpha=0.5, quiet=quiet)  # type: ignore[arg-type]


def test_update_accepts_numpy_bool_from_comparisons() -> None:
    base = make_baseline()
    assert update_baseline(base, base.median, alpha=0.5, quiet=np.bool_(False)) is base
    assert update_baseline(base, base.median, alpha=0.5, quiet=np.bool_(True)) is not base


def test_update_with_quiet_true_rejects_bad_vectors() -> None:
    base = make_baseline()
    x = base.median.copy()
    x[3] = NAN
    with pytest.raises(CalibrationError):
        update_baseline(base, x, alpha=0.5, quiet=True)
    with pytest.raises(CalibrationError, match="forma"):
        update_baseline(base, np.zeros(10), alpha=0.5, quiet=True)


def test_quiet_loop_with_is_quiet_ignores_movement_and_follows_slow_drift() -> None:
    base = make_baseline()
    moving = moved(base, CHECK, 20.0)
    quiet = is_quiet(base, moving, z_max=4.0, features_to_check=CHECK)
    assert update_baseline(base, moving, alpha=0.5, quiet=quiet) is base  # movimiento: no aprende
    drift = moved(base, FEATURE_NAMES, 1.0)
    quiet = is_quiet(base, drift, z_max=4.0, features_to_check=CHECK)
    assert update_baseline(base, drift, alpha=0.5, quiet=quiet) is not base


# --------------------------------------------------------------------- JSON I/O


def test_round_trip_save_load(tmp_path: Path) -> None:
    base = make_baseline(environment_note="sala-101 vacía, pared norte")
    path = tmp_path / "baseline.json"
    save_baseline(base, path)
    loaded = load_baseline(path)
    assert loaded.node_id == base.node_id
    assert loaded.feature_names == base.feature_names == FEATURE_NAMES
    assert np.array_equal(loaded.median, base.median)  # exacto
    assert np.array_equal(loaded.mad, base.mad)
    assert loaded.n_windows == base.n_windows
    assert loaded.created_at == base.created_at
    assert loaded.environment_note == base.environment_note
    x = quiet_features(1, seed=8)[0]
    assert np.array_equal(deviation(loaded, x), deviation(base, x))


def test_round_trip_keeps_the_missing_rssi_baseline_as_null(tmp_path: Path) -> None:
    X = quiet_features()
    X[:, RSSI] = NAN
    X[:, CHANGE] = 0.0
    base = fit_baseline("n1", X, "sin RSSI", created_at=5.0)
    path = tmp_path / "baseline.json"
    save_baseline(base, path)
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw  # JSON estricto
    record = json.loads(raw)
    assert record["median"][RSSI] is None and record["mad"][RSSI] is None
    loaded = load_baseline(path)
    assert np.isnan(loaded.median[RSSI]) and np.isnan(loaded.mad[RSSI])
    assert np.array_equal(loaded.median, base.median, equal_nan=True)
    assert np.array_equal(loaded.mad, base.mad, equal_nan=True)
    assert loaded.floored_features == ("change_points",)


def test_saved_file_layout(tmp_path: Path) -> None:
    base = make_baseline(environment_note="calibración: sala")
    path = tmp_path / "baseline.json"
    save_baseline(base, path)
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert "calibración".encode("utf-8") in raw  # UTF-8 real
    record = json.loads(raw)
    assert set(record) == {
        "baseline_version",
        "node_id",
        "feature_names",
        "median",
        "mad",
        "n_windows",
        "created_at",
        "environment_note",
    }
    assert record["baseline_version"] == BASELINE_VERSION == 1
    assert record["feature_names"] == list(FEATURE_NAMES)
    assert len(record["median"]) == len(record["mad"]) == 15


def test_save_accepts_str_path_leaves_no_temp_files_and_overwrites(tmp_path: Path) -> None:
    path = str(tmp_path / "baseline.json")
    save_baseline(make_baseline(node_id="a"), path)
    save_baseline(make_baseline(node_id="b"), path)
    assert load_baseline(path).node_id == "b"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["baseline.json"]


def test_save_to_a_missing_directory_fails_cleanly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        save_baseline(make_baseline(), tmp_path / "no-existe" / "b.json")


def test_save_rejects_a_non_baseline(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Baseline"):
        save_baseline({"node_id": "n1"}, tmp_path / "b.json")  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_baseline(tmp_path / "no-existe.json")


def valid_record() -> dict[str, Any]:
    base = make_baseline()
    return {
        "baseline_version": 1,
        "node_id": "n1",
        "feature_names": list(FEATURE_NAMES),
        "median": base.median.tolist(),
        "mad": base.mad.tolist(),
        "n_windows": 60,
        "created_at": 1_700_000_000.5,
        "environment_note": "nota",
    }


def write_record(path: Path, record: Any) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_valid_record_loads(tmp_path: Path) -> None:
    loaded = load_baseline(write_record(tmp_path / "b.json", valid_record()))
    assert loaded.n_windows == 60 and loaded.node_id == "n1"


@pytest.mark.parametrize(
    "missing",
    [
        "baseline_version",
        "node_id",
        "feature_names",
        "median",
        "mad",
        "n_windows",
        "created_at",
        "environment_note",
    ],
)
def test_load_requires_every_field(tmp_path: Path, missing: str) -> None:
    record = valid_record()
    del record[missing]
    with pytest.raises(CalibrationError, match=missing):
        load_baseline(write_record(tmp_path / "b.json", record))


def test_load_rejects_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="extra"):
        load_baseline(write_record(tmp_path / "b.json", {**valid_record(), "extra": 1}))


@pytest.mark.parametrize("version", [2, 0])
def test_load_rejects_unsupported_versions(tmp_path: Path, version: int) -> None:
    record = {**valid_record(), "baseline_version": version}
    with pytest.raises(CalibrationError, match=f"baseline_version {version} no soportada"):
        load_baseline(write_record(tmp_path / "b.json", record))


@pytest.mark.parametrize("version", ["1", 1.0, True, None])
def test_load_rejects_non_integer_versions(tmp_path: Path, version: object) -> None:
    record = {**valid_record(), "baseline_version": version}
    with pytest.raises(CalibrationError, match="baseline_version debe ser un entero"):
        load_baseline(write_record(tmp_path / "b.json", record))


def test_load_rejects_feature_names_that_differ_from_the_current_ones(tmp_path: Path) -> None:
    reordered = list(FEATURE_NAMES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    for names in (reordered, list(FEATURE_NAMES[:-1]), [*FEATURE_NAMES, "nueva"], "variance"):
        record = {**valid_record(), "feature_names": names}
        with pytest.raises(CalibrationError, match="feature_names"):
            load_baseline(write_record(tmp_path / "b.json", record))


def test_load_rejects_null_outside_mean_rssi(tmp_path: Path) -> None:
    record = valid_record()
    record["median"][3] = None
    with pytest.raises(CalibrationError, match="spectral_power"):
        load_baseline(write_record(tmp_path / "b.json", record))
    record = valid_record()
    record["mad"][RSSI] = None  # null en el MAD pero no en la mediana: incoherente
    with pytest.raises(CalibrationError, match="mean_rssi"):
        load_baseline(write_record(tmp_path / "b.json", record))


@pytest.mark.parametrize(
    "override, match",
    [
        ({"median": [1.0] * 14}, "median"),
        ({"mad": [1.0] * 16}, "mad"),
        ({"median": "no es lista"}, "median"),
        ({"median": ["x"] * 15}, "median"),
        ({"mad": [True] * 15}, "mad"),
        ({"node_id": ""}, "node_id"),
        ({"node_id": 3}, "node_id"),
        ({"n_windows": 0}, "n_windows"),
        ({"n_windows": 5.0}, "n_windows"),
        ({"n_windows": True}, "n_windows"),
        ({"created_at": "ayer"}, "created_at"),
        ({"created_at": None}, "created_at"),
        ({"environment_note": 5}, "environment_note"),
    ],
)
def test_load_rejects_incoherent_values(
    tmp_path: Path, override: dict[str, Any], match: str
) -> None:
    with pytest.raises(CalibrationError, match=match):
        load_baseline(write_record(tmp_path / "b.json", {**valid_record(), **override}))


def test_load_rejects_non_positive_mad(tmp_path: Path) -> None:
    for bad in (0.0, -1.0):
        record = valid_record()
        record["mad"][2] = bad
        with pytest.raises(CalibrationError, match="mad debe ser > 0"):
            load_baseline(write_record(tmp_path / "b.json", record))


def test_load_rejects_invalid_files(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    path.write_text("{roto", encoding="utf-8")
    with pytest.raises(CalibrationError, match="JSON inválido"):
        load_baseline(path)
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(CalibrationError, match="objeto JSON"):
        load_baseline(path)
    path.write_bytes(b'{"x": "\xff\xfe"}')
    with pytest.raises(CalibrationError, match="UTF-8"):
        load_baseline(path)
    text = json.dumps(valid_record()).replace('"n_windows": 60', '"n_windows": NaN')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CalibrationError, match="NaN"):
        load_baseline(path)


# ------------------------------------------------------------- Baseline directo


def test_direct_construction_is_validated() -> None:
    base = make_baseline()
    args: dict[str, Any] = {
        "node_id": "n1",
        "feature_names": FEATURE_NAMES,
        "median": base.median,
        "mad": base.mad,
        "n_windows": 3,
        "created_at": 1.0,
        "environment_note": "",
    }
    assert Baseline(**args).n_windows == 3
    with pytest.raises(CalibrationError, match="feature_names"):
        Baseline(**{**args, "feature_names": tuple(reversed(FEATURE_NAMES))})
    with pytest.raises(CalibrationError, match="mad debe ser > 0"):
        Baseline(**{**args, "mad": np.zeros(15)})
    bad_median = base.median.copy()
    bad_median[0] = NAN
    with pytest.raises(CalibrationError, match="variance"):
        Baseline(**{**args, "median": bad_median})
    with pytest.raises(CalibrationError, match="forma"):
        Baseline(**{**args, "median": np.zeros(3)})