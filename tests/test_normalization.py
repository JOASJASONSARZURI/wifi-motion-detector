"""Pruebas de wifi_motion.normalization.

SYNTHETIC: todos los datos de este archivo se generan aquí con numpy (semilla
fija) o se escriben a mano; no son mediciones reales. Las pruebas de archivos
usan ``tmp_path`` de pytest.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wifi_motion.normalization import (
    CONSTANT_STD_THRESHOLD,
    NORMALIZER_VERSION,
    Normalizer,
    NotFittedError,
)

NAN = float("nan")
INF = float("inf")
NAMES = ("variance", "amp_mean", "dominant_freq_hz")


def synthetic(n: int = 200, seed: int = 0) -> np.ndarray:
    """SYNTHETIC: 3 columnas con medias y escalas muy distintas."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.normal(50.0, 10.0, n),
            rng.normal(-3.0, 0.01, n),
            rng.uniform(0.0, 5000.0, n),
        ]
    )


def fitted(X: np.ndarray | None = None, names: tuple[str, ...] = NAMES) -> Normalizer:
    return Normalizer().fit(synthetic() if X is None else X, names)


# ------------------------------------------------------------------ fit básico


def test_version_constant() -> None:
    assert NORMALIZER_VERSION == 1


def test_fit_returns_self_and_learns_population_statistics() -> None:
    X = synthetic()
    norm = Normalizer()
    assert norm.is_fitted is False
    assert norm.fit(X, NAMES) is norm
    assert norm.is_fitted is True
    assert norm.feature_names == NAMES
    assert np.allclose(norm.mean, X.mean(axis=0))
    assert np.allclose(norm.std, X.std(axis=0))  # ddof = 0
    assert norm.constant_features == []


def test_fit_transform_gives_zero_mean_and_unit_std() -> None:
    X = synthetic()
    Z = Normalizer().fit_transform(X, NAMES)
    assert Z.shape == X.shape
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-12)


def test_fit_accepts_lists_and_a_list_of_names() -> None:
    norm = Normalizer().fit([[1.0, 10.0], [3.0, 30.0]], ["a", "b"])
    assert norm.feature_names == ("a", "b")
    assert norm.mean.tolist() == [2.0, 20.0]
    assert norm.std.tolist() == [1.0, 10.0]


def test_two_rows_is_enough() -> None:
    Z = Normalizer().fit_transform([[1.0], [3.0]], ("a",))
    assert Z.ravel().tolist() == [-1.0, 1.0]


def test_fit_does_not_keep_a_reference_to_the_input() -> None:
    X = synthetic()
    norm = fitted(X)
    mean_before = norm.mean.copy()
    X[:] = 0.0  # mutar el original no debe afectar al ajuste
    assert np.array_equal(norm.mean, mean_before)


def test_statistics_are_read_only() -> None:
    norm = fitted()
    with pytest.raises(ValueError):
        norm.mean[0] = 0.0
    with pytest.raises(ValueError):
        norm.std[0] = 1.0


# -------------------------------------------------------------------- constantes


def test_constant_column_uses_std_one_and_is_listed() -> None:
    X = synthetic()
    X[:, 1] = 7.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # una división por cero daría RuntimeWarning
        norm = Normalizer().fit(X, NAMES)
        Z = norm.transform(X)
    assert norm.constant_features == ["amp_mean"]
    assert isinstance(norm.constant_features, list)
    assert norm.std[1] == 1.0 and norm.mean[1] == 7.0
    assert np.all(Z[:, 1] == 0.0)
    assert np.all(np.isfinite(Z))
    # Las demás columnas siguen normalizándose bien.
    assert np.allclose(Z[:, [0, 2]].mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z[:, [0, 2]].std(axis=0), 1.0, atol=1e-12)


def test_constant_column_with_inexact_float_is_still_constant() -> None:
    X = synthetic()
    X[:, 0] = 0.1  # el residuo de redondeo de la std queda por debajo del umbral
    norm = Normalizer().fit(X, NAMES)
    assert norm.constant_features == ["variance"]
    assert norm.std[0] == 1.0


def test_std_below_threshold_counts_as_constant_and_above_does_not() -> None:
    assert CONSTANT_STD_THRESHOLD == 1e-12
    tiny = np.array([[1.0, 5.0], [1.0 + 2e-13, 6.0], [1.0, 7.0], [1.0 + 2e-13, 8.0]])
    norm = Normalizer().fit(tiny, ("tiny", "ok"))
    assert norm.constant_features == ["tiny"]  # std ~ 1e-13 < 1e-12
    big = np.array([[1.0], [1.0 + 2e-9]])
    assert Normalizer().fit(big, ("x",)).constant_features == []  # std = 1e-9 > 1e-12


def test_all_columns_constant() -> None:
    X = np.full((5, 3), 2.0)
    norm = Normalizer().fit(X, NAMES)
    assert norm.constant_features == list(NAMES)
    assert np.all(norm.transform(X) == 0.0)


def test_constant_feature_in_new_data_shows_the_deviation() -> None:
    X = synthetic()
    X[:, 1] = 7.0
    norm = Normalizer().fit(X, NAMES)
    new = np.array([[50.0, 9.5, 100.0]])
    assert norm.transform(new)[0, 1] == pytest.approx(2.5)  # (9.5 - 7.0) / 1.0


def test_constant_features_is_recomputed_on_refit() -> None:
    X = synthetic()
    X[:, 2] = 1.0
    norm = Normalizer().fit(X, NAMES)
    assert norm.constant_features == ["dominant_freq_hz"]
    norm.fit(synthetic(), NAMES)
    assert norm.constant_features == []


# ----------------------------------------------------------------- fit rechaza


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_fit_rejects_non_finite_values(bad: float) -> None:
    X = synthetic()
    X[10, 1] = bad
    with pytest.raises(ValueError, match="no finito"):
        Normalizer().fit(X, NAMES)


@pytest.mark.parametrize("rows", [0, 1])
def test_fit_rejects_fewer_than_two_rows(rows: int) -> None:
    with pytest.raises(ValueError, match="al menos 2 filas"):
        Normalizer().fit(synthetic(rows), NAMES)


def test_fit_rejects_columns_that_do_not_match_names() -> None:
    with pytest.raises(ValueError, match="3 columna.*2 nombre"):
        Normalizer().fit(synthetic(), ("a", "b"))
    with pytest.raises(ValueError, match="3 columna.*4 nombre"):
        Normalizer().fit(synthetic(), ("a", "b", "c", "d"))


def test_fit_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        Normalizer().fit(np.arange(5.0), ("a",))
    with pytest.raises(ValueError, match="2-D"):
        Normalizer().fit(np.zeros((2, 2, 2)), ("a", "b"))


@pytest.mark.parametrize(
    "names",
    [(), ("a", "a", "b"), ("a", "", "b"), ("a", "  ", "b"), ("a", 1, "b"), "abc", None],
)
def test_fit_rejects_bad_feature_names(names: Any) -> None:
    with pytest.raises(ValueError, match="feature_names"):
        Normalizer().fit(synthetic(), names)


def test_fit_rejects_non_numeric_data() -> None:
    with pytest.raises(ValueError):
        Normalizer().fit([["x", "y"], ["z", "w"]], ("a", "b"))
    with pytest.raises(ValueError):
        Normalizer().fit([[1.0, None], [2.0, 3.0]], ("a", "b"))


def test_fit_rejects_values_that_overflow_statistics() -> None:
    X = np.array([[1e308], [-1e308], [1e308], [-1e308]])
    with pytest.raises(ValueError, match="desbordan"):
        Normalizer().fit(X, ("a",))


def test_failed_fit_keeps_the_previous_state() -> None:
    norm = fitted()
    mean_before = norm.mean.copy()
    bad = synthetic()
    bad[0, 0] = NAN
    with pytest.raises(ValueError):
        norm.fit(bad, NAMES)
    assert np.array_equal(norm.mean, mean_before)
    assert norm.feature_names == NAMES


# -------------------------------------------------------------------- transform


def test_transform_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        Normalizer().transform(synthetic())


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_transform_rejects_non_finite_values(bad: float) -> None:
    norm = fitted()
    X = synthetic(5, seed=1)
    X[2, 0] = bad
    with pytest.raises(ValueError, match="no finito"):
        norm.transform(X)


@pytest.mark.parametrize("cols", [1, 2, 4])
def test_transform_rejects_wrong_number_of_columns(cols: int) -> None:
    norm = fitted()
    with pytest.raises(ValueError, match="columna"):
        norm.transform(np.zeros((5, cols)))
    with pytest.raises(ValueError, match="columna"):
        norm.transform(np.zeros(cols))  # vector 1-D de longitud incorrecta


def test_transform_rejects_3d_input() -> None:
    with pytest.raises(ValueError, match="3-D"):
        fitted().transform(np.zeros((2, 2, 3)))


def test_transform_rejects_a_result_that_overflows() -> None:
    norm = Normalizer().fit([[0.0], [1e-3]], ("a",))  # std = 5e-4
    with pytest.raises(ValueError, match="no es finito"):
        norm.transform([[1e308]])


def test_transform_uses_fit_statistics_not_the_new_data() -> None:
    train = synthetic(seed=0)
    norm = fitted(train)
    new = synthetic(50, seed=1) * 3.0 + 100.0  # distribución muy distinta
    Z = norm.transform(new)
    expected = (new - train.mean(axis=0)) / train.std(axis=0)
    assert np.allclose(Z, expected)
    # Si recalculara con los datos nuevos, la media sería 0 y la std 1:
    assert not np.allclose(Z.mean(axis=0), 0.0, atol=1e-3)
    assert not np.allclose(Z.std(axis=0), 1.0, atol=1e-3)


def test_transform_does_not_change_the_statistics() -> None:
    norm = fitted()
    mean_before, std_before = norm.mean.copy(), norm.std.copy()
    norm.transform(synthetic(7, seed=4) * 10.0)
    norm.transform(synthetic(3, seed=5) - 1000.0)
    assert np.array_equal(norm.mean, mean_before)
    assert np.array_equal(norm.std, std_before)


def test_a_single_row_is_normalized_with_fit_statistics() -> None:
    train = synthetic()
    norm = fitted(train)
    row = np.array([[train[:, 0].mean(), train[:, 1].mean(), train[:, 2].mean() + 1000.0]])
    Z = norm.transform(row)  # una sola fila: recalcular daría std 0
    assert Z[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert Z[0, 2] == pytest.approx(1000.0 / train[:, 2].std())


def test_transform_accepts_a_1d_vector_and_matches_the_2d_row() -> None:
    norm = fitted()
    X = synthetic(4, seed=2)
    vector = norm.transform(X[1])
    assert vector.shape == (3,)
    assert np.array_equal(vector, norm.transform(X)[1])


def test_transform_does_not_modify_its_input() -> None:
    norm = fitted()
    X = synthetic(6, seed=3)
    before = X.copy()
    norm.transform(X)
    assert np.array_equal(X, before)


def test_transform_accepts_an_empty_batch() -> None:
    assert fitted().transform(np.zeros((0, 3))).shape == (0, 3)


# ---------------------------------------------------------------- save / load


def test_round_trip_save_load(tmp_path: Path) -> None:
    X = synthetic()
    X[:, 2] = 4.2  # incluye una constante
    norm = Normalizer().fit(X, NAMES)
    path = tmp_path / "norm.json"
    norm.save(path)
    loaded = Normalizer.load(path)
    assert loaded.is_fitted
    assert loaded.feature_names == norm.feature_names
    assert np.array_equal(loaded.mean, norm.mean)  # exacto: repr de float va y vuelve sin pérdida
    assert np.array_equal(loaded.std, norm.std)
    assert loaded.constant_features == norm.constant_features == ["dominant_freq_hz"]
    new = synthetic(20, seed=9)
    assert np.array_equal(loaded.transform(new), norm.transform(new))


def test_save_accepts_str_path_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    path = str(tmp_path / "norm.json")
    fitted().save(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["norm.json"]
    assert Normalizer.load(path).is_fitted


def test_saved_file_layout(tmp_path: Path) -> None:
    names = ("variância", "amp_mean", "x")  # UTF-8 real
    X = synthetic()
    path = tmp_path / "norm.json"
    Normalizer().fit(X, names).save(path)
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert "variância".encode("utf-8") in raw
    record = json.loads(raw)
    assert set(record) == {
        "normalizer_version",
        "feature_names",
        "mean",
        "std",
        "constant_features",
    }
    assert record["normalizer_version"] == 1
    assert record["feature_names"] == list(names)
    assert len(record["mean"]) == len(record["std"]) == 3
    assert record["constant_features"] == []


def test_save_overwrites_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "norm.json"
    fitted().save(path)
    other = Normalizer().fit([[1.0], [3.0]], ("solo",))
    other.save(path)
    assert Normalizer.load(path).feature_names == ("solo",)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["norm.json"]


def test_save_before_fit_raises_and_creates_nothing(tmp_path: Path) -> None:
    with pytest.raises(NotFittedError):
        Normalizer().save(tmp_path / "norm.json")
    assert list(tmp_path.iterdir()) == []


def test_save_to_a_missing_directory_fails_cleanly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fitted().save(tmp_path / "no-existe" / "norm.json")


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Normalizer.load(tmp_path / "no-existe.json")


def valid_record() -> dict[str, Any]:
    return {
        "normalizer_version": 1,
        "feature_names": ["a", "b"],
        "mean": [1.0, 2.0],
        "std": [0.5, 1.0],
        "constant_features": ["b"],
    }


def write_record(path: Path, record: Any) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_valid_record_loads(tmp_path: Path) -> None:
    norm = Normalizer.load(write_record(tmp_path / "n.json", valid_record()))
    assert norm.feature_names == ("a", "b")
    assert norm.constant_features == ["b"]
    assert norm.transform([[2.0, 5.0]]).tolist() == [[2.0, 3.0]]


@pytest.mark.parametrize(
    "missing", ["normalizer_version", "feature_names", "mean", "std", "constant_features"]
)
def test_load_requires_every_field(tmp_path: Path, missing: str) -> None:
    record = valid_record()
    del record[missing]
    with pytest.raises(ValueError, match=missing):
        Normalizer.load(write_record(tmp_path / "n.json", record))


def test_load_rejects_unknown_fields(tmp_path: Path) -> None:
    record = {**valid_record(), "extra": 1}
    with pytest.raises(ValueError, match="extra"):
        Normalizer.load(write_record(tmp_path / "n.json", record))


@pytest.mark.parametrize("version", [2, 0, 9])
def test_load_rejects_unsupported_versions(tmp_path: Path, version: int) -> None:
    record = {**valid_record(), "normalizer_version": version}
    with pytest.raises(ValueError, match=f"normalizer_version {version} no soportada"):
        Normalizer.load(write_record(tmp_path / "n.json", record))


@pytest.mark.parametrize("version", ["1", 1.0, True, None, [1]])
def test_load_rejects_non_integer_versions(tmp_path: Path, version: object) -> None:
    record = {**valid_record(), "normalizer_version": version}
    with pytest.raises(ValueError, match="normalizer_version debe ser un entero"):
        Normalizer.load(write_record(tmp_path / "n.json", record))


@pytest.mark.parametrize(
    "override, match",
    [
        ({"mean": [1.0]}, "mean"),  # longitud distinta de los nombres
        ({"std": [0.5, 1.0, 2.0]}, "std"),
        ({"mean": [1.0, "x"]}, "mean"),
        ({"mean": [1.0, None]}, "mean"),
        ({"mean": [1.0, True]}, "mean"),
        ({"mean": "no es lista"}, "mean"),
        ({"std": [0.5, 0.0]}, "std debe ser > 0"),
        ({"std": [-0.5, 1.0]}, "std debe ser > 0"),
        ({"feature_names": ["a", "a"]}, "repetidos"),
        ({"feature_names": []}, "feature_names"),
        ({"feature_names": "ab"}, "feature_names"),
        ({"constant_features": ["zzz"]}, "zzz"),
        ({"constant_features": "b"}, "constant_features"),
        ({"constant_features": ["a"]}, "constante 'a' debe tener std 1.0"),  # a tiene std 0.5
    ],
)
def test_load_rejects_incoherent_records(
    tmp_path: Path, override: dict[str, Any], match: str
) -> None:
    record = {**valid_record(), **override}
    with pytest.raises(ValueError, match=match):
        Normalizer.load(write_record(tmp_path / "n.json", record))


def test_load_rejects_invalid_files(tmp_path: Path) -> None:
    path = tmp_path / "n.json"
    path.write_text("{roto", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON inválido"):
        Normalizer.load(path)
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="objeto JSON"):
        Normalizer.load(path)
    path.write_bytes(b'{"x": "\xff\xfe"}')
    with pytest.raises(ValueError, match="UTF-8"):
        Normalizer.load(path)
    path.write_text(
        '{"normalizer_version":1,"feature_names":["a"],"mean":[NaN],"std":[1.0],'
        '"constant_features":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="NaN"):
        Normalizer.load(path)


def test_loaded_normalizer_can_be_refit_without_touching_the_file(tmp_path: Path) -> None:
    path = tmp_path / "n.json"
    fitted().save(path)
    original = path.read_bytes()
    loaded = Normalizer.load(path)
    loaded.fit([[1.0], [3.0]], ("solo",))
    assert loaded.feature_names == ("solo",)
    assert path.read_bytes() == original