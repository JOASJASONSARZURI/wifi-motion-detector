"""Pruebas SYNTHETIC del clasificador softmax multiclase."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wifi_motion.classifiers.base import (
    ABSENT,
    CLASS_STATES,
    PRESENT_MOVING,
    PRESENT_STILL,
    ClassifierError,
)
from wifi_motion.classifiers.softmax import (
    SoftmaxClassifier,
    SoftmaxConfig,
    _softmax,
    confusion_matrix,
    macro_f1,
)
from wifi_motion.features import FEATURE_NAMES

RSSI = FEATURE_NAMES.index("mean_rssi")


def synthetic_data(
    seed: int = 7,
    counts: tuple[int, int, int] = (90, 90, 90),
    noise: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Tres nubes gaussianas separables, generadas exclusivamente aquí."""
    rng = np.random.default_rng(seed)
    centers = (-3.0, 0.0, 3.0)
    batches = []
    labels: list[str] = []
    for state, center, count in zip(CLASS_STATES, centers, counts):
        batch = rng.normal(0.0, noise, size=(count, len(FEATURE_NAMES)))
        batch[:, 0] += center
        batch[:, 1] += center * 0.6
        batches.append(batch)
        labels.extend([state] * count)
    X = np.vstack(batches)
    y = np.asarray(labels)
    order = rng.permutation(len(y))
    return X[order], y[order]


def split_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = synthetic_data()
    train_indices: list[int] = []
    val_indices: list[int] = []
    for state in CLASS_STATES:
        indices = np.flatnonzero(y == state)
        train_indices.extend(indices[:60])
        val_indices.extend(indices[60:])
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def config(**overrides: object) -> SoftmaxConfig:
    values: dict[str, object] = {
        "lr": 0.08,
        "epochs": 250,
        "batch_size": 24,
        "l2": 1e-4,
        "seed": 19,
        "patience": 35,
    }
    values.update(overrides)
    return SoftmaxConfig(**values)  # type: ignore[arg-type]


def train_model(
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    settings: SoftmaxConfig | None = None,
) -> SoftmaxClassifier:
    if X_train is None or y_train is None or X_val is None or y_val is None:
        X_train, y_train, X_val, y_val = split_data()
    return SoftmaxClassifier().fit(
        X_train,
        y_train.tolist(),
        X_val,
        y_val.tolist(),
        config() if settings is None else settings,
    )


def test_converges_on_separable_data_and_validation_loss_decreases() -> None:
    X_train, y_train, X_val, y_val = split_data()
    model = train_model(X_train, y_train, X_val, y_val)
    predictions = model.predict_batch(X_val)
    score = macro_f1(y_val.tolist(), [item.state for item in predictions], CLASS_STATES)
    assert score > 0.98
    assert len(model.history) > 1
    assert model.history[1]["val_loss"] < model.history[0]["val_loss"]
    assert all(
        set(item) == {"epoch", "train_loss", "val_loss", "val_macro_f1"}
        for item in model.history
    )


def test_same_seed_is_reproducible() -> None:
    X_train, y_train, X_val, y_val = split_data()
    first = train_model(X_train, y_train, X_val, y_val)
    second = train_model(X_train, y_train, X_val, y_val)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.bias, second.bias)
    assert first.history == second.history


def test_fit_rejects_missing_training_class() -> None:
    X_train, y_train, X_val, y_val = split_data()
    keep = y_train != PRESENT_MOVING
    with pytest.raises(ValueError, match=PRESENT_MOVING):
        train_model(X_train[keep], y_train[keep], X_val, y_val)


def test_fit_rejects_unknown_label() -> None:
    X_train, y_train, X_val, y_val = split_data()
    bad_labels = y_train.tolist()
    bad_labels[0] = "unknown-state"
    with pytest.raises(ValueError, match="unknown-state"):
        SoftmaxClassifier().fit(X_train, bad_labels, X_val, y_val.tolist(), config())


def test_mean_rssi_is_imputed_from_training_median() -> None:
    X_train, y_train, X_val, y_val = split_data()
    training_rssi = np.arange(len(X_train), dtype=np.float64) + 10.0
    X_train[:, RSSI] = training_rssi
    X_train[::4, RSSI] = np.nan
    expected_median = float(np.nanmedian(X_train[:, RSSI]))
    X_val[:, RSSI] = np.nan
    model = train_model(X_train, y_train, X_val, y_val)
    assert model.rssi_median == expected_median
    assert model.predict(X_val[0]).state in CLASS_STATES


def test_nan_in_non_rssi_feature_is_rejected() -> None:
    X_train, y_train, X_val, y_val = split_data()
    X_train[0, 0] = np.nan
    with pytest.raises(ClassifierError, match="variance"):
        train_model(X_train, y_train, X_val, y_val)


def test_save_load_round_trip_preserves_predictions(tmp_path: Path) -> None:
    X_train, y_train, X_val, y_val = split_data()
    model = train_model(X_train, y_train, X_val, y_val)
    path = tmp_path / "softmax.json"
    model.save(path)
    loaded = SoftmaxClassifier.load(path)
    original = model.predict_batch(X_val)
    restored = loaded.predict_batch(X_val)
    assert [item.state for item in restored] == [item.state for item in original]
    assert [item.scores for item in restored] == [item.scores for item in original]


def test_load_rejects_missing_unknown_and_invalid_model_fields(tmp_path: Path) -> None:
    X_train, y_train, X_val, y_val = split_data()
    model = train_model(X_train, y_train, X_val, y_val)
    path = tmp_path / "softmax.json"
    model.save(path)
    record = json.loads(path.read_text(encoding="utf-8"))

    del record["bias"]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="bias"):
        SoftmaxClassifier.load(path)

    record = json.loads(json.dumps(record))
    record["bias"] = [0.0] * len(CLASS_STATES)
    record["unexpected"] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="desconocidos"):
        SoftmaxClassifier.load(path)


def test_softmax_is_stable_for_extreme_logits() -> None:
    scores = _softmax(np.array([[1e308, 0.0, -1e308], [-1e308, 1e308, 0.0]]))
    assert np.all(np.isfinite(scores))
    assert np.allclose(scores.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(scores, axis=1), [0, 1])


def test_balanced_class_weight_improves_rare_class_macro_f1() -> None:
    X_train, y_train = synthetic_data(seed=23, counts=(240, 12, 12), noise=1.0)
    X_val, y_val = synthetic_data(seed=24, counts=(60, 60, 60), noise=1.0)
    settings = config(epochs=300, patience=50, lr=0.04)
    plain = train_model(X_train, y_train, X_val, y_val, settings)
    balanced = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        config(epochs=300, patience=50, lr=0.04, class_weight="balanced"),
    )
    plain_f1 = macro_f1(
        y_val.tolist(), [prediction.state for prediction in plain.predict_batch(X_val)], CLASS_STATES
    )
    balanced_f1 = macro_f1(
        y_val.tolist(),
        [prediction.state for prediction in balanced.predict_batch(X_val)],
        CLASS_STATES,
    )
    assert balanced_f1 > plain_f1


def test_confusion_matrix_and_macro_f1_match_hand_calculation() -> None:
    actual = ["a", "a", "b", "b", "c", "c"]
    predicted = ["a", "b", "b", "c", "c", "a"]
    classes = ("a", "b", "c")
    expected = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert np.array_equal(confusion_matrix(actual, predicted, classes), expected)
    assert macro_f1(actual, predicted, classes) == pytest.approx(0.5)