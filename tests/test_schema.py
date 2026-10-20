"""Pruebas de wifi_motion.schema.

SYNTHETIC: todos los valores de este archivo están escritos a mano solo para
probar la estructura; no son mediciones reales.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields
from typing import Any

import pytest

from wifi_motion.schema import (
    LABELS,
    SCHEMA_VERSION,
    CsiSample,
    SessionMetadata,
    is_valid_label,
)


def make_sample(**overrides: Any) -> CsiSample:
    """SYNTHETIC: muestra mínima válida, con campos sobrescribibles."""
    base: dict[str, Any] = {
        "timestamp": 1_700_000_000.25,
        "node_id": "node-a",
        "amplitudes": (1.0, 2.5, 3.0),
    }
    base.update(overrides)
    return CsiSample(**base)


def make_metadata(**overrides: Any) -> SessionMetadata:
    """SYNTHETIC: metadatos de ejemplo, marcados como sintéticos."""
    base: dict[str, Any] = {
        "session_id": "sess-001",
        "environment": "sala-101",
        "sensor_layout": "nodo A a 2 m del nodo B, ambos a 1 m de altura",
        "nominal_rate_hz": 100.0,
        "synthetic": True,
    }
    base.update(overrides)
    return SessionMetadata(**base)


# --------------------------------------------------------------- constantes


def test_schema_version_is_1() -> None:
    assert SCHEMA_VERSION == 1


def test_labels_constant() -> None:
    assert LABELS == ("absent", "present_still", "present_moving")


@pytest.mark.parametrize("label", ["absent", "present_still", "present_moving"])
def test_is_valid_label_accepts_known_labels(label: str) -> None:
    assert is_valid_label(label) is True


@pytest.mark.parametrize(
    "label",
    ["unknown", "degraded", "unknown/degraded", "", "Absent", " absent", None, 1],
)
def test_is_valid_label_rejects_everything_else(label: object) -> None:
    assert is_valid_label(label) is False


# ---------------------------------------------------------------- CsiSample


def test_csi_sample_minimal_construction_and_defaults() -> None:
    sample = make_sample()
    assert sample.timestamp == 1_700_000_000.25
    assert sample.node_id == "node-a"
    assert sample.amplitudes == (1.0, 2.5, 3.0)
    assert sample.rssi is None
    assert sample.phase is None
    assert sample.label is None


def test_csi_sample_full_construction() -> None:
    sample = make_sample(rssi=-42.5, phase=(0.1, 0.2, 0.3), label="present_still")
    assert sample.rssi == -42.5
    assert sample.phase == (0.1, 0.2, 0.3)
    assert sample.label == "present_still"


def test_csi_sample_fields_are_exactly_the_documented_ones() -> None:
    # Guarda contra añadir campos nuevos (p. ej. identidades) sin querer.
    assert [f.name for f in fields(CsiSample)] == [
        "timestamp",
        "node_id",
        "amplitudes",
        "rssi",
        "phase",
        "label",
    ]


def test_csi_sample_is_immutable() -> None:
    sample = make_sample()
    with pytest.raises(FrozenInstanceError):
        sample.node_id = "otro"  # type: ignore[misc]
    # Atributo nuevo: según la versión de Python, frozen+slots lanza
    # FrozenInstanceError (subclase de AttributeError) o TypeError.
    with pytest.raises((AttributeError, TypeError)):
        sample.extra = 1  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        sample.amplitudes[0] = 9.9  # type: ignore[index]


def test_csi_sample_uses_slots_and_is_hashable() -> None:
    sample = make_sample()
    assert not hasattr(sample, "__dict__")
    assert hash(sample) == hash(make_sample())


def test_csi_sample_round_trip_minimal() -> None:
    sample = make_sample()
    assert CsiSample.from_dict(sample.to_dict()) == sample


def test_csi_sample_round_trip_full() -> None:
    sample = make_sample(rssi=-42.5, phase=(0.1, 0.2, 0.3), label="present_moving")
    data = sample.to_dict()
    assert data["amplitudes"] == [1.0, 2.5, 3.0]
    assert isinstance(data["amplitudes"], list)
    assert isinstance(data["phase"], list)
    restored = CsiSample.from_dict(data)
    assert restored == sample
    assert isinstance(restored.amplitudes, tuple)
    assert isinstance(restored.phase, tuple)


def test_csi_sample_round_trip_through_json() -> None:
    sample = make_sample(rssi=-40.0, phase=(0.5, 0.6, 0.7), label="absent")
    restored = CsiSample.from_dict(json.loads(json.dumps(sample.to_dict())))
    assert restored == sample


@pytest.mark.parametrize("missing", ["timestamp", "node_id", "amplitudes"])
def test_csi_sample_from_dict_missing_required_field(missing: str) -> None:
    data = make_sample().to_dict()
    del data[missing]
    with pytest.raises(ValueError, match=missing):
        CsiSample.from_dict(data)


def test_csi_sample_from_dict_optional_fields_may_be_absent() -> None:
    data = {"timestamp": 1.0, "node_id": "n", "amplitudes": [1.0]}
    sample = CsiSample.from_dict(data)
    assert sample.rssi is None and sample.phase is None and sample.label is None


def test_csi_sample_from_dict_unknown_field_is_an_error() -> None:
    data = make_sample().to_dict()
    data["person_name"] = "x"
    with pytest.raises(ValueError, match="person_name"):
        CsiSample.from_dict(data)


@pytest.mark.parametrize("bad", ["1.0,2.0", None, 3.0, {"a": 1}])
def test_csi_sample_from_dict_amplitudes_must_be_list_or_tuple(bad: object) -> None:
    data = make_sample().to_dict()
    data["amplitudes"] = bad
    with pytest.raises(ValueError, match="amplitudes"):
        CsiSample.from_dict(data)


def test_csi_sample_from_dict_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="mapeo"):
        CsiSample.from_dict([1, 2, 3])  # type: ignore[arg-type]


# ------------------------------------------------------------ SessionMetadata


def test_metadata_construction_and_defaults() -> None:
    meta = SessionMetadata(
        session_id="sess-001",
        environment="sala-101",
        sensor_layout="layout",
        nominal_rate_hz=50.0,
    )
    assert meta.schema_version == SCHEMA_VERSION
    assert meta.notes == ""
    assert meta.synthetic is False


def test_metadata_full_construction() -> None:
    meta = make_metadata(notes="prueba", schema_version=SCHEMA_VERSION)
    assert meta.session_id == "sess-001"
    assert meta.environment == "sala-101"
    assert meta.nominal_rate_hz == 100.0
    assert meta.notes == "prueba"
    assert meta.synthetic is True


def test_metadata_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        SessionMetadata(1, "s", "e", "l", 10.0)  # type: ignore[misc]


def test_metadata_fields_are_exactly_the_documented_ones() -> None:
    assert [f.name for f in fields(SessionMetadata)] == [
        "schema_version",
        "session_id",
        "environment",
        "sensor_layout",
        "nominal_rate_hz",
        "notes",
        "synthetic",
    ]


def test_metadata_is_immutable() -> None:
    meta = make_metadata()
    with pytest.raises(FrozenInstanceError):
        meta.synthetic = False  # type: ignore[misc]
    # Ver nota en test_csi_sample_is_immutable sobre frozen+slots.
    with pytest.raises((AttributeError, TypeError)):
        meta.extra = 1  # type: ignore[attr-defined]
    assert not hasattr(meta, "__dict__")


def test_metadata_round_trip() -> None:
    meta = make_metadata(notes="notas de prueba")
    assert SessionMetadata.from_dict(meta.to_dict()) == meta


def test_metadata_round_trip_through_json() -> None:
    meta = make_metadata()
    restored = SessionMetadata.from_dict(json.loads(json.dumps(meta.to_dict())))
    assert restored == meta


@pytest.mark.parametrize(
    "missing",
    ["schema_version", "session_id", "environment", "sensor_layout", "nominal_rate_hz"],
)
def test_metadata_from_dict_missing_required_field(missing: str) -> None:
    data = make_metadata().to_dict()
    del data[missing]
    with pytest.raises(ValueError, match=missing):
        SessionMetadata.from_dict(data)


def test_metadata_from_dict_optional_fields_use_defaults() -> None:
    data = make_metadata().to_dict()
    del data["notes"]
    del data["synthetic"]
    meta = SessionMetadata.from_dict(data)
    assert meta.notes == ""
    assert meta.synthetic is False


def test_metadata_from_dict_unknown_field_is_an_error() -> None:
    data = make_metadata().to_dict()
    data["operator_name"] = "x"
    with pytest.raises(ValueError, match="operator_name"):
        SessionMetadata.from_dict(data)


def test_metadata_from_dict_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="mapeo"):
        SessionMetadata.from_dict("no es un dict")  # type: ignore[arg-type]