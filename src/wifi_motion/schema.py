"""Estructuras inmutables de datos: muestras CSI y metadatos de sesión.

Este módulo solo define estructura, constantes y helpers triviales
(``to_dict`` / ``from_dict``). La validación real (finitud, longitudes, orden
temporal, versión de esquema, coherencia de ``synthetic``) vive en
``validation.py``.

Privacidad: ningún campo debe contener identidades de personas. ``node_id``
identifica un sensor, no a un individuo; ``environment``, ``sensor_layout`` y
``notes`` son texto libre y NO deben incluir nombres ni datos personales.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

SCHEMA_VERSION: int = 1

# Etiquetas de verdad-terreno (solo en grabaciones). "unknown/degraded" NO está
# aquí a propósito: es una salida del detector cuando faltan datos, no una
# actividad observable que se pueda etiquetar.
LABELS: tuple[str, ...] = ("absent", "present_still", "present_moving")


def is_valid_label(label: object) -> bool:
    """Devuelve True solo si ``label`` es exactamente una de ``LABELS``."""
    return isinstance(label, str) and label in LABELS


def _to_plain(value: Any) -> Any:
    """Convierte tuplas en listas para que el dict sea compatible con JSON."""
    return list(value) if isinstance(value, tuple) else value


def _to_tuple(value: object, name: str) -> tuple[Any, ...]:
    """Convierte lista/tupla en tupla; cualquier otro tipo es un error explícito."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise ValueError(
        f"'{name}' debe ser una lista o tupla, no {type(value).__name__}"
    )


def _check_keys(
    data: object,
    cls_name: str,
    required: tuple[str, ...],
    optional: tuple[str, ...],
) -> Mapping[str, Any]:
    """Exige un mapeo con todos los campos requeridos y sin campos desconocidos.

    Decisión de diseño (la opción más simple y explícita): los campos
    desconocidos también lanzan ValueError en lugar de descartarse en silencio.
    """
    if not isinstance(data, Mapping):
        raise ValueError(
            f"{cls_name}.from_dict espera un mapeo, no {type(data).__name__}"
        )
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"{cls_name}.from_dict: faltan campos requeridos: {', '.join(missing)}"
        )
    unknown = sorted(str(k) for k in data if k not in required and k not in optional)
    if unknown:
        raise ValueError(
            f"{cls_name}.from_dict: campos desconocidos: {', '.join(unknown)}"
        )
    return data


@dataclass(frozen=True, slots=True)
class CsiSample:
    """Una muestra CSI de un nodo: una amplitud por subportadora.

    ``timestamp`` son segundos de época Unix (con fracciones). ``phase`` es
    opcional y todavía no se usa. ``label`` solo aparece en grabaciones.
    """

    timestamp: float
    node_id: str
    amplitudes: tuple[float, ...]
    rssi: float | None = None
    phase: tuple[float, ...] | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Dict compatible con JSON (las tuplas pasan a listas)."""
        return {f.name: _to_plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CsiSample:
        """Reconstruye la muestra; ValueError si falta un campo requerido."""
        d = _check_keys(
            data,
            "CsiSample",
            required=("timestamp", "node_id", "amplitudes"),
            optional=("rssi", "phase", "label"),
        )
        phase = d.get("phase")
        return cls(
            timestamp=d["timestamp"],
            node_id=d["node_id"],
            amplitudes=_to_tuple(d["amplitudes"], "amplitudes"),
            rssi=d.get("rssi"),
            phase=None if phase is None else _to_tuple(phase, "phase"),
            label=d.get("label"),
        )


# Decisión de diseño: kw_only=True para poder dar a schema_version un valor por
# defecto (SCHEMA_VERSION) sin romper el orden de campos pedido.
@dataclass(frozen=True, slots=True, kw_only=True)
class SessionMetadata:
    """Metadatos NO sensibles del montaje de una sesión.

    ``synthetic`` debe ser True en todo dato generado por código; esa regla se
    comprueba en ``validation.py``, no aquí.
    """

    schema_version: int = SCHEMA_VERSION
    session_id: str
    environment: str
    sensor_layout: str
    nominal_rate_hz: float
    notes: str = ""
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Dict compatible con JSON."""
        return {f.name: _to_plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionMetadata:
        """Reconstruye los metadatos; ValueError si falta un campo requerido.

        ``schema_version`` es requerido al deserializar (un archivo debe
        declarar su versión); su valor se conserva tal cual y su
        compatibilidad la comprueba ``validation.py``.
        """
        d = _check_keys(
            data,
            "SessionMetadata",
            required=(
                "schema_version",
                "session_id",
                "environment",
                "sensor_layout",
                "nominal_rate_hz",
            ),
            optional=("notes", "synthetic"),
        )
        return cls(
            schema_version=d["schema_version"],
            session_id=d["session_id"],
            environment=d["environment"],
            sensor_layout=d["sensor_layout"],
            nominal_rate_hz=d["nominal_rate_hz"],
            notes=d.get("notes", ""),
            synthetic=d.get("synthetic", False),
        )