"""Grabaciones JSONL versionadas: escritura atómica y lectura perezosa.

Formato (una línea por registro, UTF-8, cada línea termina en "\\n")::

    {"type":"header","schema_version":1,"session_id":"...", ...}
    {"type":"sample","timestamp":1700000000.25,"node_id":"n1", ...}

- La primera línea es el header (``SessionMetadata.to_dict()``); las demás son
  muestras (``CsiSample.to_dict()``).
- No hay comentarios ni líneas en blanco: cualquier línea no reconocida es un
  ``RecordingFormatError`` con su número de línea.
- JSON estricto: ``NaN``/``Infinity`` no existen en el estándar, así que no se
  pueden escribir ni se aceptan al leer. (Decisión de diseño: es preferible un
  error visible a un archivo que otras herramientas no puedan abrir.)
- La lectura tolera finales de línea CRLF (p. ej. archivos tocados en Windows).

Este módulo solo comprueba estructura y versión. El contenido de la señal
(vacías, negativas, etiquetas, orden temporal...) lo revisa ``validation.py``.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from wifi_motion.schema import SCHEMA_VERSION, CsiSample, SessionMetadata

__all__ = [
    "RecordingFormatError",
    "SampleIterator",
    "iter_recording",
    "write_recording",
]


class RecordingFormatError(ValueError):
    """El archivo no cumple el formato; indica la línea (desde 1) y el motivo."""

    def __init__(self, line_number: int, reason: str) -> None:
        super().__init__(line_number, reason)
        self.line_number = line_number
        self.reason = reason

    def __str__(self) -> str:
        return f"línea {self.line_number}: {self.reason}"


# ------------------------------------------------------------------ lectura


def _reject_constant(name: str) -> Any:
    raise ValueError(f"constante no permitida en JSON estricto: {name}")


def _decode_record(raw: bytes, line_number: int) -> dict[str, Any]:
    """Decodifica una línea a un objeto JSON (dict) o lanza RecordingFormatError."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecordingFormatError(
            line_number, f"no es UTF-8 válido (byte {exc.start} de la línea)"
        ) from exc
    try:
        record = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise RecordingFormatError(
            line_number, f"JSON inválido: {exc.msg} (columna {exc.colno})"
        ) from exc
    except ValueError as exc:  # constante NaN/Infinity rechazada
        raise RecordingFormatError(line_number, f"JSON inválido: {exc}") from exc
    if not isinstance(record, dict):
        raise RecordingFormatError(
            line_number,
            f"el registro debe ser un objeto JSON, no {type(record).__name__}",
        )
    return record


def _split_type(record: dict[str, Any], line_number: int) -> tuple[Any, dict[str, Any]]:
    """Separa el campo 'type' del resto de los campos."""
    if "type" not in record:
        raise RecordingFormatError(line_number, "falta el campo 'type'")
    fields = {k: v for k, v in record.items() if k != "type"}
    return record["type"], fields


def _unknown_type(line_number: int, rtype: Any) -> RecordingFormatError:
    return RecordingFormatError(line_number, f"tipo de registro desconocido: {rtype!r}")


def _parse_header(raw: bytes) -> SessionMetadata:
    rtype, fields = _split_type(_decode_record(raw, 1), 1)
    if rtype == "sample":
        raise RecordingFormatError(
            1, "falta el header: la primera línea es un registro 'sample'"
        )
    if rtype != "header":
        raise _unknown_type(1, rtype)
    if "schema_version" in fields:
        version = fields["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise RecordingFormatError(
                1, f"schema_version debe ser un entero, no {version!r}"
            )
        if version != SCHEMA_VERSION:
            raise RecordingFormatError(
                1,
                f"schema_version {version} no soportada "
                f"(esta versión del código solo soporta {SCHEMA_VERSION})",
            )
    try:
        return SessionMetadata.from_dict(fields)
    except ValueError as exc:
        raise RecordingFormatError(1, f"header inválido: {exc}") from exc


def _parse_sample(raw: bytes, line_number: int) -> CsiSample:
    rtype, fields = _split_type(_decode_record(raw, line_number), line_number)
    if rtype == "header":
        raise RecordingFormatError(
            line_number, "header repetido: solo puede estar en la primera línea"
        )
    if rtype != "sample":
        raise _unknown_type(line_number, rtype)
    try:
        return CsiSample.from_dict(fields)
    except ValueError as exc:
        raise RecordingFormatError(line_number, f"muestra inválida: {exc}") from exc


class SampleIterator(Iterator[CsiSample]):
    """Iterador perezoso de muestras: lee y analiza una línea por cada ``next``.

    Mantiene el archivo abierto hasta agotarse, fallar o cerrarse (``close()``,
    ``with`` o recolección). Tras un error de formato queda cerrado y agotado.
    """

    def __init__(self, file: BinaryIO, first_line_number: int) -> None:
        self._file: BinaryIO | None = file
        self._line_number = first_line_number

    @property
    def closed(self) -> bool:
        """True cuando el archivo ya se cerró."""
        return self._file is None

    def close(self) -> None:
        """Cierra el archivo (idempotente); ``next`` pasa a agotarse."""
        file, self._file = self._file, None
        if file is not None:
            file.close()

    def __iter__(self) -> SampleIterator:
        return self

    def __next__(self) -> CsiSample:
        file = self._file
        if file is None:
            raise StopIteration
        try:
            raw = file.readline()
            if not raw:
                raise StopIteration
            line_number = self._line_number
            self._line_number += 1
            return _parse_sample(raw, line_number)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> SampleIterator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def iter_recording(
    path: str | os.PathLike[str],
) -> tuple[SessionMetadata, SampleIterator]:
    """Abre una grabación: lee el header ya y entrega las muestras de forma perezosa.

    Los errores del header se lanzan aquí; los de las muestras, al llegar a su
    línea. Consume el iterador por completo o ciérralo (``close()`` / ``with``)
    para liberar el archivo. Un archivo inexistente lanza FileNotFoundError.
    """
    file = open(path, "rb")
    try:
        raw = file.readline()
        if not raw:
            raise RecordingFormatError(1, "falta el header: el archivo está vacío")
        metadata = _parse_header(raw)
    except BaseException:
        file.close()
        raise
    return metadata, SampleIterator(file, first_line_number=2)


# ---------------------------------------------------------------- escritura


def _encode(record: dict[str, Any]) -> bytes:
    """Una línea JSON estricta, UTF-8, terminada en "\\n"."""
    text = json.dumps(
        record, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return (text + "\n").encode("utf-8")


def write_recording(
    path: str | os.PathLike[str],
    metadata: SessionMetadata,
    samples: Iterable[CsiSample],
) -> int:
    """Escribe la grabación de forma atómica y devuelve cuántas muestras escribió.

    Escribe en un archivo temporal del mismo directorio y, solo si todo salió
    bien, lo renombra sobre ``path`` (``os.replace``). Si algo falla, el destino
    queda como estaba (o sin crear) y el temporal se elimina. El header conserva
    ``metadata.synthetic``. Lanza TypeError si recibe algo que no sea
    ``SessionMetadata``/``CsiSample`` y ValueError si un valor no cabe en JSON
    estricto (NaN, infinito).
    """
    if not isinstance(metadata, SessionMetadata):
        raise TypeError(f"metadata debe ser SessionMetadata, no {type(metadata).__name__}")
    target = Path(path)
    # Decisión de diseño: nombre temporal propio y apertura exclusiva ("xb"), en
    # lugar de mkstemp, para que el archivo final conserve los permisos normales.
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    try:
        with open(tmp, "xb") as out:
            try:
                out.write(_encode({"type": "header", **metadata.to_dict()}))
            except ValueError as exc:
                raise ValueError(f"header: {exc}") from exc
            for sample in samples:
                if not isinstance(sample, CsiSample):
                    raise TypeError(
                        f"muestra {count + 1}: se esperaba CsiSample, "
                        f"no {type(sample).__name__}"
                    )
                try:
                    out.write(_encode({"type": "sample", **sample.to_dict()}))
                except ValueError as exc:
                    raise ValueError(f"muestra {count + 1}: {exc}") from exc
                count += 1
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return count