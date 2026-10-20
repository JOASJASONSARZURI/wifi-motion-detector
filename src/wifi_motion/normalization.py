"""Normalización z-score de vectores de características.

Solo numpy y la biblioteca estándar (``json``). La media y la desviación por
característica se aprenden UNA vez con ``Normalizer.fit`` y desde entonces
``transform`` solo las aplica: nunca las recalcula con los datos nuevos.

Reglas (la opción más simple y la que no oculta datos malos):

- Todo valor no finito (NaN, inf) se rechaza con ``ValueError``, tanto al
  ajustar como al transformar. Esto incluye ``mean_rssi`` cuando es NaN por no
  haber RSSI: quien use este módulo debe excluir esa característica o tratar
  ese caso antes, porque aquí no se imputa ni se rellena nada.
- La desviación se calcula sobre la población (ddof = 0), de modo que
  ``fit_transform`` deja media 0 y desviación 1 exactas (salvo redondeo).
- Una característica constante en el entrenamiento no se puede escalar: se usa
  ``std = 1.0`` y se registra en ``Normalizer.constant_features`` para que sea
  visible. En ``transform`` esa columna queda como ``x - media``.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CONSTANT_STD_THRESHOLD",
    "NORMALIZER_VERSION",
    "Normalizer",
    "NotFittedError",
]

NORMALIZER_VERSION: int = 1

# Desviaciones por debajo de este valor (absoluto) se tratan como cero.
CONSTANT_STD_THRESHOLD: float = 1e-12

_FIELDS: tuple[str, ...] = (
    "normalizer_version",
    "feature_names",
    "mean",
    "std",
    "constant_features",
)


class NotFittedError(RuntimeError):
    """Se usó el normalizador antes de ajustarlo (``fit``) o cargarlo (``load``)."""


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _checked_names(feature_names: object) -> tuple[str, ...]:
    if isinstance(feature_names, str) or not isinstance(feature_names, (tuple, list)):
        raise ValueError(
            "feature_names debe ser una tupla (o lista) de textos, "
            f"no {type(feature_names).__name__}"
        )
    names = tuple(feature_names)
    if not names:
        raise ValueError("feature_names no puede estar vacío")
    bad = [n for n in names if not isinstance(n, str) or not n.strip()]
    if bad:
        raise ValueError(f"feature_names solo admite textos no vacíos; inválidos: {bad!r}")
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(f"feature_names tiene nombres repetidos: {duplicated}")
    return names


def _checked_matrix(X: object, what: str) -> np.ndarray:
    """Convierte a float64 y exige valores finitos; devuelve una copia propia."""
    try:
        array = np.array(X, dtype=np.float64)  # copia: no comparte memoria con X
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}: no se pudo convertir a números: {exc}") from exc
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(f"{what} tiene {bad} valor(es) no finito(s) (NaN o infinito)")
    return array


class Normalizer:
    """Normalizador z-score ``(x - media) / std`` con estadísticas fijas.

    IMPORTANTE: ``fit`` debe recibir SOLO datos de entrenamiento. Nunca le pase
    datos de validación ni del conjunto de prueba final: ajustar con ellos
    filtraría información de la evaluación hacia el modelo y daría resultados
    optimistas e inválidos. Ajuste con el entrenamiento y use ``transform``
    (que no recalcula nada) sobre validación, prueba y datos en vivo. Este
    módulo no puede comprobar de dónde vienen los datos; es responsabilidad de
    quien lo llama.

    Atributos:

    - ``feature_names``: nombres en el orden de las columnas (None si no
      ajustado).
    - ``mean`` / ``std``: arrays de solo lectura de forma ``(d,)`` (None si no
      ajustado). ``std`` ya incluye la sustitución por 1.0 de las constantes.
    - ``constant_features``: lista PÚBLICA con los nombres de las
      características cuya desviación fue 0 (o < ``CONSTANT_STD_THRESHOLD`` =
      1e-12) en el entrenamiento; para ellas se usa ``std = 1.0``. Vacía si no
      hay ninguna. Se reasigna en cada ``fit``.

    Volver a llamar a ``fit`` descarta por completo el ajuste anterior.
    """

    def __init__(self) -> None:
        self.feature_names: tuple[str, ...] | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.constant_features: list[str] = []

    @property
    def is_fitted(self) -> bool:
        """True si ya hay estadísticas (por ``fit`` o ``load``)."""
        return self.mean is not None

    # ------------------------------------------------------------------ fit

    def fit(self, X: Any, feature_names: tuple[str, ...]) -> Normalizer:
        """Aprende media y desviación (ddof = 0) por columna; devuelve ``self``.

        Solo con datos de ENTRENAMIENTO (ver el docstring de la clase).
        ``ValueError`` si ``X`` no es una matriz 2-D ``(n, d)``, si tiene menos
        de 2 filas, si contiene valores no finitos, si ``d`` no coincide con
        ``len(feature_names)``, si los nombres están vacíos, repetidos o no son
        textos, o si las estadísticas se desbordan. Si falla, el estado previo
        del normalizador no cambia.
        """
        names = _checked_names(feature_names)
        data = _checked_matrix(X, "X")
        if data.ndim != 2:
            raise ValueError(f"X debe ser una matriz 2-D (n, d), no de {data.ndim} dimensión(es)")
        n_rows, n_cols = data.shape
        if n_rows < 2:
            raise ValueError(f"se necesitan al menos 2 filas para ajustar, hay {n_rows}")
        if n_cols != len(names):
            raise ValueError(
                f"X tiene {n_cols} columna(s) pero feature_names tiene {len(names)} nombre(s)"
            )

        with np.errstate(all="ignore"):
            mean = data.mean(axis=0)
            std = data.std(axis=0)
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
            raise ValueError("las estadísticas se desbordan: los valores de X son demasiado grandes")

        constant = std < CONSTANT_STD_THRESHOLD
        std = np.where(constant, 1.0, std)
        self._set_state(names, mean, std, [n for n, c in zip(names, constant) if c])
        return self

    def _set_state(
        self,
        names: tuple[str, ...],
        mean: np.ndarray,
        std: np.ndarray,
        constant_features: list[str],
    ) -> None:
        mean = np.array(mean, dtype=np.float64)
        std = np.array(std, dtype=np.float64)
        mean.flags.writeable = False
        std.flags.writeable = False
        self.feature_names = names
        self.mean = mean
        self.std = std
        self.constant_features = list(constant_features)

    # ------------------------------------------------------------ transform

    def transform(self, X: Any) -> np.ndarray:
        """Aplica ``(X - media) / std`` con las estadísticas del ``fit``.

        Acepta una matriz ``(n, d)`` o un vector ``(d,)`` (devuelve la misma
        forma). Devuelve un array nuevo; no modifica ``X`` ni recalcula media o
        desviación con estos datos. ``ValueError`` si hay valores no finitos,
        si el número de columnas no coincide o si el resultado se desborda;
        ``NotFittedError`` si aún no hay ajuste.
        """
        if self.mean is None or self.std is None:
            raise NotFittedError("el Normalizer no está ajustado: llame a fit() o load() primero")
        data = _checked_matrix(X, "X")
        if data.ndim not in (1, 2):
            raise ValueError(f"X debe ser un vector (d,) o una matriz (n, d), no {data.ndim}-D")
        n_cols = data.shape[-1]
        if n_cols != self.mean.shape[0]:
            raise ValueError(
                f"X tiene {n_cols} columna(s) pero el Normalizer se ajustó con "
                f"{self.mean.shape[0]}"
            )
        with np.errstate(all="ignore"):
            result = (data - self.mean) / self.std
        if not np.all(np.isfinite(result)):
            raise ValueError("el resultado no es finito (valores fuera de rango numérico)")
        return result

    def fit_transform(self, X: Any, feature_names: tuple[str, ...]) -> np.ndarray:
        """``fit(X, feature_names)`` seguido de ``transform(X)``; solo para ENTRENAMIENTO."""
        return self.fit(X, feature_names).transform(X)

    # ------------------------------------------------------------ save/load

    def save(self, path: str | os.PathLike[str]) -> None:
        """Guarda el ajuste como JSON (UTF-8, estricto) de forma atómica.

        Campos: ``normalizer_version`` (1), ``feature_names``, ``mean``,
        ``std`` y ``constant_features``. Escribe en un temporal del mismo
        directorio y lo renombra sobre ``path`` solo si todo salió bien;
        si falla, el destino queda como estaba. ``NotFittedError`` si no hay
        ajuste.
        """
        if self.feature_names is None or self.mean is None or self.std is None:
            raise NotFittedError("el Normalizer no está ajustado: no hay nada que guardar")
        record = {
            "normalizer_version": NORMALIZER_VERSION,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "constant_features": list(self.constant_features),
        }
        payload = (
            json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
        target = Path(path)
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "xb") as out:
                out.write(payload)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Normalizer:
        """Carga un ajuste guardado con ``save``.

        Verifica que existan TODOS los campos, que no haya campos
        desconocidos, que ``normalizer_version`` sea exactamente 1 y que los
        datos sean coherentes (longitudes iguales, números finitos, ``std`` > 0,
        ``constant_features`` incluido en los nombres y con ``std == 1.0``).
        ``ValueError`` si algo falla; ``FileNotFoundError`` si no existe.
        """
        raw = Path(path).read_bytes()
        try:
            record = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path}: no es UTF-8 válido") from exc
        except ValueError as exc:  # JSONDecodeError y constantes NaN/Infinity
            raise ValueError(f"{path}: JSON inválido: {exc}") from exc
        return cls._from_record(record, str(path))

    @classmethod
    def _from_record(cls, record: object, where: str) -> Normalizer:
        if not isinstance(record, dict):
            raise ValueError(f"{where}: el contenido debe ser un objeto JSON")
        missing = [f for f in _FIELDS if f not in record]
        if missing:
            raise ValueError(f"{where}: faltan campos: {', '.join(missing)}")
        unknown = sorted(str(k) for k in record if k not in _FIELDS)
        if unknown:
            raise ValueError(f"{where}: campos desconocidos: {', '.join(unknown)}")

        version = record["normalizer_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError(f"{where}: normalizer_version debe ser un entero, no {version!r}")
        if version != NORMALIZER_VERSION:
            raise ValueError(
                f"{where}: normalizer_version {version} no soportada "
                f"(esta versión del código solo soporta {NORMALIZER_VERSION})"
            )

        names = _checked_names(record["feature_names"])
        mean = _checked_numbers(where, "mean", record["mean"], len(names))
        std = _checked_numbers(where, "std", record["std"], len(names))
        if np.any(std <= 0.0):
            raise ValueError(f"{where}: std debe ser > 0 en todas las características")

        constant = record["constant_features"]
        if not isinstance(constant, list) or not all(isinstance(c, str) for c in constant):
            raise ValueError(f"{where}: constant_features debe ser una lista de textos")
        unknown_names = [c for c in constant if c not in names]
        if unknown_names:
            raise ValueError(
                f"{where}: constant_features contiene nombres que no están en "
                f"feature_names: {unknown_names}"
            )
        for name in constant:
            if std[names.index(name)] != 1.0:
                raise ValueError(f"{where}: la característica constante {name!r} debe tener std 1.0")

        normalizer = cls()
        normalizer._set_state(names, mean, std, constant)
        return normalizer


def _reject_constant(name: str) -> Any:
    raise ValueError(f"constante no permitida en JSON estricto: {name}")


def _checked_numbers(where: str, field: str, value: object, length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where}: {field} debe ser una lista de {length} número(s)")
    if not all(_is_finite_number(v) for v in value):
        raise ValueError(f"{where}: {field} debe contener solo números finitos")
    return np.array(value, dtype=np.float64)