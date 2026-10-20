"""Línea base de la sala vacía, por nodo, con estadísticas robustas.

Solo numpy y la biblioteca estándar (``json``). Una ``Baseline`` resume cómo se
ven las características (``FEATURE_NAMES``) de UN nodo, en su ubicación
instalada, con la sala vacía: mediana y MAD (desviación absoluta mediana) por
característica. Un nodo = una línea base; quien use este módulo guarda un dict
``node_id -> Baseline``.

Reglas (la opción más simple y la que no oculta datos malos):

- Mediana y MAD en lugar de media y desviación: un movimiento puntual durante
  la calibración (menos de la mitad de las ventanas) no arrastra la línea base.
- Todo valor no finito se rechaza con ``CalibrationError``, con una única
  excepción: ``mean_rssi`` puede ser NaN (la ventana no traía RSSI). Esos NaN
  se ignoran al calibrar (``nanmedian``). Si TODAS las ventanas carecen de
  RSSI, la línea base de ``mean_rssi`` queda en NaN (no hay línea base de
  RSSI), y su desvío también es NaN; nunca se inventa un valor.
- Las puntuaciones de ``deviation`` miden desvío respecto a la sala vacía; NO
  son probabilidades ni umbrales universales. Qué desvío es "quieto" lo decide
  quien llama, con datos medidos.
- Las pruebas usan datos SYNTHETIC: no demuestran exactitud de campo.

Errores: ``CalibrationError`` (subclase de ``ValueError``) para datos o
líneas base inválidos; ``ValueError`` simple para argumentos de configuración
mal formados (``min_windows``, ``alpha``, ``z_max``, ``features_to_check``).
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
import warnings
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from wifi_motion.features import FEATURE_NAMES

__all__ = [
    "BASELINE_VERSION",
    "MAD_EPSILON",
    "MAD_TO_SIGMA",
    "Baseline",
    "CalibrationError",
    "deviation",
    "fit_baseline",
    "is_quiet",
    "load_baseline",
    "save_baseline",
    "update_baseline",
]

BASELINE_VERSION: int = 1

# Factor que vuelve el MAD comparable con una desviación típica si los datos
# fueran normales. Solo es una escala de referencia, no un supuesto probado.
MAD_TO_SIGMA: float = 1.4826

# Sustituto VISIBLE de un MAD exactamente 0 (característica sin variación en la
# calibración, p. ej. change_points siempre 0). Es un valor inicial arbitrario:
# hace que cualquier desvío respecto a esa característica dé una puntuación muy
# alta en lugar de dividir por cero. Ver ``Baseline.floored_features``.
MAD_EPSILON: float = 1e-6

_N_FEATURES = len(FEATURE_NAMES)
_RSSI_INDEX = FEATURE_NAMES.index("mean_rssi")

_FIELDS: tuple[str, ...] = (
    "baseline_version",
    "node_id",
    "feature_names",
    "median",
    "mad",
    "n_windows",
    "created_at",
    "environment_note",
)


class CalibrationError(ValueError):
    """Datos de calibración o línea base inválidos."""


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _readonly_vector(value: object, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float64)  # copia propia
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} no se pudo convertir a números: {exc}") from exc
    if array.shape != (_N_FEATURES,):
        raise CalibrationError(f"{name} debe tener forma ({_N_FEATURES},), no {array.shape}")
    array.flags.writeable = False
    return array


# ------------------------------------------------------------------- Baseline


@dataclass(frozen=True, slots=True, eq=False)
class Baseline:
    """Línea base inmutable de un nodo: mediana y MAD por característica.

    - ``node_id``: sensor al que pertenece (texto no vacío; no es una persona).
    - ``feature_names``: debe ser exactamente ``FEATURE_NAMES`` (mismo orden).
    - ``median``: mediana por característica, forma ``(15,)``, solo lectura.
    - ``mad``: MAD SIN escalar (mediana de ``|x - mediana|``), forma ``(15,)``,
      solo lectura. Un MAD exactamente 0 se sustituyó por ``MAD_EPSILON``.
      Todos los valores son > 0 (o NaN junto con la mediana de ``mean_rssi``
      si no hubo RSSI).
    - ``n_windows``: ventanas usadas en la calibración (filas, incluidas las
      que no traían RSSI); no cambia con ``update_baseline``.
    - ``created_at``: instante de la calibración (época Unix, s); no cambia con
      ``update_baseline``.
    - ``environment_note``: texto libre sobre el entorno; no debe contener
      nombres ni datos personales.

    Los arrays se copian al construir y quedan de solo lectura. ``eq=False``:
    la igualdad es por identidad (comparar arrays con ``==`` es ambiguo);
    compare los campos explícitamente. Los valores inválidos lanzan
    ``CalibrationError`` al construir.
    """

    node_id: str
    feature_names: tuple[str, ...]
    median: np.ndarray
    mad: np.ndarray
    n_windows: int
    created_at: float
    environment_note: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise CalibrationError(f"node_id debe ser un texto no vacío, no {self.node_id!r}")
        names = self.feature_names
        if not isinstance(names, (tuple, list)) or tuple(names) != FEATURE_NAMES:
            raise CalibrationError(
                "feature_names no coincide con FEATURE_NAMES (mismos nombres y orden): "
                f"{names!r}"
            )
        median = _readonly_vector(self.median, "median")
        mad = _readonly_vector(self.mad, "mad")
        # NaN solo se admite en mean_rssi, y a la vez en mediana y MAD.
        allowed = np.zeros(_N_FEATURES, dtype=bool)
        allowed[_RSSI_INDEX] = bool(np.isnan(median[_RSSI_INDEX]) and np.isnan(mad[_RSSI_INDEX]))
        for label, array in (("median", median), ("mad", mad)):
            bad = ~np.isfinite(array) & ~allowed
            if bad.any():
                culprits = [FEATURE_NAMES[i] for i in np.flatnonzero(bad)]
                raise CalibrationError(f"{label} tiene valores no finitos en: {culprits}")
        if np.any(mad[~np.isnan(mad)] <= 0.0):
            raise CalibrationError("mad debe ser > 0 en todas las características")
        if (
            isinstance(self.n_windows, bool)
            or not isinstance(self.n_windows, int)
            or self.n_windows < 1
        ):
            raise CalibrationError(f"n_windows debe ser un entero >= 1, no {self.n_windows!r}")
        if not _is_finite_number(self.created_at):
            raise CalibrationError(f"created_at debe ser un número finito, no {self.created_at!r}")
        if not isinstance(self.environment_note, str):
            raise CalibrationError("environment_note debe ser un texto")
        object.__setattr__(self, "feature_names", FEATURE_NAMES)
        object.__setattr__(self, "median", median)
        object.__setattr__(self, "mad", mad)
        object.__setattr__(self, "created_at", float(self.created_at))

    @property
    def floored_features(self) -> tuple[str, ...]:
        """Características cuyo MAD fue 0 en la calibración y vale ``MAD_EPSILON``.

        Para ellas ``deviation`` da puntuaciones enormes ante cualquier
        variación; revíselas antes de fiarse de esa característica.
        """
        return tuple(n for n, m in zip(FEATURE_NAMES, self.mad) if m == MAD_EPSILON)


# ------------------------------------------------------------------ validación


def _checked_baseline(baseline: object) -> Baseline:
    if not isinstance(baseline, Baseline):
        raise TypeError(f"baseline debe ser Baseline, no {type(baseline).__name__}")
    return baseline


def _checked_matrix(features: object) -> np.ndarray:
    try:
        X = np.array(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"features no se pudo convertir a números: {exc}") from exc
    if X.ndim != 2 or X.shape[1] != _N_FEATURES:
        raise CalibrationError(
            f"features debe tener forma (n, {_N_FEATURES}) en el orden de FEATURE_NAMES, "
            f"no {X.shape}"
        )
    return X


def _non_finite_names(values: np.ndarray) -> list[str]:
    """Nombres con valores no finitos; el NaN de mean_rssi se tolera, el inf no."""
    bad = ~np.isfinite(values)
    nan_ok = np.zeros(values.shape, dtype=bool)
    nan_ok[..., _RSSI_INDEX] = np.isnan(values[..., _RSSI_INDEX])
    bad &= ~nan_ok
    columns = np.flatnonzero(bad.reshape(-1, _N_FEATURES).any(axis=0))
    return [FEATURE_NAMES[i] for i in columns]


def _checked_vector(feature_vector: object) -> np.ndarray:
    try:
        x = np.array(feature_vector, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"feature_vector no se pudo convertir a números: {exc}") from exc
    if x.shape != (_N_FEATURES,):
        raise CalibrationError(
            f"feature_vector debe tener forma ({_N_FEATURES},) en el orden de "
            f"FEATURE_NAMES, no {x.shape}"
        )
    bad = _non_finite_names(x)
    if bad:
        raise CalibrationError(f"feature_vector tiene valores no finitos en: {bad}")
    return x


# ----------------------------------------------------------------------- fit


def fit_baseline(
    node_id: str,
    features: Any,
    environment_note: str,
    min_windows: int = 30,
    created_at: float | None = None,
) -> Baseline:
    """Calibra la línea base de un nodo con ventanas de la SALA VACÍA.

    ``features`` es una matriz ``(n, 15)``: un vector de ``extract_features``
    por fila, en el orden de ``FEATURE_NAMES`` (opcionalmente normalizado o no;
    la línea base queda en las mismas unidades que reciba). Debe contener solo
    ventanas con la sala realmente vacía: la mediana y el MAD toleran algún
    movimiento puntual (menos de la mitad de las ventanas), pero no una
    calibración con gente.

    ``min_windows`` (entero >= 2, defecto 30) es el mínimo de filas;
    ``created_at`` (defecto: ``time.time()``) permite fijar el instante.
    Lanza ``CalibrationError`` si hay menos filas, si la forma no es
    ``(n, 15)`` o si hay valores no finitos (salvo NaN en ``mean_rssi``).
    ``MAD == 0`` se sustituye por ``MAD_EPSILON`` y la característica queda en
    ``Baseline.floored_features``.
    """
    if isinstance(min_windows, bool) or not isinstance(min_windows, int) or min_windows < 2:
        raise ValueError(f"min_windows debe ser un entero >= 2, no {min_windows!r}")
    X = _checked_matrix(features)
    n = X.shape[0]
    if n < min_windows:
        raise CalibrationError(
            f"hay {n} ventana(s) de calibración; se necesitan al menos {min_windows}"
        )
    bad = _non_finite_names(X)
    if bad:
        raise CalibrationError(
            f"features tiene valores no finitos (solo se tolera NaN en mean_rssi): {bad}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columna toda NaN -> NaN, no aviso
        median = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - median), axis=0)
    mad = np.where(mad == 0.0, MAD_EPSILON, mad)  # NaN (sin RSSI) se conserva

    return Baseline(
        node_id=node_id,
        feature_names=FEATURE_NAMES,
        median=median,
        mad=mad,
        n_windows=n,
        created_at=time.time() if created_at is None else created_at,
        environment_note=environment_note,
    )


# ------------------------------------------------------------------ deviation


def deviation(baseline: Baseline, feature_vector: Any) -> np.ndarray:
    """Puntuaciones robustas ``(x - mediana) / (1.4826 * MAD)``, forma ``(15,)``.

    Es SOLO una medida de desvío respecto a la línea base: NO es una
    probabilidad ni una confianza, y su escala depende de la calibración. El
    signo indica la dirección. Si ``mean_rssi`` es NaN en el vector, o la línea
    base no tiene RSSI, su puntuación es NaN (nunca un valor inventado).
    ``CalibrationError`` si el vector no es ``(15,)``, tiene valores no
    finitos (salvo NaN en ``mean_rssi``) o la puntuación se desborda.
    """
    base = _checked_baseline(baseline)
    x = _checked_vector(feature_vector)
    with np.errstate(all="ignore"):
        z = (x - base.median) / (MAD_TO_SIGMA * base.mad)
    overflow = _non_finite_names(z)
    if overflow:
        raise CalibrationError(f"la puntuación se desborda (valores fuera de rango) en: {overflow}")
    return z


# -------------------------------------------------------------------- is_quiet


def is_quiet(
    baseline: Baseline,
    feature_vector: Any,
    z_max: float,
    features_to_check: tuple[str, ...],
) -> bool:
    """True si ``|deviation|`` de TODAS las características indicadas es <= ``z_max``.

    Ni ``z_max`` ni ``features_to_check`` tienen valor por defecto: dependen del
    hardware y del entorno y deben fijarse con datos medidos, no son umbrales
    universales. ``z_max`` debe ser un número finito > 0 y ``features_to_check``
    una tupla (o lista) no vacía de nombres de ``FEATURE_NAMES``; si no,
    ``ValueError``. Si alguna característica comprobada no se puede juzgar
    (su desvío es NaN: p. ej. ``mean_rssi`` sin RSSI) se lanza
    ``CalibrationError`` en vez de responder "quieto" o "no quieto" sin
    base. Que la respuesta sea True no prueba que la sala esté vacía; solo
    que esas características no se alejan de la línea base más de ``z_max``.
    """
    base = _checked_baseline(baseline)
    if not _is_finite_number(z_max) or z_max <= 0:
        raise ValueError(f"z_max debe ser un número finito > 0, no {z_max!r}")
    if isinstance(features_to_check, str) or not isinstance(features_to_check, (tuple, list)):
        raise ValueError("features_to_check debe ser una tupla de nombres de característica")
    if not features_to_check:
        raise ValueError("features_to_check no puede estar vacío")
    unknown = [n for n in features_to_check if n not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"features_to_check tiene nombres desconocidos: {unknown}")

    z = deviation(base, feature_vector)
    checked = z[[FEATURE_NAMES.index(n) for n in features_to_check]]
    if np.isnan(checked).any():
        missing = [n for n, v in zip(features_to_check, checked) if np.isnan(v)]
        raise CalibrationError(
            f"no se puede juzgar la quietud: desvío NaN (sin dato o sin línea base) en {missing}"
        )
    return bool(np.all(np.abs(checked) <= float(z_max)))


# ------------------------------------------------------------ update_baseline


def update_baseline(
    baseline: Baseline, feature_vector: Any, alpha: float, quiet: bool
) -> Baseline:
    """Línea base nueva con la mediana movida por EMA, SOLO si ``quiet`` es True.

    Con ``quiet=False`` devuelve la MISMA línea base (el mismo objeto), sin
    validar ``feature_vector`` ni tocar nada. Con ``quiet=True`` devuelve una
    nueva: ``mediana = (1 - alpha) * mediana + alpha * x``; la original no se
    modifica. ``alpha`` debe estar en (0, 1] (``ValueError``, se valida siempre)
    y ``quiet`` debe ser bool (``TypeError``: un valor "casi booleano" como
    ``1`` o ``"False"`` no se interpreta).

    Decisiones de diseño (la opción más simple):

    - Solo se actualiza la mediana. El MAD, ``n_windows`` y ``created_at`` no
      cambian: una sola muestra no da una estimación fiable del MAD, y una EMA
      de ``|x - mediana|`` lo sesgaría hacia arriba (≈ 1.85 veces con ruido
      normal), lo que reduce la sensibilidad en silencio. Para refrescar la
      escala hay que recalibrar con ``fit_baseline``.
    - ``mean_rssi`` solo se actualiza si el vector y la línea base lo tienen;
      si no, se deja como está (no se crea una línea base de RSSI a partir de
      una muestra).
    - Esta función no decide si hay quietud: ``quiet`` lo calcula quien llama
      (p. ej. con ``is_quiet``). Actualizar con ``quiet`` mal calculado deriva
      la línea base hacia la presencia; con ``alpha`` pequeño la deriva es
      lenta pero real.
    """
    base = _checked_baseline(baseline)
    if not _is_finite_number(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha debe estar en (0, 1], no {alpha!r}")
    if not isinstance(quiet, (bool, np.bool_)):
        raise TypeError(f"quiet debe ser bool, no {type(quiet).__name__}")
    if not quiet:
        return base

    x = _checked_vector(feature_vector)
    a = float(alpha)
    new_median = (1.0 - a) * base.median + a * x
    keep = ~(np.isfinite(base.median) & np.isfinite(x))  # solo puede ser mean_rssi
    new_median = np.where(keep, base.median, new_median)
    return Baseline(
        node_id=base.node_id,
        feature_names=base.feature_names,
        median=new_median,
        mad=base.mad,
        n_windows=base.n_windows,
        created_at=base.created_at,
        environment_note=base.environment_note,
    )


# ------------------------------------------------------------------- JSON I/O


def _to_json_list(array: np.ndarray) -> list[float | None]:
    """JSON estricto no admite NaN: el NaN de mean_rssi se guarda como null."""
    return [None if math.isnan(v) else v for v in array.tolist()]


def save_baseline(baseline: Baseline, path: str | os.PathLike[str]) -> None:
    """Guarda la línea base como JSON versionado (UTF-8, estricto), de forma atómica.

    Campos: ``baseline_version`` (1), ``node_id``, ``feature_names``,
    ``median``, ``mad``, ``n_windows``, ``created_at``, ``environment_note``.
    El NaN de ``mean_rssi`` (sin RSSI) se escribe como ``null``. Escribe en un
    temporal del mismo directorio y lo renombra sobre ``path`` solo si todo
    salió bien; si falla, el destino queda como estaba.
    """
    base = _checked_baseline(baseline)
    record = {
        "baseline_version": BASELINE_VERSION,
        "node_id": base.node_id,
        "feature_names": list(base.feature_names),
        "median": _to_json_list(base.median),
        "mad": _to_json_list(base.mad),
        "n_windows": base.n_windows,
        "created_at": base.created_at,
        "environment_note": base.environment_note,
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


def _reject_constant(name: str) -> Any:
    raise ValueError(f"constante no permitida en JSON estricto: {name}")


def _numbers_or_none(where: str, field: str, value: object) -> list[float]:
    if not isinstance(value, list) or len(value) != _N_FEATURES:
        raise CalibrationError(f"{where}: {field} debe ser una lista de {_N_FEATURES} valores")
    out: list[float] = []
    for v in value:
        if v is None:
            out.append(float("nan"))  # solo mean_rssi lo admite: lo comprueba Baseline
        elif _is_finite_number(v):
            out.append(float(v))
        else:
            raise CalibrationError(f"{where}: {field} debe contener números finitos o null")
    return out


def load_baseline(path: str | os.PathLike[str]) -> Baseline:
    """Carga una línea base guardada con ``save_baseline``.

    Verifica que existan TODOS los campos y que no haya desconocidos, que
    ``baseline_version`` sea exactamente 1, que ``feature_names`` coincida con
    ``FEATURE_NAMES`` actuales (una línea base de otro conjunto de
    características no se reinterpreta en silencio) y que los valores sean
    coherentes (ver ``Baseline``). ``CalibrationError`` si algo falla;
    ``FileNotFoundError`` si el archivo no existe.
    """
    where = str(path)
    raw = Path(path).read_bytes()
    try:
        record = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except UnicodeDecodeError as exc:
        raise CalibrationError(f"{where}: no es UTF-8 válido") from exc
    except ValueError as exc:  # JSONDecodeError y constantes NaN/Infinity
        raise CalibrationError(f"{where}: JSON inválido: {exc}") from exc
    if not isinstance(record, dict):
        raise CalibrationError(f"{where}: el contenido debe ser un objeto JSON")
    missing = [f for f in _FIELDS if f not in record]
    if missing:
        raise CalibrationError(f"{where}: faltan campos: {', '.join(missing)}")
    unknown = sorted(str(k) for k in record if k not in _FIELDS)
    if unknown:
        raise CalibrationError(f"{where}: campos desconocidos: {', '.join(unknown)}")

    version = record["baseline_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise CalibrationError(f"{where}: baseline_version debe ser un entero, no {version!r}")
    if version != BASELINE_VERSION:
        raise CalibrationError(
            f"{where}: baseline_version {version} no soportada "
            f"(esta versión del código solo soporta {BASELINE_VERSION})"
        )
    names = record["feature_names"]
    if not isinstance(names, list):
        raise CalibrationError(f"{where}: feature_names debe ser una lista de textos")
    try:
        return Baseline(
            node_id=record["node_id"],
            feature_names=tuple(names),
            median=_numbers_or_none(where, "median", record["median"]),  # type: ignore[arg-type]
            mad=_numbers_or_none(where, "mad", record["mad"]),  # type: ignore[arg-type]
            n_windows=record["n_windows"],
            created_at=record["created_at"],
            environment_note=record["environment_note"],
        )
    except CalibrationError as exc:
        if str(exc).startswith(where):
            raise
        raise CalibrationError(f"{where}: {exc}") from exc