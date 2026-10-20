"""Extracción de características de una ``Window``: vector fijo de 15 valores.

Solo numpy (``numpy.fft``), sin scipy. Cada característica se define en el
docstring de ``extract_features``; el orden del vector es ``FEATURE_NAMES``.

Reglas (ver también ``extract_features``):

- Solo se aceptan ventanas con ``quality == WindowQuality.OK``. Con
  ``HAS_GAP`` o ``INSUFFICIENT_SAMPLES``, con valores no finitos, con una
  banda por encima de Nyquist o con una banda sin ningún bin de FFT se lanza
  ``FeatureError``: nunca se devuelve un número falso en silencio.
- No se interpola ni se rellena nada. El espectro supone muestreo uniforme a
  ``window.estimated_rate_hz`` (1 / mediana de las diferencias de timestamps);
  el jitter de las marcas de tiempo introduce un error que no se corrige.
- Ninguna característica es NaN salvo ``mean_rssi`` cuando la ventana no trae
  RSSI (documentado, no es un error).

Estos datos NO demuestran exactitud de campo: las pruebas usan señales
SYNTHETIC. Las bandas y el umbral por defecto son valores iniciales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from wifi_motion.windowing import Window, WindowQuality

__all__ = [
    "FEATURE_NAMES",
    "FeatureConfig",
    "FeatureError",
    "extract_features",
    "extract_features_dict",
]

FEATURE_NAMES: tuple[str, ...] = (
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


class FeatureError(ValueError):
    """La ventana no permite calcular características fiables."""


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _checked_band(name: str, band: object) -> tuple[float, float]:
    if (
        not isinstance(band, (tuple, list))
        or len(band) != 2
        or not all(_is_finite_number(v) for v in band)
    ):
        raise ValueError(f"{name} debe ser un par (bajo, alto) de números finitos, no {band!r}")
    low, high = float(band[0]), float(band[1])
    if not 0.0 <= low < high:
        raise ValueError(f"{name} debe cumplir 0 <= bajo < alto, no {band!r}")
    return low, high


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Parámetros de ``extract_features``.

    VALORES INICIALES A AJUSTAR CON DATOS MEDIDOS, NO UMBRALES UNIVERSALES:
    ninguno de los defaults sale de mediciones; dependen del hardware, del
    entorno y de la tasa de muestreo.

    - ``motion_band_hz``: banda de "movimiento" en Hz, ``(bajo, alto)``.
      Inicial: ``(1.0, 5.0)``.
    - ``breathing_band_hz``: banda de "respiración" en Hz. Inicial:
      ``(0.1, 0.5)``.
    - ``change_k``: factor del umbral de cambios bruscos (> 0). Inicial: 5.0.
    - ``entropy_bins``: número de bins del histograma de ``amp_entropy``
      (entero >= 2). Inicial: 16.

    Las bandas son intervalos ``[bajo, alto)`` en Hz, salvo que ``alto`` sea
    exactamente la frecuencia de Nyquist de la ventana: entonces incluyen
    también ese extremo (no hay nada por encima). La resolución espectral es
    ``rate / n`` Hz; una banda más estrecha que eso, o una ventana demasiado
    corta para ella, puede no contener ningún bin y da ``FeatureError``.
    Los valores inválidos lanzan ``ValueError`` al construir la configuración.
    """

    motion_band_hz: tuple[float, float] = (1.0, 5.0)
    breathing_band_hz: tuple[float, float] = (0.1, 0.5)
    change_k: float = 5.0
    entropy_bins: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "motion_band_hz", _checked_band("motion_band_hz", self.motion_band_hz)
        )
        object.__setattr__(
            self,
            "breathing_band_hz",
            _checked_band("breathing_band_hz", self.breathing_band_hz),
        )
        if not _is_finite_number(self.change_k) or self.change_k <= 0:
            raise ValueError(f"change_k debe ser un número finito > 0, no {self.change_k!r}")
        if (
            isinstance(self.entropy_bins, bool)
            or not isinstance(self.entropy_bins, int)
            or self.entropy_bins < 2
        ):
            raise ValueError(f"entropy_bins debe ser un entero >= 2, no {self.entropy_bins!r}")


# ---------------------------------------------------------------- validación


def _checked_arrays(window: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Devuelve (timestamps, amplitudes, rssi) como float64, o lanza FeatureError."""
    try:
        timestamps = np.asarray(window.timestamps, dtype=np.float64)
        amplitudes = np.asarray(window.amplitudes, dtype=np.float64)
        rssi = None if window.rssi is None else np.asarray(window.rssi, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FeatureError(f"la ventana contiene valores no numéricos: {exc}") from exc
    if amplitudes.ndim != 2 or amplitudes.shape[1] < 1:
        raise FeatureError(
            f"amplitudes debe tener forma (n, subportadoras >= 1), no {amplitudes.shape}"
        )
    n = amplitudes.shape[0]
    if n < 2:
        raise FeatureError(f"la ventana tiene {n} muestra(s); se necesitan al menos 2")
    if timestamps.shape != (n,):
        raise FeatureError(
            f"timestamps debe tener forma ({n},), no {timestamps.shape}"
        )
    if rssi is not None and rssi.shape != (n,):
        raise FeatureError(f"rssi debe tener forma ({n},), no {rssi.shape}")
    if not np.all(np.isfinite(timestamps)):
        raise FeatureError("la ventana tiene timestamps no finitos")
    if not np.all(np.isfinite(amplitudes)):
        raise FeatureError("la ventana tiene amplitudes no finitas")
    if rssi is not None and not np.all(np.isfinite(rssi)):
        raise FeatureError("la ventana tiene valores de RSSI no finitos")
    return timestamps, amplitudes, rssi


def _checked_rate(window: Window) -> float:
    rate = window.estimated_rate_hz
    if not _is_finite_number(rate) or rate <= 0:
        raise FeatureError(
            f"estimated_rate_hz debe ser un número finito > 0, no {rate!r}; "
            "no se puede calcular el espectro"
        )
    return float(rate)


# ------------------------------------------------------------------ espectro


def _power_spectrum(centered: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Frecuencias (Hz) y potencia unilateral por bin de la señal sin media.

    Normalización de Parseval: ``power.sum()`` (DC incluido, que es ~0 porque
    la señal ya no tiene media) es igual a ``mean(centered ** 2)``, es decir,
    la varianza temporal de la señal. Cada bin intermedio suma su espejo
    negativo (factor 2); DC y Nyquist (n par) no tienen espejo.
    """
    n = centered.size
    power = np.abs(np.fft.rfft(centered)) ** 2 / n**2
    if n % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    freqs = np.fft.rfftfreq(n, d=1.0 / rate_hz)
    return freqs, power


def _check_band_vs_nyquist(name: str, band: tuple[float, float], nyquist: float) -> None:
    if band[1] > nyquist * (1.0 + 1e-12):
        raise FeatureError(
            f"{name} {band} supera la frecuencia de Nyquist ({nyquist} Hz = "
            "estimated_rate_hz / 2)"
        )


def _band_power(
    name: str,
    band: tuple[float, float],
    freqs: np.ndarray,
    power: np.ndarray,
    nyquist: float,
    resolution_hz: float,
) -> float:
    low, high = band
    tol = 1e-9 * nyquist  # absorbe el redondeo de k * rate / n en los bordes
    mask = freqs >= low - tol
    if high < nyquist * (1.0 - 1e-12):  # si alto == Nyquist, el extremo se incluye
        mask &= freqs < high - tol
    mask[0] = False  # nunca se cuenta la componente DC
    if not mask.any():
        raise FeatureError(
            f"{name} {band} no contiene ningún bin de la FFT (resolución "
            f"{resolution_hz:.6g} Hz); ventana demasiado corta o banda demasiado estrecha"
        )
    return float(power[mask].sum())


# --------------------------------------------------------------- estadísticas


def _shape_stats(values: np.ndarray) -> tuple[float, float]:
    """(asimetría, curtosis en exceso) por momentos; (0.0, 0.0) si es constante."""
    if np.ptp(values) == 0.0:
        return 0.0, 0.0
    centered = values - values.mean()
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    return m3 / m2**1.5, m4 / m2**2 - 3.0


def _entropy_bits(values: np.ndarray, bins: int) -> float:
    if np.ptp(values) == 0.0:
        return 0.0
    counts, _ = np.histogram(values, bins=bins)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p))) + 0.0  # + 0.0 evita un -0.0


def _temporal_variance(amplitudes: np.ndarray) -> float:
    per_subcarrier = np.var(amplitudes, axis=0)
    per_subcarrier[np.ptp(amplitudes, axis=0) == 0.0] = 0.0  # sin residuo de redondeo
    return float(per_subcarrier.mean())


# ------------------------------------------------------------------ extracción


def _compute(window: Window, config: FeatureConfig) -> dict[str, float]:
    if not isinstance(window, Window):
        raise TypeError(f"window debe ser Window, no {type(window).__name__}")
    if not isinstance(config, FeatureConfig):
        raise TypeError(f"config debe ser FeatureConfig, no {type(config).__name__}")
    if window.quality is not WindowQuality.OK:
        raise FeatureError(
            f"ventana no utilizable: quality={window.quality.value!r} "
            "(solo se aceptan ventanas 'ok')"
        )
    _, amplitudes, rssi = _checked_arrays(window)
    rate_hz = _checked_rate(window)
    nyquist = rate_hz / 2.0
    _check_band_vs_nyquist("motion_band_hz", config.motion_band_hz, nyquist)
    _check_band_vs_nyquist("breathing_band_hz", config.breathing_band_hz, nyquist)

    n = amplitudes.shape[0]
    values = amplitudes.ravel()
    with np.errstate(all="ignore"):  # desbordes -> inf/nan -> se detectan al final
        series = amplitudes.mean(axis=1)  # s(t)
        if np.ptp(series) == 0.0:
            centered = np.zeros_like(series)  # sin residuo de redondeo en la media
        else:
            centered = series - series.mean()
        freqs, power = _power_spectrum(centered, rate_hz)
        resolution = rate_hz / n
        spectral_power = float(power[1:].sum())
        motion = _band_power(
            "motion_band_hz", config.motion_band_hz, freqs, power, nyquist, resolution
        )
        breathing = _band_power(
            "breathing_band_hz", config.breathing_band_hz, freqs, power, nyquist, resolution
        )
        # Sin potencia (señal constante) no hay frecuencia dominante: 0.0.
        dominant = float(freqs[1 + int(np.argmax(power[1:]))]) if spectral_power > 0.0 else 0.0

        diffs = np.abs(np.diff(series))
        change_points = int(np.count_nonzero(diffs > config.change_k * np.median(diffs)))

        amp_range = float(np.ptp(values))
        skew, kurtosis = _shape_stats(values)
        q25, q75 = np.percentile(values, [25.0, 75.0])
        result: dict[str, float] = {
            "variance": _temporal_variance(amplitudes),
            "motion_band_power": motion,
            "breathing_band_power": breathing,
            "spectral_power": spectral_power,
            "dominant_freq_hz": dominant,
            "change_points": float(change_points),
            "mean_rssi": float("nan") if rssi is None else float(rssi.mean()),
            "amp_mean": float(values.mean()),
            "amp_std": 0.0 if amp_range == 0.0 else float(values.std()),
            "amp_skew": skew,
            "amp_kurtosis": kurtosis,
            "amp_iqr": float(q75 - q25),
            "amp_entropy": _entropy_bits(values, config.entropy_bins),
            "amp_max": float(values.max()),
            "amp_range": amp_range,
        }

    for name, value in result.items():
        if name == "mean_rssi" and rssi is None:
            continue
        if not math.isfinite(value):
            raise FeatureError(
                f"la característica {name!r} no es finita (valores fuera de rango "
                "numérico); no se devuelve un número falso"
            )
    return {name: result[name] for name in FEATURE_NAMES}


def extract_features_dict(
    window: Window, config: FeatureConfig = FeatureConfig()
) -> dict[str, float]:
    """Como ``extract_features`` pero como ``dict[str, float]`` (orden de FEATURE_NAMES)."""
    return _compute(window, config)


def extract_features(window: Window, config: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Vector float64 de forma ``(15,)`` con las características, en el orden de FEATURE_NAMES.

    Lanza ``FeatureError`` si ``window.quality`` no es OK, si hay valores no
    finitos, si ``estimated_rate_hz`` no es > 0, si la ventana tiene menos de 2
    muestras, si una banda supera Nyquist (``estimated_rate_hz / 2``) o si una
    banda no contiene ningún bin de FFT.

    Definiciones (todas las estadísticas son poblacionales, ddof = 0):

    Serie ``s(t)``: media de las amplitudes sobre las subportadoras en cada
    instante (forma ``(n,)``). La FFT supone muestreo uniforme a
    ``estimated_rate_hz``, sin ventana de apodización ni detrending (solo se
    resta la media de ``s``), así que hay fuga espectral si la señal no cabe en
    un número entero de ciclos.

    Potencias: unilaterales con normalización de Parseval, de modo que la suma
    de todas las bandas no-DC es la varianza temporal de ``s(t)`` (unidades de
    amplitud al cuadrado).

    1. ``variance``: varianza temporal de cada subportadora, promediada sobre
       subportadoras. No es la varianza de ``s(t)``: es >= que esta.
    2. ``motion_band_power``: potencia de ``s(t)`` en ``config.motion_band_hz``.
    3. ``breathing_band_power``: potencia de ``s(t)`` en
       ``config.breathing_band_hz``.
    4. ``spectral_power``: potencia total de ``s(t)`` sin la componente DC
       (= varianza de ``s(t)``).
    5. ``dominant_freq_hz``: frecuencia (Hz) del bin de mayor potencia de
       ``s(t)`` sin DC; resolución ``rate / n``. Vale 0.0 si no hay potencia
       (señal constante).
    6. ``change_points``: número de instantes con
       ``|s(t) - s(t-1)| > change_k * mediana(|diferencias|)`` (estricto).
       Si la mediana es 0 (señal casi plana), cuenta toda diferencia no nula.
    7. ``mean_rssi``: media del RSSI de la ventana, o NaN si la ventana no trae
       RSSI. Es la única característica que puede ser NaN; no es un error.
    8. ``amp_mean``: media de TODAS las amplitudes de la ventana.
    9. ``amp_std``: desviación típica de todas las amplitudes.
    10. ``amp_skew``: asimetría ``m3 / m2**1.5`` (momentos centrales, a mano).
        0.0 si todas las amplitudes son iguales.
    11. ``amp_kurtosis``: curtosis EN EXCESO ``m4 / m2**2 - 3`` (normal = 0).
        0.0 si todas las amplitudes son iguales.
    12. ``amp_iqr``: rango intercuartílico (percentil 75 - 25, interpolación
        lineal).
    13. ``amp_entropy``: entropía de Shannon, en bits, del histograma de todas
        las amplitudes con ``config.entropy_bins`` bins entre su mínimo y su
        máximo (rango propio de cada ventana: mide la forma, no la escala).
        Máximo ``log2(entropy_bins)``; 0.0 si son todas iguales.
    14. ``amp_max``: máximo de todas las amplitudes.
    15. ``amp_range``: máximo - mínimo de todas las amplitudes.
    """
    values = _compute(window, config)
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)