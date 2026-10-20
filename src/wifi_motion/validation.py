"""Validación de muestras CSI: detecta problemas y los reporta, sin descartar nada.

Ninguna función de este módulo lanza excepciones por datos malos en flujo
normal: devuelven una lista de ``Issue`` visibles. Solo la configuración
inválida de ``StreamValidator`` (programación, no datos) lanza ``ValueError``.

Qué hacer con los problemas (descartar, marcar la ventana como degradada,
etc.) lo decide quien consume los resultados, p. ej. con ``is_usable``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from numbers import Real

from wifi_motion.schema import CsiSample, is_valid_label


class IssueCode(str, Enum):
    """Tipos de problema que puede reportar la validación."""

    EMPTY_AMPLITUDES = "empty_amplitudes"
    NON_FINITE_VALUE = "non_finite_value"
    NEGATIVE_AMPLITUDE = "negative_amplitude"
    SUBCARRIER_COUNT_MISMATCH = "subcarrier_count_mismatch"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    OUT_OF_ORDER = "out_of_order"
    GAP = "gap"
    MISSING_NODE_ID = "missing_node_id"
    INVALID_LABEL = "invalid_label"


@dataclass(frozen=True, slots=True)
class Issue:
    """Un problema detectado; ``timestamp`` y ``node_id`` son None si no son fiables."""

    code: IssueCode
    message: str
    timestamp: float | None
    node_id: str | None


# Problemas que impiden usar la muestra. GAP, NEGATIVE_AMPLITUDE e
# INVALID_LABEL NO la invalidan, pero siguen siendo visibles en la lista de
# issues (en especial GAP, que el consumidor de la ventana debe poder ver).
BLOCKING_CODES: frozenset[IssueCode] = frozenset(
    {
        IssueCode.NON_FINITE_VALUE,
        IssueCode.EMPTY_AMPLITUDES,
        IssueCode.SUBCARRIER_COUNT_MISMATCH,
        IssueCode.DUPLICATE_TIMESTAMP,
        IssueCode.OUT_OF_ORDER,
        IssueCode.MISSING_NODE_ID,
    }
)


def _is_finite_number(value: object) -> bool:
    """True solo para números reales finitos (bool, texto, NaN e inf dan False)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _timestamp_or_none(value: object) -> float | None:
    return float(value) if _is_finite_number(value) else None  # type: ignore[arg-type]


def _node_id_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _issue(sample: CsiSample, code: IssueCode, message: str) -> Issue:
    return Issue(
        code=code,
        message=message,
        timestamp=_timestamp_or_none(sample.timestamp),
        node_id=_node_id_or_none(sample.node_id),
    )


def validate_sample(sample: CsiSample) -> list[Issue]:
    """Valida una sola muestra. Lista vacía = ningún problema.

    Decisión de diseño: además de lo pedido, un ``timestamp`` no finito se
    reporta como NON_FINITE_VALUE, porque con NaN las comparaciones de orden
    fallan en silencio. ``rssi`` y ``phase`` no se validan: aún no se usan.
    Se asume que los tipos siguen el esquema; solo se tolera que los valores
    numéricos no lo sean (se reportan, no lanzan).
    """
    issues: list[Issue] = []

    if _node_id_or_none(sample.node_id) is None:
        issues.append(
            _issue(sample, IssueCode.MISSING_NODE_ID, "node_id vacío o no es texto")
        )

    if not _is_finite_number(sample.timestamp):
        issues.append(
            _issue(
                sample,
                IssueCode.NON_FINITE_VALUE,
                "timestamp no es un número finito",
            )
        )

    amplitudes = sample.amplitudes
    if len(amplitudes) == 0:
        issues.append(
            _issue(sample, IssueCode.EMPTY_AMPLITUDES, "la muestra no tiene amplitudes")
        )
    else:
        bad = [i for i, v in enumerate(amplitudes) if not _is_finite_number(v)]
        if bad:
            issues.append(
                _issue(
                    sample,
                    IssueCode.NON_FINITE_VALUE,
                    f"{len(bad)} amplitud(es) no finita(s) o no numérica(s); "
                    f"la primera en el índice {bad[0]}",
                )
            )
        negative = [
            i for i, v in enumerate(amplitudes) if _is_finite_number(v) and float(v) < 0
        ]
        if negative:
            issues.append(
                _issue(
                    sample,
                    IssueCode.NEGATIVE_AMPLITUDE,
                    f"{len(negative)} amplitud(es) negativa(s); "
                    f"la primera en el índice {negative[0]}",
                )
            )

    if sample.label is not None and not is_valid_label(sample.label):
        issues.append(
            _issue(sample, IssueCode.INVALID_LABEL, f"etiqueta inválida: {sample.label!r}")
        )

    return issues


@dataclass(slots=True)
class _NodeState:
    """Estado mutable de un nodo dentro de StreamValidator."""

    last_timestamp: float | None = None
    subcarriers: int | None = None


class StreamValidator:
    """Valida un flujo de muestras con estado POR ``node_id``. Solo informa.

    Detecta duplicados, muestras atrasadas, huecos (diferencia estrictamente
    mayor que ``max_gap_s``) y cambios en el número de subportadoras.
    ``check`` también incluye los problemas de ``validate_sample``, de modo que
    ``is_usable(validator.check(s))`` es el veredicto completo.

    Decisiones de diseño (la opción más simple):
    - El último timestamp de un nodo es el MÁXIMO visto: una muestra atrasada
      o duplicada no lo modifica.
    - La referencia de subportadoras es ``expected_subcarriers`` si se da; si
      no, el conteo de la primera muestra no vacía del nodo, que queda fijo.
      Un cambio persistente se sigue reportando en cada muestra (no se acepta
      en silencio como "nueva normalidad"); recomendado fijar
      ``expected_subcarriers`` para que una primera muestra defectuosa no
      defina la referencia. Para reconfigurar, crea un validador nuevo.
    - Sin ``node_id`` válido no hay estado que consultar: solo se reportan los
      problemas de la muestra suelta. Un timestamp no finito tampoco toca el
      estado temporal.
    """

    def __init__(self, max_gap_s: float, expected_subcarriers: int | None = None) -> None:
        if not _is_finite_number(max_gap_s) or max_gap_s <= 0:
            raise ValueError(f"max_gap_s debe ser un número finito > 0, no {max_gap_s!r}")
        if expected_subcarriers is not None and (
            isinstance(expected_subcarriers, bool)
            or not isinstance(expected_subcarriers, int)
            or expected_subcarriers <= 0
        ):
            raise ValueError(
                "expected_subcarriers debe ser None o un entero > 0, "
                f"no {expected_subcarriers!r}"
            )
        self._max_gap_s = float(max_gap_s)
        self._expected = expected_subcarriers
        self._nodes: dict[str, _NodeState] = {}

    def check(self, sample: CsiSample) -> list[Issue]:
        """Revisa la muestra, actualiza el estado de su nodo y devuelve los issues."""
        issues = validate_sample(sample)
        node_id = _node_id_or_none(sample.node_id)
        if node_id is None:
            return issues
        state = self._nodes.setdefault(node_id, _NodeState())
        self._check_subcarriers(sample, state, issues)
        self._check_timestamp(sample, state, issues)
        return issues

    def _check_subcarriers(
        self, sample: CsiSample, state: _NodeState, issues: list[Issue]
    ) -> None:
        count = len(sample.amplitudes)
        if count == 0:  # ya reportado como EMPTY_AMPLITUDES; no fija referencia
            return
        reference = self._expected if self._expected is not None else state.subcarriers
        if reference is None:
            state.subcarriers = count
            return
        if count != reference:
            issues.append(
                _issue(
                    sample,
                    IssueCode.SUBCARRIER_COUNT_MISMATCH,
                    f"{count} subportadoras; se esperaban {reference}",
                )
            )

    def _check_timestamp(
        self, sample: CsiSample, state: _NodeState, issues: list[Issue]
    ) -> None:
        if not _is_finite_number(sample.timestamp):  # ya reportado
            return
        ts = float(sample.timestamp)
        last = state.last_timestamp
        if last is None:
            state.last_timestamp = ts
        elif ts == last:
            issues.append(
                _issue(sample, IssueCode.DUPLICATE_TIMESTAMP, f"timestamp repetido: {ts}")
            )
        elif ts < last:
            issues.append(
                _issue(
                    sample,
                    IssueCode.OUT_OF_ORDER,
                    f"timestamp {ts} es anterior al último visto ({last})",
                )
            )
        else:
            if ts - last > self._max_gap_s:
                issues.append(
                    _issue(
                        sample,
                        IssueCode.GAP,
                        f"hueco de {ts - last} s (máximo permitido: {self._max_gap_s} s)",
                    )
                )
            state.last_timestamp = ts


def is_usable(issues: Iterable[Issue]) -> bool:
    """False si algún issue impide usar la muestra (ver ``BLOCKING_CODES``)."""
    return not any(issue.code in BLOCKING_CODES for issue in issues)