"""Pruebas de wifi_motion.validation.

SYNTHETIC: todos los datos de este archivo están escritos a mano; no son
mediciones reales. Los timestamps usan fracciones binarias exactas (0.25, 0.5,
0.75...) para que las comparaciones de huecos no dependan del redondeo.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from wifi_motion.schema import CsiSample
from wifi_motion.validation import (
    BLOCKING_CODES,
    Issue,
    IssueCode,
    StreamValidator,
    is_usable,
    validate_sample,
)

NAN = float("nan")
INF = float("inf")


def make_sample(**overrides: Any) -> CsiSample:
    """SYNTHETIC: muestra válida de 3 subportadoras."""
    base: dict[str, Any] = {
        "timestamp": 100.0,
        "node_id": "n1",
        "amplitudes": (1.0, 2.0, 3.0),
    }
    base.update(overrides)
    return CsiSample(**base)


def codes(issues: list[Issue]) -> list[IssueCode]:
    return [i.code for i in issues]


def make_issue(code: IssueCode) -> Issue:
    return Issue(code=code, message="x", timestamp=1.0, node_id="n1")


# ------------------------------------------------------------ IssueCode/Issue


def test_issue_code_has_all_required_members() -> None:
    required = {
        "EMPTY_AMPLITUDES",
        "NON_FINITE_VALUE",
        "NEGATIVE_AMPLITUDE",
        "SUBCARRIER_COUNT_MISMATCH",
        "DUPLICATE_TIMESTAMP",
        "OUT_OF_ORDER",
        "GAP",
        "MISSING_NODE_ID",
        "INVALID_LABEL",
    }
    assert required <= {c.name for c in IssueCode}


def test_issue_is_immutable_and_keeps_fields() -> None:
    issue = Issue(IssueCode.GAP, "hueco", 5.5, "n1")
    assert (issue.code, issue.message, issue.timestamp, issue.node_id) == (
        IssueCode.GAP,
        "hueco",
        5.5,
        "n1",
    )
    with pytest.raises(FrozenInstanceError):
        issue.message = "otro"  # type: ignore[misc]


# ------------------------------------------------------------ validate_sample


def test_validate_sample_clean_sample_has_no_issues() -> None:
    assert validate_sample(make_sample()) == []
    assert validate_sample(make_sample(label="absent")) == []


def test_empty_amplitudes() -> None:
    issues = validate_sample(make_sample(amplitudes=()))
    assert codes(issues) == [IssueCode.EMPTY_AMPLITUDES]
    assert is_usable(issues) is False


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_non_finite_amplitude(bad: float) -> None:
    issues = validate_sample(make_sample(amplitudes=(1.0, bad, 3.0)))
    # -inf no se reporta además como negativa: solo como no finita.
    assert codes(issues) == [IssueCode.NON_FINITE_VALUE]
    assert "índice 1" in issues[0].message  # posición del primer valor malo
    assert is_usable(issues) is False


@pytest.mark.parametrize("bad", ["x", None, True, 10**400])
def test_non_numeric_amplitude_is_reported_not_raised(bad: object) -> None:
    issues = validate_sample(make_sample(amplitudes=(1.0, bad, 3.0)))
    assert IssueCode.NON_FINITE_VALUE in codes(issues)


@pytest.mark.parametrize("bad", [NAN, INF, "x"])
def test_non_finite_timestamp(bad: object) -> None:
    issues = validate_sample(make_sample(timestamp=bad))
    assert codes(issues) == [IssueCode.NON_FINITE_VALUE]
    assert issues[0].timestamp is None  # un timestamp roto no se propaga al Issue


def test_negative_amplitude_is_reported_but_sample_stays_usable() -> None:
    issues = validate_sample(make_sample(amplitudes=(1.0, -0.5, 3.0)))
    assert codes(issues) == [IssueCode.NEGATIVE_AMPLITUDE]
    assert is_usable(issues) is True


@pytest.mark.parametrize("node_id", ["", "   "])
def test_missing_node_id(node_id: str) -> None:
    issues = validate_sample(make_sample(node_id=node_id))
    assert codes(issues) == [IssueCode.MISSING_NODE_ID]
    assert issues[0].node_id is None
    assert is_usable(issues) is False


def test_invalid_label_is_reported_but_sample_stays_usable() -> None:
    issues = validate_sample(make_sample(label="unknown"))
    assert codes(issues) == [IssueCode.INVALID_LABEL]
    assert is_usable(issues) is True


def test_no_label_is_not_an_issue() -> None:
    assert validate_sample(make_sample(label=None)) == []


def test_validate_sample_reports_all_problems_at_once() -> None:
    issues = validate_sample(
        make_sample(node_id="", amplitudes=(-1.0, NAN), label="zzz")
    )
    assert set(codes(issues)) == {
        IssueCode.MISSING_NODE_ID,
        IssueCode.NON_FINITE_VALUE,
        IssueCode.NEGATIVE_AMPLITUDE,
        IssueCode.INVALID_LABEL,
    }


def test_issue_carries_sample_timestamp_and_node() -> None:
    issues = validate_sample(make_sample(timestamp=123.5, node_id="nodo-7", amplitudes=()))
    assert issues[0].timestamp == 123.5
    assert issues[0].node_id == "nodo-7"


# ------------------------------------------------------------ StreamValidator


def test_stream_validator_rejects_bad_configuration() -> None:
    for bad_gap in (0, -1.0, NAN, INF, "1"):
        with pytest.raises(ValueError, match="max_gap_s"):
            StreamValidator(max_gap_s=bad_gap)  # type: ignore[arg-type]
    for bad_count in (0, -3, True, 2.5):
        with pytest.raises(ValueError, match="expected_subcarriers"):
            StreamValidator(max_gap_s=1.0, expected_subcarriers=bad_count)  # type: ignore[arg-type]
    StreamValidator(max_gap_s=1.0, expected_subcarriers=52)  # válido: no lanza


def test_correct_sequence_has_no_issues() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    for ts in (100.0, 100.5, 101.0, 101.5, 102.0):
        issues = validator.check(make_sample(timestamp=ts))
        assert issues == []
        assert is_usable(issues) is True


def test_duplicate_timestamp() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    assert validator.check(make_sample(timestamp=100.0)) == []
    issues = validator.check(make_sample(timestamp=100.0))
    assert codes(issues) == [IssueCode.DUPLICATE_TIMESTAMP]
    assert is_usable(issues) is False
    # El duplicado no altera el estado: la siguiente muestra normal pasa limpia.
    assert validator.check(make_sample(timestamp=100.5)) == []


def test_out_of_order_does_not_regress_state() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    validator.check(make_sample(timestamp=100.0))
    validator.check(make_sample(timestamp=101.0))
    issues = validator.check(make_sample(timestamp=100.5))
    assert codes(issues) == [IssueCode.OUT_OF_ORDER]
    assert is_usable(issues) is False
    # Respecto al máximo (101.0) la diferencia es 0.75: sin hueco. Si el estado
    # hubiera retrocedido a 100.5 sería 1.25 y aparecería un GAP falso.
    assert validator.check(make_sample(timestamp=101.75)) == []


def test_gap_is_reported_and_keeps_sample_usable() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    validator.check(make_sample(timestamp=100.0))
    validator.check(make_sample(timestamp=100.5))
    issues = validator.check(make_sample(timestamp=102.0))
    assert codes(issues) == [IssueCode.GAP]
    assert issues[0].timestamp == 102.0
    assert issues[0].node_id == "n1"
    assert is_usable(issues) is True


def test_gap_exactly_equal_to_max_is_not_a_gap() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    validator.check(make_sample(timestamp=100.0))
    assert validator.check(make_sample(timestamp=101.0)) == []
    assert codes(validator.check(make_sample(timestamp=102.25))) == [IssueCode.GAP]


def test_state_advances_after_a_gap() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    validator.check(make_sample(timestamp=100.0))
    assert codes(validator.check(make_sample(timestamp=105.0))) == [IssueCode.GAP]
    # El siguiente se mide contra 105.0, no contra 100.0.
    assert validator.check(make_sample(timestamp=105.5)) == []


def test_multiple_gaps_are_each_reported() -> None:
    validator = StreamValidator(max_gap_s=0.5)
    timestamps = (100.0, 101.0, 101.25, 103.0, 103.25)
    all_issues = [i for ts in timestamps for i in validator.check(make_sample(timestamp=ts))]
    assert codes(all_issues) == [IssueCode.GAP, IssueCode.GAP]


def test_subcarrier_count_change_uses_first_sample_as_reference() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    assert validator.check(make_sample(timestamp=100.0, amplitudes=(1.0, 2.0, 3.0))) == []
    issues = validator.check(make_sample(timestamp=100.5, amplitudes=(1.0, 2.0, 3.0, 4.0)))
    assert codes(issues) == [IssueCode.SUBCARRIER_COUNT_MISMATCH]
    assert is_usable(issues) is False
    # La referencia no se desplaza: el cambio persistente se sigue reportando...
    again = validator.check(make_sample(timestamp=101.0, amplitudes=(1.0, 2.0, 3.0, 4.0)))
    assert codes(again) == [IssueCode.SUBCARRIER_COUNT_MISMATCH]
    # ...y volver al conteo original vuelve a ser normal.
    assert validator.check(make_sample(timestamp=101.5)) == []


def test_expected_subcarriers_applies_from_the_first_sample() -> None:
    validator = StreamValidator(max_gap_s=1.0, expected_subcarriers=3)
    issues = validator.check(make_sample(timestamp=100.0, amplitudes=(1.0, 2.0, 3.0, 4.0)))
    assert codes(issues) == [IssueCode.SUBCARRIER_COUNT_MISMATCH]
    assert validator.check(make_sample(timestamp=100.5)) == []


def test_empty_amplitudes_do_not_set_subcarrier_reference() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    first = validator.check(make_sample(timestamp=100.0, amplitudes=()))
    assert codes(first) == [IssueCode.EMPTY_AMPLITUDES]  # sin MISMATCH adicional
    assert validator.check(make_sample(timestamp=100.5)) == []
    assert validator.check(make_sample(timestamp=101.0)) == []


def test_check_includes_single_sample_issues() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    issues = validator.check(make_sample(amplitudes=(1.0, NAN, 3.0), label="zzz"))
    assert set(codes(issues)) == {IssueCode.NON_FINITE_VALUE, IssueCode.INVALID_LABEL}
    assert is_usable(issues) is False


def test_check_missing_node_id_reports_and_keeps_no_state() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    for _ in range(2):  # repetir el mismo timestamp sin node_id NO es "duplicado"
        issues = validator.check(make_sample(node_id=""))
        assert codes(issues) == [IssueCode.MISSING_NODE_ID]
        assert is_usable(issues) is False


def test_non_finite_timestamp_does_not_break_the_stream() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    assert validator.check(make_sample(timestamp=100.0)) == []
    bad = validator.check(make_sample(timestamp=NAN))
    assert codes(bad) == [IssueCode.NON_FINITE_VALUE]
    assert is_usable(bad) is False
    assert validator.check(make_sample(timestamp=100.5)) == []


# ------------------------------------------------------------ estado por nodo


def test_two_interleaved_nodes_do_not_interfere() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    # n1 y n2 con relojes muy distintos y distinto número de subportadoras.
    sequence = [
        make_sample(node_id="n1", timestamp=100.0, amplitudes=(1.0, 2.0, 3.0)),
        make_sample(node_id="n2", timestamp=5000.0, amplitudes=(1.0, 2.0, 3.0, 4.0)),
        make_sample(node_id="n1", timestamp=100.5, amplitudes=(1.0, 2.0, 3.0)),
        make_sample(node_id="n2", timestamp=5000.5, amplitudes=(1.0, 2.0, 3.0, 4.0)),
        make_sample(node_id="n1", timestamp=101.0, amplitudes=(1.0, 2.0, 3.0)),
    ]
    for sample in sequence:
        assert validator.check(sample) == []  # un estado global daría OUT_OF_ORDER/GAP/MISMATCH


def test_same_timestamp_on_different_nodes_is_not_a_duplicate() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    assert validator.check(make_sample(node_id="n1", timestamp=100.0)) == []
    assert validator.check(make_sample(node_id="n2", timestamp=100.0)) == []


def test_problem_in_one_node_does_not_contaminate_the_other() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    validator.check(make_sample(node_id="n1", timestamp=100.0))
    validator.check(make_sample(node_id="n2", timestamp=100.0))
    # n1 sufre un hueco; n2 sigue sano.
    assert codes(validator.check(make_sample(node_id="n1", timestamp=105.0))) == [IssueCode.GAP]
    assert validator.check(make_sample(node_id="n2", timestamp=100.5)) == []
    # n1 sufre un duplicado; n2 sigue sano.
    assert codes(validator.check(make_sample(node_id="n1", timestamp=105.0))) == [
        IssueCode.DUPLICATE_TIMESTAMP
    ]
    assert validator.check(make_sample(node_id="n2", timestamp=101.0)) == []


def test_stale_node_gap_is_detected_independently() -> None:
    validator = StreamValidator(max_gap_s=1.0)
    validator.check(make_sample(node_id="n1", timestamp=100.0))
    validator.check(make_sample(node_id="n2", timestamp=100.0))
    validator.check(make_sample(node_id="n1", timestamp=100.5))
    # n2 calla mientras n1 avanza; al volver, SU hueco se mide contra SU último dato.
    assert validator.check(make_sample(node_id="n1", timestamp=101.0)) == []
    assert codes(validator.check(make_sample(node_id="n2", timestamp=103.0))) == [IssueCode.GAP]


# ------------------------------------------------------------------ is_usable


@pytest.mark.parametrize(
    "code",
    [
        IssueCode.NON_FINITE_VALUE,
        IssueCode.EMPTY_AMPLITUDES,
        IssueCode.SUBCARRIER_COUNT_MISMATCH,
        IssueCode.DUPLICATE_TIMESTAMP,
        IssueCode.OUT_OF_ORDER,
        IssueCode.MISSING_NODE_ID,
    ],
)
def test_is_usable_false_for_blocking_codes(code: IssueCode) -> None:
    assert is_usable([make_issue(code)]) is False


@pytest.mark.parametrize(
    "code",
    [IssueCode.GAP, IssueCode.NEGATIVE_AMPLITUDE, IssueCode.INVALID_LABEL],
)
def test_is_usable_true_for_non_blocking_codes(code: IssueCode) -> None:
    assert is_usable([make_issue(code)]) is True


def test_is_usable_empty_and_mixed() -> None:
    assert is_usable([]) is True
    assert is_usable([make_issue(IssueCode.GAP), make_issue(IssueCode.NON_FINITE_VALUE)]) is False


def test_blocking_codes_are_exactly_the_specified_ones() -> None:
    assert BLOCKING_CODES == {
        IssueCode.NON_FINITE_VALUE,
        IssueCode.EMPTY_AMPLITUDES,
        IssueCode.SUBCARRIER_COUNT_MISMATCH,
        IssueCode.DUPLICATE_TIMESTAMP,
        IssueCode.OUT_OF_ORDER,
        IssueCode.MISSING_NODE_ID,
    }