"""Pruebas de wifi_motion.io_jsonl.

SYNTHETIC: todos los datos de este archivo están escritos a mano; no son
mediciones reales. Las pruebas de archivos usan ``tmp_path`` de pytest.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from wifi_motion.io_jsonl import RecordingFormatError, iter_recording, write_recording
from wifi_motion.schema import CsiSample, SessionMetadata


def make_metadata(**overrides: Any) -> SessionMetadata:
    """SYNTHETIC: metadatos de ejemplo, marcados como sintéticos."""
    base: dict[str, Any] = {
        "session_id": "sess-001",
        "environment": "sala-101",
        "sensor_layout": "nodo A a 2 m del nodo B",
        "nominal_rate_hz": 100.0,
        "synthetic": True,
    }
    base.update(overrides)
    return SessionMetadata(**base)


def make_samples(n: int, node_id: str = "n1") -> list[CsiSample]:
    """SYNTHETIC: n muestras de 3 subportadoras con timestamps crecientes."""
    return [
        CsiSample(
            timestamp=100.0 + i * 0.5,
            node_id=node_id,
            amplitudes=(1.0 + i, 2.0 + i, 3.0 + i),
        )
        for i in range(n)
    ]


HEADER_LINE = json.dumps({"type": "header", **make_metadata().to_dict()})


def sample_line(i: int = 0, **overrides: Any) -> str:
    record: dict[str, Any] = {"type": "sample", **make_samples(i + 1)[i].to_dict()}
    record.update(overrides)
    return json.dumps(record)


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def read_all(path: Path) -> tuple[SessionMetadata, list[CsiSample]]:
    metadata, samples = iter_recording(path)
    with samples:
        return metadata, list(samples)


# ---------------------------------------------------------- RecordingFormatError


def test_format_error_is_a_value_error_with_line_and_reason() -> None:
    err = RecordingFormatError(7, "algo falló")
    assert isinstance(err, ValueError)
    assert err.line_number == 7
    assert err.reason == "algo falló"
    assert str(err) == "línea 7: algo falló"


# ----------------------------------------------------------------- round-trip


def test_round_trip_write_then_read(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    samples = [
        CsiSample(100.0, "n1", (1.0, 2.0, 3.0)),
        CsiSample(100.5, "n2", (4.0, 5.0, 6.0), rssi=-41.5, label="present_still"),
        CsiSample(101.0, "n1", (7.0, 8.0, 9.0), phase=(0.1, 0.2, 0.3), label="absent"),
    ]
    metadata = make_metadata(notes="prueba de ida y vuelta")
    assert write_recording(path, metadata, samples) == 3
    read_meta, read_samples = read_all(path)
    assert read_meta == metadata
    assert read_samples == samples


def test_round_trip_accepts_str_path_and_generator(tmp_path: Path) -> None:
    path = str(tmp_path / "rec.jsonl")
    samples = make_samples(4)
    assert write_recording(path, make_metadata(), (s for s in samples)) == 4
    assert read_all(Path(path))[1] == samples


@pytest.mark.parametrize("synthetic", [True, False])
def test_header_preserves_synthetic_flag(tmp_path: Path, synthetic: bool) -> None:
    path = tmp_path / "rec.jsonl"
    write_recording(path, make_metadata(synthetic=synthetic), make_samples(1))
    first_line = json.loads(path.read_bytes().split(b"\n")[0])
    assert first_line["synthetic"] is synthetic
    assert read_all(path)[0].synthetic is synthetic


def test_recording_without_samples(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    assert write_recording(path, make_metadata(), []) == 0
    metadata, samples = read_all(path)
    assert metadata == make_metadata()
    assert samples == []


def test_file_layout_is_header_then_samples_utf8_with_lf(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    metadata = make_metadata(notes="calibración: pared norte")
    write_recording(path, metadata, make_samples(3))
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    assert "calibración".encode("utf-8") in raw  # UTF-8 real, no \u00f3 escapado
    lines = raw.decode("utf-8").split("\n")[:-1]
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]
    assert [r["type"] for r in records] == ["header", "sample", "sample", "sample"]
    assert records[0]["schema_version"] == 1


def test_reading_tolerates_crlf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    text = "\r\n".join([HEADER_LINE, sample_line(0), sample_line(1)]) + "\r\n"
    path.write_bytes(text.encode("utf-8"))
    _, samples = read_all(path)
    assert samples == make_samples(2)


def test_reader_checks_structure_only_not_signal_content(tmp_path: Path) -> None:
    # Esto lo detectará validation.py, no el lector.
    path = tmp_path / "rec.jsonl"
    odd = [
        CsiSample(100.0, "n1", ()),
        CsiSample(100.0, "", (-1.0, -2.0), label="zzz"),
    ]
    write_recording(path, make_metadata(), odd)
    assert read_all(path)[1] == odd


# --------------------------------------------------------------- errores: header


def test_empty_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    path.write_bytes(b"")
    with pytest.raises(RecordingFormatError, match="vacío") as info:
        iter_recording(path)
    assert info.value.line_number == 1


def test_file_without_header_is_an_error(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "rec.jsonl", [sample_line(0), sample_line(1)])
    with pytest.raises(RecordingFormatError, match="header") as info:
        iter_recording(path)
    assert info.value.line_number == 1


def test_future_schema_version_is_rejected_immediately(tmp_path: Path) -> None:
    header = json.dumps({"type": "header", **make_metadata().to_dict(), "schema_version": 2})
    path = write_lines(tmp_path / "rec.jsonl", [header, sample_line(0)])
    with pytest.raises(RecordingFormatError, match="schema_version 2 no soportada") as info:
        iter_recording(path)
    assert info.value.line_number == 1


def test_future_version_wins_over_unknown_new_fields(tmp_path: Path) -> None:
    header = json.dumps(
        {"type": "header", **make_metadata().to_dict(), "schema_version": 9, "nuevo": 1}
    )
    path = write_lines(tmp_path / "rec.jsonl", [header])
    with pytest.raises(RecordingFormatError, match="no soportada"):
        iter_recording(path)


@pytest.mark.parametrize("bad_version", ["1", 1.0, True, None, [1]])
def test_non_integer_schema_version_is_rejected(tmp_path: Path, bad_version: object) -> None:
    header = json.dumps(
        {"type": "header", **make_metadata().to_dict(), "schema_version": bad_version}
    )
    path = write_lines(tmp_path / "rec.jsonl", [header])
    with pytest.raises(RecordingFormatError, match="schema_version debe ser un entero"):
        iter_recording(path)


@pytest.mark.parametrize("missing", ["schema_version", "session_id", "nominal_rate_hz"])
def test_header_with_missing_field_is_an_error(tmp_path: Path, missing: str) -> None:
    fields = make_metadata().to_dict()
    del fields[missing]
    path = write_lines(tmp_path / "rec.jsonl", [json.dumps({"type": "header", **fields})])
    with pytest.raises(RecordingFormatError, match=missing) as info:
        iter_recording(path)
    assert info.value.line_number == 1


def test_header_with_unknown_field_is_an_error(tmp_path: Path) -> None:
    header = json.dumps({"type": "header", **make_metadata().to_dict(), "operator": "x"})
    path = write_lines(tmp_path / "rec.jsonl", [header])
    with pytest.raises(RecordingFormatError, match="operator"):
        iter_recording(path)


def test_header_line_with_unknown_type(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "rec.jsonl", ['{"type":"cabecera"}'])
    with pytest.raises(RecordingFormatError, match="cabecera") as info:
        iter_recording(path)
    assert info.value.line_number == 1


def test_header_with_broken_json_reports_line_1(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "rec.jsonl", ['{"type":"header",'])
    with pytest.raises(RecordingFormatError, match="JSON inválido") as info:
        iter_recording(path)
    assert info.value.line_number == 1


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        iter_recording(tmp_path / "no-existe.jsonl")


# -------------------------------------------------------------- errores: muestras


@pytest.mark.parametrize("bad_index", [0, 2, 5])
def test_broken_json_reports_the_exact_line_number(tmp_path: Path, bad_index: int) -> None:
    good = [sample_line(i) for i in range(6)]
    good[bad_index] = '{"type":"sample","timestamp":1.0,'  # JSON truncado
    path = write_lines(tmp_path / "rec.jsonl", [HEADER_LINE, *good])
    expected_line = bad_index + 2  # línea 1 = header
    _, samples = iter_recording(path)
    with samples:
        with pytest.raises(RecordingFormatError, match="JSON inválido") as info:
            list(samples)
    assert info.value.line_number == expected_line
    assert f"línea {expected_line}" in str(info.value)


def test_unknown_record_type_reports_its_line(tmp_path: Path) -> None:
    lines = [HEADER_LINE, sample_line(0), '{"type":"marker","at":1.0}', sample_line(1)]
    path = write_lines(tmp_path / "rec.jsonl", lines)
    _, samples = iter_recording(path)
    with samples:
        with pytest.raises(RecordingFormatError, match="marker") as info:
            list(samples)
    assert info.value.line_number == 3


def test_record_without_type_field(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "rec.jsonl", [HEADER_LINE, '{"timestamp":1.0}'])
    _, samples = iter_recording(path)
    with samples:
        with pytest.raises(RecordingFormatError, match="'type'") as info:
            list(samples)
    assert info.value.line_number == 2


def test_repeated_header_is_an_error(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "rec.jsonl", [HEADER_LINE, sample_line(0), HEADER_LINE])
    _, samples = iter_recording(path)
    with samples:
        with pytest.raises(RecordingFormatError, match="header repetido") as info:
            list(samples)
    assert info.value.line_number == 3


@pytest.mark.parametrize("missing", ["timestamp", "node_id", "amplitudes"])
def test_sample_with_missing_field(tmp_path: Path, missing: str) -> None:
    fields = make_samples(1)[0].to_dict()
    del fields[missing]
    bad = json.dumps({"type": "sample", **fields})
    path = write_lines(tmp_path / "rec.jsonl", [HEADER_LINE, sample_line(0), bad])
    _, samples = iter_recording(path)
    with samples:
        with pytest.raises(RecordingFormatError, match=missing) as info:
            list(samples)
    assert info.value.line_number == 3


def test_sample_with_unknown_field(tmp_path: Path) -> None:
    path = write_lines(
        tmp_path / "rec.jsonl", [HEADER_LINE, sample_line(0, person_name="x")]
    )
    _, samples = iter_recording(path)
    with samples:
        with pytest.raises(RecordingFormatError, match="person_name") as info:
            list(samples)
    assert info.value.line_number == 2


@pytest.mark.parametrize("bad_line", ["# comentario", "// comentario", "", "   ", "[1, 2]", "3"])
def test_comments_blank_lines_and_non_objects_are_errors(tmp_path: Path, bad_line: str) -> None:
    path = write_lines(tmp_path / "rec.jsonl", [HEADER_LINE, sample_line(0), bad_line])
    _, samples = iter_recording(path)
    with samples:
        with pytest.raises(RecordingFormatError) as info:
            list(samples)
    assert info.value.line_number == 3


def test_nan_and_infinity_tokens_are_rejected_on_read(tmp_path: Path) -> None:
    nan_line = '{"type":"sample","timestamp":1.0,"node_id":"n1","amplitudes":[NaN]}'
    path = write_lines(tmp_path / "rec.jsonl", [HEADER_LINE, nan_line])
    _, samples = iter_recording(path)
    with samples:
        with pytest.raises(RecordingFormatError, match="NaN") as info:
            list(samples)
    assert info.value.line_number == 2


def test_invalid_utf8_reports_its_line(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    path.write_bytes(
        (HEADER_LINE + "\n" + sample_line(0) + "\n").encode("utf-8")
        + b'{"type":"sample","node_id":"\xff\xfe"}\n'
    )
    _, samples = iter_recording(path)
    with samples:
        with pytest.raises(RecordingFormatError, match="UTF-8") as info:
            list(samples)
    assert info.value.line_number == 3


# ------------------------------------------------------------- lectura perezosa


def test_samples_are_parsed_lazily_one_line_at_a_time(tmp_path: Path) -> None:
    # Header + 1000 muestras buenas + una línea rota al final (línea 1002).
    # Si el lector cargara o analizara todo de golpe, el error saldría al abrir
    # o con la primera muestra.
    lines = [HEADER_LINE] + [sample_line(i % 3) for i in range(1000)] + ["{roto"]
    path = write_lines(tmp_path / "rec.jsonl", lines)
    metadata, samples = iter_recording(path)  # no lanza: el error está lejos
    assert metadata == make_metadata()
    assert isinstance(samples, Iterator)
    assert iter(samples) is samples
    assert not hasattr(samples, "__len__")
    first_ten = [next(samples) for _ in range(10)]
    assert len(first_ten) == 10
    assert samples.closed is False
    with pytest.raises(RecordingFormatError) as info:
        for _ in samples:
            pass
    assert info.value.line_number == 1002


def test_header_errors_surface_at_open_but_sample_errors_do_not(tmp_path: Path) -> None:
    bad_sample = write_lines(tmp_path / "a.jsonl", [HEADER_LINE, "{roto"])
    _, samples = iter_recording(bad_sample)  # sin excepción todavía
    samples.close()
    bad_header = write_lines(tmp_path / "b.jsonl", ["{roto", sample_line(0)])
    with pytest.raises(RecordingFormatError):
        iter_recording(bad_header)


def test_iterator_closes_file_when_exhausted(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recording(path, make_metadata(), make_samples(2))
    _, samples = iter_recording(path)
    assert samples.closed is False
    assert len(list(samples)) == 2
    assert samples.closed is True
    assert list(samples) == []  # agotado: sigue sin lanzar
    path.unlink()  # en Windows fallaría si el archivo siguiera abierto


def test_iterator_close_and_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recording(path, make_metadata(), make_samples(5))
    _, samples = iter_recording(path)
    next(samples)
    samples.close()
    samples.close()  # idempotente
    assert samples.closed is True
    assert list(samples) == []
    _, other = iter_recording(path)
    with other:
        next(other)
    assert other.closed is True


def test_iterator_closes_file_after_a_format_error(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "rec.jsonl", [HEADER_LINE, "{roto"])
    _, samples = iter_recording(path)
    with pytest.raises(RecordingFormatError):
        next(samples)
    assert samples.closed is True
    path.unlink()


def test_header_error_does_not_leave_file_open(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "rec.jsonl", [sample_line(0)])
    with pytest.raises(RecordingFormatError):
        iter_recording(path)
    path.unlink()  # en Windows fallaría si quedó abierto


# ------------------------------------------------------------ escritura atómica


def only_files(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def test_successful_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recording(path, make_metadata(), make_samples(3))
    assert only_files(tmp_path) == ["rec.jsonl"]


def test_write_replaces_an_existing_file_completely(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recording(path, make_metadata(session_id="vieja"), make_samples(5))
    write_recording(path, make_metadata(session_id="nueva"), make_samples(2))
    metadata, samples = read_all(path)
    assert metadata.session_id == "nueva"
    assert len(samples) == 2
    assert only_files(tmp_path) == ["rec.jsonl"]


def test_failure_midway_keeps_existing_destination_intact(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recording(path, make_metadata(session_id="original"), make_samples(3))
    original = path.read_bytes()

    def exploding() -> Iterator[CsiSample]:
        yield from make_samples(2)
        raise RuntimeError("fallo simulado a mitad de escritura")

    with pytest.raises(RuntimeError, match="fallo simulado"):
        write_recording(path, make_metadata(session_id="nuevo"), exploding())
    assert path.read_bytes() == original
    assert only_files(tmp_path) == ["rec.jsonl"]  # el temporal se eliminó


def test_failure_midway_does_not_create_the_destination(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"

    def exploding() -> Iterator[CsiSample]:
        yield from make_samples(2)
        raise RuntimeError("fallo simulado")

    with pytest.raises(RuntimeError):
        write_recording(path, make_metadata(), exploding())
    assert not path.exists()
    assert only_files(tmp_path) == []


def test_destination_is_untouched_while_writing_is_in_progress(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recording(path, make_metadata(session_id="original"), make_samples(3))
    original = path.read_bytes()
    observed: list[bool] = []

    def watching() -> Iterator[CsiSample]:
        for i, sample in enumerate(make_samples(6)):
            if i == 3:  # a mitad de la escritura el destino sigue siendo el original
                observed.append(path.read_bytes() == original)
            yield sample

    assert write_recording(path, make_metadata(session_id="nuevo"), watching()) == 6
    assert observed == [True]
    assert read_all(path)[0].session_id == "nuevo"


def test_non_finite_value_fails_without_touching_destination(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recording(path, make_metadata(), make_samples(2))
    original = path.read_bytes()
    bad = [make_samples(1)[0], CsiSample(101.0, "n1", (1.0, float("nan"), 3.0))]
    with pytest.raises(ValueError, match="muestra 2"):
        write_recording(path, make_metadata(), bad)
    assert path.read_bytes() == original
    assert only_files(tmp_path) == ["rec.jsonl"]


def test_non_finite_header_value_fails(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    with pytest.raises(ValueError, match="header"):
        write_recording(path, make_metadata(nominal_rate_hz=float("inf")), make_samples(1))
    assert only_files(tmp_path) == []


def test_non_sample_item_fails_without_touching_destination(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recording(path, make_metadata(), make_samples(2))
    original = path.read_bytes()
    with pytest.raises(TypeError, match="muestra 2"):
        write_recording(path, make_metadata(), [make_samples(1)[0], {"timestamp": 1.0}])  # type: ignore[list-item]
    assert path.read_bytes() == original
    assert only_files(tmp_path) == ["rec.jsonl"]


def test_non_metadata_argument_is_a_type_error(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="metadata"):
        write_recording(tmp_path / "rec.jsonl", {"session_id": "x"}, [])  # type: ignore[arg-type]
    assert only_files(tmp_path) == []