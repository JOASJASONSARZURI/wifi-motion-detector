"""Pruebas SYNTHETIC a mano de las fuentes visuales."""

from __future__ import annotations

from wifi_motion.gui.sources import DemoSource


def test_demo_source_cycles_through_states_and_marks_synthetic() -> None:
    source = DemoSource()

    results = [source.poll()[0] for _ in range(4)]

    assert [result.state for result in results] == [
        "absent",
        "present_still",
        "present_moving",
        "unknown",
    ]
    assert all(result.reason and "SYNTHETIC" in result.reason for result in results)
    assert source.evidence == "SYNTHETIC"
    assert source.stats()["evidence"] == "SYNTHETIC"
