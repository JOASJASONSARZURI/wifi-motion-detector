"""SYNTHETIC: la separabilidad está por construcción; un buen resultado sobre estos datos NO valida el sistema con hardware real."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from wifi_motion.schema import LABELS, CsiSample, SessionMetadata

__all__ = ["generate_dataset", "generate_session", "inject_faults"]

_START_TIMESTAMP = 1_700_000_000.0


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} debe ser finito y mayor que cero")
    return result


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} debe ser finito")
    return result


def _fault_rate(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} debe estar entre 0 y 1")
    return result


def generate_session(
    segments: list[tuple[str, float]],
    *,
    session_id: str,
    node_id: str = "node-1",
    rate_hz: float = 100.0,
    n_subcarriers: int = 52,
    seed: int = 0,
    baseline_offset: float = 0.0,
    noise_scale: float = 1.0,
) -> tuple[SessionMetadata, list[CsiSample]]:
    """Genera una grabación determinista marcada siempre como sintética."""
    rate_hz = _positive_finite(rate_hz, "rate_hz")
    baseline_offset = _finite(baseline_offset, "baseline_offset")
    noise_scale = _finite(noise_scale, "noise_scale")
    if noise_scale < 0:
        raise ValueError("noise_scale no puede ser negativo")
    if isinstance(n_subcarriers, bool) or not isinstance(n_subcarriers, int):
        raise TypeError("n_subcarriers debe ser un entero")
    if n_subcarriers <= 0:
        raise ValueError("n_subcarriers debe ser mayor que cero")

    normalized_segments: list[tuple[str, float, int]] = []
    for label, duration in segments:
        if label not in LABELS:
            raise ValueError(f"etiqueta no válida: {label!r}")
        duration = _finite(duration, "duration")
        if duration < 0:
            raise ValueError("duration no puede ser negativa")
        normalized_segments.append((label, duration, round(duration * rate_hz)))

    rng = np.random.default_rng(seed)
    carrier_offsets = rng.uniform(-0.3, 0.3, n_subcarriers)
    baseline = 1.0 + carrier_offsets + baseline_offset
    samples: list[CsiSample] = []
    sample_index = 0
    time_step = 1.0 / rate_hz

    for label, _duration, count in normalized_segments:
        if count == 0:
            continue
        elapsed = np.arange(count, dtype=np.float64) * time_step
        amplitudes = np.broadcast_to(baseline, (count, n_subcarriers)).copy()
        amplitudes += rng.normal(0.0, 0.1 * noise_scale, amplitudes.shape)

        if label == "present_still":
            phases = rng.uniform(0.0, 2.0 * math.pi, n_subcarriers)
            frequencies = rng.uniform(0.2, 0.3, n_subcarriers)
            amplitudes += 0.25 * np.sin(
                2.0 * math.pi * elapsed[:, None] * frequencies[None, :]
                + phases[None, :]
            )
        elif label == "present_moving":
            phases = rng.uniform(0.0, 2.0 * math.pi, (3, n_subcarriers))
            frequencies = rng.uniform(0.5, 3.0, (3, n_subcarriers))
            components = np.sin(
                2.0 * math.pi * elapsed[:, None, None] * frequencies[None, :, :]
                + phases[None, :, :]
            )
            amplitudes += 0.35 * components.sum(axis=1)

            burst_count = max(1, round(_duration / 2.0))
            centers = rng.uniform(0.0, max(_duration, time_step), burst_count)
            widths = rng.uniform(0.08, 0.25, burst_count)
            burst_envelope = np.exp(
                -0.5
                * ((elapsed[:, None] - centers[None, :]) / widths[None, :]) ** 2
            ).sum(axis=1)
            burst_carriers = rng.uniform(0.0, 2.0 * math.pi, n_subcarriers)
            amplitudes += (
                0.35
                * burst_envelope[:, None]
                * np.sin(
                    2.0 * math.pi * elapsed[:, None] * 1.7 + burst_carriers[None, :]
                )
            )

        for row in amplitudes:
            samples.append(
                CsiSample(
                    timestamp=_START_TIMESTAMP + sample_index * time_step,
                    node_id=node_id,
                    amplitudes=tuple(float(value) for value in row),
                    label=label,
                )
            )
            sample_index += 1

    if len(samples) > 1:
        intervals = rng.uniform(0.995, 1.005, len(samples) - 1) * time_step
        timestamps = _START_TIMESTAMP + np.concatenate(
            (np.zeros(1), np.cumsum(intervals))
        )
        samples = [
            replace(sample, timestamp=float(timestamp))
            for sample, timestamp in zip(samples, timestamps)
        ]

    metadata = SessionMetadata(
        session_id=session_id,
        environment="SYNTHETIC",
        sensor_layout="SYNTHETIC",
        nominal_rate_hz=rate_hz,
        synthetic=True,
    )
    return metadata, samples


def generate_dataset(
    n_sessions: int,
    *,
    seed: int = 0,
    **kwargs: object,
) -> list[tuple[SessionMetadata, list[CsiSample]]]:
    """Genera sesiones independientes con parámetros de ruido y baseline variados."""
    if isinstance(n_sessions, bool) or not isinstance(n_sessions, int):
        raise TypeError("n_sessions debe ser un entero")
    if n_sessions < 0:
        raise ValueError("n_sessions no puede ser negativo")

    rng = np.random.default_rng(seed)
    base_offset = _finite(float(kwargs.pop("baseline_offset", 0.0)), "baseline_offset")
    base_noise = _finite(float(kwargs.pop("noise_scale", 1.0)), "noise_scale")
    if base_noise < 0:
        raise ValueError("noise_scale no puede ser negativo")
    dataset = []
    for index in range(n_sessions):
        offset_jitter = float(rng.uniform(-0.05, 0.05))
        noise_factor = float(rng.uniform(0.8, 1.2))
        dataset.append(
            generate_session(
                session_id=f"syn-{index:03d}",
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                baseline_offset=base_offset + index * 0.25 + offset_jitter,
                noise_scale=base_noise * noise_factor,
                **kwargs,
            )
        )
    return dataset


def inject_faults(
    samples: list[CsiSample],
    *,
    nan_rate: float = 0.0,
    drop_rate: float = 0.0,
    duplicate_rate: float = 0.0,
    swap_rate: float = 0.0,
    seed: int = 0,
) -> list[CsiSample]:
    """Devuelve una copia con defectos aleatorios para probar validación."""
    nan_rate = _fault_rate(nan_rate, "nan_rate")
    drop_rate = _fault_rate(drop_rate, "drop_rate")
    duplicate_rate = _fault_rate(duplicate_rate, "duplicate_rate")
    swap_rate = _fault_rate(swap_rate, "swap_rate")
    rng = np.random.default_rng(seed)
    copied = list(samples)

    if nan_rate:
        corrupted = []
        for sample in copied:
            if rng.random() < nan_rate and sample.amplitudes:
                index = int(rng.integers(len(sample.amplitudes)))
                amplitudes = list(sample.amplitudes)
                amplitudes[index] = float("nan")
                sample = replace(sample, amplitudes=tuple(amplitudes))
            corrupted.append(sample)
        copied = corrupted

    if drop_rate:
        copied = [sample for sample in copied if rng.random() >= drop_rate]

    if duplicate_rate:
        duplicated = []
        for sample in copied:
            duplicated.append(sample)
            if rng.random() < duplicate_rate:
                duplicated.append(sample)
        copied = duplicated

    index = 0
    while index + 1 < len(copied):
        if rng.random() < swap_rate:
            copied[index], copied[index + 1] = copied[index + 1], copied[index]
            index += 2
        else:
            index += 1
    return copied