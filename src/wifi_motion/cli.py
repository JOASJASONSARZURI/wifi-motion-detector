"""Interfaz de línea de comandos para grabaciones y pipeline Wi-Fi CSI."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import sys
from pathlib import Path

import numpy as np

from wifi_motion.calibration import fit_baseline, load_baseline, save_baseline
from wifi_motion.classifiers.base import CLASS_STATES, Classifier, Prediction
from wifi_motion.classifiers.softmax import (
    SoftmaxClassifier,
    SoftmaxConfig,
    macro_f1,
)
from wifi_motion.classifiers.threshold import (
    ThresholdClassifier,
    ThresholdConfig,
    tune_thresholds,
)
from wifi_motion.dataset import build_dataset, split_by_session
from wifi_motion.detector import Detector, run_samples
from wifi_motion.evaluation import (
    check_disjoint_sessions,
    majority_baseline,
    report_label,
    stream_evaluation,
    window_metrics,
)
from wifi_motion.features import FeatureConfig, FeatureError, extract_features
from wifi_motion.io_jsonl import iter_recording, write_recording
from wifi_motion.schema import CsiSample, SessionMetadata
from wifi_motion.smoothing import SmoothingConfig, StateSmoother
from wifi_motion.synthetic import generate_dataset
from wifi_motion.windowing import SlidingWindower, WindowQuality, make_windows

_SYNTHETIC_SEGMENTS = [
    ("absent", 12.0),
    ("present_still", 12.0),
    ("present_moving", 12.0),
]
_EVAL_THRESHOLD_GRID = (
    0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0,
    20.0, 30.0, 40.0, 50.0, 80.0, 120.0,
)


def _add_window_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-s", type=float, default=4.0)
    parser.add_argument("--hop-s", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--max-gap-s", type=float, default=0.15)


def _window_config(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "window_s": args.window_s,
        "hop_s": args.hop_s,
        "min_samples": args.min_samples,
        "max_gap_s": args.max_gap_s,
    }


def _read_recording(
    path: str | Path,
) -> tuple[SessionMetadata, list[CsiSample]]:
    metadata, iterator = iter_recording(path)
    with iterator:
        samples = list(iterator)
    return metadata, samples


def _generate_synthetic(args: argparse.Namespace) -> int:
    if args.sessions < 1:
        raise ValueError("--sessions debe ser al menos 1")
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    recordings = generate_dataset(
        args.sessions,
        seed=args.seed,
        segments=_SYNTHETIC_SEGMENTS,
    )
    for metadata, samples in recordings:
        write_recording(destination / f"{metadata.session_id}.jsonl", metadata, samples)
    source_metadata, source_samples = recordings[0]
    empty_samples = []
    for sample in source_samples:
        if sample.label != "absent":
            break
        empty_samples.append(sample)
    if len(empty_samples) < 2:
        raise ValueError("la sesión sintética no contiene suficientes muestras absent")
    period = len(empty_samples) / source_metadata.nominal_rate_hz
    repeats = math.ceil(80.0 / period)
    empty_samples = [
        replace(
            sample,
            timestamp=sample.timestamp + repeat * period,
            label="absent",
        )
        for repeat in range(repeats)
        for sample in empty_samples
    ]
    empty_metadata = replace(
        source_metadata, session_id="syn-calibration-absent"
    )
    write_recording(
        destination / f"{empty_metadata.session_id}.jsonl",
        empty_metadata,
        empty_samples,
    )
    print(
        f"{len(recordings)} grabación(es) SYNTHETIC y una de calibración "
        f"(solo absent) en {destination}"
    )
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    metadata, samples = _read_recording(args.input)
    if any(sample.label != "absent" for sample in samples):
        raise ValueError("calibrate requiere una grabación etiquetada solo como absent")
    node_samples = [sample for sample in samples if sample.node_id == args.node_id]
    if not node_samples:
        raise ValueError(f"no hay muestras del nodo {args.node_id!r}")

    windows = make_windows(node_samples, **_window_config(args))
    vectors = []
    skipped = 0
    for window in windows:
        if window.quality is not WindowQuality.OK:
            skipped += 1
            continue
        try:
            vectors.append(extract_features(window, FeatureConfig()))
        except FeatureError:
            skipped += 1
    if not vectors:
        raise ValueError("no hay ventanas válidas para calibrar")
    baseline = fit_baseline(
        args.node_id,
        np.stack(vectors),
        environment_note=metadata.environment,
    )
    save_baseline(baseline, args.out)
    print(
        f"Línea base guardada en {args.out} ({baseline.n_windows} ventanas; "
        f"{skipped} descartadas)"
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    train_paths = [Path(value) for value in args.train.split(",") if value]
    val_paths = [Path(value) for value in args.val.split(",") if value]
    if not train_paths or not val_paths:
        raise ValueError("--train y --val deben incluir al menos un archivo")
    train_recordings = [_read_recording(path) for path in train_paths]
    val_recordings = [_read_recording(path) for path in val_paths]
    train_ids = [metadata.session_id for metadata, _ in train_recordings]
    val_ids = [metadata.session_id for metadata, _ in val_recordings]
    combined = train_recordings + val_recordings
    dataset = build_dataset(combined, _window_config(args), FeatureConfig())
    train_data, val_data, _ = split_by_session(dataset, train_ids, val_ids, [])
    if not train_data.y:
        raise ValueError("el conjunto train no contiene ventanas etiquetadas válidas")
    if not val_data.y:
        raise ValueError("el conjunto val no contiene ventanas etiquetadas válidas")

    model = SoftmaxClassifier().fit(
        train_data.X,
        train_data.y,
        val_data.X,
        val_data.y,
        SoftmaxConfig(seed=args.seed, epochs=args.epochs),
    )
    model.save(args.out)
    predictions = model.predict_batch(val_data.X)
    score = macro_f1(
        val_data.y, [prediction.state for prediction in predictions], CLASS_STATES
    )
    synthetic_marker = (
        ", SYNTHETIC"
        if all(metadata.synthetic for metadata, _ in combined)
        else ""
    )
    print(f"F1 macro (validación{synthetic_marker}): {score:.6f}")
    print(f"Modelo guardado en {args.out}")
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.classifier == "threshold":
        if not args.baseline:
            raise ValueError("--baseline es obligatorio con --classifier threshold")
        classifier = ThresholdClassifier(
            load_baseline(args.baseline),
            ThresholdConfig(
                motion_features=("motion_band_power",),
                presence_features=("breathing_band_power",),
                motion_z=args.motion_z,
                presence_z=args.presence_z,
            ),
        )
    else:
        if not args.model:
            raise ValueError("--model es obligatorio con --classifier softmax")
        classifier = SoftmaxClassifier.load(args.model)

    detector = Detector(
        SlidingWindower(**_window_config(args)),
        FeatureConfig(),
        classifier,
        StateSmoother(
            SmoothingConfig(
                ema_alpha=args.ema_alpha,
                confirm_n=args.confirm_n,
                confirm_n_to_absent=args.confirm_n_to_absent,
                margin=args.margin,
            )
        ),
        stale_timeout_s=args.stale_timeout_s,
    )
    _metadata, iterator = iter_recording(args.input)
    with iterator:
        for result in run_samples(detector, iterator):
            print(
                json.dumps(
                    {
                        "node_id": result.node_id,
                        "timestamp": result.timestamp,
                        "state": result.state,
                        "scores": dict(result.scores),
                        "confidence": result.confidence,
                        "changed": result.changed,
                        "input_quality": result.input_quality,
                        "reason": result.reason,
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
    stats = detector.stats()
    stats["rejected_by_issue"] = {
        code.value: count for code, count in stats["rejected_by_issue"].items()
    }
    print(json.dumps(stats, allow_nan=False, separators=(",", ":")), file=sys.stderr)
    return 0


class _MajorityClassifier:
    name = "majority-baseline"

    def __init__(self, state: str) -> None:
        self.state = state

    def predict(self, _features: np.ndarray) -> Prediction:
        return Prediction(
            state=self.state,
            scores={label: float(label == self.state) for label in CLASS_STATES},
            confidence=1.0,
            source=self.name,
        )

    def predict_batch(self, X: np.ndarray) -> list[Prediction]:
        return [self.predict(row) for row in X]


def _eval_paths(value: str | None, name: str, *, required: bool) -> list[Path]:
    paths = [Path(part.strip()) for part in (value or "").split(",") if part.strip()]
    if required and not paths:
        raise ValueError(f"--{name} debe incluir al menos un archivo JSONL")
    return paths


def _check_unique_session_ids(
    recordings: list[tuple[SessionMetadata, list[CsiSample]]], split: str
) -> list[str]:
    ids = [metadata.session_id for metadata, _samples in recordings]
    if len(set(ids)) != len(ids):
        raise ValueError(f"hay session_id repetido dentro de --{split}")
    return ids


def _evaluation_detector(
    classifier: Classifier, args: argparse.Namespace
) -> Detector:
    return Detector(
        SlidingWindower(**_window_config(args)),
        FeatureConfig(),
        classifier,
        StateSmoother(
            SmoothingConfig(
                ema_alpha=args.ema_alpha,
                confirm_n=args.confirm_n,
                confirm_n_to_absent=args.confirm_n_to_absent,
                margin=args.margin,
            )
        ),
        stale_timeout_s=args.stale_timeout_s,
    )


def _eval_log_has_match(
    path: Path, test_ids: list[str], model_paths: dict[str, str]
) -> bool:
    if not path.exists():
        return False
    expected_ids = sorted(test_ids)
    with path.open(encoding="utf-8") as log:
        for line_number, line in enumerate(log, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: JSON inválido en línea {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: la línea {line_number} debe ser un objeto JSON")
            if (
                sorted(record.get("test_ids", [])) == expected_ids
                and record.get("models") == model_paths
            ):
                return True
    return False


def _eval(args: argparse.Namespace) -> int:
    train_paths = _eval_paths(args.train, "train", required=True)
    val_paths = _eval_paths(args.val, "val", required=False)
    test_paths = _eval_paths(args.test, "test", required=True)
    if not args.baseline:
        raise ValueError("--baseline es obligatorio para evaluar el clasificador por umbral")
    if not args.model:
        raise ValueError("--model es obligatorio para evaluar el clasificador softmax")

    train_recordings = [_read_recording(path) for path in train_paths]
    val_recordings = [_read_recording(path) for path in val_paths]
    test_recordings = [_read_recording(path) for path in test_paths]
    train_ids = _check_unique_session_ids(train_recordings, "train")
    val_ids = _check_unique_session_ids(val_recordings, "val")
    test_ids = _check_unique_session_ids(test_recordings, "test")
    check_disjoint_sessions(train_ids, val_ids, test_ids)

    train_data = build_dataset(train_recordings, _window_config(args), FeatureConfig())
    val_data = build_dataset(val_recordings, _window_config(args), FeatureConfig())
    test_data = build_dataset(test_recordings, _window_config(args), FeatureConfig())
    if not train_data.y:
        raise ValueError("el conjunto train no contiene ventanas etiquetadas válidas")
    if val_recordings and not val_data.y:
        raise ValueError("el conjunto val no contiene ventanas etiquetadas válidas")
    if not test_data.y:
        raise ValueError("el conjunto test no contiene ventanas etiquetadas válidas")

    baseline = load_baseline(args.baseline)
    if val_recordings:
        threshold_config = tune_thresholds(
            baseline,
            val_data.X,
            val_data.y,
            _EVAL_THRESHOLD_GRID,
            _EVAL_THRESHOLD_GRID,
        )
    else:
        threshold_config = ThresholdConfig(
            motion_z=args.motion_z,
            presence_z=args.presence_z,
        )
    softmax = SoftmaxClassifier.load(args.model)
    majority_state = majority_baseline(train_data.y, ["absent"])[0]
    true_labels = test_data.y
    predictions: dict[str, list[str]] = {
        "baseline": [majority_state] * len(true_labels),
        "threshold": [
            item.state
            for item in ThresholdClassifier(baseline, threshold_config).predict_batch(
                test_data.X
            )
        ],
        "softmax": [item.state for item in softmax.predict_batch(test_data.X)],
    }

    evidence = report_label([metadata for metadata, _samples in test_recordings])
    window_counts = Counter(true_labels)
    counts_by_class = {state: window_counts[state] for state in CLASS_STATES}
    comparison: dict[str, dict[str, object]] = {}
    classifier_factories = {
        "baseline": lambda: _MajorityClassifier(majority_state),
        "threshold": lambda: ThresholdClassifier(baseline, threshold_config),
        "softmax": lambda: SoftmaxClassifier.load(args.model),
    }
    for name in ("baseline", "threshold", "softmax"):
        report = window_metrics(true_labels, predictions[name])
        report = replace(
            report,
            evidence=evidence,
            n_sessions=len(test_ids),
            n_windows_by_class=tuple(
                (state, counts_by_class[state]) for state in CLASS_STATES
            ),
            session_ids=tuple(test_ids),
        )
        stream_report = stream_evaluation(
            lambda factory=classifier_factories[name]: _evaluation_detector(
                factory(), args
            ),
            test_recordings,
            transition_tolerance_s=args.transition_tolerance_s,
        )
        comparison[name] = {
            "window": report.to_dict(),
            "stream": stream_report.to_dict(),
        }

    model_paths = {
        "baseline": str(Path(args.baseline).resolve()),
        "softmax": str(Path(args.model).resolve()),
    }
    parameters = {
        "window": _window_config(args),
        "smoothing": {
            "ema_alpha": args.ema_alpha,
            "confirm_n": args.confirm_n,
            "confirm_n_to_absent": args.confirm_n_to_absent,
            "margin": args.margin,
            "stale_timeout_s": args.stale_timeout_s,
        },
        "transition_tolerance_s": args.transition_tolerance_s,
        "threshold_config": {
            "motion_features": list(threshold_config.motion_features),
            "presence_features": list(threshold_config.presence_features),
            "motion_z": threshold_config.motion_z,
            "presence_z": threshold_config.presence_z,
            "tuned_on_validation": bool(val_recordings),
            "grid": list(_EVAL_THRESHOLD_GRID) if val_recordings else None,
        },
    }
    report_record = {
        "evidence": evidence,
        "n_sessions": len(test_ids),
        "session_ids": test_ids,
        "n_windows": len(true_labels),
        "n_windows_by_class": counts_by_class,
        "parameters": parameters,
        "models": model_paths,
        "comparison": comparison,
    }

    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(
        json.dumps(report_record, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = [
        f"Evidence: {evidence}",
        "",
        f"Sesiones TEST: {len(test_ids)} ({', '.join(test_ids)})",
        f"Ventanas TEST: {len(true_labels)}; por clase: "
        + ", ".join(f"{label}={counts_by_class[label]}" for label in CLASS_STATES),
        "",
        "| Clasificador | Macro F1 (ventanas) | Macro F1 (flujo) | Falsas alarmas de flujo | Latencia missed |",
        "|---|---:|---:|---:|---:|",
    ]
    if evidence == "SYNTHETIC":
        markdown.insert(
            1,
            "La separabilidad es por construcción; esto no valida el sistema con hardware real",
        )
    for name in ("baseline", "threshold", "softmax"):
        window = comparison[name]["window"]
        flow = comparison[name]["stream"]
        assert isinstance(window, dict) and isinstance(flow, dict)
        missed = sum(
            int(item["missed"])
            for item in flow["latency_by_transition"].values()  # type: ignore[union-attr]
        )
        markdown.append(
            f"| {name} | {window['macro_f1']} | "
            f"{flow['window_metrics']['macro_f1']} | "
            f"{flow['flow_false_alarm_count']} | {missed} |"  # type: ignore[index]
        )
    markdown.extend(
        [
            "",
            "## Parámetros",
            "",
            "```json",
            json.dumps(parameters, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    (destination / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    log_path = destination / "eval_log.jsonl"
    if _eval_log_has_match(log_path, test_ids, model_paths):
        print(
            "advertencia: este conjunto de prueba ya se usó; el resultado deja de ser una evaluación única",
            file=sys.stderr,
        )
    log_record = {
        "date": datetime.now(timezone.utc).isoformat(),
        "test_ids": sorted(test_ids),
        "models": model_paths,
    }
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(log_record, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Evaluación guardada en {destination}")
    return 0


def _record(_args: argparse.Namespace) -> int:
    print(
        "record requiere hardware y el formato real de la fuente CSI; aún no implementado",
        file=sys.stderr,
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wifi-motion")
    commands = parser.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser("generate-synthetic")
    synthetic.add_argument("--out", required=True)
    synthetic.add_argument("--sessions", type=int, required=True)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.set_defaults(handler=_generate_synthetic)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--node-id", required=True)
    calibrate.add_argument("--out", required=True)
    _add_window_options(calibrate)
    calibrate.set_defaults(handler=_calibrate)

    train = commands.add_parser("train")
    train.add_argument("--train", required=True)
    train.add_argument("--val", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=int, default=300)
    _add_window_options(train)
    train.set_defaults(handler=_train)

    run = commands.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument(
        "--classifier", required=True, choices=("threshold", "softmax")
    )
    run.add_argument("--baseline")
    run.add_argument("--model")
    run.add_argument("--ema-alpha", type=float, default=0.4)
    run.add_argument("--confirm-n", type=int, default=3)
    run.add_argument("--confirm-n-to-absent", type=int, default=5)
    run.add_argument("--margin", type=float, default=0.05)
    run.add_argument("--stale-timeout-s", type=float, default=5.0)
    run.add_argument(
        "--motion-z",
        type=float,
        default=3.0,
        help="umbral inicial de movimiento; debe ajustarse con datos medidos",
    )
    run.add_argument(
        "--presence-z",
        type=float,
        default=120.0,
        help="umbral inicial de presencia; no es universal ni valida exactitud de campo",
    )
    _add_window_options(run)
    run.set_defaults(handler=_run)

    evaluate = commands.add_parser(
        "eval",
        description=(
            "Evalúa en test (una sola vez) un baseline mayoritario, un clasificador "
            "por umbral y un modelo softmax, tanto por ventana como en flujo. "
            "Los argumentos --train, --val y --test son listas separadas por coma "
            "de rutas de grabaciones JSONL etiquetadas; las sesiones deben ser "
            "disjuntas. Si se proporciona "
            "--val, tune_thresholds ajusta motion_z/presence_z exclusivamente con "
            "sus ventanas. El modelo softmax se carga sin reentrenar. El resultado "
            "se guarda en report.json/report.md y cada ejecución test se agrega a "
            "eval_log.jsonl; reutilizar modelo y sesiones de test se advierte."
        ),
    )
    evaluate.add_argument("--train", required=True, help="rutas JSONL separadas por coma")
    evaluate.add_argument(
        "--val",
        help="rutas JSONL separadas por coma; solo se usan para ajustar umbrales",
    )
    evaluate.add_argument("--test", required=True, help="rutas JSONL separadas por coma")
    evaluate.add_argument("--baseline", required=True, help="línea base calibrada JSON")
    evaluate.add_argument("--model", required=True, help="modelo softmax JSON ya entrenado")
    evaluate.add_argument("--out-dir", required=True)
    evaluate.add_argument(
        "--transition-tolerance-s",
        type=float,
        default=0.5,
        help="exclusión alrededor de transiciones reales para métricas de flujo",
    )
    evaluate.add_argument("--ema-alpha", type=float, default=0.4)
    evaluate.add_argument("--confirm-n", type=int, default=3)
    evaluate.add_argument("--confirm-n-to-absent", type=int, default=5)
    evaluate.add_argument("--margin", type=float, default=0.05)
    evaluate.add_argument("--stale-timeout-s", type=float, default=5.0)
    evaluate.add_argument("--motion-z", type=float, default=3.0)
    evaluate.add_argument("--presence-z", type=float, default=120.0)
    _add_window_options(evaluate)
    evaluate.set_defaults(handler=_eval)

    record = commands.add_parser("record")
    record.set_defaults(handler=_record)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la CLI y devuelve un código de salida."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())