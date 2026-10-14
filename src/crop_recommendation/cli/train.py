"""Train classical, stacking and deep models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from crop_recommendation.cli.common import add_data_arguments, add_output_argument, load_data
from crop_recommendation.config import METRICS_NAME, MODEL_BUNDLE_NAME
from crop_recommendation.data import PreparedData
from crop_recommendation.models.classical import MODEL_SPECS
from crop_recommendation.models.stacking import BEST_META, STACK_SPECS
from crop_recommendation.persistence import write_json
from crop_recommendation.training import (
    TrainingResult,
    search_classical,
    train_classical,
    train_stack,
)

LOGGER = logging.getLogger(__name__)

DEEP_CHOICES = ("cnn", "lstm", "cnn_lstm", "cnn_bilstm", "tabnet")


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Train a model.", description=__doc__)
    sub = parser.add_subparsers(dest="family", required=True, metavar="{classical,stack,deep}")

    classical = sub.add_parser("classical", help="Single scikit-learn / boosting classifier.")
    classical.add_argument("--model", choices=[*MODEL_SPECS, "all"], default="rf")
    classical.add_argument(
        "--variant",
        choices=["default", "tuned", "search"],
        default="tuned",
        help="default = library defaults (Table 4); tuned = recorded best parameters; "
        "search = re-run the hyper-parameter search (default: %(default)s).",
    )
    classical.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for searches.")
    _finish_parser(classical)

    stack = sub.add_parser("stack", help="Stacking ensemble (RF + GB + CatBoost base learners).")
    stack.add_argument(
        "--meta",
        choices=[*STACK_SPECS, "all"],
        default=BEST_META,
        help="Meta-learner (default: %(default)s, the paper's best).",
    )
    stack.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for the search.")
    _finish_parser(stack)

    deep = sub.add_parser("deep", help="Deep model (requires the 'deep' extra).")
    deep.add_argument("--model", choices=[*DEEP_CHOICES, "all"], default="tabnet")
    deep.add_argument(
        "--tune", action="store_true", help="Run the Optuna search first (not for TabNet)."
    )
    deep.add_argument(
        "--trials", type=int, default=20, help="Optuna trials (default: %(default)s)."
    )
    deep.add_argument("--device", default="auto", help="auto, cpu or cuda (default: %(default)s).")
    _finish_parser(deep)

    parser.set_defaults(handler=run)


def _finish_parser(parser: argparse.ArgumentParser) -> None:
    add_data_arguments(parser)
    add_output_argument(parser, "models")


def _save(result: TrainingResult, data: PreparedData, directory: Path) -> None:
    bundle_path = result.to_bundle(data).save(directory / MODEL_BUNDLE_NAME)
    write_json(directory / METRICS_NAME, result.to_dict())
    LOGGER.info("%s -> %s", result.name, bundle_path)


def _run_classical(args: argparse.Namespace, data: PreparedData) -> None:
    keys = list(MODEL_SPECS) if args.model == "all" else [args.model]
    for key in keys:
        if args.variant == "search":
            result = search_classical(data, key, seed=args.seed, n_jobs=args.n_jobs)
        else:
            result = train_classical(data, key, tuned=args.variant == "tuned", seed=args.seed)
        _save(result, data, args.output_dir / f"{key}-{args.variant}")


def _run_stack(args: argparse.Namespace, data: PreparedData) -> None:
    keys = list(STACK_SPECS) if args.meta == "all" else [args.meta]
    for key in keys:
        result = train_stack(data, key, seed=args.seed, n_jobs=args.n_jobs)
        _save(result, data, args.output_dir / f"stack-{key}")


def _run_deep(args: argparse.Namespace, data: PreparedData) -> None:
    try:
        import torch

        from crop_recommendation.models.deep.registry import DEEP_SPECS
        from crop_recommendation.models.deep.tabnet import fit_tabnet
        from crop_recommendation.models.deep.trainer import fit_deep, resolve_device
        from crop_recommendation.models.deep.tuning import tune
    except ImportError as error:
        raise SystemExit(
            "Deep models need the extra: pip install 'crop-recommendation[deep]'"
        ) from error

    device = resolve_device(args.device)
    keys = list(DEEP_CHOICES) if args.model == "all" else [args.model]
    for key in keys:
        directory = args.output_dir / key
        directory.mkdir(parents=True, exist_ok=True)
        if key == "tabnet":
            tabnet = fit_tabnet(data, seed=args.seed, device_name=args.device)
            tabnet.model.save_model(str(directory / "tabnet"))
            write_json(directory / METRICS_NAME, tabnet.to_dict())
            continue
        spec = DEEP_SPECS[key]
        overrides: dict[str, Any] | None = None
        if args.tune:
            study = tune(spec, data, seed=args.seed, n_trials=args.trials, device=device)
            overrides = dict(study.best_params)
        result = fit_deep(spec, data, seed=args.seed, device=device, overrides=overrides)
        torch.save(result.model.state_dict(), directory / "weights.pt")
        write_json(directory / METRICS_NAME, result.to_dict())


def run(args: argparse.Namespace) -> int:
    data = load_data(args)
    if args.family == "classical":
        _run_classical(args, data)
    elif args.family == "stack":
        _run_stack(args, data)
    else:
        _run_deep(args, data)
    return 0
