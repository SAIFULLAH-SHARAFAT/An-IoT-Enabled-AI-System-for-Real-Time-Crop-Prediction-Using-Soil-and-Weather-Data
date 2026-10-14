"""Re-run the paper's experiments and compare with the published numbers."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass

from crop_recommendation.cli.common import add_data_arguments, add_output_argument, load_data
from crop_recommendation.data import PreparedData
from crop_recommendation.models.classical import MODEL_SPECS
from crop_recommendation.models.stacking import BEST_META, STACK_SPECS
from crop_recommendation.persistence import write_json
from crop_recommendation.reference import (
    FIG13_TUNED_ACCURACY,
    TABLE4_DEFAULT_ACCURACY,
    TABLE5_STACKING,
    TABLE6_DEEP,
)
from crop_recommendation.training import train_classical, train_stack

LOGGER = logging.getLogger(__name__)

SECTIONS = ("table4", "fig13", "table5", "table6")


@dataclass(frozen=True)
class Row:
    section: str
    item: str
    paper: float
    reproduced: float
    note: str = ""

    @property
    def delta_points(self) -> float:
        return (self.reproduced - self.paper) * 100


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "reproduce", help="Reproduce the paper's result tables.", description=__doc__
    )
    parser.add_argument(
        "--only", nargs="+", choices=SECTIONS, default=list(SECTIONS), help="Sections to run."
    )
    parser.add_argument(
        "--all-meta",
        action="store_true",
        help=f"Table 5: run every meta-learner instead of only '{BEST_META}' (slow).",
    )
    add_data_arguments(parser)
    add_output_argument(parser, "reports/reproduction")
    parser.set_defaults(handler=run)


def _table4(data: PreparedData, seed: int) -> list[Row]:
    rows = []
    for key, paper in TABLE4_DEFAULT_ACCURACY.items():
        result = train_classical(data, key, tuned=False, seed=seed)
        rows.append(
            Row("Table 4 (default)", MODEL_SPECS[key].label, paper, result.test_metrics.accuracy)
        )
    return rows


def _fig13(data: PreparedData, seed: int) -> list[Row]:
    rows = []
    for key, paper in FIG13_TUNED_ACCURACY.items():
        result = train_classical(data, key, tuned=True, seed=seed)
        rows.append(
            Row("Fig. 13 (tuned)", MODEL_SPECS[key].label, paper, result.test_metrics.accuracy)
        )
    return rows


def _table5(data: PreparedData, seed: int, *, all_meta: bool) -> list[Row]:
    keys = list(STACK_SPECS) if all_meta else [BEST_META]
    rows = []
    for key in keys:
        result = train_stack(data, key, seed=seed)
        paper_test, paper_cv = TABLE5_STACKING[key]
        label = STACK_SPECS[key].label
        rows.append(
            Row("Table 5 (test)", f"Stack -> {label}", paper_test, result.test_metrics.accuracy)
        )
        rows.append(
            Row(
                "Table 5 (CV)",
                f"Stack -> {label}",
                paper_cv,
                float(result.details["best_cv_accuracy"]),
            )
        )
    return rows


def _table6(data: PreparedData, seed: int) -> list[Row]:
    try:
        from crop_recommendation.models.deep.registry import DEEP_SPECS
        from crop_recommendation.models.deep.tabnet import fit_tabnet
        from crop_recommendation.models.deep.trainer import fit_deep
    except ImportError:
        LOGGER.warning("Skipping Table 6: install 'crop-recommendation[deep]'")
        return []
    rows = []
    tabnet = fit_tabnet(data, seed=seed)
    rows.append(
        Row(
            "Table 6 (deep)",
            "TabNet",
            TABLE6_DEEP["tabnet"][0],
            tabnet.test_metrics.accuracy,
            f"{tabnet.n_parameters} params",
        )
    )
    for key, spec in DEEP_SPECS.items():
        result = fit_deep(spec, data, seed=seed)
        epoch_mean = result.epoch_mean_test_accuracy * 100
        note = f"{result.n_parameters} params; mean over epochs {epoch_mean:.1f}%"
        if spec.reconstructed:
            note += "; hyper-parameters reconstructed"
        rows.append(
            Row(
                "Table 6 (deep)",
                spec.label,
                TABLE6_DEEP[key][0],
                result.final_metrics.accuracy,
                note,
            )
        )
    return rows


def render_markdown(rows: list[Row]) -> str:
    lines = [
        "| Section | Item | Paper | Reproduced | Delta (pt) | Note |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines += [
        f"| {r.section} | {r.item} | {r.paper * 100:.1f} | {r.reproduced * 100:.1f} | "
        f"{r.delta_points:+.1f} | {r.note} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    data = load_data(args)
    runners: dict[str, Callable[[], list[Row]]] = {
        "table4": lambda: _table4(data, args.seed),
        "fig13": lambda: _fig13(data, args.seed),
        "table5": lambda: _table5(data, args.seed, all_meta=args.all_meta),
        "table6": lambda: _table6(data, args.seed),
    }
    rows: list[Row] = []
    for section in SECTIONS:
        if section in args.only:
            rows += runners[section]()

    markdown = render_markdown(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    write_json(
        args.output_dir / "comparison.json",
        [{**asdict(row), "delta_points": round(row.delta_points, 2)} for row in rows],
    )
    print(markdown)  # noqa: T201 - primary command output
    return 0
