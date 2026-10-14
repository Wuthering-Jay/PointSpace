"""Paired source-block bootstrap confidence intervals for Section 4.3.

The resampling unit is the original Swiss source block, never an individual
point.  Paired methods reuse exactly the same test rows, so each bootstrap
replicate measures a paired mIoU difference with spatial dependence kept
inside blocks.  Configuration is module-level for direct IDE execution.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from paper_cvrl.section43_metrics import mean_iou


AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
PREDICTION_DIR = AUDIT_ROOT / "analysis" / "predictions"
REPORT_PATH = AUDIT_ROOT / "analysis" / "section_43_report.json"
OUTPUT_PATH = AUDIT_ROOT / "analysis" / "section_43_block_bootstrap.json"
OUTPUT_CSV = Path(__file__).resolve().parent / "results/section_43/section_43_block_bootstrap.csv"
CLASS_OUTPUT_CSV = Path(__file__).resolve().parent / "results/section_43/section_43_class_bootstrap.csv"
N_BOOTSTRAP = 2000
RANDOM_STATE = 42
REGIONS = {"all": None, "P_inv": 0, "P_amb": 1, "P_vis": 2}

COMPARISONS = {
    "vpsd_minus_scratch": (
        "regional_vpsd_pca128_linear.npz",
        "regional_vpsd_scratch_pca128_linear.npz",
    ),
    "full_minus_visual": (
        "regional_cvrl_full200_linear.npz",
        "regional_cvrl_visual_pca128_linear.npz",
    ),
    "full_minus_vpsd": (
        "regional_cvrl_full200_linear.npz",
        "regional_vpsd_pca128_linear.npz",
    ),
    "full_minus_rrcl_independent": (
        "regional_cvrl_full200_linear.npz",
        "regional_rrcl_independent72_linear.npz",
    ),
    "joint_return_minus_rrcl_independent": (
        "regional_cvrl_return72_linear.npz",
        "regional_rrcl_independent72_linear.npz",
    ),
}
CLASS_IDS = (2, 3, 6, 17, 26)
CLASS_COMPARISONS = {
    "rrcl_independent_minus_vpsd": (
        "regional_rrcl_independent72_linear.npz",
        "regional_vpsd_pca128_linear.npz",
    ),
    "full_minus_vpsd": COMPARISONS["full_minus_vpsd"],
}


def _source_block_lookup():
    with (AUDIT_ROOT / "audit_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        return {row["stem"]: row["source_block"] for row in csv.DictReader(stream)}


def _load(name: str):
    with np.load(PREDICTION_DIR / name) as handle:
        return {key: handle[key] for key in handle.files}


def _macro_f1(confusion: np.ndarray) -> float:
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    return float(f1.mean())


def _single_class_f1(confusion: np.ndarray, index: int) -> float:
    true_positive = float(confusion[index, index])
    denominator = float(confusion[index].sum() + confusion[:, index].sum())
    return 2 * true_positive / denominator if denominator else 0.0


def _bootstrap_class_pair(first, second, blocks, class_id, rng):
    for key in ("stem", "point_index", "target", "region"):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"Paired class prediction rows differ for {key}")
    mask = first["region"] == 0
    labels = np.unique(np.r_[first["target"], first["prediction"], second["prediction"]])
    class_index = int(np.flatnonzero(labels == class_id)[0])
    block_names = np.unique(blocks[mask])
    confusions = []
    for prediction in (first["prediction"], second["prediction"]):
        confusions.append(
            np.stack([
                confusion_matrix(
                    first["target"][mask & (blocks == block)],
                    prediction[mask & (blocks == block)],
                    labels=labels,
                )
                for block in block_names
            ])
        )
    point_estimate = (
        _single_class_f1(confusions[0].sum(axis=0), class_index)
        - _single_class_f1(confusions[1].sum(axis=0), class_index)
    )
    differences = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample = rng.integers(0, len(block_names), size=len(block_names))
        differences[index] = (
            _single_class_f1(confusions[0][sample].sum(axis=0), class_index)
            - _single_class_f1(confusions[1][sample].sum(axis=0), class_index)
        )
    return {
        "class_id": class_id,
        "P_inv_support": int(np.sum(mask & (first["target"] == class_id))),
        "point_estimate": float(point_estimate),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "blocks": len(block_names),
    }


def _bootstrap_pair(first, second, blocks, region_id, rng):
    for key in ("stem", "point_index", "target", "region"):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"Paired prediction rows differ for {key}")
    mask = np.ones(first["target"].shape, dtype=bool)
    if region_id is not None:
        mask &= first["region"] == region_id
    labels = np.unique(
        np.r_[first["target"][mask], first["prediction"][mask], second["prediction"][mask]]
    )
    block_names = np.unique(blocks[mask])
    first_confusions = []
    second_confusions = []
    points_per_block = []
    for block in block_names:
        take = mask & (blocks == block)
        first_confusions.append(
            confusion_matrix(first["target"][take], first["prediction"][take], labels=labels)
        )
        second_confusions.append(
            confusion_matrix(first["target"][take], second["prediction"][take], labels=labels)
        )
        points_per_block.append(int(take.sum()))
    first_confusions = np.stack(first_confusions)
    second_confusions = np.stack(second_confusions)
    point_difference = mean_iou(first_confusions.sum(axis=0)) - mean_iou(
        second_confusions.sum(axis=0)
    )
    differences = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for index in range(N_BOOTSTRAP):
        sample = rng.integers(0, block_names.size, size=block_names.size)
        differences[index] = mean_iou(first_confusions[sample].sum(axis=0)) - mean_iou(
            second_confusions[sample].sum(axis=0)
        )
    return {
        "blocks": int(block_names.size),
        "points": int(sum(points_per_block)),
        "point_estimate": float(point_difference),
        "bootstrap_mean": float(differences.mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "probability_positive": float((differences > 0).mean()),
        "n_bootstrap": N_BOOTSTRAP,
    }


def main():
    lookup = _source_block_lookup()
    rng = np.random.default_rng(RANDOM_STATE)
    result = {
        "schema": "cvrl_section_43_paired_block_bootstrap_v2",
        "metric": "mIoU for regional comparisons; F1 for class comparisons",
        "resampling_unit": "original Swiss source block",
        "random_state": RANDOM_STATE,
        "comparisons": {},
        "class_comparisons": {},
    }
    for name, (first_name, second_name) in COMPARISONS.items():
        first = _load(first_name)
        second = _load(second_name)
        blocks = np.asarray([lookup[str(stem)] for stem in first["stem"]], dtype=str)
        result["comparisons"][name] = {
            region_name: _bootstrap_pair(first, second, blocks, region_id, rng)
            for region_name, region_id in REGIONS.items()
        }
    for name, (first_name, second_name) in CLASS_COMPARISONS.items():
        first = _load(first_name)
        second = _load(second_name)
        blocks = np.asarray([lookup[str(stem)] for stem in first["stem"]], dtype=str)
        result["class_comparisons"][name] = {
            str(class_id): _bootstrap_class_pair(
                first, second, blocks, class_id, rng
            )
            for class_id in CLASS_IDS
        }
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    report["paired_block_bootstrap"] = result
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["comparison", "metric", "region", "point_estimate", "ci95_low", "ci95_high", "p_positive", "blocks", "points"]
        )
        for comparison, regions in result["comparisons"].items():
            for region, values in regions.items():
                writer.writerow(
                    [
                        comparison,
                        "mIoU",
                        region,
                        values["point_estimate"],
                        values["ci95_low"],
                        values["ci95_high"],
                        values["probability_positive"],
                        values["blocks"],
                        values["points"],
                    ]
                )
    with CLASS_OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("comparison", "metric", "class_id", "P_inv_support", "point_estimate",
             "ci95_low", "ci95_high", "blocks")
        )
        for comparison, classes in result["class_comparisons"].items():
            for values in classes.values():
                writer.writerow(
                    (comparison, "class_F1", values["class_id"], values["P_inv_support"],
                     values["point_estimate"], values["ci95_low"],
                     values["ci95_high"], values["blocks"])
                )
    print(
        f"Wrote mIoU block bootstrap for {len(result['comparisons'])} paired "
        f"comparisons and class-F1 bootstrap for "
        f"{len(result['class_comparisons'])} comparisons: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
