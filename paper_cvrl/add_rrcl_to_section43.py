"""Add independently pretrained RRCL to the existing Swiss-Audit protocol.

This incremental run reuses the saved Section 4.3 settings and sample seeds;
it does not refit the already completed teacher/VPSD/CVRL probes. Run in IDE.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from analyze_cvrl_43 import (
    LOGGER,
    _build_rows,
    _fit_probe,
    _load_feature_source,
    _manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
REPORT_PATH = AUDIT_ROOT / "analysis" / "section_43_report.json"
CHECKPOINT = (
    REPOSITORY_ROOT
    / "exp/rrcl/swiss/pretrain-v1m1-litept-v1m4-100/model/model_last.pth"
)
FEATURE_DIR = AUDIT_ROOT / "features" / "rrcl_independent"
SOURCE_NAME = "rrcl_independent72"
PREDICTION_PATH = (
    AUDIT_ROOT / "analysis" / "predictions"
    / f"regional_{SOURCE_NAME}_linear.npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    settings = {
        key: report["protocol"][key]
        for key in (
            "test_fraction", "max_train_points", "max_test_points",
            "max_knn_train_points", "max_knn_test_points", "linear_c",
            "linear_max_iter", "knn_k", "random_state",
        )
    }
    manifest = _manifest(AUDIT_ROOT / "audit_manifest.csv")
    actual = {path.stem for path in FEATURE_DIR.glob("*.safetensors")}
    expected = {row["stem"] for row in manifest}
    if actual != expected:
        raise ValueError(
            f"RRCL export does not match manifest: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    oldest_export = min(
        (FEATURE_DIR / f"{stem}.safetensors").stat().st_mtime
        for stem in expected
    )
    if oldest_export < CHECKPOINT.stat().st_mtime:
        raise ValueError("Some RRCL features predate the selected checkpoint")

    LOGGER.info("Building %s common audit rows", len(manifest))
    rows = _build_rows(AUDIT_ROOT, manifest)
    feature = _load_feature_source(
        AUDIT_ROOT,
        {"kind": "point", "folder": "features/rrcl_independent"},
        rows,
    )
    if feature.shape != (rows["label"].size, 72):
        raise ValueError(f"Unexpected independent RRCL shape: {feature.shape}")
    if not np.isfinite(feature).all():
        raise ValueError("Independent RRCL feature contains NaN/Inf")
    LOGGER.info("Fitting independent RRCL frozen linear and cosine kNN")
    result = _fit_probe(
        feature,
        rows,
        np.ones(rows["label"].shape, dtype=bool),
        settings,
        seed_offset=100,
        prediction_path=PREDICTION_PATH,
    )
    reference_path = (
        AUDIT_ROOT / "analysis" / "predictions"
        / "regional_cvrl_full200_linear.npz"
    )
    with np.load(PREDICTION_PATH) as current, np.load(reference_path) as reference:
        for key in ("audit_row_index", "stem", "point_index", "target", "region"):
            if not np.array_equal(current[key], reference[key]):
                raise ValueError(f"RRCL/CVRL probe rows differ: {key}")

    report["regional_compensation"][SOURCE_NAME] = result
    report["schema"] = "cvrl_section_43_v3_rrcl_independent"
    report["independent_rrcl_provenance"] = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "feature_directory": str(FEATURE_DIR),
        "feature_files": len(actual),
        "source_block_split": "same as existing regional probes",
        "sample_rows": "exactly matched to regional_cvrl_full200_linear.npz",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = REPORT_PATH.with_name(f".{REPORT_PATH.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(REPORT_PATH)
    LOGGER.info(
        "RRCL independent: linear macro-F1 %.4f, kNN macro-F1 %.4f",
        result["linear"]["all"]["macro_F1"],
        result["knn"]["all"]["macro_F1"],
    )
    return result


if __name__ == "__main__":
    run()
