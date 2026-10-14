"""Read-only integrity check for the organized Section 4.3 deliverables."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paper_cvrl.section43_metrics import mean_iou


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results" / "section_43"
AUDIT_REPORT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit\analysis\section_43_report.json")
DOCUMENTS = (
    BASE / "CVRL_4.3_实验分析报告.md",
    BASE / "section43_main_text_figures.md",
)
FINAL_FIGURES = (
    "figure_5_visibility_bottleneck.png",
    "figure_6_projection_conflict_mechanism.png",
)


def main():
    for name in FINAL_FIGURES:
        if not (RESULTS / name).is_file():
            raise FileNotFoundError(RESULTS / name)
    linked = []
    for document in DOCUMENTS:
        content = document.read_text(encoding="utf-8")
        for path in re.findall(r"!\[[^]]*\]\(([^)]+)\)", content):
            target = document.parent / path
            if not target.is_file():
                raise FileNotFoundError(f"{document.name}: {target}")
            linked.append(target)
    report = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    metric_blocks = 0
    for family in ("internalization", "regional_compensation"):
        for source in report[family].values():
            for probe in ("linear", "knn"):
                for block in source[probe].values():
                    if "confusion_matrix" not in block:
                        continue
                    if abs(block["mIoU"] - mean_iou(block["confusion_matrix"])) > 1e-12:
                        raise ValueError("Stale mIoU in Swiss-Audit JSON")
                    metric_blocks += 1
    with (RESULTS / "section_43_block_bootstrap.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["metric"] != "mIoU" for row in rows):
        raise ValueError("Block bootstrap CSV is not explicitly mIoU")
    print(
        f"Section 4.3 assets valid: {len(linked)} linked figures, "
        f"{metric_blocks} fixed-row mIoU blocks, {len(rows)} paired CI rows."
    )


if __name__ == "__main__":
    main()
