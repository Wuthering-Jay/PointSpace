"""Add exact mIoU to existing Swiss-Audit probe results without refitting probes.

Run directly from an IDE after ``analyze_cvrl_43.py``. The stored confusion
matrices and fixed test rows are reused; legacy macro-F1 remains in the raw
JSON as an auxiliary metric, while Section 4.3 reports mIoU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paper_cvrl.section43_metrics import mean_iou, per_class_iou


REPORT_PATH = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit\analysis\section_43_report.json")


def main():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    updated = 0
    for family in ("internalization", "regional_compensation"):
        for source in report[family].values():
            for probe in ("linear", "knn"):
                for metrics in source[probe].values():
                    if "confusion_matrix" not in metrics:
                        continue
                    metrics["mIoU"] = mean_iou(metrics["confusion_matrix"])
                    class_iou = per_class_iou(metrics["confusion_matrix"])
                    for index, values in enumerate(metrics["per_class"].values()):
                        values["IoU"] = (
                            float(class_iou[index])
                            if np.isfinite(class_iou[index]) else None
                        )
                    updated += 1
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Updated {updated} fixed-row probe metric blocks: {REPORT_PATH}")


if __name__ == "__main__":
    main()
