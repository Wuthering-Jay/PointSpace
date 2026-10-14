"""Audit 100 m Vaihingen test windows for a positive Fig. 8 row."""

import json
from pathlib import Path

import laspy

from plot_section45_positive_plan import _choose_local_crop, _evaluate
from plot_section45_qualitative import CACHE, _configuration, _predict


OUTPUT = CACHE / "vaihingen_100m_positive_scan.json"


def main():
    folder = Path(_configuration("vaihingen", "cvrl").test_data_dir)
    paths = sorted(folder.glob("*.las")) + sorted(folder.glob("*.laz"))
    screened = []
    for path in paths:
        with laspy.open(path) as source:
            header = source.header
            width = float(header.maxs[0] - header.mins[0])
            height = float(header.maxs[1] - header.mins[1])
        if min(width, height) >= 99:
            screened.append((path, width, height))
    results = []
    for index, (path, width, height) in enumerate(screened, 1):
        row = _evaluate("vaihingen", path)
        row["extent_xy_m"] = [round(width, 3), round(height, 3)]
        predictions = {method: _predict("vaihingen", method, path)
                       for method in ("sonata", "vpsd", "cvrl")}
        try:
            row["best_100m_crop"] = _choose_local_crop(
                "vaihingen", path, predictions, size_m=100,
                minimum_coverage=.60, minimum_span=95,
            )
        except RuntimeError:
            row["best_100m_crop"] = None
        results.append(row)
        print(index, path.name, row["coverage_10x10"],
              row["gain_vs_strong_baseline"],
              None if row["best_100m_crop"] is None else
              row["best_100m_crop"]["gain_vs_strong_baseline"], flush=True)
    results.sort(key=lambda row: -1 if row["best_100m_crop"] is None else
                 row["best_100m_crop"]["gain_vs_strong_baseline"], reverse=True)
    OUTPUT.write_text(json.dumps({
        "protocol": "GT-size-screened Vaihingen test tiles; 100 m crop with at "
                    "least 60% 10x10 grid occupancy and 95 m point span per axis; "
                    "decoder-probe mIoU; targeted positive selection, "
                    "not representative test accuracy",
        "test_tiles": len(paths), "size_screened_tiles": len(screened),
        "results": results,
    }, indent=2), encoding="utf-8")
    print("best 100 m crops:", flush=True)
    for row in results[:15]:
        crop = row["best_100m_crop"]
        if crop is not None:
            print(Path(row["path"]).name, crop["gain_vs_strong_baseline"],
                  crop["coverage_10x10"], crop["points"], flush=True)


if __name__ == "__main__":
    main()
