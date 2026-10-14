"""Screen OpenGF test sections for structurally varied qualitative cases.

Geometry and labels define the initial tile pool; model predictions are then
used only for the explicitly targeted Fig. 9 success-case selection. This is
not an unbiased evaluation protocol.
"""

import json
from pathlib import Path

import numpy as np

from plot_section45_qualitative import CACHE, _configuration, _mapped_labels, _predict, _raw_las
from plot_section45_vertical_qualitative import _section_candidates, _section_mask


OUTPUT = CACHE / "opengf_urban_profile_scan.json"


def main():
    folder = Path(_configuration("opengf", "cvrl").test_data_dir)
    paths = sorted(folder.glob("*.las")) + sorted(folder.glob("*.laz"))
    screened = []
    for path in paths:
        xyz, raw = _raw_las(path)
        gt = _mapped_labels(raw, "opengf")
        valid = gt >= 0
        if valid.sum() < 1500:
            continue
        non_ground = float(np.mean(gt[valid] == 0))
        relief = float(np.percentile(xyz[valid, 2], 95) - np.percentile(xyz[valid, 2], 5))
        if .20 <= non_ground <= .85 and relief >= 6:
            screened.append((path, non_ground, relief))
    print("GT-screened", len(screened), "of", len(paths), flush=True)
    eligible = []
    for index, (path, _, _) in enumerate(screened, 1):
        predictions = {method: _predict("opengf", method, path)
                       for method in ("sonata", "vpsd", "cvrl")}
        rows = _section_candidates("opengf", path, predictions)
        xyz, raw = _raw_las(path)
        gt = _mapped_labels(raw, "opengf")
        for row in rows:
            if row["points"] < 230 or row["z_p95_p05"] < 6:
                continue
            non_ground = row["class_proportions"][0]
            if not (.25 <= non_ground <= .8):
                continue
            acc = row["accuracy"]
            gain = acc["cvrl"] - max(acc["sonata"], acc["vpsd"])
            if gain < .055:
                continue
            # An urban-like section should not be a single uniformly rising
            # slope: require ground and non-ground to coexist in several bins.
            inside = _section_mask(xyz, gt, row)
            axis = 0 if row["horizontal_axis"] == "x" else 1
            along = xyz[inside, axis] - row["start"]
            label = gt[inside]
            overlap_bins = 0
            mixed_bins = 0
            for bin_index in range(10):
                in_bin = (along >= bin_index * 5) & (along < (bin_index + 1) * 5)
                if in_bin.sum() < 8:
                    continue
                n_non_ground = int(np.sum(label[in_bin] == 0))
                n_ground = int(np.sum(label[in_bin] == 1))
                if n_non_ground >= 3 and n_ground >= 3:
                    mixed_bins += 1
                    z_non_ground = xyz[inside, 2][in_bin & (label == 0)]
                    z_ground = xyz[inside, 2][in_bin & (label == 1)]
                    if np.median(z_non_ground) - np.median(z_ground) >= 2:
                        overlap_bins += 1
            if overlap_bins < 2:
                continue
            row["gain_vs_strongest"] = round(gain, 6)
            row["mixed_bins"] = mixed_bins
            row["vertical_overlap_bins"] = overlap_bins
            row["screen_score"] = round(gain * np.sqrt(row["points"]) *
                                        (1 + .1 * overlap_bins), 6)
            eligible.append(row)
        if index % 100 == 0 or index == len(screened):
            print("screened", index, flush=True)
    eligible.sort(key=lambda row: row["screen_score"], reverse=True)
    OUTPUT.write_text(json.dumps({"protocol": "GT-first varied-terrain screen, "
                                       "then targeted CVRL-positive selection; not "
                                       "representative of the OpenGF test set",
                                  "test_tiles": len(paths),
                                  "gt_screened_tiles": len(screened),
                                  "eligible": eligible}, indent=2), encoding="utf-8")
    print("eligible", len(eligible), flush=True)
    for row in eligible[:25]:
        print(Path(row["path"]).name, row["gain_vs_strongest"],
              row["points"], row["z_p95_p05"],
              row["class_proportions"][0], row["vertical_overlap_bins"], flush=True)


if __name__ == "__main__":
    main()
