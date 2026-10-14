"""Rank scanned OpenGF mountain sections by visible spatial corrections."""

import json
from pathlib import Path

import numpy as np

from plot_section45_qualitative import CACHE, _mapped_labels, _predict, _raw_las
from plot_section45_vertical_qualitative import _section_mask


def main():
    source = json.loads((CACHE / "opengf_mountain_profile_scan.json").read_text(encoding="utf-8"))
    rows = [row for row in source["eligible"]
            if row["z_p95_p05"] >= 20 and row["gain_vs_strongest_visual_or_3d"] >= .04]
    tile_cache = {}
    for row in rows:
        path = Path(row["path"])
        if path not in tile_cache:
            xyz, raw = _raw_las(path)
            gt = _mapped_labels(raw, "opengf")
            pred = {method: _predict("opengf", method, path)
                    for method in ("scratch", "sonata", "d-ditr", "vpsd", "cvrl")}
            tile_cache[path] = xyz, gt, pred
        xyz, gt, pred = tile_cache[path]
        inside = _section_mask(xyz, gt, row)
        indices = np.flatnonzero(inside)
        axis = 0 if row["horizontal_axis"] == "x" else 1
        along = xyz[indices, axis] - row["start"]
        cvrl_correct = pred["cvrl"][indices] == gt[indices]
        row["all_model_accuracy"] = {
            method: round(float(np.mean(pred[method][indices] == gt[indices])), 6)
            for method in pred
        }
        row["gain_vs_all"] = round(
            row["all_model_accuracy"]["cvrl"] -
            max(row["all_model_accuracy"][method] for method in
                ("scratch", "sonata", "d-ditr", "vpsd")), 6)
        counts = {}
        for method in ("sonata", "vpsd"):
            corrected = cvrl_correct & (pred[method][indices] != gt[indices])
            bins = np.bincount(np.clip((along[corrected] / 5).astype(int), 0, 9),
                               minlength=10)
            counts[method] = {"total": int(corrected.sum()),
                              "max_5m": int(bins.max()),
                              "bins": bins.tolist()}
        row["corrected_clusters"] = counts
    rows.sort(key=lambda row: (
        min(row["corrected_clusters"][method]["max_5m"] for method in ("sonata", "vpsd")),
        row["gain_vs_strongest_visual_or_3d"],
    ), reverse=True)
    for row in rows:
        print(Path(row["path"]).name,
              "gain", row["gain_vs_strongest_visual_or_3d"],
              "relief", row["z_p95_p05"],
              "veg", row["class_proportions"][0],
              "gain_all", row["gain_vs_all"],
              "sonata", row["corrected_clusters"]["sonata"],
              "vpsd", row["corrected_clusters"]["vpsd"])
    (CACHE / "opengf_profile_cluster_rank.json").write_text(
        json.dumps({"rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
