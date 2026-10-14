"""Audit mountainous OpenGF sections for a visual-vs-CVRL comparison.

This only scans already GT-screened test tiles and cached decoder-probe
predictions. It does not change the final Fig. 9 selection audit.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_section45_qualitative import CACHE, _configuration, _predict, _raw_las, _scene_score
from plot_section45_vertical_qualitative import _render_row, _section_candidates


MOUNTAIN_POOL = 1100
MIN_TILE_RELIEF = 20.0
MIN_TILE_VEGETATION = 0.10
MIN_SECTION_RELIEF = 10.0
MIN_SECTION_VEGETATION = 0.10
MIN_CVRL_ADVANTAGE = 0.04
OUTPUT = CACHE / "opengf_mountain_profile_scan.json"
PREVIEW = CACHE / "opengf_mountain_profile_preview.png"


def main():
    folder = Path(_configuration("opengf", "cvrl").test_data_dir)
    paths = sorted(folder.glob("*.las")) + sorted(folder.glob("*.laz"))
    candidates = []
    for path in paths:
        xyz, raw = _raw_las(path)
        _, stats = _scene_score("opengf", xyz, raw)
        relief = stats.get("z_p95_p05", 0)
        vegetation = stats.get("proportions", [0])[0]
        if relief >= MIN_TILE_RELIEF and vegetation >= MIN_TILE_VEGETATION:
            candidates.append((vegetation * min(relief / 30, 2), path, stats))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    print("GT-screened tiles", len(candidates), "of", len(paths), flush=True)
    sections = []
    for index, (_, path, _) in enumerate(candidates[:MOUNTAIN_POOL], start=1):
        predictions = {method: _predict("opengf", method, path)
                       for method in ("sonata", "vpsd", "cvrl")}
        found = _section_candidates("opengf", path, predictions)
        sections.extend(found)
        if index % 50 == 0 or index == len(candidates[:MOUNTAIN_POOL]):
            print("scanned", index, "tiles", flush=True)
    eligible = []
    for row in sections:
        if row["z_p95_p05"] < MIN_SECTION_RELIEF:
            continue
        if row["class_proportions"][0] < MIN_SECTION_VEGETATION:
            continue
        acc = row["accuracy"]
        advantage = acc["cvrl"] - max(acc["sonata"], acc["vpsd"])
        if advantage < MIN_CVRL_ADVANTAGE:
            continue
        row["gain_vs_strongest_visual_or_3d"] = round(advantage, 6)
        row["net_corrected_points_vs_strongest"] = round(advantage * row["points"], 2)
        row["mountain_score"] = round(
            advantage * row["points"] ** .5 * min(row["z_p95_p05"] / 20, 1.5)
            + .012 * row["cvrl_correct_other_two_wrong"] ** .5, 6)
        eligible.append(row)
    eligible.sort(key=lambda row: row["mountain_score"], reverse=True)
    OUTPUT.write_text(json.dumps({"protocol": "All OpenGF test tiles GT-scanned; "
                                       "tiles with >=20 m relief and >=10% non-ground "
                                       "selected without model outputs; 50 m slab; "
                                       "CVRL gain vs max(Sonata,VPSD); "
                                       "not a full-test-set estimate",
                                  "tiles_scanned": len(candidates[:MOUNTAIN_POOL]),
                                  "sections_scanned": len(sections),
                                  "eligible": eligible}, indent=2), encoding="utf-8")
    print("ELIGIBLE", len(eligible), flush=True)
    for row in eligible[:20]:
        print(Path(row["path"]).name, row["gain_vs_strongest_visual_or_3d"],
              row["points"], row["class_proportions"][0], row["z_p95_p05"],
              row["cvrl_correct_other_two_wrong"], row["mountain_score"], flush=True)
    if eligible:
        preview_rows = []
        seen_tiles = set()
        for candidate in eligible:
            tile = candidate["path"]
            if tile not in seen_tiles:
                preview_rows.append(candidate)
                seen_tiles.add(tile)
            if len(preview_rows) >= 5:
                break
        fig, axes = plt.subplots(len(preview_rows), 6,
                                 figsize=(14, 2.6 * len(preview_rows)),
                                 squeeze=False)
        titles = ("GT", "Scratch", "Sonata", "D-DITR", "VPSD", "CVRL")
        for ax, title in zip(axes[0], titles):
            ax.set_title(title)
        for row_index, choice in enumerate(preview_rows):
            _render_row(axes[row_index], "opengf", choice)
        fig.tight_layout()
        fig.savefig(PREVIEW, dpi=200)
        plt.close(fig)
        print(PREVIEW, flush=True)


if __name__ == "__main__":
    main()
