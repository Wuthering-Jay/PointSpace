"""Verify and preview leading OpenGF urban-like Fig. 9 candidates."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_section45_qualitative import CACHE, _predict
from plot_section45_vertical_qualitative import _render_row


def main():
    source = json.loads((CACHE / "opengf_urban_profile_scan.json").read_text(encoding="utf-8"))
    candidates = []
    seen = set()
    for row in source["eligible"]:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        predictions = {method: _predict("opengf", method, Path(row["path"]))
                       for method in ("scratch", "sonata", "d-ditr", "vpsd", "cvrl")}
        from plot_section45_qualitative import _mapped_labels, _raw_las
        from plot_section45_vertical_qualitative import _section_mask
        xyz, raw = _raw_las(Path(row["path"]))
        gt = _mapped_labels(raw, "opengf")
        inside = _section_mask(xyz, gt, row)
        row["all_model_accuracy"] = {
            method: round(float((prediction[inside] == gt[inside]).mean()), 6)
            for method, prediction in predictions.items()
        }
        row["gain_vs_all"] = round(
            row["all_model_accuracy"]["cvrl"] -
            max(row["all_model_accuracy"][method] for method in
                ("scratch", "sonata", "d-ditr", "vpsd")), 6)
        candidates.append(row)
        if len(candidates) >= 8:
            break
    candidates.sort(key=lambda row: row["gain_vs_all"] * row["points"] ** .5,
                    reverse=True)
    (CACHE / "opengf_urban_fullmodel_rank.json").write_text(
        json.dumps({"candidates": candidates}, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(len(candidates), 6,
                             figsize=(15, 2.4 * len(candidates)), squeeze=False)
    for index, row in enumerate(candidates):
        _render_row(axes[index], "opengf", row)
        axes[index, 0].text(.02, .96, Path(row["path"]).stem,
                            transform=axes[index, 0].transAxes,
                            va="top", ha="left", fontsize=9)
        print(Path(row["path"]).name, row["gain_vs_all"],
              row["all_model_accuracy"], flush=True)
    for ax, title in zip(axes[0],
                         ("GT", "Scratch", "Sonata", "D-DITR", "VPSD", "CVRL")):
        ax.set_title(title)
    fig.tight_layout()
    path = CACHE / "opengf_urban_profile_preview.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
