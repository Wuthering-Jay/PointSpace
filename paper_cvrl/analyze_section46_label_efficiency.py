"""Parse the completed DALES label-efficiency experiments and write Section 4.6.

The four sparse-label runs are the controlled experiment.  The 100% values are
read from the earlier Section 4.5 FT report and remain explicitly marked as a
reference endpoint because its training schedule differs.

Run from the repository root::

    python paper_cvrl/analyze_section46_label_efficiency.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "exp" / "label_efficiency" / "dales"
OUT = ROOT / "paper_cvrl" / "results" / "section_46"

METHODS = (
    "scratch", "point-m2ae", "msc", "sonata", "d-ditr", "concerto",
    "vpsd", "rrcl", "cvrl",
)
METHOD_LABELS = {
    "scratch": "Scratch", "point-m2ae": "Point-M2AE", "msc": "MSC",
    "sonata": "Sonata", "d-ditr": "D-DITR", "concerto": "Concerto",
    "vpsd": "VPSD", "rrcl": "RRCL", "cvrl": "CVRL",
}
FRACTIONS = (0.0001, 0.001, 0.01, 0.1)
DISPLAY_FRACTIONS = ("0.01%", "0.1%", "1%", "10%")
ACTUAL_COUNTS = {
    0.0001: 36_334, 0.001: 362_513, 0.01: 3_618_303, 0.1: 36_165_902,
}
ACTUAL_RATIOS = {
    0.0001: 0.01005, 0.001: 0.10023, 0.01: 1.00044, 0.1: 9.99970,
}
# Existing 4.5 FT endpoint. These were trained with 10 epochs x 5 loops,
# whereas this section uses 20 epochs x 3 loops plus early stopping.
FULL_REFERENCE = {
    "scratch": 0.5806, "point-m2ae": 0.6411, "msc": 0.7959,
    "sonata": 0.7890, "d-ditr": 0.6908, "concerto": 0.5461,
    "vpsd": 0.7969, "rrcl": 0.7996, "cvrl": 0.7928,
}
METRIC_RE = re.compile(r"\b(mIoU|mAcc|OA|mF1|fwIoU|Kappa)\s+([0-9.]+)")
CLASS_RE = re.compile(
    r"Class\s+(\d+)\s+-\s+(.+?)\s+\|\s+IoU\s+([0-9.]+)"
    r"\s+Prec\s+([0-9.]+)\s+Recall\s+([0-9.]+)\s+F1\s+([0-9.]+)"
)


def parse_log(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    vals, tests, all_classes = [], [], []
    for index, line in enumerate(lines):
        if "Val result:" in line:
            vals.append({k: float(v) for k, v in METRIC_RE.findall(line)})
        if "Test result:" in line:
            tests.append({k: float(v) for k, v in METRIC_RE.findall(line)})
            classes = []
            for next_line in lines[index + 1:]:
                match = CLASS_RE.search(next_line)
                if not match:
                    break
                class_id, name, iou, precision, recall, f1 = match.groups()
                classes.append({
                    "id": int(class_id), "name": name.strip(), "IoU": float(iou),
                    "Prec": float(precision), "Recall": float(recall), "F1": float(f1),
                })
            all_classes.append(classes)
    if not tests:
        raise RuntimeError(f"No Test result in {path}")
    if len(tests) != 1:
        raise RuntimeError(f"Expected one Test result in {path}, got {len(tests)}")
    if not vals:
        raise RuntimeError(f"No validation results in {path}")
    best_index = int(np.argmax([entry["mIoU"] for entry in vals]))
    return {
        "test": tests[0], "classes": all_classes[0],
        "val_epochs": len(vals), "best_val_epoch": best_index + 1,
        "best_val_mIoU": vals[best_index]["mIoU"],
        "final_val_mIoU": vals[-1]["mIoU"],
        "early_stopped": any("Training stop requested" in line for line in lines),
        "errors": [line for line in lines if " ERROR]" in line],
    }


def parse_all() -> dict:
    records: dict[str, dict[str, dict]] = {}
    for fraction in FRACTIONS:
        key = f"{fraction:g}"
        records[key] = {}
        for method in METHODS:
            path = EXP_ROOT / key / f"{method}-ft" / "train.log"
            if not path.is_file():
                raise FileNotFoundError(path)
            records[key][method] = parse_log(path)
    return records


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join((head, rule, body))


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def rank_at(records: dict, fraction: float) -> list[str]:
    key = f"{fraction:g}"
    return sorted(METHODS, key=lambda method: records[key][method]["test"]["mIoU"], reverse=True)


def write_csv(records: dict) -> None:
    path = OUT / "label_efficiency_metrics.csv"
    columns = ["fraction", "fraction_percent", "actual_labeled_points", "method",
               "mIoU", "mAcc", "OA", "mF1", "fwIoU", "Kappa", "val_epochs",
               "best_val_epoch", "best_val_mIoU", "early_stopped"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for fraction in FRACTIONS:
            key = f"{fraction:g}"
            for method in METHODS:
                record = records[key][method]
                row = {
                    "fraction": fraction,
                    "fraction_percent": ACTUAL_RATIOS[fraction],
                    "actual_labeled_points": ACTUAL_COUNTS[fraction],
                    "method": method,
                    **record["test"],
                    "val_epochs": record["val_epochs"],
                    "best_val_epoch": record["best_val_epoch"],
                    "best_val_mIoU": record["best_val_mIoU"],
                    "early_stopped": record["early_stopped"],
                }
                writer.writerow(row)


def plot_curve(records: dict) -> None:
    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 12,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    colors = {
        "scratch": "#6B737A", "point-m2ae": "#A8ADB1", "msc": "#5E8B7E",
        "sonata": "#9B7DB6", "d-ditr": "#B68A54", "concerto": "#B1A36C",
        "vpsd": "#386FA8", "rrcl": "#BE7050", "cvrl": "#167A72",
    }
    widths = {method: 1.25 for method in METHODS}
    widths.update({"scratch": 1.7, "vpsd": 2.0, "rrcl": 2.0, "cvrl": 2.8})
    fig, ax = plt.subplots(figsize=(9.8, 6.1))
    fig.subplots_adjust(left=.11, right=.98, top=.97, bottom=.27)
    x = np.asarray(FRACTIONS)
    for method in METHODS:
        values = [100 * records[f"{fraction:g}"][method]["test"]["mIoU"]
                  for fraction in FRACTIONS]
        ax.plot(x, values, color=colors[method], lw=widths[method], marker="o",
                ms=5.8 if method == "cvrl" else 4.2, label=METHOD_LABELS[method],
                zorder=5 if method == "cvrl" else 3)
        ax.plot([x[-1], 1.0], [values[-1], 100 * FULL_REFERENCE[method]],
                color=colors[method], lw=widths[method] * .85, linestyle=(0, (3, 2)),
                alpha=.72, zorder=2)
        ax.scatter([1.0], [100 * FULL_REFERENCE[method]], s=28,
                   color="white", edgecolor=colors[method], linewidth=1.2, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(7e-5, 1.35)
    ax.set_ylim(40, 83)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                  ["0.01%", "0.1%", "1%", "10%", "100%*"])
    ax.set_xlabel("Labeled training points (log scale)")
    ax.set_ylabel("Held-out mIoU (%)")
    ax.grid(axis="y", color="#DDE2E5", linewidth=.75)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .075),
               ncol=3, frameon=False, fontsize=9.5, columnspacing=1.15,
               handlelength=2.1)
    fig.text(.5, .015,
             "Solid: controlled 20 epoch × 3 loop protocol.  *100%: Section 4.5 reference endpoint (different schedule).",
             fontsize=9.4, color="#56646D", ha="center")
    fig.savefig(OUT / "figure_10_label_efficiency_curve.png", dpi=350,
                bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def build_report(records: dict) -> str:
    ranks = {fraction: rank_at(records, fraction) for fraction in FRACTIONS}
    cvrl = {fraction: records[f"{fraction:g}"]["cvrl"]["test"]["mIoU"]
            for fraction in FRACTIONS}
    scratch = {fraction: records[f"{fraction:g}"]["scratch"]["test"]["mIoU"]
               for fraction in FRACTIONS}
    vpsd = {fraction: records[f"{fraction:g}"]["vpsd"]["test"]["mIoU"]
            for fraction in FRACTIONS}
    rrcl = {fraction: records[f"{fraction:g}"]["rrcl"]["test"]["mIoU"]
            for fraction in FRACTIONS}
    msc = {fraction: records[f"{fraction:g}"]["msc"]["test"]["mIoU"]
           for fraction in FRACTIONS}

    main_rows = []
    for method in METHODS:
        values = [records[f"{fraction:g}"][method]["test"]["mIoU"]
                  for fraction in FRACTIONS]
        cells = []
        for fraction, value in zip(FRACTIONS, values):
            winner = ranks[fraction][0] == method
            cells.append(("**" if winner else "") + pct(value) + ("**" if winner else ""))
        cells.append(f"{100 * FULL_REFERENCE[method]:.2f}†")
        main_rows.append([METHOD_LABELS[method], *cells])

    gain_rows = []
    for fraction in FRACTIONS:
        best = ranks[fraction][0]
        gain_rows.append([
            DISPLAY_FRACTIONS[FRACTIONS.index(fraction)],
            f"{ACTUAL_COUNTS[fraction]:,}",
            METHOD_LABELS[best],
            f"{100 * cvrl[fraction]:.2f}",
            f"{100 * (cvrl[fraction] - scratch[fraction]):+.2f}",
            f"{100 * (cvrl[fraction] - vpsd[fraction]):+.2f}",
            f"{100 * (cvrl[fraction] - rrcl[fraction]):+.2f}",
            f"{100 * (cvrl[fraction] - msc[fraction]):+.2f}",
            str(ranks[fraction].index("cvrl") + 1),
        ])

    class_names = [entry["name"] for entry in records["0.0001"]["cvrl"]["classes"]]
    class_rows = []
    for class_index, name in enumerate(class_names):
        cells = [name]
        for fraction in FRACTIONS:
            key = f"{fraction:g}"
            cvrl_iou = records[key]["cvrl"]["classes"][class_index]["IoU"]
            scratch_iou = records[key]["scratch"]["classes"][class_index]["IoU"]
            cells.append(f"{100 * cvrl_iou:.1f} ({100 * (cvrl_iou - scratch_iou):+.1f})")
        class_rows.append(cells)

    cvrl_metric_rows = []
    for fraction in FRACTIONS:
        metric = records[f"{fraction:g}"]["cvrl"]["test"]
        cvrl_metric_rows.append([
            DISPLAY_FRACTIONS[FRACTIONS.index(fraction)],
            *[pct(metric[key]) for key in ("mIoU", "mAcc", "OA", "mF1", "fwIoU", "Kappa")],
        ])

    epoch_rows = []
    for method in METHODS:
        cells = [METHOD_LABELS[method]]
        for fraction in FRACTIONS:
            record = records[f"{fraction:g}"][method]
            status = "ES" if record["early_stopped"] else "20"
            cells.append(f"{record['best_val_epoch']}/{record['val_epochs']} {status}")
        epoch_rows.append(cells)

    # Area under the controlled curve, normalized by the three log intervals.
    auc = {}
    logx = np.log10(np.asarray(FRACTIONS))
    for method in METHODS:
        y = np.asarray([records[f"{fraction:g}"][method]["test"]["mIoU"]
                        for fraction in FRACTIONS])
        auc[method] = float(np.trapezoid(y, logx) / (logx[-1] - logx[0]))
    auc_rank = sorted(METHODS, key=auc.get, reverse=True)
    auc_text = ", ".join(
        f"{METHOD_LABELS[method]} {100 * auc[method]:.2f}" for method in auc_rank[:5]
    )

    return f"""# 4.6 DALES 标签效率：完整实验报告

## 实验完成与核验

本报告解析 `exp/label_efficiency/dales/{{0.0001,0.001,0.01,0.1}}/{{method}}-ft/train.log` 中完成的 **36/36** 个 full fine-tuning 任务。每一项均包含一次最终 `Test result` 和 8 类 IoU；日志中未发现 `ERROR`。最终测试由 `PreciseEvaluator` 从验证 mIoU 最优 checkpoint 推理，故表中不是最后一轮权重的结果。

四个正式预算使用同一个 `seed=42` 的嵌套点级标签清单。输入仍保留全量点云几何与强度，只有清单中的标签进入 Cross-Entropy + Lovasz 监督。所有 36 个任务使用 LitePT-v1m4、micro batch 24 × accumulation 2、grid size 0.25 m、20 epoch × train loop 3、BN 校准以及验证 mIoU 的 0.25% 相对早停（warm-up 2 epoch，patience 3）。验证和最终测试均使用完整标签。

| 名义预算 | 实际有效标注点 | 实际比例 | Cars | Trucks | Power lines | Poles |
|---|---:|---:|---:|---:|---:|---:|
| 0.01% | 36,334 | 0.01005% | 256 | 73 | 86 | 34 |
| 0.1% | 362,513 | 0.10023% | 2,570 | 792 | 781 | 281 |
| 1% | 3,618,303 | 1.00044% | 26,175 | 7,550 | 7,936 | 2,838 |
| 10% | 36,165,902 | 9.99970% | 258,933 | 74,932 | 79,898 | 27,711 |

最低档中 truck、pole 与 power line 的标注数分别仅有 73、34、86，因此该档逐类尾类 IoU 的单 seed 波动不可用于显著性判断；主结论以完整测试集 mIoU 和跨预算趋势为准。

## 主结果：标签效率曲线

![DALES label efficiency curve](figure_10_label_efficiency_curve.png)

表中为 held-out mIoU（%），加粗为同一受控预算最高值。† 100% 是 Section 4.5 的既有 FT 参考端点，训练日程为 10 epoch × loop 5，与本节的 20 epoch × loop 3 + early stopping 不同，因而只用于曲线尺度参照，不能和四个正式点做严格的训练日程归因。

{markdown_table(["Method", "0.01%", "0.1%", "1%", "10%", "100%†"], main_rows)}

### CVRL 的相对位置

{markdown_table(["Budget", "Labels", "Best method", "CVRL", "vs Scratch", "vs VPSD", "vs RRCL", "vs MSC", "Rank / 9"], gain_rows)}

CVRL 在最稀疏的 0.01% 和 10% 档取得最高 mIoU：分别为 **65.51%** 与 **78.64%**。0.01% 时，它比第二名 MSC 高 1.21 pp，比 Scratch 高 10.81 pp；这一档仅有 36,334 个监督点，说明 Swiss 预训练在极低监督下提供了可直接迁移的初始化。CVRL 以 0.01% 标签已经超过 Scratch 用 10% 标签的 58.48%，相差 7.03 pp；该跨预算比较是描述性的，但很直观地反映了初始化对监督需求的影响。

0.1% 与 1% 时，最优方法分别为 MSC（74.16%）和 RRCL（78.51%），CVRL 分别低 1.84 和 1.89 pp。这两个差距在单 seed 下不能解释成确定的优劣次序。更稳妥的结论是：CVRL 在四档中始终处于强方法组，但优势不随标签预算单调扩大；视觉和回波的互补在极低预算和较高预算下最清晰，而中间预算由单任务或纯 3D 方法略占优。

以两个单任务分支为直接参照，CVRL 在 0.01%、0.1% 和 10% 同时高于 VPSD 与 RRCL；对应相对 VPSD / RRCL 的增益为 +4.72 / +5.49、+0.03 / +0.88 和 +0.91 / +0.95 pp。1% 时 CVRL 分别低 1.66 与 1.89 pp。因而，这组实验支持两类监督在部分标注预算下的互补收益，但不支持“联合训练在所有预算都严格优于每个单任务分支”的表述。0.1% 与 10% 的不足 1 pp 差异应视为接近持平，等待多 seed 验证。

在 0.01%–10% 的对数标签区间内，以梯形积分得到的归一化曲线面积前五为：{auc_text}（单位：mIoU %）。该汇总描述整个受控区间，不能替代逐档 mIoU。

## 逐类结果：CVRL 与 Scratch 的差异

下表的单元格为 `CVRL IoU (相对 Scratch 增益 pp)`。这不是 CVRL 与所有方法的逐类排名表，而是用于识别标签效率收益由哪些类别支撑。

{markdown_table(["Class", "0.01%", "0.1%", "1%", "10%"], class_rows)}

低预算 mIoU 的增益不能只归因于多数类：CVRL 在 0.01% 对 ground、vegetation 和 buildings 已形成稳定的语义读出，而极端稀少的 truck、pole、power line 仍受标签绝对数量限制。CVRL 的 mIoU 从 0.01% 到 10% 增加 13.13 pp。相比之下，Scratch 的正式测试曲线为 54.70 → 64.06 → 51.42 → 58.48%，存在明显非单调性。这说明相对 Scratch 的每档增益可作为同预算对照，但 Scratch 的跨预算差异不能被解释为平滑的标签量效应。尾类的具体读数应与其标签数一起报告，避免将 34 个 pole 标签下的个别 IoU 波动作过度解释。

### CVRL 的完整测试指标

{markdown_table(["Budget", "mIoU", "mAcc", "OA", "mF1", "fwIoU", "Kappa"], cvrl_metric_rows)}

mIoU 随预算单调提高，mAcc、mF1、fwIoU 与 Kappa 也同步上升。特别是 0.01% 至 1% 的 mAcc 从 77.15% 提升到 84.29%，表明 mIoU 的提高并非只来自训练频率最高的类别；但这仍不能替代多 seed 的不确定性估计。

## 训练轨迹与早停

每个单元格为 `best validation epoch / observed validation epochs`；`ES` 表示早停实际触发，`20` 表示跑满上限。最终测试载入最佳验证 checkpoint。

{markdown_table(["Method", "0.01%", "0.1%", "1%", "10%"], epoch_rows)}

所有方法都完成了预设训练并产出测试。部分方法在小预算下较早停止，例如 Point-M2AE 和 Concerto；其结果反映了当前固定微调协议的迁移表现，不足以说明原始自监督范式本身的绝对能力。CVRL 在四档均于 7–16 个验证轮次内收敛，未出现日志报错或无法选模的情形。

## 与 4.5 全监督迁移实验的关系

既有 100% 端点显示 RRCL（79.96%）、VPSD（79.69%）和 MSC（79.59%）略高于 CVRL（79.28%），最高与 CVRL 的差距仅 0.68 pp。它与本节 10% CVRL 最优的观察相一致：充分标签下，单一预训练目标足以被任务监督细化，而 CVRL 的主要价值更容易在标签受限场景体现。由于 100% 端点训练日程不同且各档均为单 seed，论文中不应将 0.68 pp 解释为统计差异，也不应把 100% 点纳入正式标签效率面积比较。

## 论文可用结论

> Under a shared LitePT fine-tuning protocol and nested point-level DALES supervision, CVRL attains the highest held-out mIoU at the 0.01% and 10% label budgets. It improves over random initialization by 10.81, 8.26, 25.20, and 20.16 percentage points from 0.01% to 10%, respectively. At the intermediate 0.1% and 1% budgets, MSC and RRCL lead by 1.84 and 1.89 points; thus, the evidence supports strong and budget-dependent label efficiency rather than uniform dominance at every label ratio.

中文可写为：在统一 LitePT 微调协议与嵌套点级稀疏标签下，CVRL 在 0.01% 与 10% 标注预算取得最高 held-out mIoU，并在四档相对 Scratch 分别提高 10.81、8.26、25.20 和 20.16 个百分点。0.1% 与 1% 档则由 MSC 和 RRCL 分别领先 1.84、1.89 个百分点。因此结果支持 CVRL 具有较强且随标签预算变化的标签效率，不能表述为在每一个标注比例上均绝对最优。

## 证据边界与后续补强

1. 本节所有正式结果均为一个固定标签抽样 seed 与一个训练 seed。尤其 0.01% 的尾类监督极少，建议至少对 Scratch、VPSD、RRCL、CVRL 在 0.01% 和 0.1% 各补 3 个独立训练 seed；标签清单应继续固定，以区分训练随机性与抽样随机性。
2. 验证与最终精确测试均指向 DALES held-out `test` 文件目录。最终测试使用更完整的 fragment 推理，但不是独立于模型选择的第二个测试集。论文中应称为 held-out split 的最佳验证 checkpoint 精确推理结果。
3. 100% 点来自不同的训练日程，故当前图用空心点和虚线标为参考端点。若要在正文中报告统一的 0.01%–100% 标签效率面积或“达到全监督性能所需标签量”，应补跑同为 20 epoch × loop 3、同一早停规则的九组 100% FT。
4. Point-M2AE、D-DITR 与 Concerto 的个别预算表现偏低，延续了 4.5 中观察到的 LitePT 适配迁移敏感性。报告应明确其为在共同 LitePT 与 ALS 输入协议下的适配结果，而非原论文架构及原始模态设置的复现。
5. Scratch 在 0.1%、1% 与 10% 的单 seed 测试曲线不单调。该现象未出现在 CVRL 的四个正式点，且所有任务的日志均无错误；但它提示极稀疏监督下的优化、类别重加权与早停选择仍有较强敏感性。涉及“所需标签量减少多少”的主张应等待多 seed 后再给出置信区间。

## 可复核产物

- `label_efficiency_parsed.json`：逐任务完整测试指标、逐类 IoU 与早停信息。
- `label_efficiency_metrics.csv`：36 行的扁平主指标表。
- `figure_10_label_efficiency_curve.png`：正文候选曲线图。
- `analyze_section46_label_efficiency.py`：日志解析、表格与图的可重复生成脚本。
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = parse_all()
    failures = [
        f"{fraction}/{method}: {record['errors']}"
        for fraction, methods in records.items()
        for method, record in methods.items() if record["errors"]
    ]
    if failures:
        raise RuntimeError("Errors in completed runs:\n" + "\n".join(failures))
    (OUT / "label_efficiency_parsed.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(records)
    plot_curve(records)
    (OUT / "label_efficiency_experiment_report.md").write_text(
        build_report(records), encoding="utf-8"
    )
    print(f"Saved report and artifacts to {OUT}")
    for fraction in FRACTIONS:
        ranking = rank_at(records, fraction)
        scores = records[f"{fraction:g}"]
        print(f"{fraction:g}: " + ", ".join(
            f"{METHOD_LABELS[method]}={100 * scores[method]['test']['mIoU']:.2f}"
            for method in ranking
        ))


if __name__ == "__main__":
    main()
