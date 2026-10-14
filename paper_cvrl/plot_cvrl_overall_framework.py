"""Render the compact, code-faithful CVRL overview (Fig. 1).

Run from the repository root in the pointspace environment. The two input
thumbnails are derived from one paired Swiss ALS/orthophoto tile; all diagram
elements remain editable vector graphics in the SVG.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

import laspy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.pointcloud_render import camera_basis, render_point_cloud_image


DATA = Path(r"G:\data\swiss\joint_tiles_100")
STEM = "swisssurface3d_2024_2679-1248_2056_5728.copc_0033"
OUT = ROOT / "paper_cvrl/results/method"
ASSETS = OUT / "assets"
BASE = OUT / "figure_1_cvrl_overall_framework"

INK = "#27333A"
MUTED = "#61717A"
SOFT = "#A9B5BB"
BLUE = "#3978A9"
BLUE_BG = "#F0F6FA"
GREEN = "#267E77"
GREEN_BG = "#F0F8F5"
NEUTRAL_BG = "#F4F6F6"
WHITE = "#FFFFFF"


def box(ax, x, y, w, h, *, edge=SOFT, fill=WHITE, lw=1.4,
        dashed=False, radius=15, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        linestyle=(0, (5, 4)) if dashed else "solid",
        zorder=z,
    )
    ax.add_patch(patch)


def txt(ax, x, y, value, *, size=19, color=INK, bold=False,
        align="center", z=5):
    ax.text(x, y, value, ha=align, va="center", fontsize=size,
            fontweight="bold" if bold else "normal", color=color,
            zorder=z)


def line(ax, points, *, color=INK, width=1.7, dashed=False, z=3):
    p = np.asarray(points)
    ax.plot(p[:, 0], p[:, 1], color=color, linewidth=width,
            linestyle=(0, (5, 4)) if dashed else "solid", zorder=z)


def arrow(ax, start, end, *, color=INK, width=2.0, dashed=False,
          head=15, z=4):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=head,
        linewidth=width, color=color,
        linestyle=(0, (5, 4)) if dashed else "solid", zorder=z,
    ))


def feature_strip(ax, x, y, w, h, color, label, dim, *, vertical=False):
    """Compact feature glyph: operation is encoded by geometry, not a box."""
    if vertical:
        for i in range(6):
            alpha = .38 + .10 * i
            ax.add_patch(Rectangle((x, y + i * h / 6), w, h / 6 - 1.5,
                                   facecolor=color, edgecolor="none",
                                   alpha=min(alpha, .9), zorder=4))
        txt(ax, x + w + 12, y + h * .40, label, size=19, color=color,
            bold=True, align="left")
        txt(ax, x + w + 12, y + h * .68, dim, size=14, color=MUTED,
            align="left")
    else:
        for i in range(8):
            alpha = .34 + .075 * i
            ax.add_patch(Rectangle((x + i * w / 8, y), w / 8 - 1.5, h,
                                   facecolor=color, edgecolor="none",
                                   alpha=min(alpha, .9), zorder=4))
        txt(ax, x + w / 2, y + h + 21, label, size=18, color=INK,
            bold=True)
        txt(ax, x + w / 2, y + h + 43, dim, size=14, color=MUTED)


def token(ax, x, y, *, visible=True, radius=8):
    ax.add_patch(Circle(
        (x, y), radius=radius,
        facecolor=BLUE if visible else WHITE,
        edgecolor=BLUE if visible else SOFT,
        linewidth=1.4, zorder=5,
    ))


def build_input_assets():
    """Make matched real-data icons, not illustrative or semantic renderings."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    cloud_path = DATA / "pointcloud" / f"{STEM}.las"
    if not cloud_path.is_file():
        cloud_path = DATA / "Swiss-Audit/pointcloud" / f"{STEM}.las"
    dom_path = DATA / "image" / f"{STEM}.jpg"
    if not cloud_path.is_file() or not dom_path.is_file():
        raise FileNotFoundError(f"Paired ALS/DOM tile missing: {cloud_path}, {dom_path}")

    cloud = laspy.read(cloud_path)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    xy = xyz[:, :2] - xyz[:, :2].min(axis=0)
    rgba = np.empty((len(xyz), 4), dtype=np.float32)
    rgba[:] = (.48, .58, .64, 1.0)
    right, up, view = camera_basis(xy[:, 0], xy[:, 1],
                                   azimuth=-125.0, elevation=35.0)
    camera = (right, up, view,
              np.median(np.column_stack((xy, xyz[:, 2])), axis=0))
    rendered, _, _, _ = render_point_cloud_image(
        xy[:, 0], xy[:, 1], xyz[:, 2], rgba, camera=camera,
        width=760, style="lit_edl", point_scale=.82,
        fractional_splat=True, square_frame=True,
    )
    visible = np.min(rendered[..., :3], axis=2) < .985
    rendered[..., 3] = visible.astype(np.float32)
    yy, xx = np.where(visible)
    if not len(xx):
        raise RuntimeError("Real ALS thumbnail rendered no visible points")
    pad = 12
    bounds = (max(0, xx.min() - pad), max(0, yy.min() - pad),
              min(rendered.shape[1], xx.max() + pad + 1),
              min(rendered.shape[0], yy.max() + pad + 1))
    als = Image.fromarray((np.clip(rendered, 0, 1) * 255).astype(np.uint8))
    als_path = OUT / "als_pointcloud.png"
    als.crop(bounds).save(als_path)

    with Image.open(dom_path) as source:
        dom = source.convert("RGB")
        dom.thumbnail((760, 760), Image.Resampling.LANCZOS)
        dom_out = OUT / "orthophoto_crop.png"
        dom.save(dom_out)
    return als_path, dom_out, cloud_path, dom_path


def draw_framework(ax, als_path, dom_path):
    # Two publication-level regions establish the first visual hierarchy.
    box(ax, 28, 70, 1370, 842, edge=SOFT, fill=WHITE, lw=1.5,
        dashed=True, radius=16, z=0)
    box(ax, 1425, 70, 347, 842, edge=SOFT, fill=WHITE, lw=1.5,
        dashed=True, radius=16, z=0)
    txt(ax, 62, 92, "CVRL Pre-training", size=24, bold=True,
        align="left")
    txt(ax, 1458, 92, "Downstream Usage", size=24, bold=True,
        align="left")
    txt(ax, 1600, 126, "LiDAR only", size=17, color=MUTED, bold=True)

    # ── Upper layer: visual-semantic supervision ──────────────────────────
    txt(ax, 59, 137, "VISUAL SEMANTIC SUPERVISION", size=13,
        color=BLUE, bold=True, align="left")
    ax.imshow(Image.open(dom_path), extent=(66, 238, 286, 166),
              aspect="auto", zorder=2)
    txt(ax, 152, 310, "Orthophoto", size=17, bold=True)

    box(ax, 272, 181, 168, 82, fill=NEUTRAL_BG, edge=SOFT,
        dashed=True, radius=10)
    txt(ax, 356, 210, "Frozen DINOv3", size=17, bold=True)
    txt(ax, 356, 239, "visual teacher", size=14, color=MUTED)
    arrow(ax, (240, 224), (269, 224), color=SOFT, dashed=True,
          width=1.4, head=11)

    # DINO patch grid; colors encode target embeddings rather than semantics.
    patch_colors = ["#5A86B3", "#7AA0C4", "#406E9D", "#91AFCB",
                    "#6D95BC", "#4779A9", "#83A7C8", "#5A88B4"]
    for row in range(2):
        for col in range(4):
            ax.add_patch(Rectangle(
                (478 + col * 29, 190 + row * 29), 23, 23,
                facecolor=patch_colors[row * 4 + col], edgecolor=WHITE,
                linewidth=.8, zorder=4,
            ))
    txt(ax, 524, 267, "2D patch features", size=14, color=MUTED)
    arrow(ax, (442, 222), (474, 222), color=SOFT, dashed=True,
          width=1.4, head=11)

    # Visible and excluded 3D tokens: only filled tokens receive visual edges.
    token_xy = [(683, 189), (717, 207), (751, 186), (784, 220),
                (671, 242), (711, 255), (753, 244), (798, 264)]
    visibility = [True, True, False, True, False, True, True, False]
    for (x, y), is_visible in zip(token_xy, visibility):
        token(ax, x, y, visible=is_visible, radius=7)
    for patch_x, point_idx in ((489, 0), (518, 1), (547, 5), (576, 6)):
        line(ax, [(patch_x, 237), token_xy[point_idx]], color=BLUE,
             width=.85, dashed=True, z=2)
    box(ax, 650, 166, 180, 122, edge=BLUE, fill="none", lw=1.1,
        dashed=True, radius=19, z=1)
    txt(ax, 740, 307, "visible token set", size=14, color=BLUE)
    token(ax, 667, 329, visible=True, radius=5)
    txt(ax, 680, 329, r"$\mathcal{P}_{vis}$", size=12, color=BLUE,
        align="left")
    token(ax, 752, 329, visible=False, radius=5)
    txt(ax, 765, 329, "excluded", size=12, color=MUTED, align="left")

    # A single mechanism block, not a text-heavy card.
    txt(ax, 948, 178, "VPSD", size=25, color=BLUE, bold=True)
    txt(ax, 948, 207, "Patch-set distillation", size=16, color=INK)
    ax.add_patch(Polygon(
        [(877, 236), (1019, 236), (997, 288), (899, 288)],
        closed=True, facecolor=BLUE_BG, edgecolor=BLUE,
        linewidth=1.7, zorder=3,
    ))
    txt(ax, 948, 260, r"supervise $\mathcal{P}_{vis}$", size=15,
        color=BLUE)
    arrow(ax, (831, 228), (875, 250), color=BLUE, width=1.8)
    arrow(ax, (948, 291), (948, 346), color=BLUE, width=1.8)

    # ── Middle layer: one shared LiDAR hierarchy ──────────────────────────
    txt(ax, 59, 374, "SHARED LIDAR REPRESENTATION", size=13,
        color=INK, bold=True, align="left")
    ax.imshow(Image.open(als_path), extent=(65, 285, 593, 400),
              aspect="auto", zorder=2)
    txt(ax, 175, 618, "ALS point cloud", size=18, bold=True)

    txt(ax, 645, 402, "Shared LiDAR Encoder", size=25, bold=True)
    txt(ax, 645, 430, "LitePT", size=15, color=MUTED)
    stages = [
        (330, 470, 94, 100, "Stem"),
        (458, 462, 108, 116, "P1"),
        (600, 452, 120, 136, "P2"),
        (754, 442, 132, 156, "P3"),
        (920, 432, 144, 176, "P4"),
    ]
    for x, y, w, h, name in stages:
        ax.add_patch(Polygon(
            [(x, y + 10), (x + w - 10, y), (x + w, y + h - 10),
             (x + 10, y + h)], closed=True,
            facecolor="#F7F7F5", edgecolor=INK, linewidth=1.5, zorder=3,
        ))
        txt(ax, x + w / 2, y + h / 2, name, size=17, bold=True)
    for left, right in zip(stages[:-1], stages[1:]):
        arrow(ax, (left[0] + left[2] + 3, left[1] + left[3] / 2),
              (right[0] - 4, right[1] + right[3] / 2),
              color=INK, width=1.7, head=12)
    arrow(ax, (286, 505), (326, 518), color=INK, width=2.5, head=16)

    # P2-P4 jointly feed the visual branch, encoded graphically by a bracket.
    for x in (660, 820, 992):
        line(ax, [(x, 445), (x, 382)], color=BLUE, width=1.2)
    line(ax, [(660, 382), (992, 382)], color=BLUE, width=1.2)
    arrow(ax, (948, 382), (948, 347), color=BLUE, width=1.7, head=11)
    txt(ax, 822, 365, "multi-scale tokens", size=13, color=BLUE)

    # ── Lower layer: return-structure supervision ────────────────────────
    txt(ax, 59, 653, "RETURN-STRUCTURE SUPERVISION", size=13,
        color=GREEN, bold=True, align="left")
    box(ax, 67, 690, 210, 84, fill=NEUTRAL_BG, edge=SOFT,
        dashed=True, radius=10)
    txt(ax, 172, 719, "Return metadata", size=17, bold=True)
    txt(ax, 172, 749, r"$(r_i,n_i)$ · targets only", size=14,
        color=MUTED)

    # Four role glyphs and a context histogram summarize the two targets.
    role_x = [335, 380, 425, 470]
    role_names = ["S", "F", "M", "L"]
    role_heights = [1, 3, 2, 1]
    for x, name, count in zip(role_x, role_names, role_heights):
        for j in range(count):
            ax.add_patch(Circle((x, 729 - j * 17), radius=5.3,
                                facecolor=GREEN, edgecolor=WHITE,
                                linewidth=.5, zorder=4))
        txt(ax, x, 757, name, size=13, color=MUTED)
    txt(ax, 403, 680, "return roles", size=14, color=INK, bold=True)
    arrow(ax, (280, 731), (318, 731), color=SOFT, dashed=True,
          width=1.4, head=11)

    hist = [.70, .46, .23, .58]
    for i, height in enumerate(hist):
        ax.add_patch(Rectangle((529 + i * 25, 754 - height * 64), 15,
                               height * 64, facecolor=GREEN, edgecolor="none",
                               alpha=.82, zorder=4))
        txt(ax, 536 + i * 25, 770, role_names[i], size=11, color=MUTED)
    line(ax, [(520, 754), (624, 754)], color=INK, width=.8)
    txt(ax, 572, 680, "local context", size=14, color=INK, bold=True)
    arrow(ax, (485, 731), (515, 731), color=SOFT, dashed=True,
          width=1.4, head=11)

    # Structural readout from the shared hierarchy, supervised by both targets.
    arrow(ax, (660, 592), (660, 692), color=GREEN, width=1.8)
    box(ax, 660, 692, 194, 70, edge=GREEN, fill=WHITE, lw=1.4,
        radius=9)
    txt(ax, 757, 716, "structural embedding", size=13, bold=True)
    txt(ax, 757, 744, "shared 72D bottleneck", size=12, color=MUTED)
    arrow(ax, (628, 726), (657, 726), color=SOFT, dashed=True,
          width=1.35, head=10)

    # Two tiny heads expose the RRCL mechanism without architecture clutter.
    box(ax, 875, 675, 92, 56, edge=GREEN, fill=WHITE, lw=1.2, radius=8)
    box(ax, 875, 745, 92, 56, edge=GREEN, fill=WHITE, lw=1.2, radius=8)
    txt(ax, 921, 703, "Role", size=15, color=GREEN, bold=True)
    txt(ax, 921, 773, "Context", size=15, color=GREEN, bold=True)
    arrow(ax, (856, 714), (872, 704), color=GREEN, width=1.35, head=10)
    arrow(ax, (856, 741), (872, 770), color=GREEN, width=1.35, head=10)
    txt(ax, 1039, 705, "RRCL", size=25, color=GREEN, bold=True)
    txt(ax, 1039, 738, "role + context", size=15, color=INK)
    line(ax, [(970, 703), (995, 703)], color=GREEN, width=1.4)
    line(ax, [(970, 773), (982, 773), (982, 703), (995, 703)],
         color=GREEN, width=1.4)

    # Feature bars make readout and concatenation visually self-explanatory.
    feature_strip(ax, 1167, 235, 22, 104, BLUE, r"$z^V$", "128D",
                  vertical=True)
    arrow(ax, (1022, 261), (1163, 278), color=BLUE, width=1.8)
    feature_strip(ax, 1167, 694, 22, 82, GREEN, r"$z^R$", "72D",
                  vertical=True)
    arrow(ax, (1100, 704), (1163, 724), color=GREEN, width=1.8)

    # Same point-wise concat as code: 128D blue + 72D green = 200D.
    line(ax, [(1246, 292), (1282, 292), (1282, 487)],
         color=BLUE, width=1.6)
    line(ax, [(1246, 735), (1300, 735), (1300, 516)],
         color=GREEN, width=1.6)
    arrow(ax, (1282, 487), (1322, 487), color=BLUE, width=1.6, head=11)
    arrow(ax, (1300, 516), (1322, 516), color=GREEN, width=1.6, head=11)
    ax.add_patch(Rectangle((1324, 474), 45, 26, facecolor=BLUE,
                           edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((1324, 500), 45, 15, facecolor=GREEN,
                           edgecolor="none", zorder=4))
    txt(ax, 1346, 544, r"$z^{CVRL}$", size=18, bold=True)
    txt(ax, 1346, 568, "200D", size=13, color=MUTED)

    # ── Downstream region: the multimodal sources disappear ──────────────
    # The retained CVRL feature crosses the pretraining boundary; no image or
    # return-target path enters downstream usage.
    arrow(ax, (1372, 495), (1507, 367), color=INK, width=2.2, head=15)

    # One retained encoder feeds three alternative usage modes.
    box(ax, 1510, 330, 180, 72, edge=INK, fill="#F7F7F5", lw=1.4,
        radius=9)
    txt(ax, 1600, 359, "LiDAR backbone", size=18, bold=True)
    txt(ax, 1600, 386, "pretrained", size=13, color=MUTED)

    use_labels = [("Frozen", "representation"),
                  ("Decoder", "adaptation"),
                  ("Full", "fine-tuning")]
    use_y = [505, 633, 761]
    line(ax, [(1510, 366), (1470, 366), (1470, 797)], color=INK, width=1.25)
    for y, (first, second) in zip(use_y, use_labels):
        box(ax, 1508, y, 184, 73, edge="#829097", fill=WHITE,
            lw=1.2, radius=8)
        txt(ax, 1600, y + 27, first, size=16, bold=True)
        txt(ax, 1600, y + 51, second, size=14, color=MUTED)
        arrow(ax, (1470, y + 36), (1505, y + 36), color=INK,
              width=1.2, head=9)

    txt(ax, 713, 884,
        "Dashed paths: pretraining targets   ·   Solid paths: retained LiDAR representation",
        size=14, color=MUTED)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    als_path, dom_path, cloud_path, source_dom = build_input_assets()
    html_path = BASE.with_suffix(".html")
    if not html_path.is_file():
        raise FileNotFoundError(f"Inline SVG source missing: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    match = re.search(r"<svg\b[^>]*id=\"cvrl-framework\"[\s\S]*?</svg>", html)
    if match is None:
        raise RuntimeError("Could not locate #cvrl-framework in HTML")
    svg = match.group(0)
    for filename, path, mime in (
        ("orthophoto_crop.png", dom_path, "image/png"),
        ("als_pointcloud.png", als_path, "image/png"),
    ):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        svg = svg.replace(
            f'href="{filename}"', f'href="data:{mime};base64,{encoded}"'
        )
    svg_path = BASE.with_suffix(".svg")
    svg_path.write_text(svg, encoding="utf-8")

    async def render_png():
        from playwright.async_api import async_playwright

        candidates = [
            Path(os.environ.get("PROGRAMFILES", ""))
            / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", ""))
            / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("PROGRAMFILES", ""))
            / "Microsoft/Edge/Application/msedge.exe",
        ]
        executable = next((path for path in candidates if path.is_file()), None)
        async with async_playwright() as playwright:
            launch_kwargs = {"headless": True}
            if executable is not None:
                launch_kwargs["executable_path"] = str(executable)
            browser = await playwright.chromium.launch(**launch_kwargs)
            page = await browser.new_page(
                viewport={"width": 1800, "height": 1000},
                device_scale_factor=2.3622,
            )
            await page.goto(html_path.resolve().as_uri(), wait_until="load")
            await page.locator("#cvrl-framework").screenshot(
                path=str(BASE.with_suffix(".png")),
                animations="disabled",
            )
            await browser.close()

    asyncio.run(render_png())

    metadata = {
        "source_las": str(cloud_path),
        "source_dom": str(source_dom),
        "config": "configs/cvrl/pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py",
        "model": "pointspace/models/backbone/cvrl/cvrl_v1m1.py",
        "visual_dim": 128,
        "return_dim": 72,
        "cvrl_dim": 200,
        "canvas": [1800, 900],
        "purpose": "architecture plus one mechanism layer for VPSD and RRCL",
    }
    BASE.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(svg_path)
    print(BASE.with_suffix(".png"))


if __name__ == "__main__":
    main()
