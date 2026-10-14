"""Publication-oriented, code-aligned CVRL framework figure.

The topology mirrors the official CVRL-v1m1 implementation. The script uses
a paired real Swiss-Audit ALS/orthophoto tile for the two input glyphs and
exports self-contained SVG/HTML plus a high-resolution PNG.
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import os
import sys
from pathlib import Path

import laspy
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pointcloud_render import camera_basis, render_point_cloud_image


AUDIT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
PAIRED = Path(r"G:\data\swiss\joint_tiles_100")
STEM = "swisssurface3d_2024_2679-1248_2056_5728.copc_0033"
OUT = ROOT / "paper_cvrl/results/method"
BASE = OUT / "figure_1_cvrl_overall_framework"

INK = "#2F3A40"
MUTED = "#69777E"
FAINT = "#A9B4BB"
PALE = "#E2E8EB"
BLUE = "#3E7BAA"
BLUE_MID = "#79A6C4"
BLUE_LIGHT = "#AFCBE0"
BLUE_BG = "#F8FBFD"
GREEN = "#2C8A7D"
GREEN_MID = "#72B2A8"
GREEN_LIGHT = "#A8D0C9"
GREEN_BG = "#F7FBF9"
ORANGE = "#C98A4A"


def make_assets():
    """Build paired real-data input glyphs from Swiss-Audit."""
    OUT.mkdir(parents=True, exist_ok=True)
    cloud = AUDIT / "pointcloud" / f"{STEM}.las"
    ortho = PAIRED / "image" / f"{STEM}.jpg"
    if not cloud.is_file() or not ortho.is_file():
        raise FileNotFoundError(f"Swiss-Audit pair missing: {cloud}, {ortho}")

    las = laspy.read(cloud)
    x, y, z = (np.asarray(axis, dtype=np.float64) for axis in (las.x, las.y, las.z))
    low, high = np.percentile(z, [1, 99])

    # Real 3D ALS glyph rendered by the repository's deterministic point
    # renderer. White background pixels become transparent for SVG placement.
    right, up, view = camera_basis(x, y, azimuth=34.0, elevation=61.0)
    camera = (
        right,
        up,
        view,
        np.asarray([np.median(x), np.median(y), np.median(z)]),
    )
    rgba = np.empty((z.shape[0], 4), dtype=np.float32)
    rgba[:, :3] = np.asarray([0.48, 0.61, 0.68], dtype=np.float32)
    rgba[:, 3] = 1.0
    image, _, _, radius = render_point_cloud_image(
        x,
        y,
        z,
        rgba,
        camera=camera,
        width=980,
        style="lit_edl",
        point_scale=0.92,
        fractional_splat=True,
        square_frame=False,
        strength=2.6,
    )
    rgb = np.clip(image[..., :3], 0, 1)
    white_distance = 1.0 - np.min(rgb, axis=-1)
    alpha = np.clip((white_distance - 0.012) / 0.08, 0, 1)
    out = np.dstack((rgb, alpha))
    Image.fromarray((out * 255).astype(np.uint8), "RGBA").save(
        OUT / "als_pointcloud.png"
    )

    with Image.open(ortho) as source:
        source.convert("RGB").resize((900, 900), Image.Resampling.LANCZOS).save(
            OUT / "orthophoto_crop.png"
        )
    return cloud, ortho, low, high, radius


def make_svg_unboxed():
    marks: list[str] = []

    def put(value: str):
        marks.append(value)

    def box(x, y, w, h, fill="none", stroke="none", r=0, sw=1.4, dash=""):
        put(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-dasharray="{dash}"/>'
        )

    def line(d, color=INK, sw=2.0, dash="", arrow=False, marker=None):
        if marker is None:
            marker = "blue" if color == BLUE else "green" if color == GREEN else "ink"
        end = f' marker-end="url(#{marker})"' if arrow else ""
        put(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-dasharray="{dash}" stroke-linecap="round" '
            f'stroke-linejoin="round"{end}/>'
        )

    def label(x, y, text, size=16, color=INK, bold=False, anchor="middle", italic=False):
        put(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{700 if bold else 400}" '
            f'font-style="{"italic" if italic else "normal"}" fill="{color}">'
            f'{html.escape(text)}</text>'
        )

    def circle(x, y, r=4, fill=INK, stroke="none", sw=1):
        put(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def feature_strip(x, y, w, h, color, segments=8, gap=1.4, border="none"):
        segment = w / segments
        for index in range(segments):
            box(x + index * segment, y, segment - gap, h, color, border, 1, 0.7)

    def stage_stack(x, y, name, dim, layers, color="#DDE5E9", edge=INK, w=76, h=58):
        for index in range(layers - 1, -1, -1):
            box(x + index * 4, y - index * 4, w, h, color, edge, 3, 1.3)
        label(x + w / 2, y + 25, name, 18, INK, True)
        label(x + w / 2, y + 46, f"{dim}D", 13, MUTED)

    def mini_stack(x, y, count, color, w=10, h=27, dx=7):
        for index in range(count):
            box(x + index * dx, y - index * 1.6, w, h, color, "#FFFFFF", 0, 0.6)

    def snowflake(x, y, radius=8, color=BLUE, sw=1.5):
        for angle in (0, 60, 120):
            rad = np.deg2rad(angle)
            dx, dy = radius * np.cos(rad), radius * np.sin(rad)
            line(f"M{x-dx:.2f} {y-dy:.2f} L{x+dx:.2f} {y+dy:.2f}", color, sw)

    def protocol_glyph(x, y, mode):
        frozen = mode != "full"
        backbone_color = "#DCE5E9" if frozen else "#E8C7A5"
        for index, height in enumerate((20, 25, 31, 37)):
            box(x + index * 18, y + 18 - height / 2, 13, height, backbone_color, INK, 1, 0.8)
        line(f"M{x+69} {y+18} H{x+83}", INK, 1.3, arrow=True)
        if mode == "frozen":
            box(x + 87, y + 7, 30, 22, "#E8C7A5", ORANGE, 2, 1)
            snowflake(x - 10, y + 18, 7, INK, 1.1)
        else:
            for index, height in enumerate((18, 25, 32)):
                box(x + 88 + index * 12, y + 18 - height / 2, 9, height, "#E8C7A5", ORANGE, 1, 0.8)
            if mode == "decoder":
                snowflake(x - 10, y + 18, 7, INK, 1.1)

    put(
        '<svg id="cvrl-framework" xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 1900 950" width="1900" height="950" role="img" '
        'aria-labelledby="title desc">'
    )
    put('<title id="title">Overall framework of CVRL</title>')
    put(
        '<desc id="desc">A single LitePT encoder is trained with asymmetric '
        'visual patch-set distillation and return role-context supervision. '
        'The retained 128D visual and 72D return readouts form a 200D LiDAR-only '
        'CVRL representation.</desc>'
    )
    put('<defs>')
    for key, color in (("ink", INK), ("blue", BLUE), ("green", GREEN), ("orange", ORANGE)):
        put(
            f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0 0 L10 5 L0 10" fill="{color}"/></marker>'
        )
    put('</defs>')

    box(0, 0, 1900, 950, "#FFFFFF")
    box(20, 28, 1495, 875, "none", FAINT, 10, 1.5, "8 7")
    box(1538, 28, 342, 875, "none", FAINT, 10, 1.5, "8 7")
    box(32, 89, 1470, 280, BLUE_BG)
    box(32, 608, 1470, 280, GREEN_BG)
    label(55, 67, "CVRL Pre-training", 30, INK, True, "start")
    label(754, 66, "One shared encoder under asymmetric supervision", 17, MUTED)
    label(1709, 66, "LiDAR-only Downstream", 26, INK, True)

    # VPSD: real orthophoto and frozen patch teacher.
    label(52, 118, "VPSD", 27, BLUE, True, "start")
    label(52, 140, "Visual Patch-Set Distillation", 15, MUTED, False, "start")
    put('<image href="orthophoto_crop.png" x="52" y="153" width="150" height="150"/>')
    label(127, 324, "Orthophoto", 17, INK, True)
    line("M204 228 H235", FAINT, 1.4, "5 4", True)
    put(
        f'<polygon points="237,181 350,192 350,270 237,282" fill="{BLUE_BG}" '
        f'stroke="{BLUE}" stroke-width="1.8"/>'
    )
    label(293, 222, "DINOv3", 22, BLUE, True)
    label(293, 249, "frozen visual teacher", 13, MUTED)
    snowflake(339, 190, 8, BLUE)
    line("M352 229 H380", FAINT, 1.4, "5 4", True)
    put('<image href="orthophoto_crop.png" x="384" y="163" width="132" height="132"/>')
    for index in range(1, 5):
        line(f"M{384 + 26.4 * index:.1f} 163 V295", "#FFFFFF", 0.9)
        line(f"M384 {163 + 26.4 * index:.1f} H516", "#FFFFFF", 0.9)
    box(384, 163, 132, 132, "none", BLUE, 0, 1.2)
    label(450, 318, "DINO patch teacher", 16, BLUE, True)
    label(450, 338, "M × 1024", 13, MUTED)

    # Enlarged many-to-many visible token-patch correspondence graph.
    label(545, 126, "Visible token–patch correspondence", 17, BLUE, True, "start")
    label(545, 147, "image_valid + patch_index", 13, MUTED, False, "start")
    patch_cells = [(550 + 34 * col, 174 + 38 * row) for row in range(3) for col in range(2)]
    token_points = [
        (690, 170), (722, 181), (754, 167), (783, 191), (704, 217),
        (742, 229), (778, 235), (695, 263), (731, 274), (769, 267),
    ]
    for px, py in patch_cells:
        box(px, py, 26, 26, BLUE_LIGHT, BLUE, 0, 1.1)
    edge_pairs = [
        (0, 0), (0, 1), (0, 4), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 4), (2, 5), (3, 3), (3, 5), (3, 6),
        (4, 4), (4, 7), (4, 8), (5, 5), (5, 6), (5, 8), (5, 9),
    ]
    for patch_index, token_index in edge_pairs:
        px, py = patch_cells[patch_index]
        tx, ty = token_points[token_index]
        line(f"M{px+26} {py+13} L{tx} {ty}", BLUE_MID, 0.9)
    for tx, ty in token_points:
        circle(tx, ty, 4.2, BLUE)
    label(668, 318, "P_vis valid edges", 14, BLUE, True)

    # Three separate aggregate-then-project paths.
    aggregate_rows = (
        (175, "P2", 144, 7, BLUE_LIGHT),
        (225, "P3", 252, 5, BLUE_MID),
        (275, "P4", 504, 3, BLUE),
    )
    for y, name, dim, token_count, color in aggregate_rows:
        label(824, y + 5, name, 16, BLUE, True, "start")
        mini_stack(856, y - 12, token_count, BLUE_MID, 8, 24, 6)
        line(f"M906 {y} H930", BLUE, 1.9, arrow=True)
        feature_strip(936, y - 12, 101, 24, color, 8)
        label(1047, y + 5, f"{dim}D", 14, MUTED, False, "start")
        line(f"M1091 {y} H1110", BLUE, 1.5)
    line("M1110 175 V275", BLUE, 1.6)
    line("M1110 225 H1130", BLUE, 2.0, arrow=True)
    feature_strip(1137, 204, 114, 42, BLUE, 11)
    label(1194, 268, "Patch-set feature", 16, BLUE, True)
    label(1194, 287, "144 + 252 + 504 = 900D", 13, MUTED)
    line("M1254 225 H1271", BLUE, 2.0, arrow=True)
    box(1275, 180, 135, 91, BLUE_BG, BLUE, 5, 1.7)
    label(1342, 205, "Native DINO projector", 16, BLUE, True)
    label(1342, 231, "LN → Linear", 13, MUTED)
    label(1342, 250, "GELU → Linear", 13, MUTED)
    label(1342, 268, "900 → 1024", 13, MUTED)
    line("M1412 225 H1426", BLUE, 2.0, arrow=True)
    feature_strip(1432, 206, 57, 38, BLUE, 7)
    label(1461, 263, "1024D", 13, MUTED)

    # Explicit double-track cosine supervision in native DINO space.
    feature_strip(1330, 112, 60, 17, BLUE_LIGHT, 7)
    label(1322, 125, "teacher", 12, MUTED, False, "end")
    line("M516 155 V111 H1328", BLUE, 1.3, "5 4", True)
    feature_strip(1430, 112, 60, 17, BLUE, 7)
    label(1422, 125, "student", 12, MUTED, False, "end")
    line("M1461 205 V133 H1460", BLUE, 1.8)
    line("M1392 120 H1428", BLUE, 1.3, "5 4", True)
    label(1410, 103, "Cosine distillation", 15, BLUE, True)
    label(1410, 151, "L_VPSD", 14, BLUE, True)

    # Parallel retained point readout; PCA is not on the loss path.
    line("M931 319 H1131", BLUE, 1.9, arrow=True)
    label(1018, 311, "P2-aligned hierarchy", 13, MUTED)
    feature_strip(1138, 302, 82, 31, BLUE_MID, 9)
    label(1179, 350, "visual-native point feature · 1024D", 14, BLUE, True)
    line("M1223 318 H1250", BLUE, 1.9, arrow=True)
    box(1254, 298, 92, 54, BLUE_BG, BLUE, 5, 1.5)
    label(1300, 320, "Fixed PCA", 15, BLUE, True)
    label(1300, 340, "1024 → 128", 12, MUTED)
    label(1300, 365, "non-trainable", 12, MUTED)
    line("M1348 325 H1372", BLUE, 2.3, arrow=True)
    feature_strip(1378, 296, 87, 55, BLUE, 9)
    label(1421, 374, "zV · 128D", 20, BLUE, True)

    # One shared LiDAR encoder.
    label(740, 402, "Shared LiDAR Encoder", 27, INK, True)
    label(740, 424, "LitePT-v1m4", 15, MUTED)
    put('<image href="als_pointcloud.png" x="51" y="422" width="200" height="148"/>')
    label(151, 584, "ALS point cloud", 17, INK, True)
    label(151, 603, "coord + intensity", 13, MUTED)
    line("M248 499 H301", INK, 2.6, arrow=True)
    stage_x = (308, 449, 590, 731, 872, 1013)
    stage_names = ("Stem", "P0", "P1", "P2", "P3", "P4")
    stage_dims = (36, 36, 72, 144, 252, 504)
    stage_layers = (2, 2, 3, 3, 6, 3)
    for x, name, dim, depth in zip(stage_x, stage_names, stage_dims, stage_layers):
        stage_stack(x, 469, name, dim, depth)
    for left, right in zip(stage_x[:-1], stage_x[1:]):
        line(f"M{left+91} 498 H{right-9}", INK, 2.3, arrow=True)
    for sx, ey in ((770, 175), (911, 225), (1052, 275)):
        line(f"M{sx} 463 V{ey} H812", BLUE, 1.7)
    line("M770 463 V319 H929", BLUE, 1.7, arrow=True)
    line("M911 463 V304 H929", BLUE, 1.2)
    line("M1052 463 V289 H929", BLUE, 1.2)
    line("M307 548 V565 H678 V548", FAINT, 1.3)
    label(493, 581, "shared-gradient region · Stem / P0 / P1", 13, MUTED)
    label(493, 598, "return warm-up → accumulated stage-wise PCGrad after 10%", 12, MUTED)

    # RRCL full native decoder.
    label(52, 643, "RRCL", 27, GREEN, True, "start")
    label(52, 665, "Return Role and Context Learning", 15, MUTED, False, "start")
    label(820, 650, "LitePT decoder", 21, GREEN, True)
    decoder = ((1040, "D3", 252, 2), (899, "D2", 144, 3), (758, "D1", 72, 4), (617, "D0", 72, 5))
    # P4 is the detached decoder seed. P3/P2 are detached skips; P1/P0
    # remain differentiable and join D1/D0, respectively.
    line("M1052 527 V672 H1090 V700", GREEN, 2.2, "5 4", arrow=True)
    label(1090, 696, "⊥", 21, GREEN, True)
    for index, (x, name, dim, layers) in enumerate(decoder):
        stage_stack(x, 704, name, dim, layers, GREEN_LIGHT, GREEN, 75, 50)
        if index + 1 < len(decoder):
            line(f"M{x-8} 729 H{decoder[index+1][0]+90}", GREEN, 2.1, arrow=True)
    for sx, dx, route_y in ((911, 1063, 681), (770, 937, 690)):
        line(f"M{sx} 527 V{route_y} H{dx} V699", GREEN, 1.4, "5 4")
        label(dx, 695, "⊥", 21, GREEN, True)
    line("M629 527 V681 H796 V699", GREEN, 1.4)
    line("M488 527 V670 H655 V699", GREEN, 1.4)
    label(946, 672, "stop-grad: P2 / P3 / P4", 13, GREEN, True)
    line("M615 754 V777 H1114", GREEN, 2.6, arrow=True)
    box(1118, 753, 100, 49, GREEN_BG, GREEN, 5, 1.6)
    label(1168, 783, "LayerNorm", 15, GREEN, True)
    line("M1220 777 H1242", GREEN, 2.4, arrow=True)
    feature_strip(1248, 750, 106, 54, GREEN, 9)
    label(1301, 826, "zR · 72D", 20, GREEN, True)

    # Return metadata is target-only.
    box(52, 714, 165, 55, "#F4F6F7", FAINT, 5, 1.2, "5 4")
    label(134, 738, "LAS return metadata", 15, INK, True)
    label(134, 758, "r_i, n_i · targets only", 12, MUTED)
    line("M219 741 H242", GREEN, 1.4, "5 4", True)
    box(246, 694, 159, 112, "none", GREEN, 4, 1.3, "5 4")
    label(326, 714, "Role target", 16, GREEN, True)
    label(326, 731, "pulse return formation", 12, MUTED)
    line("M282 741 V762", GREEN, 1.2, arrow=True)
    line("M263 771 H301", GREEN_LIGHT, 1.2)
    circle(282, 767, 4, GREEN)
    label(292, 773, "S", 13, GREEN, True, "start")
    line("M349 740 V758", GREEN, 1.2, arrow=True)
    for yy, role in ((758, "F"), (776, "M"), (794, "L")):
        line(f"M331 {yy+4} H386", GREEN_LIGHT, 1)
        circle(349, yy, 3.5, GREEN)
        label(362, yy + 4, role, 12, GREEN, True, "start")
    line("M408 750 H431", GREEN, 1.4, "5 4", True)
    put(f'<circle cx="482" cy="750" r="39" fill="none" stroke="{GREEN_MID}" '
        'stroke-width="1.2" stroke-dasharray="4 3"/>')
    circle(482, 750, 6, "#FFFFFF", GREEN, 1.7)
    for dx, dy, radius in ((-25,-8,4),(-17,19,3),(5,-27,4),(25,-17,3),(31,8,4),(14,25,3),(-8,31,4),(-31,10,3),(2,10,3)):
        circle(482 + dx, 750 + dy, radius, GREEN_MID)
    label(482, 806, "XYZ KNN", 15, GREEN, True)
    label(482, 823, "K = 48 · center excluded", 12, MUTED)
    line("M523 750 H552", GREEN, 1.4, "5 4", True)
    for index, height in enumerate((34, 52, 22, 41)):
        box(562 + index * 25, 790 - height, 16, height, GREEN if index != 2 else GREEN_MID)
        label(570 + index * 25, 809, ("S", "F", "M", "L")[index], 12, MUTED)
    label(607, 830, "Context target", 15, GREEN, True)
    label(607, 847, "local role histogram", 12, MUTED)

    # Heads consume zR and are discarded after pretraining.
    line("M1356 777 H1371 V742 H1387", GREEN, 1.9, arrow=True)
    line("M1371 777 V817 H1387", GREEN, 1.9, arrow=True)
    for y, title, loss in ((718, "Role head", "L_role"), (793, "Context head", "L_ctx")):
        box(1391, y, 102, 49, GREEN_BG, GREEN, 5, 1.3, "5 4")
        label(1442, y + 21, title, 15, GREEN, True)
        label(1442, y + 40, "72 → 72 → 4", 12, MUTED)
        label(1498, y + 30, loss, 13, GREEN, True, "end")
    label(1442, 866, "pretraining-only heads", 12, MUTED)
    line("M405 768 V873 H1418 V767", GREEN, 1.2, "5 4", True)
    line("M658 750 V885 H1477 V842", GREEN, 1.2, "5 4", True)

    # Retained visual-return representation.
    line("M1468 324 V444 H1422 V459", BLUE, 2.6, arrow=True)
    line("M1356 777 H1368 V533 H1480 V519", GREEN, 2.6, arrow=True)
    feature_strip(1383, 461, 76, 55, BLUE, 8)
    feature_strip(1459, 461, 43, 55, GREEN, 5)
    label(1442, 544, "zCVRL = [zV | zR]", 19, INK, True)
    label(1442, 565, "200D", 18, INK, True)
    label(1442, 584, "128D visual + 72D return", 12, MUTED)
    line("M1504 489 H1536", INK, 2.7, arrow=True)

    # LiDAR-only downstream protocols as mini network-state glyphs.
    label(1709, 112, "CVRL backbone", 21, INK, True)
    for index, height in enumerate((22, 29, 36, 42, 35)):
        box(1593 + index * 25, 151 - height / 2, 18, height, PALE, INK, 1, 0.9)
    line("M1716 151 H1742", INK, 1.8, arrow=True)
    feature_strip(1748, 132, 83, 38, BLUE, 8)
    feature_strip(1831, 132, 37, 38, GREEN, 4)
    label(1730, 196, "encoder + visual / return readouts", 13, MUTED)
    line("M1538 489 H1561 V151 H1589", INK, 2.0, arrow=True)
    protocol_y = (305, 505, 705)
    protocol_info = (
        ("Frozen representation", "linear probe / cosine kNN", "frozen"),
        ("Decoder probe", "frozen backbone + trainable decoder", "decoder"),
        ("Full fine-tuning", "all trainable", "full"),
    )
    # Three mutually exclusive transfer protocols branch from the same
    # pretrained backbone; the vertical rail is not a serial computation.
    line("M1709 205 V224 H1604 V703", INK, 1.4)
    for y, (title, description, mode) in zip(protocol_y, protocol_info):
        line(f"M1604 {y-2} H1626", INK, 1.4, arrow=True)
        protocol_glyph(1636, y - 20, mode)
        label(1709, y + 50, title, 18, INK, True)
        label(1709, y + 71, description, 13, MUTED)
    label(1709, 850, "LiDAR input only", 18, INK, True)
    label(1709, 873, "no orthophoto or return targets", 13, MUTED)

    # Narrow footer.
    line("M31 904 H1880", FAINT, 1)
    label(52, 931, "Training objective", 15, INK, True, "start")
    label(245, 931, "L_CVRL = L_VPSD + λ_R(t) (L_role + λ_ctx L_ctx)", 15, MUTED, False, "start")
    line("M990 926 H1020", INK, 2.6, arrow=True)
    label(1031, 931, "retained feature flow", 12, MUTED, False, "start")
    line("M1195 926 H1225", GREEN, 1.3, "5 4")
    label(1236, 931, "supervision / target", 12, MUTED, False, "start")
    label(1395, 932, "⊥", 18, GREEN, True)
    label(1412, 931, "stop-gradient", 12, MUTED, False, "start")
    put('</svg>')
    return "\n".join(marks)


def make_svg():
    """Build the containerized paper figure without changing model topology."""
    marks: list[str] = []

    def put(value: str):
        marks.append(value)

    def box(x, y, w, h, fill="none", stroke="none", r=0, sw=1.3, dash=""):
        put(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-dasharray="{dash}"/>'
        )

    def line(d, color=INK, sw=2.0, dash="", arrow=False, marker=None):
        if marker is None:
            marker = "blue" if color == BLUE else "green" if color == GREEN else "ink"
        end = f' marker-end="url(#{marker})"' if arrow else ""
        put(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-dasharray="{dash}" stroke-linecap="round" '
            f'stroke-linejoin="round"{end}/>'
        )

    def label(x, y, text, size=16, color=INK, bold=False, anchor="middle", italic=False):
        put(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{700 if bold else 400}" '
            f'font-style="{"italic" if italic else "normal"}" fill="{color}">'
            f'{html.escape(text)}</text>'
        )

    def circle(x, y, r=4, fill=INK, stroke="none", sw=1):
        put(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def port(x, y, color):
        circle(x, y, 4.2, "#FFFFFF", color, 1.6)

    def feature_strip(x, y, w, h, color, segments=8, gap=1.3, border="none"):
        segment = w / segments
        for index in range(segments):
            box(x + index * segment, y, segment - gap, h, color, border, 1, 0.6)

    def stage_stack(x, y, name, dim, layers, color="#DDE5E9", edge=INK, w=74, h=54):
        for index in range(layers - 1, -1, -1):
            box(x + index * 4, y - index * 4, w, h, color, edge, 3, 1.2)
        label(x + w / 2, y + 23, name, 17, INK, True)
        label(x + w / 2, y + 43, f"{dim}D", 12, MUTED)

    def container(x, y, w, h, title, subtitle, color, fill):
        box(x, y, w, h, fill, color, 7, 1.15)
        label(x + 15, y + 25, title, 17, color, True, "start")
        if subtitle:
            label(x + 15, y + 44, subtitle, 12, MUTED, False, "start")

    def snowflake(x, y, radius=7, color=BLUE, sw=1.4):
        for angle in (0, 60, 120):
            rad = np.deg2rad(angle)
            dx, dy = radius * np.cos(rad), radius * np.sin(rad)
            line(f"M{x-dx:.2f} {y-dy:.2f} L{x+dx:.2f} {y+dy:.2f}", color, sw)

    def mini_network(x, y, mode):
        frozen = mode != "full"
        backbone_color = PALE if frozen else "#E8C7A5"
        for index, height in enumerate((19, 25, 31, 37)):
            box(x + index * 19, y + 20 - height / 2, 13, height, backbone_color, INK, 1, 0.8)
        line(f"M{x+73} {y+20} H{x+88}", INK, 1.3, arrow=True)
        if mode == "frozen":
            box(x + 92, y + 9, 31, 22, "#E8C7A5", ORANGE, 2, 0.9)
        else:
            for index, height in enumerate((18, 25, 32)):
                box(x + 93 + index * 13, y + 20 - height / 2, 9, height, "#E8C7A5", ORANGE, 1, 0.8)
        if frozen:
            snowflake(x - 11, y + 20, 6.5, INK, 1.1)

    put(
        '<svg id="cvrl-framework" xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 1900 950" width="1900" height="950" role="img" '
        'aria-labelledby="title desc">'
    )
    put('<title id="title">Overall framework of CVRL</title>')
    put(
        '<desc id="desc">Containerized code-aligned CVRL framework. A single '
        'LitePT encoder supplies visual patch-set distillation and a protected '
        'return decoder. The retained 128D and 72D readouts form a 200D '
        'LiDAR-only representation.</desc>'
    )
    put('<defs>')
    for key, color in (("ink", INK), ("blue", BLUE), ("green", GREEN), ("orange", ORANGE)):
        put(
            f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0 0 L10 5 L0 10" fill="{color}"/></marker>'
        )
    put('</defs>')

    box(0, 0, 1900, 950, "#FFFFFF")
    box(18, 24, 1495, 880, "none", FAINT, 10, 1.5, "8 7")
    box(1534, 24, 348, 880, "none", FAINT, 10, 1.5, "8 7")
    label(48, 61, "CVRL Pre-training", 29, INK, True, "start")
    label(756, 60, "One shared encoder under asymmetric supervision", 17, MUTED)
    label(1708, 60, "LiDAR-only Downstream", 25, INK, True)

    # ---------------------------------------------------------------- VPSD
    label(42, 93, "VPSD", 25, BLUE, True, "start")
    label(128, 93, "Visual Patch-Set Distillation", 14, MUTED, False, "start")
    # V1: orthophoto -> frozen teacher -> patch teacher.
    put('<g id="V1_visual_teacher">')
    container(40, 105, 335, 230, "Visual teacher", "paired image supervision", BLUE, BLUE_BG)
    put('<image href="orthophoto_crop.png" x="55" y="158" width="108" height="108"/>')
    label(109, 286, "Orthophoto", 14, INK, True)
    line("M166 212 H183", FAINT, 1.3, "5 4", True)
    put(
        f'<polygon points="185,178 264,185 264,239 185,247" fill="#FFFFFF" '
        f'stroke="{BLUE}" stroke-width="1.5"/>'
    )
    label(224, 207, "DINOv3", 18, BLUE, True)
    label(224, 227, "frozen", 12, MUTED)
    snowflake(255, 184, 6.5, BLUE)
    line("M266 212 H278", FAINT, 1.3, "5 4", True)
    put('<image href="orthophoto_crop.png" x="281" y="171" width="78" height="78"/>')
    for index in range(1, 4):
        line(f"M{281+19.5*index:.1f} 171 V249", "#FFFFFF", 0.8)
        line(f"M281 {171+19.5*index:.1f} H359", "#FFFFFF", 0.8)
    box(281, 171, 78, 78, "none", BLUE, 0, 1)
    label(320, 270, "patch teacher", 13, BLUE, True)
    label(320, 288, "M × 1024", 12, MUTED)
    port(375, 212, BLUE)
    line("M359 212 H367 V147 H371", BLUE, 1.1, "5 4", True)
    port(375, 147, BLUE)
    put('</g>')

    # V2: local many-to-many relation and three independent patch aggregations.
    put('<g id="V2_correspondence_aggregation">')
    container(
        390, 105, 600, 230,
        "Patch correspondence and aggregation",
        "visible token–patch edges · P2/P3/P4 → 900D",
        BLUE, BLUE_BG,
    )
    port(390, 212, BLUE)
    line("M379 212 H386", BLUE, 1.6, arrow=True)
    patch_cells = [(410 + 27 * col, 169 + 31 * row) for row in range(3) for col in range(2)]
    token_points = [(505,166),(532,178),(558,164),(582,187),(517,214),(548,225),(577,231),(507,253),(539,266),(570,258)]
    for px, py in patch_cells:
        box(px, py, 21, 21, BLUE_LIGHT, BLUE, 0, 1)
    for patch_index, token_index in ((0,0),(0,1),(0,4),(1,1),(1,2),(1,3),(2,0),(2,4),(2,5),(3,3),(3,5),(3,6),(4,4),(4,7),(4,8),(5,5),(5,6),(5,8),(5,9)):
        px, py = patch_cells[patch_index]
        tx, ty = token_points[token_index]
        line(f"M{px+21} {py+10} L{tx} {ty}", BLUE_MID, 0.75)
    for tx, ty in token_points:
        circle(tx, ty, 3.7, BLUE)
    label(493, 299, "P_vis correspondence", 12, BLUE, True)
    line("M586 219 H610", BLUE, 1.5, arrow=True)

    # Fixed bottom ports accept P2/P3/P4 from the shared encoder.
    for px, name in ((708, "P2"), (848, "P3"), (988, "P4")):
        port(px, 335, BLUE)
        label(px, 322, name, 13, BLUE, True)
    # Internal routing is absorbed by the container.
    row_data = ((174,"P2",144,BLUE_LIGHT),(219,"P3",252,BLUE_MID),(264,"P4",504,BLUE))
    for y, name, dim, color in row_data:
        label(616, y + 5, name, 14, BLUE, True, "start")
        for index in range(4 if name == "P2" else 3 if name == "P3" else 2):
            box(651 + index * 8, y - 10 - index * 1.4, 11, 22, BLUE_MID, "#FFFFFF", 0, 0.5)
        line(f"M692 {y} H711", BLUE, 1.5, arrow=True)
        feature_strip(717, y - 10, 92, 21, color, 8)
        label(817, y + 5, f"{dim}D", 12, MUTED, False, "start")
        line(f"M858 {y} H875", BLUE, 1.2)
    line("M875 174 V264", BLUE, 1.3)
    line("M875 219 H889", BLUE, 1.7, arrow=True)
    feature_strip(894, 199, 79, 40, BLUE, 9)
    label(934, 256, "patch-set 900D", 13, BLUE, True)
    # A second, pointwise P2-aligned 900D port feeds the same projector weights.
    feature_strip(891, 286, 67, 20, BLUE_MID, 8)
    label(924, 321, "P2-aligned 900D", 12, MUTED)
    port(990, 219, BLUE)
    port(990, 296, BLUE)
    line("M994 219 H1001", BLUE, 1.8, arrow=True)
    line("M994 296 H1001", BLUE, 1.8, arrow=True)
    put('</g>')

    # V3: two local lanes through the same projector weights.
    put('<g id="V3_projection_readout">')
    container(
        1005, 105, 340, 230,
        "Projection and visual readout",
        "native-space distillation + retained point feature",
        BLUE, BLUE_BG,
    )
    port(1005, 219, BLUE)
    port(1005, 296, BLUE)
    box(1030, 158, 109, 139, "#FFFFFF", BLUE, 5, 1.4)
    label(1085, 181, "Native DINO", 15, BLUE, True)
    label(1085, 201, "projector", 15, BLUE, True)
    label(1085, 228, "LN → Linear", 12, MUTED)
    label(1085, 246, "GELU → Linear", 12, MUTED)
    label(1085, 269, "900 → 1024", 12, MUTED)
    line("M1009 219 H1026", BLUE, 1.8, arrow=True)
    line("M1009 296 H1018 V273 H1026", BLUE, 1.8, arrow=True)
    line("M1142 201 H1154", BLUE, 1.8, arrow=True)
    feature_strip(1160, 186, 55, 30, BLUE, 7)
    label(1187, 232, "student 1024D", 12, MUTED)
    # Teacher supervision enters only through the top container port.
    port(1236, 105, BLUE)
    line("M1236 109 V170", BLUE, 1.2, "5 4", arrow=True)
    feature_strip(1222, 175, 45, 16, BLUE_LIGHT, 6)
    line("M1217 201 H1228", BLUE, 1.2, "5 4", True)
    label(1260, 202, "cosine", 14, BLUE, True)
    label(1260, 220, "L_VPSD", 12, BLUE, True)
    # Point lane: 1024D native point feature -> fixed PCA.
    line("M1142 273 H1154", BLUE, 1.8, arrow=True)
    feature_strip(1160, 258, 54, 27, BLUE_MID, 7)
    label(1187, 304, "point 1024D", 12, MUTED)
    line("M1217 272 H1227", BLUE, 1.8, arrow=True)
    box(1231, 249, 77, 48, "#FFFFFF", BLUE, 4, 1.2)
    label(1269, 269, "Fixed PCA", 14, BLUE, True)
    label(1269, 287, "1024 → 128", 12, MUTED)
    port(1345, 273, BLUE)
    line("M1309 273 H1341", BLUE, 2.3, arrow=True)
    put('</g>')

    # Teacher target takes one orthogonal edge route along the VPSD boundary.
    line("M375 147 V96 H1236 V101", BLUE, 1.2, "5 4", arrow=True)

    # --------------------------------------------------------- shared encoder
    put('<g id="shared_lidar_encoder">')
    label(735, 375, "Shared LiDAR Encoder", 26, INK, True)
    label(735, 396, "LitePT-v1m4", 14, MUTED)
    put('<image href="als_pointcloud.png" x="45" y="408" width="180" height="140"/>')
    label(135, 566, "ALS point cloud", 16, INK, True)
    label(135, 585, "coord + intensity", 12, MUTED)
    line("M223 480 H246", INK, 2.6, arrow=True)
    stage_x = (250, 390, 530, 670, 810, 950)
    stage_names = ("Stem", "P0", "P1", "P2", "P3", "P4")
    stage_dims = (36, 36, 72, 144, 252, 504)
    stage_layers = (2, 2, 3, 3, 6, 3)
    for x, name, dim, depth in zip(stage_x, stage_names, stage_dims, stage_layers):
        stage_stack(x, 449, name, dim, depth)
    for left, right in zip(stage_x[:-1], stage_x[1:]):
        line(f"M{left+89} 476 H{right-8}", INK, 2.3, arrow=True)
    # Three visual taps terminate at V2's fixed bottom ports.
    for sx, px in ((707,708),(847,848),(987,988)):
        line(f"M{sx} 443 V352 H{px} V339", BLUE, 1.6, arrow=True)
    line("M249 528 V544 H606 V528", FAINT, 1.2)
    label(428, 559, "shared-gradient region · Stem / P0 / P1", 12, MUTED)
    label(428, 577, "return warm-up → accumulated PCGrad after 10%", 12, MUTED)
    put('</g>')

    # -------------------------------------------------------------- RRCL
    label(42, 628, "RRCL", 25, GREEN, True, "start")
    label(126, 628, "Return Role and Context Learning", 14, MUTED, False, "start")
    # R1: target construction remains self-contained.
    put('<g id="R1_return_target_construction">')
    container(
        40, 642, 410, 225,
        "Return target construction",
        "target-only metadata · no encoder input",
        GREEN, GREEN_BG,
    )
    box(56, 696, 112, 46, "#F4F6F7", FAINT, 4, 1, "4 3")
    label(112, 716, "LAS metadata", 13, INK, True)
    label(112, 734, "r_i, n_i", 12, MUTED)
    line("M170 719 H186", GREEN, 1.2, "5 4", True)
    box(191, 685, 92, 91, "#FFFFFF", GREEN, 4, 1.1, "4 3")
    label(237, 703, "Role target", 13, GREEN, True)
    # Single and multiple pulse formation.
    line("M216 711 V730", GREEN, 1.1, arrow=True)
    circle(216, 735, 3.5, GREEN)
    label(226, 740, "S", 12, GREEN, True, "start")
    line("M256 711 V726", GREEN, 1.1, arrow=True)
    for yy, role in ((727,"F"),(744,"M"),(761,"L")):
        circle(256, yy, 3, GREEN)
        label(267, yy+4, role, 12, GREEN, True, "start")
    line("M286 730 H300", GREEN, 1.2, "5 4", True)
    put(f'<circle cx="330" cy="730" r="30" fill="none" stroke="{GREEN_MID}" '
        'stroke-width="1.1" stroke-dasharray="4 3"/>')
    circle(330,730,5,"#FFFFFF",GREEN,1.4)
    for dx, dy in ((-18,-8),(-12,17),(4,-20),(20,-10),(22,11),(8,21),(-20,10)):
        circle(330+dx,730+dy,3,GREEN_MID)
    label(330, 773, "XYZ KNN · K=48", 12, MUTED)
    line("M362 730 H376", GREEN, 1.2, "5 4", True)
    for index, height in enumerate((24,38,16,29)):
        box(381+index*14, 759-height, 9, height, GREEN if index != 2 else GREEN_MID)
        label(385+index*14, 775, ("S","F","M","L")[index], 12, MUTED)
    label(409, 793, "Context target", 12, GREEN, True)
    # Role and context targets merge into one external supervision bus.
    line("M237 776 V810 H430 V820", GREEN, 1.1, "5 4")
    line("M409 797 V830 H430 V820", GREEN, 1.1, "5 4")
    circle(430, 820, 3.2, GREEN)
    line("M430 820 H446", GREEN, 1.2, "5 4", arrow=True)
    label(347, 852, "role + context targets", 12, MUTED)
    port(450, 820, GREEN)
    put('</g>')

    # R2: fixed top ports feed the protected decoder locally.
    put('<g id="R2_return_decoder_readout">')
    container(
        465, 642, 700, 225,
        "Return decoder readout",
        "LitePT decoder · stop-grad at P2/P3/P4",
        GREEN, GREEN_BG,
    )
    decoder = ((1010,"D3",252,2),(860,"D2",144,3),(710,"D1",72,4),(560,"D0",72,5))
    top_ports = ((988, "P4", True, 1048),(848, "P3", True, 1028),(708, "P2", True, 898),(568, "P1", False, 748),(428, "P0", False, 598))
    for px, _name, detached, dx in top_ports:
        # P0 lies left of R2; use the nearest fixed container port.
        target_x = max(px, 482)
        port(target_x, 642, GREEN)
        route_y = 678 + (0 if detached else 8)
        line(f"M{target_x} 646 V{route_y} H{dx} V700", GREEN, 1.25, "5 4" if detached else "")
        if detached:
            label(dx, 696, "⊥", 18, GREEN, True)
    for index, (x, name, dim, layers) in enumerate(decoder):
        stage_stack(x, 704, name, dim, layers, GREEN_LIGHT, GREEN, 70, 47)
        if index + 1 < len(decoder):
            line(f"M{x-7} 728 H{decoder[index+1][0]+85}", GREEN, 1.9, arrow=True)
    # D0 returns to a pointwise 72D readout within the same container.
    line("M558 751 V810 H966", GREEN, 2.2, arrow=True)
    box(971, 786, 85, 47, "#FFFFFF", GREEN, 4, 1.2)
    label(1013, 815, "LayerNorm", 13, GREEN, True)
    line("M1058 810 H1070", GREEN, 2.0, arrow=True)
    feature_strip(1075, 786, 73, 48, GREEN, 7)
    label(1111, 852, "zR · 72D", 16, GREEN, True)
    port(1165, 810, GREEN)
    line("M1149 810 H1161", GREEN, 2.3, arrow=True)
    put('</g>')

    # R3: zR enters once and branches locally to both heads.
    put('<g id="R3_pretraining_heads">')
    container(
        1180, 700, 165, 167,
        "Pretraining-only heads",
        "discarded after pretraining",
        GREEN, GREEN_BG,
    )
    port(1180, 781, GREEN)
    line("M1184 781 H1197", GREEN, 1.8, arrow=True)
    line("M1198 781 V834", GREEN, 1.5)
    for y, title in ((760,"Role head"),(813,"Context head")):
        box(1210, y, 116, 42, "#FFFFFF", GREEN, 4, 1.1, "4 3")
        label(1268, y+18, title, 13, GREEN, True)
        label(1268, y+35, "72 → 72 → 4", 12, MUTED)
        line(f"M1198 {y+21} H1207", GREEN, 1.4, arrow=True)
    label(1334, 782, "L_role", 12, GREEN, True, "end")
    label(1334, 835, "L_ctx", 12, GREEN, True, "end")
    # The single target bus enters at the bottom and splits locally.
    port(1262, 867, GREEN)
    line("M1262 863 V848 H1336 V782 H1329", GREEN, 1.1, "5 4", True)
    line("M1336 848 V835 H1329", GREEN, 1.1, "5 4", True)
    put('</g>')
    # One edge-routed supervision bus from R1 to R3.
    line("M454 820 V884 H1262 V871", GREEN, 1.25, "6 5", True)

    # Encoder-to-decoder edges terminate only at R2's top ports.
    for sx, px in ((987,988),(847,848),(707,708),(567,568),(427,482)):
        line(f"M{sx} 503 V618 H{px} V638", GREEN, 1.35, "5 4" if sx >= 707 else "", arrow=True)

    # ------------------------------------------------ representation band
    put('<g id="representation_outputs">')
    container(
        1360, 105, 140, 762,
        "Representation outputs",
        "retained after pretraining",
        INK, "#FAFBFB",
    )
    # zV
    port(1360, 273, BLUE)
    line("M1349 273 H1356", BLUE, 2.5, arrow=True)
    feature_strip(1383, 238, 92, 50, BLUE, 9)
    label(1429, 307, "zV · 128D", 18, BLUE, True)
    # zCVRL
    feature_strip(1381, 450, 61, 54, BLUE, 7)
    feature_strip(1442, 450, 35, 54, GREEN, 4)
    label(1429, 527, "zCVRL", 20, INK, True)
    label(1429, 548, "[zV | zR] · 200D", 14, INK, True)
    # zR
    port(1360, 735, GREEN)
    feature_strip(1391, 705, 76, 48, GREEN, 7)
    label(1429, 774, "zR · 72D", 18, GREEN, True)
    # Internal merge only; no crossing through other mechanisms.
    line("M1429 310 V446", BLUE, 2.4, arrow=True)
    line("M1429 701 V508", GREEN, 2.4, arrow=True)
    # V3 output and R2 retained output reach fixed band ports.
    line("M1345 273 H1356", BLUE, 2.5, arrow=True)
    line("M1165 810 H1172 V680 H1350 V735 H1356", GREEN, 2.5, arrow=True)
    # A short local branch sends zR to pretraining heads without routing through them.
    line("M1172 810 V781 H1176", GREEN, 1.8, arrow=True)
    port(1500, 477, INK)
    line("M1479 477 H1496", INK, 2.7, arrow=True)
    line("M1504 477 H1532", INK, 2.7, arrow=True)
    put('</g>')

    # ---------------------------------------------------------- downstream
    put('<g id="lidar_only_downstream">')
    put('<g id="down_backbone">')
    container(
        1555, 88, 304, 162,
        "CVRL backbone",
        "shared encoder + visual / return readouts",
        INK, "#FAFBFB",
    )
    port(1555, 169, INK)
    line("M1536 477 H1543 V169 H1551", INK, 2.4, arrow=True)
    for index, height in enumerate((21,27,34,40,34)):
        box(1583+index*24, 174-height/2, 17, height, PALE, INK, 1, 0.8)
    line("M1703 174 H1720", INK, 1.7, arrow=True)
    feature_strip(1725, 154, 77, 39, BLUE, 8)
    feature_strip(1802, 154, 37, 39, GREEN, 4)
    put('</g>')
    # A shared protocol bus branches to three independent boxes.
    line("M1581 250 H1552 V743", INK, 1.3)
    protocols = (
        (284, "Frozen representation", "linear probe / cosine kNN", "frozen", "down_frozen"),
        (478, "Decoder probe", "frozen backbone + trainable decoder", "decoder", "down_decoder"),
        (672, "Full fine-tuning", "all trainable", "full", "down_full_finetuning"),
    )
    for y, title, subtitle, mode, group_id in protocols:
        put(f'<g id="{group_id}">')
        box(1570, y, 274, 142, "#FAFBFB", FAINT, 6, 1.1)
        line(f"M1552 {y+71} H1567", INK, 1.3, arrow=True)
        mini_network(1638, y+31, mode)
        label(1707, y+96, title, 17, INK, True)
        label(1707, y+117, subtitle, 12, MUTED)
        put('</g>')
    label(1707, 850, "LiDAR input only", 17, INK, True)
    label(1707, 872, "no orthophoto or return targets", 12, MUTED)
    put('</g>')

    # --------------------------------------------------------------- footer
    line("M28 908 H1872", FAINT, 1)
    label(45, 935, "Training objective", 15, INK, True, "start")
    label(237, 935, "L_CVRL = L_VPSD + λ_R(t) (L_role + λ_ctx L_ctx)", 15, MUTED, False, "start")
    line("M982 930 H1014", INK, 2.6, arrow=True)
    label(1025, 935, "retained feature flow", 12, MUTED, False, "start")
    line("M1190 930 H1222", GREEN, 1.2, "5 4")
    label(1233, 935, "supervision / target", 12, MUTED, False, "start")
    label(1396, 936, "⊥", 18, GREEN, True)
    label(1413, 935, "stop-gradient", 12, MUTED, False, "start")
    put('</svg>')
    return "\n".join(marks)


def make_html(svg: str) -> str:
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fig. 1. Overall framework of CVRL</title>
<style>body{{margin:0;background:white;font-family:"Times New Roman",serif}}
main{{max-width:1900px;margin:auto;padding:12px}}button{{font:16px "Times New Roman",serif;padding:8px 14px;margin-bottom:10px}}
svg{{display:block;width:100%;height:auto}}svg text{{font-family:"Times New Roman","Liberation Serif",serif}}</style>
</head><body><main><button id="download-svg" type="button">Download SVG</button>{svg}</main>
<script>document.getElementById('download-svg').addEventListener('click',()=>{{
const clone=document.getElementById('cvrl-framework').cloneNode(true);
const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)],{{type:'image/svg+xml'}}));
const a=document.createElement('a');a.href=url;a.download='figure_1_cvrl_overall_framework.svg';a.click();
setTimeout(()=>URL.revokeObjectURL(url),1000)}});</script></body></html>'''


async def screenshot(html_path: Path, png_path: Path):
    from playwright.async_api import async_playwright

    candidates = (
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
    )
    executable = next((path for path in candidates if path.is_file()), None)
    async with async_playwright() as playwright:
        kwargs = {"headless": True}
        if executable:
            kwargs["executable_path"] = str(executable)
        browser = await playwright.chromium.launch(**kwargs)
        page = await browser.new_page(
            viewport={"width": 1940, "height": 1040}, device_scale_factor=2.3622
        )
        await page.goto(html_path.resolve().as_uri(), wait_until="load")
        await page.locator("#cvrl-framework").screenshot(
            path=str(png_path), animations="disabled"
        )
        await browser.close()


def main():
    cloud, ortho, low, high, radius = make_assets()
    svg = make_svg()
    embedded = svg
    for name in ("als_pointcloud.png", "orthophoto_crop.png"):
        data = base64.b64encode((OUT / name).read_bytes()).decode("ascii")
        embedded = embedded.replace(
            f'href="{name}"', f'href="data:image/png;base64,{data}"'
        )
    html_path = BASE.with_suffix(".html")
    html_path.write_text(make_html(embedded), encoding="utf-8")
    BASE.with_suffix(".svg").write_text(embedded, encoding="utf-8")
    asyncio.run(screenshot(html_path, BASE.with_suffix(".png")))
    BASE.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_las": str(cloud),
                "source_orthophoto": str(ortho),
                "dataset": "Swiss-Audit",
                "cloud_view": "orthographic oblique, transparent background",
                "cloud_render": {
                    "renderer": "utils.pointcloud_render",
                    "style": "lit_edl",
                    "color": "uniform muted blue-grey",
                    "source_height_percentiles_m": [low, high],
                    "point_scale": 0.92,
                    "point_radius_px": radius,
                },
                "config": "configs/cvrl/pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py",
                "model": "pointspace/models/backbone/cvrl/cvrl_v1m1.py",
                "canvas": [1900, 950],
                "retained_dimensions": [128, 72, 200],
                "topology_locked": True,
                "layout": "containerized orthogonal routing",
                "containers": {
                    "VPSD": [
                        "Visual teacher",
                        "Patch correspondence and aggregation",
                        "Projection and visual readout",
                    ],
                    "RRCL": [
                        "Return target construction",
                        "Return decoder readout",
                        "Pretraining-only heads",
                    ],
                    "output": "Representation outputs",
                },
                "svg_groups": [
                    "V1_visual_teacher",
                    "V2_correspondence_aggregation",
                    "V3_projection_readout",
                    "shared_lidar_encoder",
                    "R1_return_target_construction",
                    "R2_return_decoder_readout",
                    "R3_pretraining_heads",
                    "representation_outputs",
                    "down_backbone",
                    "down_frozen",
                    "down_decoder",
                    "down_full_finetuning",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(BASE.with_suffix(".html"))
    print(BASE.with_suffix(".svg"))
    print(BASE.with_suffix(".png"))


if __name__ == "__main__":
    main()
