"""Paper Fig. 4: measured return roles and local return context.

Run from the repository root in the pointspace environment::

    python paper_cvrl/plot_fig4_rrcl_mechanism.py

The profile, pulse examples, and 48-NN neighborhoods are from one real
Swiss-Audit LAS tile. LAS classes only select two display examples; all
rendered colors denote return roles, and RRCL targets use return metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import laspy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Ellipse
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.pointcloud_render import camera_basis, render_point_cloud_image, to_image_pixels


STEM = "swisssurface3d_2024_2679-1248_2056_5728.copc_0033"
POINT_PATH = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit\pointcloud") / f"{STEM}.las"
OUT = ROOT / "paper_cvrl" / "results" / "method"
PNG = OUT / "figure_4_rrcl_return_physics.png"
SVG = OUT / "figure_4_rrcl_return_physics.svg"
META = OUT / "figure_4_rrcl_return_physics.json"

ROLE_CODES = ("S", "F", "M", "L")
ROLE_NAMES = ("Single", "First", "Middle", "Last")
ROLE_COLORS = ("#4C78A8", "#F58518", "#B279A2", "#54A24B")
INK = "#19232B"
GRID = "#E2E7EB"
K = 48
GRID_SIZE = 0.25
PROFILE_START = np.asarray((2679300.0, 1248386.0))
PROFILE_END = np.asarray((2679392.0, 1248306.0))
PROFILE_FROM_M = 15.0
PROFILE_TO_M = 100.0
PROFILE_HALF_WIDTH_M = 0.5
CONTEXT_DISPLAY_HALF_WIDTH_M = 8.0
CONTEXT_DISPLAY_Z_RANGE_M = (-5.0, 20.0)
CONTEXT_PROJECTION_BOUNDS = (-10.0, 10.0, -6.0, 12.0)
CONTEXT_DISPLAY_CAP = 8000
CONTEXT_RENDER_WIDTH = 1100
CONTEXT_POINT_SCALE = 0.75
_render_height = round(CONTEXT_RENDER_WIDTH * (
    CONTEXT_PROJECTION_BOUNDS[3] - CONTEXT_PROJECTION_BOUNDS[2]
) / (CONTEXT_PROJECTION_BOUNDS[1] - CONTEXT_PROJECTION_BOUNDS[0]))
CONTEXT_SHARED_RADIUS = int(np.clip(np.rint(
    CONTEXT_POINT_SCALE * np.sqrt(
        (CONTEXT_RENDER_WIDTH - 16) * (_render_height - 16)
        / (CONTEXT_DISPLAY_CAP + K + 1)
    )
), 1, 5))


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#65717B",
        "axes.linewidth": 0.85,
        "xtick.color": INK,
        "ytick.color": INK,
    })


def _load():
    if not POINT_PATH.is_file():
        raise FileNotFoundError(POINT_PATH)
    las = laspy.read(POINT_PATH)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    rn = np.asarray(las.return_number, dtype=np.int16)
    nr = np.asarray(las.number_of_returns, dtype=np.int16)
    gps = np.asarray(las.gps_time, dtype=np.float64)
    cls = np.asarray(las.classification, dtype=np.int16)
    role = np.full(len(xyz), -1, dtype=np.int8)
    valid = (rn >= 1) & (nr >= 1) & (rn <= nr)
    role[valid & (nr == 1)] = 0
    role[valid & (nr > 1) & (rn == 1)] = 1
    role[valid & (nr > 1) & (rn > 1) & (rn < nr)] = 2
    role[valid & (nr > 1) & (rn == nr)] = 3
    return xyz, rn, nr, gps, cls, role


def _profile(xyz):
    direction = PROFILE_END - PROFILE_START
    direction /= np.linalg.norm(direction)
    normal = np.asarray((-direction[1], direction[0]))
    relative = xyz[:, :2] - PROFILE_START
    along = relative @ direction
    across = relative @ normal
    mask = ((along >= PROFILE_FROM_M) & (along <= PROFILE_TO_M)
            & (np.abs(across) <= PROFILE_HALF_WIDTH_M))
    return mask, along - PROFILE_FROM_M, np.abs(across)


def _select_pulse(xyz, rn, nr, gps, profile_mask, across):
    """Exact GPS-time group with F/M/L and a compact XY footprint."""
    order = np.argsort(gps, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(gps[order])) + 1]
    stops = np.r_[starts[1:], len(order)]
    starts = starts[stops - starts == 3]
    groups = order[starts[:, None] + np.arange(3)]
    valid = (np.all(nr[groups] == 3, axis=1)
             & np.all(np.sort(rn[groups], axis=1) == (1, 2, 3), axis=1)
             & np.all(profile_mask[groups], axis=1))
    groups = groups[valid]
    if not len(groups):
        raise RuntimeError("No complete three-return pulse on the profile")
    xy = xyz[groups, :2]
    diameter = np.maximum.reduce([
        np.linalg.norm(xy[:, i] - xy[:, j], axis=1)
        for i, j in ((0, 1), (0, 2), (1, 2))
    ])
    span = np.ptp(xyz[groups, 2], axis=1)
    score = np.where(diameter <= 1.0,
                     span - 0.15 * diameter - 0.05 * across[groups].mean(axis=1),
                     -np.inf)
    if not np.isfinite(score).any():
        raise RuntimeError("No XY-consistent complete three-return pulse")
    selected = groups[np.argmax(score)]
    return selected[np.argsort(rn[selected])]


def _single_examples(xyz, cls, role, profile_mask, distance):
    selected = []
    for class_id, desired_distance in ((2, 16.0), (6, 41.0)):
        candidates = np.flatnonzero(profile_mask & (cls == class_id) & (role == 0))
        if not len(candidates):
            raise RuntimeError(f"No single return for LAS class {class_id}")
        selected.append(int(candidates[np.argmin(
            np.abs(distance[candidates] - desired_distance)
        )]))
    return selected


def _voxel_subset(xyz, role):
    grid = np.floor((xyz - xyz.min(axis=0)) / GRID_SIZE).astype(np.int32)
    _, original = np.unique(grid, axis=0, return_index=True)
    original.sort()
    return xyz[original], role[original], original


def _select_contexts(xyz, role):
    """Select same-role S centers with contrasting actual 48-NN histograms."""
    current_xyz, current_role, source_index = _voxel_subset(xyz, role)
    tree = cKDTree(current_xyz)
    candidates = np.flatnonzero(current_role == 0)
    if len(candidates) > 18000:
        candidates = np.sort(np.random.default_rng(44).choice(
            candidates, 18000, replace=False
        ))
    distance, index = tree.query(current_xyz[candidates], k=K + 1, workers=-1)
    not_center = index != candidates[:, None]
    neighbors = index[not_center].reshape(len(candidates), K)
    neighbor_distance = distance[not_center].reshape(len(candidates), K)
    neighbor_roles = current_role[neighbors]
    count = np.stack([(neighbor_roles == code).sum(axis=1)
                      for code in range(4)], axis=1)
    valid_count = ((neighbor_roles >= 0) & (neighbor_roles < 4)).sum(axis=1)
    fraction = count / np.maximum(valid_count[:, None], 1)
    z_span = np.ptp(current_xyz[neighbors, 2], axis=1)
    radius = neighbor_distance[:, -1]
    layered = (fraction[:, 1] + fraction[:, 2]
               + 0.55 * fraction[:, 2]
               + 0.06 * np.minimum(z_span, 8) - 0.08 * radius)
    layered[~((valid_count >= 46) & (fraction[:, 1] + fraction[:, 2] > 0.47)
              & (count[:, 2] >= 3) & (z_span > 2.0)
              & (radius < 2.5))] = -np.inf
    surface = fraction[:, 0] + fraction[:, 3] - 0.045 * np.minimum(z_span, 8)
    surface[~((valid_count >= 46) & (fraction[:, 0] + fraction[:, 3] > 0.78)
              & (fraction[:, 0] > 0.10) & (fraction[:, 3] > 0.10)
              & (radius > 0.7) & (radius < 2.5))] = -np.inf
    if not np.isfinite(layered).any() or not np.isfinite(surface).any():
        raise RuntimeError("No suitable same-role neighborhood pair")
    first = int(np.argmax(layered))
    separation = np.linalg.norm(
        current_xyz[candidates] - current_xyz[candidates[first]], axis=1
    )
    surface[separation < 15.0] = -np.inf
    if not np.isfinite(surface).any():
        raise RuntimeError("No separated second neighborhood")
    second = int(np.argmax(surface))
    examples = []
    for pos in (first, second):
        center = int(candidates[pos])
        near = neighbors[pos]
        examples.append({
            "center": center,
            "neighbors": near,
            "counts": count[pos],
            "radius_m": float(radius[pos]),
            "z_span_m": float(z_span[pos]),
        })
    return current_xyz, current_role, source_index, examples


def _profile_panel(ax, xyz, role, mask, distance, singles, pulse):
    """Narrow source index, not the principal visual evidence."""
    ids = np.flatnonzero(mask & (role >= 0))
    ax.scatter(distance[ids], xyz[ids, 2], s=3.4, color="#758188",
               alpha=0.78, lw=0, rasterized=True, zorder=2)
    for index in singles:
        ax.scatter(distance[index], xyz[index, 2], s=42,
                   facecolor=ROLE_COLORS[0], edgecolor="white",
                   linewidth=0.8, zorder=6)
    for index in pulse:
        ax.scatter(distance[index], xyz[index, 2], s=42,
                   facecolor=ROLE_COLORS[role[index]], edgecolor="white",
                   linewidth=0.8, zorder=6)
    ax.set_xlim(0, PROFILE_TO_M - PROFILE_FROM_M)
    ax.set_ylim(408.8, 431.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _pulse_card(ax, xyz, role, rn, distance, profile_mask, selected, zlim):
    """A larger real-profile crop; only highlighted returns use role colors."""
    center = float(distance[selected].mean())
    vicinity = np.flatnonzero(profile_mask & (np.abs(distance - center) <= 3.5)
                               & (xyz[:, 2] >= zlim[0]) & (xyz[:, 2] <= zlim[1]))
    ax.scatter(distance[vicinity] - center, xyz[vicinity, 2], s=12,
               color="#657179", alpha=0.72, lw=0, zorder=1)
    sequence = selected[np.argsort(xyz[selected, 2])[::-1]]
    if len(sequence) > 1:
        ax.plot(distance[sequence] - center, xyz[sequence, 2],
                color="#515B61", lw=1.1, linestyle=(0, (3, 2)))
    for index in sequence:
        code = int(role[index])
        ax.scatter(distance[index] - center, xyz[index, 2], s=128,
                   color=ROLE_COLORS[code], edgecolors="white", lw=1.0, zorder=4)
        label = (f"{ROLE_CODES[code]}  $r={int(rn[index])}$"
                 if len(sequence) > 1 else ROLE_CODES[code])
        ax.annotate(label,
                    (distance[index] - center, xyz[index, 2]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=13.5, fontweight="bold",
                    bbox=(dict(facecolor="white", edgecolor="none",
                               alpha=0.88, pad=0.35)
                          if len(sequence) > 1 else None))
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(*zlim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _context_panel(fig, cloud_box, hist_box, xyz, current_xyz,
                   current_role, example, *, azimuth=55.0, elevation=35.0):
    center = current_xyz[example["center"]]
    half = CONTEXT_DISPLAY_HALF_WIDTH_M
    local = xyz - center
    background = np.flatnonzero(
        (np.abs(local[:, 0]) < half) & (np.abs(local[:, 1]) < half)
        & (local[:, 2] > CONTEXT_DISPLAY_Z_RANGE_M[0])
        & (local[:, 2] < CONTEXT_DISPLAY_Z_RANGE_M[1])
    )
    if len(background) > CONTEXT_DISPLAY_CAP:
        background = np.sort(np.random.default_rng(63).choice(
            background, CONTEXT_DISPLAY_CAP, replace=False
        ))
    ax = fig.add_axes(cloud_box)
    near = current_xyz[example["neighbors"]]
    near_local = near - center
    cloud = np.vstack((local[background], near_local, np.zeros((1, 3))))
    rgba = np.tile(np.array([0.70, 0.73, 0.75, 1.0], dtype=np.float32),
                   (len(cloud), 1))
    gray = 0.74 - 0.19 * np.clip(
        (local[background, 2] - CONTEXT_DISPLAY_Z_RANGE_M[0])
        / (CONTEXT_DISPLAY_Z_RANGE_M[1] - CONTEXT_DISPLAY_Z_RANGE_M[0]),
        0.0, 1.0,
    )
    rgba[:len(background), :3] = gray[:, None]
    for code, color in enumerate(ROLE_COLORS):
        selected = current_role[example["neighbors"]] == code
        rgba[len(background):len(background) + K][selected] = mpl.colors.to_rgba(color)
    rgba[-1] = (1.0, 1.0, 1.0, 1.0)
    right, up, view = camera_basis(cloud[:, 0], cloud[:, 1],
                                   azimuth=azimuth, elevation=elevation)
    camera = (right, up, view, np.zeros(3))
    image, camera, projection, radius = render_point_cloud_image(
        cloud[:, 0], cloud[:, 1], cloud[:, 2], rgba,
        camera=camera, width=CONTEXT_RENDER_WIDTH, style="lit_edl",
        strength=3.2,
        point_scale=CONTEXT_POINT_SCALE,
        min_point_radius=CONTEXT_SHARED_RADIUS,
        max_point_radius=CONTEXT_SHARED_RADIUS, fractional_splat=True,
        projection_bounds=CONTEXT_PROJECTION_BOUNDS,
    )
    ax.imshow(image, origin="lower", interpolation="nearest")
    # Preserve all 48 return colors after depth compositing. These circles
    # have exactly the same radius as the gray point-render splats.
    px, py = to_image_pixels(near_local[:, 0], near_local[:, 1],
                             near_local[:, 2], camera, projection)
    for x, y, code in zip(px, py, current_role[example["neighbors"]]):
        ax.add_patch(Circle((x, y), radius=radius,
                            facecolor=ROLE_COLORS[int(code)],
                            edgecolor="none", zorder=5))
    cx, cy = to_image_pixels([0.0], [0.0], [0.0], camera, projection)
    ax.add_patch(Circle((cx[0], cy[0]), radius=radius,
                        facecolor="white", edgecolor=INK,
                        linewidth=0.8, zorder=6))
    near_center_x = float(np.mean(px))
    near_center_y = float(np.mean(py))
    near_width = max(38.0, float(np.ptp(px)) * 1.25)
    near_height = max(38.0, float(np.ptp(py)) * 1.25)
    ax.add_patch(Ellipse(
        (near_center_x, near_center_y), near_width, near_height,
        fill=False, edgecolor="#344B59", linewidth=1.15,
        linestyle=(0, (3, 2)), zorder=7,
    ))
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_facecolor("white")

    hist = fig.add_axes(hist_box)
    counts = example["counts"]
    fractions = counts / counts.sum()
    hist.barh(np.arange(4), fractions, height=0.62, color=ROLE_COLORS,
              linewidth=0, zorder=3)
    for code, fraction in enumerate(fractions):
        hist.text(max(float(fraction), 0.01) + 0.035, code, f"{fraction:.2f}",
                  ha="left", va="center", fontsize=11.7)
    hist.set_xlim(0, 1.10)
    hist.set_ylim(3.55, -0.55)
    hist.set_yticks(np.arange(4), ROLE_CODES)
    hist.set_xticks((0, 0.5, 1.0))
    hist.tick_params(labelsize=11.5, length=0)
    hist.grid(axis="x", color=GRID, linewidth=0.7)
    hist.set_axisbelow(True)
    hist.spines[["top", "right", "left", "bottom"]].set_visible(False)
    hist.set_facecolor("white")
    fig.add_artist(ConnectionPatch(
        xyA=(near_center_x + near_width / 2, near_center_y),
        coordsA=ax.transData, xyB=(-0.04, 0.5), coordsB=hist.transAxes,
        arrowstyle="-|>", mutation_scale=11, shrinkA=1.5, shrinkB=1.5,
        linewidth=1.0, color="#526B78", zorder=8,
    ))
    return len(background)


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    xyz, rn, nr, gps, cls, role = _load()
    profile_mask, distance, across = _profile(xyz)
    singles = _single_examples(xyz, cls, role, profile_mask, distance)
    pulse = _select_pulse(xyz, rn, nr, gps, profile_mask, across)
    current_xyz, current_role, source_index, examples = _select_contexts(xyz, role)
    assert all(current_role[e["center"]] == 0 for e in examples)

    fig = plt.figure(figsize=(11.4, 6.0))
    # One compact role legend; the two panels are independent evidence units.
    role_handles = [Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor=color, markeredgecolor="none",
                           markersize=9.5, label=f"{name} ({code})")
                    for code, name, color in zip(ROLE_CODES, ROLE_NAMES, ROLE_COLORS)]
    fig.legend(handles=role_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.986), frameon=False,
               fontsize=13.4, handletextpad=0.35, columnspacing=1.7)
    for x0, x1, y0, y1 in (
        (0.159, 0.159, 0.49, 0.88),
        (0.309, 0.309, 0.49, 0.88),
        (0.470, 0.470, 0.14, 0.89),
        (0.485, 0.982, 0.522, 0.522),
    ):
        fig.add_artist(Line2D(
            (x0, x1), (y0, y1), transform=fig.transFigure,
            color="#CBD3D8", linewidth=0.8, linestyle=(0, (4, 4)),
        ))

    card_specs = (
        (0.012, "Single: bare ground", np.asarray([singles[0]]),
         (407.5, 412.5), r"$r=1,\ n=1\ \rightarrow\ S$"),
        (0.162, "Single: roof", np.asarray([singles[1]]),
         (427.5, 432.5), r"$r=1,\ n=1\ \rightarrow\ S$"),
        (0.312, "Multiple returns", pulse,
         (408.8, 418.5), r"$n=3\ \rightarrow\ F,M,L$"),
    )
    for x, title, selected, zlim, notation in card_specs:
        fig.text(x + 0.072, 0.861, title, ha="center", va="center",
                 fontsize=11.1, fontweight="bold")
        card_ax = fig.add_axes((x + 0.008, 0.540, 0.128, 0.284))
        _pulse_card(card_ax, xyz, role, rn, distance, profile_mask,
                    selected, zlim)
        fig.text(x + 0.072, 0.513, notation, ha="center", va="center",
                 fontsize=11.2)
    fig.text(0.234, 0.448, "Source ALS profile", ha="center", va="center",
             fontsize=11.7, fontweight="bold")
    overview = fig.add_axes((0.013, 0.140, 0.442, 0.275))
    _profile_panel(overview, xyz, role, profile_mask, distance, singles, pulse)
    fig.text(0.234, 0.064, "(a)  Return roles from laser–scene interaction",
             ha="center", va="center", fontsize=14.4, fontweight="bold")

    background_counts = []
    for example, y, descriptor, camera_angles in zip(
        examples, (0.535, 0.151),
        ("Mixed-return neighborhood", "Single/last-dominant neighborhood"),
        ((55.0, 35.0), (15.0, 52.0)),
    ):
        fig.text(0.494, y + 0.316, descriptor,
                 ha="left", va="center", fontsize=11.9, fontweight="bold")
        background_counts.append(_context_panel(
            fig, (0.493, y + 0.033, 0.280, 0.246),
            (0.809, y + 0.047, 0.162, 0.215),
            xyz, current_xyz, current_role, example,
            azimuth=camera_angles[0], elevation=camera_angles[1],
        ))
    fig.text(0.734, 0.064, "(b)  Local return-role context",
             ha="center", va="center", fontsize=14.4, fontweight="bold")

    fig.savefig(PNG, dpi=400, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(SVG, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    metadata = {
        "source_tile": STEM,
        "pointcloud": str(POINT_PATH),
        "point_count": int(len(xyz)),
        "role_colors_S_F_M_L": list(ROLE_COLORS),
        "class_usage": "LAS class selects the ground and roof single-return examples only; the profile and context clouds are rendered in neutral gray",
        "profile": {
            "start_xy": PROFILE_START.tolist(),
            "end_xy": PROFILE_END.tolist(),
            "distance_range_m": [PROFILE_FROM_M, PROFILE_TO_M],
            "half_width_m": PROFILE_HALF_WIDTH_M,
            "point_count": int(profile_mask.sum()),
            "class_counts": {str(int(code)): int(np.sum(profile_mask & (cls == code)))
                             for code in np.unique(cls[profile_mask])},
            "single_ground_source_index": singles[0],
            "single_roof_source_index": singles[1],
            "multiple_source_indices": pulse.tolist(),
            "multiple_return_numbers": rn[pulse].tolist(),
            "multiple_same_gps_time": bool(np.all(gps[pulse] == gps[pulse[0]])),
            "multiple_xy_diameter_m": float(max(
                np.linalg.norm(xyz[pulse[i], :2] - xyz[pulse[j], :2])
                for i in range(3) for j in range(i + 1, 3)
            )),
            "multiple_z_span_m": float(np.ptp(xyz[pulse, 2])),
        },
        "context": {
            "grid_sample_m": GRID_SIZE,
            "grid_sample_method": "deterministic first point per occupied voxel; illustration, not an augmented training batch",
            "current_set_point_count": int(len(current_xyz)),
            "knn": K,
            "center_excluded": True,
            "display_xy_half_width_m": CONTEXT_DISPLAY_HALF_WIDTH_M,
            "display_relative_z_range_m": list(CONTEXT_DISPLAY_Z_RANGE_M),
            "point_scale": CONTEXT_POINT_SCALE,
            "shared_density_calibrated_point_radius_px": CONTEXT_SHARED_RADIUS,
            "shared_projection_bounds_uv": list(CONTEXT_PROJECTION_BOUNDS),
            "render_style": "lit_edl",
            "camera_azimuth_elevation_by_example": [[55.0, 35.0], [15.0, 52.0]],
            "renderer": "PointSpace static Lit-EDL point renderer; both examples share projection bounds and point size but use individually selected oblique views",
            "background_display_counts": background_counts,
            "examples": [{
                "sampled_center_index": int(e["center"]),
                "source_point_index": int(source_index[e["center"]]),
                "center_xyz": current_xyz[e["center"]].tolist(),
                "center_role": ROLE_CODES[int(current_role[e["center"]])],
                "counts_S_F_M_L": e["counts"].tolist(),
                "fractions_S_F_M_L": (e["counts"] / e["counts"].sum()).tolist(),
                "knn_radius_m": e["radius_m"],
                "z_span_m": e["z_span_m"],
            } for e in examples],
        },
        "caution": "LAS class chooses ground/roof single-return examples only. It is not an RRCL target. The voxel subset is a deterministic current-set analogue, not an exact stochastic training crop.",
    }
    META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(PNG)
    print(SVG)
    print("profile classes:", metadata["profile"]["class_counts"])
    print("context counts:", [e["counts"].tolist() for e in examples])


if __name__ == "__main__":
    main()
