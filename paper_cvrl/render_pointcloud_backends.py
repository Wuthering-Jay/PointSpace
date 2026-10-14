"""Render matched NumPy, Open3D Filament and Blender Cycles comparisons.

Configuration is module-level for direct IDE execution.  Every backend uses
the same Swiss-Audit tile, deterministic point sample, near-nadir camera and
q_geo color mapping.  This script is an experimental renderer benchmark; it
does not replace the figures selected for the paper.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import colors as mcolors
from open3d.visualization import gui, rendering


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.pointcloud_render import camera_basis, render_point_cloud_image


AUDIT_ROOT = Path(r"G:\data\swiss\joint_tiles_100\Swiss-Audit")
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "section_43"
BLENDER_EXECUTABLE = Path(
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
)
BLENDER_SCRIPT = Path(__file__).resolve().parent / "blender_cycles_pointcloud.py"
ENVIRONMENTS = ("urban", "dense_vegetation")
MAX_POINTS = 50_000
WIDTH = 900
HEIGHT = 700
RANDOM_STATE = 42
BLENDER_ENGINE = "CYCLES"
_OPEN3D_APP_INITIALIZED = False


def _select_tiles(summary):
    selected = []
    for environment in ENVIRONMENTS:
        candidates = [
            tile for tile in summary["tile_stats"]
            if tile["environment"] == environment
        ]
        candidates.sort(key=lambda tile: tile["region_count"][0] / tile["points"])
        if not candidates:
            raise ValueError(f"No Swiss-Audit tile for environment={environment}")
        selected.append(candidates[len(candidates) // 2])
    return selected


def _prepare(tile, rng):
    cloud = laspy.read(AUDIT_ROOT / "q_geo_pointcloud" / f"{tile['stem']}.laz")
    count = len(cloud.points)
    index = (
        np.arange(count)
        if count <= MAX_POINTS
        else np.sort(rng.choice(count, size=MAX_POINTS, replace=False))
    )
    points = np.column_stack(
        (
            np.asarray(cloud.x)[index],
            np.asarray(cloud.y)[index],
            np.asarray(cloud.z)[index],
        )
    ).astype(np.float64)
    q_geo = np.asarray(cloud.q_geo)[index].astype(np.float32)
    colors = plt.get_cmap("viridis")(np.clip(q_geo, 0, 1))[:, :3].astype(np.float32)
    right, up, view = camera_basis(points[:, 0], points[:, 1])
    center = np.median(points, axis=0)
    centered = points - center
    u = centered @ right
    v = centered @ up
    u_low, u_high = np.quantile(u, [0.002, 0.998])
    v_low, v_high = np.quantile(v, [0.002, 0.998])
    u_span = max(float(u_high - u_low), 1e-6)
    v_span = max(float(v_high - v_low), 1e-6)
    aspect = WIDTH / HEIGHT
    ortho_scale = 1.08 * max(v_span, u_span / aspect)
    point_spacing = np.sqrt(max(u_span * v_span, 1e-6) / max(points.shape[0], 1))
    return {
        "points_world": points,
        "points": centered.astype(np.float32),
        "colors": colors,
        "right": right.astype(np.float32),
        "up": up.astype(np.float32),
        "view": view.astype(np.float32),
        "camera": (right, up, view, center),
        "ortho_scale": ortho_scale,
        "point_radius": float(np.clip(0.78 * point_spacing, 0.12, 0.52)),
        "point_size": float(np.clip(1.9 * np.sqrt(WIDTH * HEIGHT / points.shape[0]), 3.0, 9.0)),
    }


def _render_numpy(scene):
    rgba = np.column_stack(
        (scene["colors"], np.ones(scene["colors"].shape[0], dtype=np.float32))
    )
    image, _, _, _ = render_point_cloud_image(
        scene["points_world"][:, 0],
        scene["points_world"][:, 1],
        scene["points_world"][:, 2],
        rgba,
        camera=scene["camera"],
        width=WIDTH,
        strength=5.5,
        style="lit_edl",
    )
    return np.rint(np.clip(image[:, :, :3], 0, 1) * 255).astype(np.uint8)


def _render_open3d(scene):
    global _OPEN3D_APP_INITIALIZED
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(scene["points"].astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(scene["colors"].astype(np.float64))
    diameter = max(float(np.ptp(scene["points"], axis=0).max()), 1.0)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=diameter * 0.035, max_nn=36)
    )
    eye = scene["view"].astype(np.float64) * diameter * 3.0
    cloud.orient_normals_towards_camera_location(eye)

    application = gui.Application.instance
    if not _OPEN3D_APP_INITIALIZED:
        application.initialize()
        _OPEN3D_APP_INITIALIZED = True
    # The Windows wheel does not expose EGL headless rendering.  A hidden GUI
    # window supplies Filament's native Windows graphics context while
    # Application.render_to_image still returns a deterministic image.
    window = application.create_window(
        "PointSpace Filament render", WIDTH, HEIGHT, -10000, -10000
    )
    window.show(False)
    filament_scene = rendering.Open3DScene(window.renderer)
    filament_scene.set_background(
        np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    )
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.point_size = scene["point_size"]
    material.base_roughness = 0.85
    material.base_metallic = 0.0
    filament_scene.add_geometry("ALS point cloud", cloud, material)
    filament_scene.set_lighting(
        rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
        np.asarray([-0.55, 0.60, -0.58], dtype=np.float32),
    )
    filament_scene.view.set_post_processing(True)
    filament_scene.view.set_ambient_occlusion(True, True)
    filament_scene.view.set_antialiasing(True, False)
    filament_scene.view.set_sample_count(4)
    filament_scene.view.set_shadowing(True, rendering.View.ShadowType.PCF)
    filament_scene.camera.look_at(
        np.zeros(3, dtype=np.float32), eye.astype(np.float32), scene["up"]
    )
    aspect = WIDTH / HEIGHT
    vertical = scene["ortho_scale"]
    horizontal = vertical * aspect
    filament_scene.camera.set_projection(
        rendering.Camera.Ortho,
        -horizontal / 2,
        horizontal / 2,
        -vertical / 2,
        vertical / 2,
        0.1,
        diameter * 8.0,
    )
    application.run_one_tick()
    image = np.asarray(
        application.render_to_image(filament_scene, WIDTH, HEIGHT)
    ).copy()
    filament_scene.clear_geometry()
    window.close()
    application.run_one_tick()
    return np.flipud(image[:, :, :3])


def _render_blender(scene, stem):
    cache_dir = OUTPUT_DIR / "backend_render_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    input_path = cache_dir / f"{stem}_blender_input.npz"
    output_path = cache_dir / f"{stem}_blender_cycles.png"
    np.savez_compressed(
        input_path,
        points=scene["points"],
        colors=scene["colors"],
        camera_right=scene["right"],
        camera_up=scene["up"],
        camera_view=scene["view"],
        ortho_scale=np.float32(scene["ortho_scale"]),
        point_radius=np.float32(scene["point_radius"]),
        width=np.int32(WIDTH),
        height=np.int32(HEIGHT),
        engine=np.asarray(BLENDER_ENGINE),
    )
    command = [
        str(BLENDER_EXECUTABLE),
        "--background",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        str(input_path),
        str(output_path),
    ]
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    if not output_path.is_file():
        raise RuntimeError(f"Blender did not produce the expected image: {output_path}")
    return np.flipud(plt.imread(output_path)[..., :3])


def main():
    publication._set_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BLENDER_EXECUTABLE.is_file():
        raise FileNotFoundError(f"Blender executable not found: {BLENDER_EXECUTABLE}")
    summary = json.loads(
        (AUDIT_ROOT / "analysis" / "q_geo_summary.json").read_text(encoding="utf-8")
    )
    selected = _select_tiles(summary)
    rng = np.random.default_rng(RANDOM_STATE)
    fig, axes = plt.subplots(
        len(selected), 3, figsize=(14.6, 4.0 * len(selected)), constrained_layout=True
    )
    for row, tile in enumerate(selected):
        scene = _prepare(tile, rng)
        images = (
            _render_numpy(scene),
            _render_open3d(scene),
            _render_blender(scene, tile["stem"]),
        )
        for axis, image in zip(axes[row], images):
            axis.imshow(image, origin="lower")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        ratio = tile["region_count"][0] / tile["points"] * 100
        axes[row, 0].set_ylabel(
            f"{tile['environment'].replace('_', ' ').title()}\n"
            rf"$P_{{inv}}$={ratio:.1f}%",
            rotation=0, ha="right", va="center", labelpad=12,
        )
    for axis, title in zip(
        axes[0],
        ("(a) NumPy Lit-EDL", "(b) Open3D Filament", "(c) Blender Cycles"),
    ):
        axis.set_title(title)
    scalar = plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap="viridis")
    fig.colorbar(
        scalar, ax=axes, fraction=0.012, pad=0.012,
        label=r"Geometry observability $q_{geo}$",
    )
    fig.savefig(OUTPUT_DIR / "figure_43_renderer_backend_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
