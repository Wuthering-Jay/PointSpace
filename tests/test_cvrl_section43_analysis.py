import numpy as np
import pytest
from sklearn.metrics import f1_score

from paper_cvrl.analyze_cvrl_43 import _fit_probe
from paper_cvrl.bootstrap_cvrl_43 import _bootstrap_pair, _macro_f1
from paper_cvrl.section43_metrics import mean_iou, per_class_iou
from utils.pointcloud_render import (
    apply_lit_edl,
    apply_pcv_edl,
    compute_pcv,
    modulate_rgba_with_pcv,
    render_point_cloud_image,
    sample_hemisphere_directions,
)


def test_section43_probe_reports_all_visibility_regions():
    rng = np.random.default_rng(7)
    points_per_block = 60
    blocks = np.repeat(np.asarray([f"block_{i}" for i in range(10)]), points_per_block)
    label = np.tile(np.repeat(np.arange(3), points_per_block // 3), 10)
    region = np.tile(np.arange(3), label.size // 3 + 1)[: label.size]
    feature = np.eye(3, dtype=np.float32)[label] + 0.02 * rng.normal(
        size=(label.size, 3)
    )
    rows = {
        "source_block": blocks,
        "label": label,
        "region": region,
        "environment": np.repeat("synthetic", label.size),
        "height_bin": np.zeros(label.size, dtype=np.int64),
    }
    settings = {
        "test_fraction": 0.2,
        "max_train_points": 1000,
        "max_test_points": 1000,
        "max_knn_train_points": 1000,
        "max_knn_test_points": 1000,
        "linear_c": 1.0,
        "linear_max_iter": 200,
        "knn_k": 5,
        "random_state": 42,
    }
    result = _fit_probe(
        feature,
        rows,
        np.ones(label.shape, dtype=bool),
        settings,
    )
    assert result["linear"]["all"]["OA"] > 0.95
    assert set(result["linear"]) == {"all", "P_inv", "P_amb", "P_vis"}
    assert result["knn"]["all"]["OA"] > 0.95
    assert result["linear"]["all"]["mIoU"] > 0.90
    assert result["knn"]["all"]["mIoU"] > 0.90


def test_section43_bootstrap_metric_matches_sklearn():
    target = np.asarray([0, 0, 1, 1, 2, 2])
    prediction = np.asarray([0, 1, 1, 1, 2, 0])
    confusion = np.asarray([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    assert _macro_f1(confusion) == pytest.approx(
        f1_score(target, prediction, average="macro")
    )
    expected = np.asarray([1 / 3, 2 / 3, 1 / 2])
    assert np.allclose(per_class_iou(confusion), expected)
    assert mean_iou(confusion) == pytest.approx(expected.mean())


def test_section43_paired_bootstrap_rejects_different_rows():
    base = {
        "stem": np.asarray(["a", "a"]),
        "point_index": np.asarray([1, 2]),
        "target": np.asarray([0, 1]),
        "prediction": np.asarray([0, 1]),
        "region": np.asarray([0, 2]),
    }
    mismatched = {key: value.copy() for key, value in base.items()}
    mismatched["point_index"][1] = 3
    with pytest.raises(ValueError, match="point_index"):
        _bootstrap_pair(
            base,
            mismatched,
            np.asarray(["block_a", "block_a"]),
            None,
            np.random.default_rng(42),
        )


def test_lit_edl_is_finite_and_keeps_background_white():
    yy, xx = np.mgrid[:40, :48]
    occupied = (xx - 24) ** 2 + (yy - 20) ** 2 < 15**2
    depth = np.full((40, 48), np.nan, dtype=np.float32)
    depth[occupied] = (
        0.45
        + 0.25 * np.exp(-((xx[occupied] - 20) ** 2 + (yy[occupied] - 18) ** 2) / 60)
    )
    image = np.ones((40, 48, 4), dtype=np.float32)
    image[occupied, :3] = np.asarray([0.25, 0.55, 0.85], dtype=np.float32)
    result = apply_lit_edl(image.copy(), depth, occupied, radius=2, strength=5.5)
    assert np.isfinite(result).all()
    assert np.array_equal(result[~occupied], image[~occupied])
    assert not np.allclose(result[occupied, :3], image[occupied, :3])

    pcv_edl = apply_pcv_edl(
        image.copy(), depth, occupied, radius=2, strength=4.0
    )
    assert np.isfinite(pcv_edl).all()
    assert np.array_equal(pcv_edl[~occupied], image[~occupied])
    assert not np.allclose(pcv_edl[occupied, :3], image[occupied, :3])


def test_static_point_cloud_renderer_has_adaptive_circular_splats():
    yy, xx = np.mgrid[:16, :20]
    x = xx.ravel().astype(np.float32)
    y = yy.ravel().astype(np.float32)
    z = (0.15 * np.sin(xx / 3.0) + 0.1 * np.cos(yy / 2.0)).ravel()
    rgba = np.tile(
        np.asarray([[0.25, 0.55, 0.85, 1.0]], dtype=np.float32),
        (x.size, 1),
    )
    image, camera, projection, radius = render_point_cloud_image(
        x, y, z, rgba, width=240, style="lit_edl"
    )
    assert image.ndim == 3 and image.shape[2] == 4
    assert np.isfinite(image).all()
    assert len(camera) == 4 and len(projection) == 7
    assert 1 <= radius <= 5
    assert np.any(np.any(image[..., :3] < 0.99, axis=-1))

    combined, _, _, _ = render_point_cloud_image(
        x, y, z, rgba, width=240, strength=3.0, style="pcv_lit_edl"
    )
    assert np.isfinite(combined).all()
    assert not np.allclose(combined, image)

    _, _, _, small_radius = render_point_cloud_image(
        x,
        y,
        z,
        rgba,
        width=240,
        style="pcv_lit_edl",
        point_scale=0.40,
        max_point_radius=7,
    )
    _, _, _, large_radius = render_point_cloud_image(
        x,
        y,
        z,
        rgba,
        width=240,
        style="pcv_lit_edl",
        point_scale=1.70,
        max_point_radius=7,
    )
    assert small_radius < large_radius


def test_static_point_cloud_renderer_fixed_projection_for_matched_views():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = np.array([0.0, 0.3, 0.0], dtype=np.float32)
    z = np.array([0.0, 0.4, 0.8], dtype=np.float32)
    rgba = np.tile(np.array([[0.2, 0.5, 0.8, 1.0]], dtype=np.float32), (3, 1))
    bounds = (-3.0, 3.0, -2.0, 2.0)
    first = render_point_cloud_image(
        x, y, z, rgba, width=240, style="pcv",
        min_point_radius=2, max_point_radius=2, projection_bounds=bounds,
    )
    second = render_point_cloud_image(
        x + 0.5, y, z, rgba, camera=first[1], width=240, style="pcv",
        min_point_radius=2, max_point_radius=2, projection_bounds=bounds,
    )
    assert first[0].shape == second[0].shape
    assert first[2] == second[2]
    assert first[3] == second[3] == 2


def test_fractional_splats_distinguish_scales_with_same_integer_radius():
    yy, xx = np.mgrid[:16, :20]
    x = xx.ravel().astype(np.float32)
    y = yy.ravel().astype(np.float32)
    z = (0.15 * np.sin(xx / 3.0) + 0.1 * np.cos(yy / 2.0)).ravel()
    rgba = np.tile(
        np.asarray([[0.25, 0.55, 0.85, 1.0]], dtype=np.float32),
        (x.size, 1),
    )
    low, _, _, low_radius = render_point_cloud_image(
        x, y, z, rgba, width=240, style="pcv",
        point_scale=0.15, fractional_splat=True,
    )
    high, _, _, high_radius = render_point_cloud_image(
        x, y, z, rgba, width=240, style="pcv",
        point_scale=0.16, fractional_splat=True,
    )
    assert 0 < high_radius - low_radius < 0.5
    assert not np.allclose(low, high)
    for image in (low, high):
        pixels = image[..., :3].reshape(-1, 3)
        is_background = np.all(np.isclose(pixels, 1.0), axis=1)
        is_point_color = np.all(
            np.isclose(pixels, rgba[0, :3]), axis=1
        )
        assert np.all(is_background | is_point_color)


def test_pcv_hemisphere_is_uniform_and_upper_facing():
    directions = sample_hemisphere_directions(256)
    assert directions.shape == (256, 3)
    assert np.all(directions[:, 2] > 0)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)
    assert directions[:, 2].mean() == pytest.approx(0.5, abs=1e-3)


def test_pcv_darkens_points_beneath_an_occluding_surface():
    yy, xx = np.mgrid[-10:11, -10:11]
    roof = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 3.0)))
    lower = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    points = np.vstack((roof, lower)).astype(np.float32)
    pcv = compute_pcv(
        points,
        number_of_directions=64,
        resolution=256,
        point_radius=0,
    )
    assert pcv.shape == (points.shape[0],)
    assert np.all((pcv >= 0) & (pcv <= 1))
    assert pcv[: roof.shape[0]].mean() > pcv[roof.shape[0] :].mean()

    rgba = np.tile(
        np.asarray([[0.8, 0.4, 0.2, 1.0]], dtype=np.float32),
        (points.shape[0], 1),
    )
    shaded = modulate_rgba_with_pcv(rgba, pcv)
    assert np.allclose(shaded[:, 3], 1.0)
    assert shaded[roof.shape[0] :, :3].mean() < shaded[: roof.shape[0], :3].mean()


def test_pcv_nested_context_cannot_increase_visibility_on_fixed_grid():
    rng = np.random.default_rng(19)
    core = rng.uniform(-8.0, 8.0, size=(300, 3)).astype(np.float32)
    outer = rng.uniform(-30.0, 30.0, size=(800, 3)).astype(np.float32)
    origin = np.zeros(3, dtype=np.float32)
    common = dict(
        number_of_directions=32,
        point_radius=0,
        projection_center=origin,
        pixel_size=0.5,
        depth_tolerance=0.175,
        strict_bounds=True,
    )
    core_pcv = compute_pcv(core, resolution=256, **common)
    expanded_pcv = compute_pcv(
        np.vstack((core, outer)), resolution=512, **common
    )[: len(core)]
    assert np.all(expanded_pcv <= core_pcv + 1e-7)
