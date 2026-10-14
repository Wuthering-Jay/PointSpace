"""Checks for the source-data rules behind the Swiss-Audit paper figures."""

from __future__ import annotations

import sys
from pathlib import Path

import laspy
import numpy as np
import pytest
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_cvrl"))
import plot_section43_paper_figures as figures  # noqa: E402
from plot_cvrl_43_results import PAPER_CLASS_NAMES  # noqa: E402
from plot_cvrl_43_results import REGION_COLORS  # noqa: E402
from plot_projection_conflict import (  # noqa: E402
    CASES, CASE_ROOT, CASE_VISIBILITY_ROOT, VISIBILITY_REGION_COLORS,
    _conflict_patch, _read_case,
)
from utils.pointcloud_render import render_point_cloud_image  # noqa: E402


def _tile(*, ground=0.4, vegetation=0.3, roof=0.1):
    return {
        "stem": "synthetic",
        "points": 1000,
        "class_stats": {
            "2": {"points": int(ground * 1000)},
            "3": {"points": int(vegetation * 1000)},
            "6": {"points": int(roof * 1000)},
        },
        "region_count": [200, 100, 700],
    }


def test_flat_bare_ground_requires_low_cover_and_low_relief(monkeypatch):
    tile = _tile(ground=0.90, vegetation=0.03, roof=0.01)
    monkeypatch.setattr(figures, "_ground_relief", lambda _stem: 4.0)
    assert figures._scene_group(tile) == "Flat bare ground"
    monkeypatch.setattr(figures, "_ground_relief", lambda _stem: 12.0)
    assert figures._scene_group(tile) == "Mixed / other"
    tile["class_stats"]["3"]["points"] = 100
    assert figures._scene_group(tile) == "Mixed / other"


def test_vegetation_strata_are_distinct():
    assert figures._scene_group(_tile(vegetation=0.7, roof=0)) == "Dense canopy"
    assert figures._scene_group(_tile(vegetation=0.35, roof=0)) == "Sparse vegetation"


def test_building_strata_use_roof_height(monkeypatch):
    class FakeHandle:
        def __init__(self, height):
            self.height = height

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_tensor(self, name):
            return (
                np.full(100, 6, dtype=np.uint8) if name == "classification"
                else np.full(100, self.height, dtype=np.float32)
            )

    tile = _tile(roof=0.25)
    monkeypatch.setattr(figures, "safe_open", lambda *_args, **_kwargs: FakeHandle(18))
    assert figures._scene_group(tile) == "Tall-building"
    monkeypatch.setattr(figures, "safe_open", lambda *_args, **_kwargs: FakeHandle(9))
    assert figures._scene_group(tile) == "Low-rise built"


def test_scene_aggregation_covers_every_tile(monkeypatch):
    tiles = [_tile(vegetation=0.7, roof=0), _tile(vegetation=0.3, roof=0)]
    stats = figures._scene_stats({"tile_stats": tiles})
    assert sum(entry["tiles"] for _, entry in stats) == 2
    assert sum(entry["points"] for _, entry in stats) == 2000
    assert all(abs(sum(entry[f"{region}_ratio"] for region in figures.REGIONS) - 1) < 1e-9
               for _, entry in stats)


def test_shared_paper_class_names():
    assert PAPER_CLASS_NAMES["6"] == "Building roof"
    assert PAPER_CLASS_NAMES["26"] == "Building facade"
    assert PAPER_CLASS_NAMES["15"] == "Tower"


def test_visibility_profile_colors_match_conflict_bars():
    assert VISIBILITY_REGION_COLORS == (
        REGION_COLORS["P_inv"], REGION_COLORS["P_amb"], REGION_COLORS["P_vis"],
    )


def test_figure_6_columns_have_equal_spacing():
    positions = np.asarray(figures.FIG6_PANEL_X_IN)
    gaps = np.diff(positions) - figures.FIG6_PANEL_SIZE_IN
    assert len(positions) == 4
    assert np.allclose(gaps, gaps[0], atol=1e-10)


@pytest.mark.skipif(not CASE_ROOT.exists(), reason="Swiss 200 m source data unavailable")
def test_qualitative_cases_match_200m_sources_and_pure_geometry_sidecars():
    for _label, stem, _target_class in CASES:
        source = CASE_ROOT / "pointcloud" / f"{stem}.las"
        header = laspy.open(source).header
        assert np.allclose(header.maxs[:2] - header.mins[:2], 200, atol=0.1)
        with safe_open(
            str(CASE_VISIBILITY_ROOT / f"{stem}.safetensors"), framework="np"
        ) as handle:
            assert handle.get_slice("region").get_shape()[0] == header.point_count
            assert handle.metadata()["definition"] == "pure_geometry_no_return_or_class_input"


@pytest.mark.skipif(not CASE_ROOT.exists(), reason="Swiss 200 m source data unavailable")
def test_figure_6_patch_examples_reproduce_surface_proxy_conflict():
    expected = ((3, 54, 165), (6, 60, 157))
    for (_label, stem, target_class), (proxy_class, support, target_count) in zip(
        CASES, expected
    ):
        _xyz, classification, region, pixel = _read_case(stem)
        patch = _conflict_patch(stem, classification, region, pixel, target_class)
        assert patch["proxy_class"] == proxy_class
        assert patch["support"] == support
        assert patch["purity"] >= 0.70
        assert len(patch["target_index"]) == target_count
        assert np.all(classification[patch["target_index"]] == target_class)
        assert np.all(region[patch["target_index"]] == 0)


def test_square_point_projection_uses_equal_scale():
    rng = np.random.default_rng(9)
    x = rng.uniform(0, 20, 100)
    y = rng.uniform(0, 8, 100)
    z = np.zeros_like(x)
    rgba = np.ones((100, 4), dtype=np.float32)
    image, _camera, projection, _radius = render_point_cloud_image(
        x, y, z, rgba, width=128, style="pcv", square_frame=True,
    )
    assert image.shape[:2] == (128, 128)
    assert projection[1] == projection[3]
