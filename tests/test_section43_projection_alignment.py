"""Regression check for Swiss geospatial XYZ-to-image projection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_cvrl"))
from plot_projection_conflict import fit_xy_to_pixel_affine  # noqa: E402


def test_large_world_coordinate_affine_is_stable():
    rng = np.random.default_rng(7)
    xy_origin = np.asarray([2_615_500.0, 1_179_000.0])
    xy = xy_origin + rng.uniform(0, 100, size=(5000, 2))
    pixel = np.column_stack((
        1000 - 10 * (xy[:, 1] - xy_origin[1]),
        10 * (xy[:, 0] - xy_origin[0]),
    ))
    pixel += rng.normal(0, 0.25, size=pixel.shape)
    fitted_origin, affine, median_error = fit_xy_to_pixel_affine(xy, pixel)
    predicted = np.column_stack((xy - fitted_origin, np.ones(len(xy)))) @ affine

    assert np.allclose(affine[:2], [[0, 10], [-10, 0]], atol=0.02)
    assert np.median(np.linalg.norm(predicted - pixel, axis=1)) < 0.6
    assert median_error < 0.6
