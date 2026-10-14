import numpy as np

from paper_cvrl.q_geo import (
    P_AMB,
    P_INV,
    P_VIS,
    compute_q_geo,
    partition_q_geo,
)


def test_q_geo_is_geometry_only_and_separates_occluded_points():
    xy = np.stack(np.meshgrid(np.arange(6), np.arange(6)), axis=-1).reshape(-1, 2)
    top = np.c_[xy, np.ones(xy.shape[0])]
    bottom = np.c_[xy, np.zeros(xy.shape[0])]
    points = np.r_[top, bottom].astype(np.float64)
    q_geo = compute_q_geo(
        points,
        surface_cell_size=1.0,
        surface_radius=1.0,
        surface_z_tolerance=0.10,
        surface_oversample=2,
        surface_num_offsets=4,
        surface_adaptive_radius=False,
        surface_max_radius_cells=1,
        surface_slope_compensation=False,
    )
    assert q_geo.shape == (72,)
    assert np.all((0 <= q_geo) & (q_geo <= 1))
    assert q_geo[:36].mean() > 0.9
    assert q_geo[36:].mean() < 0.1


def test_q_geo_three_way_partition_has_explicit_ambiguity_band():
    region = partition_q_geo(
        np.asarray([0.0, 0.2, 0.21, 0.59, 0.6, 1.0], dtype=np.float32),
        invisible_max=0.2,
        visible_min=0.6,
    )
    assert region.tolist() == [P_INV, P_INV, P_AMB, P_AMB, P_VIS, P_VIS]
