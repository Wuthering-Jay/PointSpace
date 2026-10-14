"""Deterministic NumPy renderers for static point-cloud figures."""

from .numpy_renderer import (
    apply_lit_edl,
    apply_pcv_edl,
    camera_basis,
    project_points,
    render_point_cloud_axis,
    render_point_cloud_image,
    to_image_pixels,
)
from .pcv import (
    compute_pcv,
    modulate_rgba_with_pcv,
    pcv_to_illumination,
    render_pcv_point_cloud_axis,
    render_pcv_point_cloud_image,
    sample_hemisphere_directions,
)

__all__ = [
    "apply_lit_edl",
    "apply_pcv_edl",
    "camera_basis",
    "project_points",
    "render_point_cloud_axis",
    "render_point_cloud_image",
    "to_image_pixels",
    "compute_pcv",
    "modulate_rgba_with_pcv",
    "pcv_to_illumination",
    "render_pcv_point_cloud_axis",
    "render_pcv_point_cloud_image",
    "sample_hemisphere_directions",
]
