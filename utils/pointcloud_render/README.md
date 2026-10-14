# Static point-cloud rendering

This package contains the repository's formal deterministic renderer for paper
figures.  It needs no OpenGL context and produces the same result on headless
and desktop machines.

The default `lit_edl` style combines:

- orthographic near-nadir projection;
- circular, density-adaptive point splats and a z-buffer;
- tiny-gap coverage correction for clustered canopy samples;
- restrained multi-scale EDL boundaries;
- fine/coarse surface normals with key and fill lighting;
- broad signed canopy relief and low-opacity cavity shading;
- a hard illumination floor that prevents outline-attached black blobs.

It intentionally does not synthesize screen-space cast shadows.  Such shadows
are unstable for topology-free point sets and were the source of dark clumps
along roofs and tree crowns in the earlier experimental renderer.

```python
from utils.pointcloud_render import render_point_cloud_image

image, camera, projection, point_radius = render_point_cloud_image(
    x,
    y,
    z,
    rgba,                 # float [N, 4], range [0, 1]
    width=1200,
    strength=5.5,
    style="lit_edl",     # or "classic_edl" for an ablation baseline
)
```

Use `render_point_cloud_axis` to draw directly into a Matplotlib axis and
`to_image_pixels` when annotations must be projected into the rendered image.
The camera returned from the first call can be passed into later calls to make
matched multi-panel comparisons.

## Portion of Visible Sky (PCV)

`pcv.py` also provides view-independent hemispherical visibility shading:

```python
from utils.pointcloud_render import compute_pcv, modulate_rgba_with_pcv

pcv = compute_pcv(
    xyz,
    number_of_directions=128,
    resolution=1024,
    point_radius=0,
)
shaded_rgba = modulate_rgba_with_pcv(rgba, pcv)
```

For the combined renderer, use `style="pcv_edl"` after color modulation, or
set `post_style="pcv_edl"` in `render_pcv_point_cloud_image`.  This adds only a
locally adaptive symmetric depth edge; PCV remains responsible for global
occlusion, so ambient lighting is not applied twice.

For final color figures, compare `post_style="pcv_edl"` (edge-only, no broad
surface lighting) and `post_style="pcv_lit_edl"` (shadow-free surface lighting)
at `edl_strength=3.0`.  In the Swiss 100 m comparison the edge-only version
preserved the desired PCV color balance better.  Keep pure `post_style="pcv"`
when displayed brightness should remain closest to the PCV scalar itself.

Display point size is controlled separately with `render_point_scale`.  It is
multiplied by projected mean point spacing and does not change the PCV field.
The PCV wrapper now defaults to `render_point_scale=0.86` with
`render_fractional_splat=True`; `render_min_radius` and `render_max_radius`
bound the resulting circular splat radius. The generic NumPy renderer retains
its original integer-splat default unless explicitly requested otherwise.
The default renderer rounds this radius to an integer.  Direct calls to
`render_point_cloud_image` or `render_point_cloud_axis` can opt into
`fractional_splat=True` for a continuous radius with deterministic, fully
opaque outer-ring coverage.  It avoids the white halos caused by blending
partial point coverage into the white background.  This changes display
coverage but never the PCV computation.

Compute PCV on the complete local scene and only then take a display sample.
Computing it on a sparse publication sample creates projection holes and
overestimates visible sky.  `point_radius=0` matches a one-pixel OpenGL point;
use radius one only when the source cloud itself is sparse.  For final figures,
128 directions are usually smooth enough, while 256 is recommended for a
convergence check.

The implementation follows the geometric definition documented for
[CloudCompare qPCV](https://www.cloudcompare.org/doc/wiki/index.php/ShadeVis_%28plugin%29):
uniform hemispherical directions, an orthographic depth buffer per direction,
and a visible-direction fraction per point.  It is an independent NumPy
implementation and does not reuse CloudCompare's GPL source code.
