# Portion of Visible Sky for static point-cloud rendering

## Definition

For a point $p_i$ and a set of equal-solid-angle directions
$\{\omega_m\}_{m=1}^{M}$ on the upper hemisphere, define

$$
V_i(\omega_m)=
\begin{cases}
1, & p_i\text{ is on the front depth layer viewed from }\omega_m,\\
0, & p_i\text{ is occluded.}
\end{cases}
$$

The Portion of Visible Sky is

$$
\operatorname{PCV}_i=\frac{1}{M}\sum_{m=1}^{M}V_i(\omega_m).
$$

This is a discrete ambient-visibility integral, not a collection of finite
point lights.  Each sample represents a directional light at infinity.  Equal
weights are correct because the directions represent equal solid angles.

## Implementation

1. Generate deterministic Fibonacci samples over the $+Z$ hemisphere.
2. For every direction, construct an orthographic camera basis.
3. Project the complete point cloud into a square depth buffer.
4. Keep the maximum view-space depth for each pixel.
5. Count a point as visible when its depth is within a small quantization
   tolerance of the front layer at its projected pixel.
6. Divide the accumulated count by the number of directions.
7. Convert PCV to bounded illumination and multiply it with the linear-RGB
   version of the requested class, elevation, feature, or scalar color.

The per-point PCV field is view-independent.  Changing the final publication
camera therefore does not require recomputing it.

For a controlled comparison of nested spatial extents, pass the same
`projection_center` and `pixel_size` to `compute_pcv` for every extent.  Use
even buffer resolutions large enough to cover every projected point and set
`strict_bounds=True`.  This keeps the world-space pixel grid aligned; adding
outer context can then only preserve or reduce a core point's PCV.  Letting
each crop refit its own projection can produce artificial non-monotonic changes.

## PCV + EDL composition

The combined mode intentionally does not stack PCV with the complete Lit-EDL
illumination model.  PCV provides broad hemispherical occlusion first.  A
second, edge-only EDL pass then restores narrow roof/road discontinuities and
broad crown boundaries.  Its strength and fine/broad edge mixture decrease in
locally rough regions, preventing individual vegetation returns from becoming
black speckles.  There is no directional light, cavity term, or cast shadow in
this second pass.

The Swiss-Audit comparison also includes a presentation-oriented full
composition.  `pcv_lit_edl` first applies PCV and then the current shadow-free
Lit-EDL model at a reduced strength of 3.0.  It gave the clearest canopy crowns,
roof planes and roads.  It is recommended for qualitative color figures, while
pure PCV remains the correct mode for inspecting or reporting the actual
visible-sky scalar.  The edge-only mode is retained as a conservative ablation:
it is artifact-safe but produced only a small visual gain over PCV alone.

## Recommended settings

| Purpose | Directions | Buffer | Point radius |
|---|---:|---:|---:|
| Fast preview | 64 | 512 | 0 |
| Normal figure | 128 | 1024 | 0 |
| Convergence/final audit | 256 | 1024–1536 | 0 |
| Genuinely sparse cloud | 128 | 512–1024 | 1 |

An oversized point radius systematically lowers PCV by making every point a
larger occluder.  A buffer that is too coarse has the same effect.  Conversely,
subsampling before PCV creates holes and falsely raises PCV.  The implementation
therefore uses one-pixel points and expects PCV to be computed before display
sampling.

On the selected 494,220-point Swiss dense-vegetation audit tile, 128 directions
reached Pearson $r=0.9633$ and MAE $=0.0274$ against the 256-direction result;
64 directions reached $r=0.9362$ and MAE $=0.0369$.  Therefore 128 is the
default for routine figures, while 256 remains the final convergence setting.

## ALS-specific choice

Use the upper hemisphere for airborne LiDAR.  A full sphere is suitable only
for closed objects: on an open terrain/vegetation cloud it lets illumination
enter from below and weakens the intended relief.  Points close to tile borders
have incomplete occluder context, so quantitative use should compute on a
larger tile with a halo and crop the center.  For qualitative rendering, a
complete 100 m tile is normally sufficient.

## References

- [CloudCompare ShadeVis/qPCV documentation](https://www.cloudcompare.org/doc/wiki/index.php/ShadeVis_%28plugin%29)
- [CloudCompare qPCV implementation](https://github.com/CloudCompare/CloudCompare/tree/master/plugins/core/Standard/qPCV)
- M. Tarini, P. Cignoni, and C. Montani, “Ambient Occlusion and Edge Cueing to
  Enhance Real Time Molecular Visualization,” IEEE TVCG, 2006,
  [doi:10.1109/TVCG.2006.115](https://doi.org/10.1109/TVCG.2006.115).
