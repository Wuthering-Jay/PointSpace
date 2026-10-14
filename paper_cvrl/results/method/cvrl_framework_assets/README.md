# CVRL framework real-data assets

Selected Swiss-Audit tile:
`swisssurface3d_2020_2760-1188_2056_5728_0086`.

It contains approximately 33.3% ground, 31.6% vegetation, and 33.0%
building-roof points. The paired orthophoto has little strong shadow
(about 0.2% of pixels below luminance 0.16), making it suitable for the
framework input glyphs.

## Main PNG assets

- `01_orthophoto.png`: original paired orthophoto.
- `02_orthophoto_patch4x4.png`: 4 x 4 image patches separated by white gaps.
- `03_als_intensity_topview.png`: north-up ALS intensity rendering.
- `04_cvrl_pca_topview.png`: full-resolution frozen CVRL-200 feature PCA.
- `05_als_return_role_topview.png`: S/F/M/L return-role coloring.

The two point-cloud panels use exactly the same north-up nadir camera,
orthographic projection, and display point radius. Intensity uses a robust
1--99% logarithmic stretch. Borderless `*_raw.png` and transparent-background
`*_transparent.png` variants are included for manual SVG composition.

The feature display uses all 373,450 point features exported from the official
Swiss-100 CVRL checkpoint. Features are L2-normalized and reduced to a
24-dimensional PCA subspace using a fixed 120,000-point sample. Three Fisher
display axes within that subspace preserve Unclassified/small-object, Ground,
Vegetation, and Building-roof contrasts. The coordinates are robustly clipped,
globally oriented from class centroids, and converted through CIELAB to sRGB.
Labels only determine the global display projection; they never assign colors
to individual points. Exact transforms and parameters are recorded in
`asset_manifest.json`.

Reproduce the feature and images from the repository root:

```powershell
D:/app/anaconda3/envs/pointspace/python.exe tools/test.py --config-file paper_cvrl/config_cvrl_framework_asset_export.py
D:/app/anaconda3/envs/pointspace/python.exe paper_cvrl/prepare_cvrl_framework_assets.py
```

Exact paths, camera values, renderer settings, and PCA statistics are stored
in `asset_manifest.json`.
