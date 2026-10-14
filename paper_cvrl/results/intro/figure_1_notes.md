# Fig. 1 · Visible-Surface Bottleneck

![Visible-Surface Bottleneck](figure_1_visible_surface_bottleneck.png)

**Suggested caption (English).** Real Swiss orthophotos and their corresponding 0.5 m-wide ALS vertical sections illustrate the visible-surface bottleneck. In (a), a ground return and canopy-interior points project onto an image patch dominated by visible vegetation; in (b), building-facade returns project onto a patch dominated by visible roof points. The white dashed A–A′ lines mark the plotted sections. Point fill denotes LAS class, while blue and red outlines denote the highlighted geometrically visible ($P_{vis}$) and invisible ($P_{inv}$) points *within the same image patch*. The visible-surface class is the majority LAS label among visible points corresponding to that patch; it is a posterior projection-conflict proxy, **not** a DINO prediction. Thus, a valid 2D projection does not necessarily provide a semantically corresponding target for every 3D point.

The two cases reuse the audited 200 m Swiss source tiles, point-to-pixel correspondence, and geometric visibility sidecars from Section 4.3. The selected vegetation patch has 54 visible proxy points (100% vegetation) and 165 projected invisible ground points. The roof patch has 60 proxy points (98.3% building roof) and 157 projected invisible facade points. The section center passes through the median XY location of the corresponding invisible target points. The [case audit](figure_1_case_audit.json) records source tile IDs, patch IDs, support, purity, affine alignment error, section geometry, and the number of highlighted same-patch points.

Regenerate with `D:/app/anaconda3/envs/pointspace/python.exe paper_cvrl/plot_fig1_problem_teaser.py`.
