# Fig. 2 · CVRL overall framework

![CVRL framework](figure_2_cvrl_framework.png)

**Suggested caption (English).** CVRL uses a single LiDAR encoder under asymmetric visual and return supervision. The visual pathway reads protected P2–P4 hierarchy features and learns from DINO patch targets only where image-visible point–patch relations are valid. The return pathway decodes the same hierarchy and predicts point return roles and local return-role context on points with valid LAS return metadata. During pretraining, the orthophoto teacher and return fields supply targets only; neither enters the LiDAR encoder. The persistent 128D visual and 72D return features concatenate into a 200D LiDAR-only representation for downstream fine-tuning.

The figure is an architectural overview, not a layer-by-layer computational graph. In the Swiss LitePT-v1m4 configuration, P2/P3/P4 have 144/252/504 channels (900D when fused). The shared semantic projector produces a native 1024D visual feature; fixed PCA maps its persistent point readout to 128D. VPSD aggregates token features per visible image patch **before** the shared projector for the training loss, while the retained visual readout projects individual points. The return decoder produces 72D point features; its role and context heads are training-only. RRCL decoder gradients are detached at P2–P4, while accumulated symmetric PCGrad is applied to the shared stem/P0/P1 during the joint phase. Solid paths denote retained feature flow and dashed paths denote pretraining targets. Fig. 3 and Fig. 4 provide the detailed patch-set and return-physics mechanisms, respectively.

Source: [plot_fig2_cvrl_framework.py](../../plot_fig2_cvrl_framework.py).
