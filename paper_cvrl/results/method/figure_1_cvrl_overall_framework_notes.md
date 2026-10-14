# Fig. 1 — Overall framework of CVRL

## Deliverables and provenance

- Editable, self-contained SVG: `figure_1_cvrl_overall_framework.svg`
- Offline HTML with inline SVG and a download button: `figure_1_cvrl_overall_framework.html`
- High-resolution PNG: `figure_1_cvrl_overall_framework.png`
- Build script: `paper_cvrl/plot_cvrl_framework_code_aligned.py`
- Source and rendering parameters: `figure_1_cvrl_overall_framework.json`

Both raster inputs refer to the same real Swiss-Audit tile,
`swisssurface3d_2024_2679-1248_2056_5728.copc_0033`. The orthophoto comes
from the paired Swiss-100 image. The ALS thumbnail is generated from the
Swiss-Audit LAS using `utils.pointcloud_render` with an oblique orthographic
camera, density-adaptive circular splats, a z-buffer and Lit-EDL. It is rendered
in a uniform muted blue-grey and its white raster background is converted to
alpha, so it reads as a real 3D ALS object rather than a rectangular semantic
map. LAS semantic labels are never used. The two source images are embedded in
the standalone SVG, while all diagram lines, text and modules remain editable
vector elements.

Run from the repository root:

```powershell
D:/app/anaconda3/envs/pointspace/python.exe paper_cvrl/plot_cvrl_framework_code_aligned.py
```

## Containerized visual organization

The current paper layout preserves the same code topology but groups local
mechanisms into nested containers so that internal edges no longer dominate
the full figure:

- VPSD is divided into **Visual teacher**, **Patch correspondence and
  aggregation**, and **Projection and visual readout**.
- RRCL is divided into **Return target construction**, **Return decoder
  readout**, and **Pretraining-only heads**.
- `zV`, `zR`, and their 200D concatenation occupy a dedicated
  **Representation outputs** band.
- The three downstream protocols are independent branches from the retained
  CVRL backbone. They are not sequential processing stages.

All cross-container connections use fixed boundary ports and orthogonal
routing. Solid heavy edges denote retained representation flow, while dashed
edges remain confined to teacher/target supervision and detached decoder
inputs. Containerization changes only visual grouping; no scientific node,
dimension, loss, detach boundary, or data dependency is changed.

The standalone SVG exposes stable editable groups for local refinement:

- `V1_visual_teacher`, `V2_correspondence_aggregation`,
  `V3_projection_readout`;
- `shared_lidar_encoder`;
- `R1_return_target_construction`, `R2_return_decoder_readout`,
  `R3_pretraining_heads`;
- `representation_outputs`;
- `down_backbone`, `down_frozen`, `down_decoder`, and
  `down_full_finetuning`.

Role and context targets remain distinct inside R1 and R3, but their two
pretraining-only supervision routes share a single edge-routed bus between
the containers. This reduces global wiring without merging either target or
loss mathematically.

## Reading order

1. The ALS tile enters a **single shared LitePT-v1m4 encoder**. The central
   hierarchy contains Stem and P0–P4, with channels 36/36/72/144/252/504.
   LAS return number and return count do not enter this arrow.
2. The upper **VPSD** route makes an explicit many-to-many visible token–patch
   correspondence graph.
   P2, P3 and P4 are **separately aggregated at the patch level** into
   144D, 252D and 504D features. Their patch-level concatenation is 900D;
   the native DINO projector maps it to a 1024D patch prediction for cosine
   distillation against the frozen DINOv3 teacher. This is the
   aggregate-then-project training path.
3. The retained visual readout uses P2-aligned hierarchical token features,
   the **same projector weights**, and a fixed global PCA to form pointwise
   `zV` at 128D. PCA does not sit between teacher and student in the
   1024D distillation loss.
4. The lower **RRCL** route includes the complete P4→D3→D2→D1→D0 decoder
   with P3/P2/P1/P0 skip inputs. In full mode, the cloned P2/P3/P4 encoder
   features are detached on the RRCL decoder path. The decoder's 72D point
   feature passes through LayerNorm to become the retained `zR`.
5. The role and context heads **read from `zR`**. Return metadata constructs
   the S/F/M/L role target; current-point-set XYZ KNN with K=48, excluding
   the center and invalid roles, constructs its local histogram target.
   Both heads are pretraining-only.
6. The two retained point features concatenate as `[zV | zR] = 128+72=200D`.
   LiDAR-only downstream uses the complete CVRL backbone, including visual
   and return readouts. Frozen representation, decoder probe and full
   fine-tuning are shown as three alternative protocol branches, not a serial
   computation.

VPSD gradients can update the full encoder. RRCL directly shares Stem/P0/P1
with VPSD, while the deep RRCL skips are detached. The first 10% of training
warms up the return loss. From 10% onward, the trainer applies symmetric
PCGrad to Stem/P0/P1 **after a full gradient accumulation window**.

## Relationship to implementation

The architecture is checked against:

- `configs/cvrl/pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py`
- `pointspace/models/backbone/cvrl/cvrl_v1m1.py`
- `pointspace/models/backbone/cvrl/common.py`
- `pointspace/models/backbone/litept_v1/litept_v1m4_cvrl.py`
- `pointspace/models/backbone/rrcl/ops.py`
- `pointspace/engines/train.py` (`CVRLPCGradTrainer`)

The simplified glyphs are schematic: token counts, patch grid size and
height colors in the thumbnail explain the mechanism but are not training
tensor shapes or semantic predictions. The dimension labels, loss ordering,
gradient boundary and retained outputs reflect the code.

Suggested caption: **Fig. 1. Overall framework of CVRL.** One shared LitePT
encoder is trained with asymmetric visual and return supervision. VPSD
aggregates visible P2/P3/P4 tokens into patch-level 900D features before
1024D DINOv3 distillation, while its retained visual point readout passes
through fixed PCA to 128D. RRCL decodes the complete LiDAR hierarchy into
a 72D point representation supervised by return role and local return
context; its deep skip inputs are detached during joint training. The
retained 128D and 72D readouts form a 200D CVRL backbone for LiDAR-only
downstream tasks. The ALS and orthophoto thumbnails are a paired real
Swiss-Audit scene; the ALS glyph is rendered directly from the real point
geometry without semantic coloring.
