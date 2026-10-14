# Fig. 8 · Decoder-probe qualitative transfer

![Fig. 8](figure_8_qualitative_transfer.png)

**Suggested caption (English).** Fig. 8. Qualitative semantic segmentation under frozen-encoder decoder probing on DALES, Vaihingen, and OpenGF. Columns show ground truth and predictions from Scratch, Sonata, D-DITR, VPSD, and CVRL. All methods use their saved best decoder-probe checkpoint and the same full-tile test-fragment aggregation. Within each dataset, the north-up orthographic nadir view, displayed points, and semantic palette are identical across columns. No error boxes or other interpretive annotations are drawn. Scenes were chosen by a predeclared structural screen using ground truth, followed by the median tile-level CVRL–strong-baseline gain among seven eligible candidates; they are not maximum-gain examples. Point clouds are rendered using the repository's PCV + edge-only EDL renderer with adaptive circular splats (display scale 0.94 for DALES and 0.86 elsewhere). Quantitative comparisons are reported in Table 4 rather than in this figure.

Selection uses only test-set GT for structural eligibility. Among eligible tiles, the reference gain is `CVRL mIoU − max(Sonata mIoU, VPSD mIoU)`, with tile mIoU averaged over GT-present classes; the middle of seven sorted candidate gains is used. This is a **selection statistic**, not a substitute for the full-test-set Table 4 metric. No tile-level predictions, labels, or metrics have been edited for display. The full candidate list, scores, gains, and checkpoint paths are in [`qualitative_cache/selection_audit.json`](qualitative_cache/selection_audit.json). The audit retains previously computed per-method error regions for reproducibility, but these are not drawn in the figure.

| Dataset | Chosen test tile | CVRL − stronger baseline, tile mIoU |
| --- | --- | ---: |
| DALES | `5135_54430_0021.las` | −0.0041 |
| Vaihingen | `Vaihingen3D_EVAL_WITH_REF_0113.las` | −0.0117 |
| OpenGF | `T1_0224.las` | −0.0212 |

The negative median gains are retained intentionally: this figure illustrates spatial behavior on representative complex scenes, not a claim that CVRL wins every tile. In particular, OpenGF's selected block has substantial vertical relief and mixed ground/non-ground structure; the remaining visual differences are subtle at full-tile scale. Method rankings and statistical interpretation belong to Table 4.

Regenerate with `D:/app/anaconda3/envs/pointspace/python.exe paper_cvrl/plot_section45_qualitative.py` from the repository root. Prediction caches are stored under `qualitative_cache/{dataset}` and are not exported as LAS files.
