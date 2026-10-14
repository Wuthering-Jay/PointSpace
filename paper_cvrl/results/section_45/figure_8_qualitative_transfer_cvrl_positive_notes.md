# Additional Fig. 8 · CVRL-positive test regions

![CVRL-positive test regions](figure_8_qualitative_transfer_cvrl_positive.png)

**Suggested caption (English).** Targeted decoder-probe examples under identical nadir views and class palettes. DALES displays its complete 50 × 50 m tile, Vaihingen displays a fully covered 50 × 50 m test-region crop, and OpenGF retains its complete test tile. Columns show ground truth, Scratch, Sonata, D-DITR, VPSD, and CVRL. No error annotations are drawn. These are positive qualitative examples, not representative samples; the original median-case [Fig. 8](figure_8_qualitative_transfer.png) and the full-test-set Table 4 remain the references for typical and aggregate performance.

The source-tile protocol starts from the seven GT-only structurally screened tiles used for Fig. 8. No complete OpenGF tile among those seven has a CVRL mIoU win, so the pool was deterministically expanded to 60 additional GT-screened tiles. The OpenGF panel retains the resulting complete tile. DALES is already approximately 50 × 50 m. For Vaihingen, the displayed `Vaihingen3D_EVAL_WITH_REF_0176.las` 50 × 50 m crop occupies **100%** of a 10 × 10 XY coverage grid, with 50 m actual point span in both axes. CVRL gains **8.03 percentage points** in local mIoU over the stronger of Sonata and VPSD. The Vaihingen point splat scale is 1.22; this affects all six panels equally. This outcome-aware local selection is separate from the whole-tile selection. The exact source candidates, crop bounds, checkpoint-based metrics, and coverage checks are retained in [`qualitative_cache/positive_tile_selection_audit.json`](qualitative_cache/positive_tile_selection_audit.json).

| Dataset | Displayed region | Sonata mIoU | VPSD mIoU | CVRL mIoU | Gain over stronger baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| DALES | `5100_54490_0027.las` | 0.6825 | 0.6926 | 0.7593 | +0.0667 |
| Vaihingen | 50 × 50 m from `Vaihingen3D_EVAL_WITH_REF_0176.las` | 0.5430 | 0.5755 | 0.6557 | +0.0803 |
| OpenGF | `T2_0082.las` | 0.8346 | 0.7095 | 0.8526 | +0.0180 |

These are cherry-picked success cases by design and must not be cited as an estimate of CVRL's typical advantage. Across the Section 4.5 figures, shared physical classes use matching colors: car (teal), power line (yellow), fence/hedge (olive), DALES vegetation/Vaihingen tree (dark green), DALES building/Vaihingen facade (blue), and ground where directly comparable (tan). DALES pole is charcoal to avoid confusion with car. Regenerate this additional figure with `D:/app/anaconda3/envs/pointspace/python.exe paper_cvrl/plot_section45_positive_plan.py`.
