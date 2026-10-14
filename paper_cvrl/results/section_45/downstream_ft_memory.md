# Downstream full-fine-tuning memory calibration

## Protocol

- Device: NVIDIA GeForce RTX 4080, 15.99 GiB.
- Safety criterion: peak CUDA **reserved** memory no greater than 14.5 GiB.
- Workload: complete semantic-segmentation forward pass, CrossEntropy + Lovasz
  backward pass, and the first AdamW update under BF16 AMP.
- Input: the densest transformed training sample selected from the 32 largest
  source files of each dataset. The selected samples contain 160,000 points
  for DALES, 75,289 for Vaihingen, and 160,000 for OpenGF.
- FT is used because it retains all activations and optimizer states and is the
  maximum-memory transfer protocol. The resulting limit is conservatively
  reused for decoder and linear probing.

The standard family comprises Scratch, Point-M2AE, MSC, Sonata, D-DITR,
Concerto, VPSD, and RRCL. After checkpoint conversion these methods use the
same downstream LitePT graph and produced identical FT memory peaks. CVRL is
measured separately because it retains the 200-D visual/return representation.

## Absolute measured ceilings

| Dataset | Family | Maximum micro batch | Peak allocated (GiB) | Peak reserved (GiB) | Next value |
| --- | --- | ---: | ---: | ---: | --- |
| DALES | Standard LitePT | 8 | 13.082 | 14.182 | 9: OOM at the 14.5-GiB allocator limit |
| DALES | CVRL | 6 | 13.187 | 14.199 | 7: OOM at the 14.5-GiB allocator limit |
| Vaihingen | Standard LitePT | 11 | 14.215 | 14.486 | 12: OOM at the 14.5-GiB allocator limit |
| Vaihingen | CVRL | 7 | 12.190 | 13.869 | 8: 16.285 GiB reserved without the enforced limit |
| OpenGF | Standard LitePT | 13 | 13.417 | 14.281 | 14: OOM at the 14.5-GiB allocator limit |
| OpenGF | CVRL | 10 | 13.331 | 14.172 | 11: OOM at the 14.5-GiB allocator limit |

These are capacity ceilings, not the final controlled experimental setting.
Using them directly would change the effective optimizer batch between methods
(for example, 16 cannot be divided exactly into micro batches 6, 7, 10, 11, or
13), introducing a training-protocol confound.

## Configured controlled setting

All generated configurations use micro batch 6 and gradient accumulation 2,
giving a uniform effective batch 12 across every dataset, method, and transfer
protocol. DALES-CVRL is the limiting measured case: micro batch 6 reserves
14.199 GiB. Linear and decoder probing conservatively reuse the FT-tested
setting. The absolute ceilings above remain useful for future throughput-only
experiments, but are not mixed into the controlled comparison.

Raw measurements are stored in `downstream_ft_memory.json`; the reproducible
scripts are `paper_cvrl/benchmark_downstream_ft_memory.py` and
`paper_cvrl/refine_downstream_ft_memory.py`.
