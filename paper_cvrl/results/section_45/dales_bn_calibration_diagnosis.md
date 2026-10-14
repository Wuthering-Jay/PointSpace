# DALES Swiss-BN Calibration Diagnosis

## Scope

This report compares the original one-epoch/one-loop DALES smoke run in
`exp/downstream_smoke_e1l1_all/dales` with the Swiss clean-input BatchNorm
recalibration run in `exp/downstream_smoke_bncal/dales`.  Scratch is unchanged
and acts as a determinism/control check.  All calibrated runs were verified
from their logs to load the corresponding `*_bncal_swiss.pth` checkpoint.

## Main result

Swiss BN recalibration fixes a large part of the decoder-probe collapse, but
does not explain or fix the full fine-tuning problem.  Large BN running values
were therefore a real transfer/interface issue, but were not the sole cause
and were not automatically invalid statistics.

| Method | Probe | Original mIoU | BN-calibrated mIoU | Change (pp) |
|---|---|---:|---:|---:|
| Point-M2AE | Linear | 0.2943 | 0.2536 | -4.07 |
| Point-M2AE | Decoder | 0.1092 | 0.4531 | **+34.39** |
| Point-M2AE | Fine-tuning | 0.3248 | 0.3235 | -0.13 |
| D-DITR | Linear | 0.2624 | 0.2311 | -3.13 |
| D-DITR | Decoder | 0.1436 | 0.3849 | **+24.13** |
| D-DITR | Fine-tuning | 0.1861 | 0.1791 | -0.70 |
| Concerto | Linear | 0.2884 | 0.2697 | -1.87 |
| Concerto | Decoder | 0.1670 | 0.1792 | +1.22 |
| Concerto | Fine-tuning | 0.2367 | 0.2384 | +0.17 |
| MSC | Decoder | 0.6431 | 0.6536 | +1.05 |
| VPSD | Decoder | 0.6714 | 0.6690 | -0.24 |
| RRCL | Decoder | 0.6506 | 0.6476 | -0.30 |
| CVRL | Decoder | 0.6688 | 0.6673 | -0.15 |

Scratch reproducibility is within 0.03 pp for all three protocols.  The
differences above therefore do not come from a changed seed, dataset mapping,
or evaluator.

## What calibration did fix

Point-M2AE decoder probe recovers every previously collapsed class.  Its DALES
mIoU rises from 0.1092 to 0.4531; vegetation IoU rises from 0.0006 to 0.8116,
cars from 0 to 0.3954, power lines from 0.1133 to 0.5790, and buildings from
0.2289 to 0.5206.

D-DITR decoder probe behaves similarly: mIoU rises from 0.1436 to 0.3849,
vegetation IoU from 0.0018 to 0.8547, and all formerly zero-IoU classes become
non-zero.  This is direct evidence that the pretrained encoder BN state was a
major cause of the first decoder collapse.

Concerto improves only slightly in aggregate.  It recovers non-zero output for
all classes, but loses 18.8 pp vegetation IoU while gaining several rare
classes.  Its problem is therefore not reducible to the encoder running
statistics.

## Why the remaining decoder gap persists

The decoder probe freezes the pretrained encoder in eval mode and trains a new
decoder.  A diagnostic replay on the same first 40 DALES validation batches
kept the encoder fixed and changed only the trained decoder BN layers from
running-stat inference to current-batch statistics:

| Method | Normal eval mIoU | Decoder batch-stat mIoU | Change (pp) |
|---|---:|---:|---:|
| Scratch | 0.5325 | 0.4841 | -4.84 |
| Point-M2AE | 0.4242 | 0.5093 | **+8.51** |
| D-DITR | 0.3608 | 0.5022 | **+14.14** |
| Concerto | 0.1925 | 0.4452 | **+25.27** |
| MSC | 0.6513 | 0.6258 | -2.56 |
| CVRL | 0.6626 | 0.6428 | -1.98 |

Thus the remaining short-run anomaly is mainly in the newly trained decoder's
running statistics, not in label mapping or checkpoint loading.  Batch-stat
inference is only a diagnostic: it is unsuitable as the final evaluator
because predictions would depend on fragment grouping.  The correct remedy is
post-training precise-BN recomputation from the clean downstream training set,
followed by ordinary eval-mode inference.

Even under batch-stat inference, the three affected methods only approach the
Scratch decoder after one epoch.  This remaining difference is representation
and optimization related: their frozen multiscale features need more decoder
updates than Scratch random features under this particular short schedule.

## Why linear probes slightly worsen

Linear probing uses the complete pretrained decoder representation in eval
mode.  Its original BN statistics and learned decoder weights are co-adapted.
Replacing those statistics with clean-input moments changes that fixed
representation without retraining it.  The 1--4 pp decreases show that large
running values alone do not prove corruption; some are legitimate properties
of the learned feature scale.

Consequently, Swiss BN calibration should not automatically replace the
original checkpoint for the strict linear-probe headline result.  It is best
treated as a decoder-interface/normalization ablation unless the checkpoint
export protocol defines clean-input precise BN in advance for every method.

## Why fine-tuning is almost unchanged

During full fine-tuning, BatchNorm uses current mini-batch statistics, so its
training forward pass is nearly independent of the initial running buffers.
This is confirmed by almost identical original and calibrated training losses
and mIoUs.  Replaying 20 validation batches with all BN layers using current
batch statistics gives:

| Method | Normal eval mIoU | All-BN batch-stat mIoU | Change (pp) |
|---|---:|---:|---:|
| Scratch | 0.5359 | 0.5162 | -1.97 |
| Point-M2AE | 0.3237 | 0.3485 | +2.48 |
| D-DITR | 0.1739 | 0.2988 | **+12.49** |
| Concerto | 0.2703 | 0.3324 | **+6.21** |
| MSC | 0.6177 | 0.5961 | -2.16 |
| CVRL | 0.6516 | 0.6329 | -1.87 |

BN mismatch remains part of D-DITR/Concerto FT, but correcting it cannot close
the full gap.  The stronger cause is the present FT initialization and learning
rate policy: the pretext decoder is retained, while every backbone parameter,
including that decoder, receives only `1e-4`; Scratch trains its whole backbone
at `1e-3`.  A pretext-specific decoder can therefore be harmful and too slow
to adapt.

For encoder-centric SSL methods, the cleaner FT protocol is to load the
pretrained stem/encoder only, initialize a fresh task decoder, train the
encoder at a low learning rate, and train decoder/head at the base learning
rate.  This also makes the distinction from decoder probe precise: the decoder
is fresh in both, but the encoder is frozen only in decoder probe.

## Interpreting the one-epoch result

One epoch is a stability and convergence-speed test, not a final performance
comparison.  Historical 10-epoch DALES-100 runs (different tile size and thus
not directly comparable numerically) show decoder probes continuing to improve:

- Point-M2AE: 0.5215 at epoch 1 to 0.6470 at epoch 10;
- D-DITR: 0.5084 to 0.6467;
- Concerto: 0.4802 to 0.6092.

The decoder methods therefore should not be rejected from the one-epoch smoke
ranking.  In contrast, the historical D-DITR and Concerto FT runs stayed low
after 10 epochs, so their FT initialization/optimization problem is substantive.

## Recommended next experiment

1. Keep original checkpoints as the strict linear-probe inputs.
2. For decoder probe, compare the calibrated encoder with and without a
   post-training downstream precise-BN pass; always use normal eval mode for
   reported metrics.
3. For FT, test an encoder-only initialization with a fresh decoder and three
   learning-rate groups: encoder `1e-4`, decoder `1e-3`--`2e-3`, head `1e-3`--`2e-3`.
4. Run this focused ablation first on Point-M2AE, D-DITR, and Concerto, with
   Scratch, MSC, and CVRL as controls.  Do not rerun the complete 81-task matrix
   until this protocol is fixed.
5. Use the full 10-epoch schedule for method ranking; retain one-epoch results
   only as optimization diagnostics.

## Targeted-transfer validation (2026-09-27)

> **Critical schedule caveat.** The trainer overwrites
> `scheduler.total_steps` with
> `len(train_loader) * epoch / gradient_accumulation_steps`.  Consequently,
> setting `epoch=1` compresses the entire cosine schedule into the smoke epoch
> and reaches zero learning rate at its end.  It is not equivalent to epoch 1
> of the formal ten-epoch schedule.  All values in this section are therefore
> interface/stability diagnostics only and must not be used as the paper's
> method ranking.

The proposed policies were subsequently tested with one DALES epoch and one
dataset loop.  Linear probing uses the original checkpoint; decoder probing
uses downstream precise BN for the new decoder; full fine-tuning loads only
the embedding/encoder, assigns `1e-4` to them and `2e-3` to the new
decoder/head, and applies precise BN before validation.

| Method | Probe | Previous mIoU | Targeted mIoU | Change (pp) |
|---|---|---:|---:|---:|
| Point-M2AE | Linear | 0.2943 | 0.2943 | +0.00 |
| Point-M2AE | Decoder | 0.4531 | 0.5229 | **+6.98** |
| Point-M2AE | Fine-tuning | 0.3235 | 0.5310 | **+20.75** |
| D-DITR | Linear | 0.2624 | 0.2624 | +0.00 |
| D-DITR | Decoder | 0.3849 | 0.5249 | **+14.00** |
| D-DITR | Fine-tuning | 0.1791 | 0.3861 | **+20.70** |
| Concerto | Linear | 0.2884 | 0.2884 | +0.00 |
| Concerto | Decoder | 0.1792 | 0.4656 | **+28.64** |
| Concerto | Fine-tuning | 0.2384 | 0.3389 | **+10.05** |

All nine jobs completed without runtime or numerical failures.  Every precise
BN pass used the same 64 batches and 26,635,113 clean DALES voxel points.
Decoder scope updated eight BN modules; full scope updated thirteen.

### Causal interpretation

Decoder training losses before and after downstream precise BN are essentially
identical: Point-M2AE 1.0563/1.0555, D-DITR 1.0623/1.0624, and Concerto
1.1484/1.1478.  The large validation gains therefore cannot be attributed to
different optimization.  They are caused by replacing inaccurate running
statistics in the newly trained decoder before eval-mode inference.  The
method is especially decisive for Concerto (+28.64 pp).

Encoder-only FT changes the optimization itself.  Relative to loading the
pretext decoder, training loss falls from 1.8269 to 1.0165 for Point-M2AE,
1.5499 to 0.9706 for D-DITR, and 1.6588 to 1.0626 for Concerto.  Thus the old
pretext decoder was not a neutral initialization: combined with a uniformly
small backbone learning rate, it obstructed adaptation.

The exact reproduction of all three original linear-probe mIoUs confirms both
determinism and the decision not to rewrite BN statistics for a strictly
frozen, co-adapted representation.

### Remaining limitation

The fixes remove the pathological collapse, but the one-epoch experiment is
not a valid final ranking.  The one-epoch Scratch references are 0.5389 for
decoder probe and 0.5481 for FT.  Point-M2AE nearly reaches them
(0.5229/0.5310), as does D-DITR decoder probing (0.5249).  Concerto decoder is
lower at 0.4656.  D-DITR and Concerto FT also remain below their own
frozen-decoder results (0.3861 versus 0.5249 and 0.3389 versus 0.4656).
This establishes a transfer-schedule problem, not evidence that the SSL
methods intrinsically underperform random initialization.

The class results support this interpretation.  D-DITR FT retains vegetation
IoU 0.8247 but reduces building IoU from the frozen-decoder value 0.8301 to
0.3388.  Concerto FT similarly reduces power-line IoU from 0.4681 to 0.0740
and building IoU from 0.7675 to 0.5396.  Under this one-epoch cosine schedule,
immediate encoder updates therefore damage useful pretrained structure
unevenly, especially for structural and rare classes.  This is a working
diagnosis until gradual unfreezing and full convergence are tested.

Archived ten-epoch DALES-100 runs further rule out a general representation
failure: Point-M2AE and D-DITR decoder probes reached test mIoU 0.6441 and
0.6419, both above Scratch decoder at 0.6233; Concerto reached 0.5938.  Their
linear probes (0.3768, 0.3462, and 0.3601) were also far above Scratch linear
at 0.1628.  Those runs used a different tile protocol and cannot be headline
numbers, but they demonstrate that one-epoch ordering is not the converged
ordering.  Only the old FT path remained abnormally low, consistent with the
invalid pretext-decoder initialization diagnosed above.

### Decision and next minimal ablation

- Adopt original checkpoints for strict linear probes.
- Adopt downstream precise decoder BN for decoder probes.
- Do not use any of these one-epoch values as paper comparisons.
- Point-M2AE encoder-only FT is sufficiently stable to enter a convergence
  run, but should use the same staged protocol as the other pretrained models.
- Do not yet freeze the current D-DITR/Concerto FT protocol as final.

For D-DITR and Concerto, the next focused experiment should compare only:

1. encoder-only FT with precise BN restricted to the decoder;
2. a total-budget-preserving one-epoch frozen-encoder decoder warm-up,
   followed by encoder unfreezing at `1e-4` (or `1e-5` if forgetting persists).

This separates encoder BN rewriting from early gradient-induced forgetting.
For the paper, the chosen FT recipe must be applied to all applicable
pretrained baselines rather than selected post hoc per method.  Scratch remains
the end-to-end random-initialization control.  MSC, Sonata, VPSD, RRCL, and
CVRL should be rerun under the same final transfer definition if that
definition changes which parts of the backbone are initialized or frozen.
