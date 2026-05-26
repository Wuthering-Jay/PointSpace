# Efficiency Benchmark for Point Cloud Semantic Segmentation Models

## 1. Experimental Setting

We evaluate the computational efficiency of representative point cloud semantic segmentation networks under a unified synthetic input protocol. The comparison focuses on model size, training latency, inference latency, and peak GPU memory consumption. Accuracy is not evaluated in this benchmark because all models are measured on synthetic point clouds rather than a held-out semantic segmentation dataset.

| Item | Setting |
|---|---|
| Hardware | `NVIDIA GeForce RTX 4080` |
| Host | `Jay` |
| Platform | `Windows-10-10.0.26200-SP0` |
| Software | Python `3.10.20 | packaged by Anaconda, Inc. | (main, Mar 11 2026, 17:42:35) [MSC v.1942 64 bit (AMD64)]`, PyTorch `2.8.0+cu128`, CUDA `12.8` |
| Input size | `16384` points per scene |
| Input channels | `5` point features plus XYZ coordinates where required |
| Number of classes | `8` |
| Benchmark profile | `balanced` |
| Precision | AMP `float16` |
| Timing protocol | `3` warmup iterations and `10` measured iterations |
| Run time | `2026-05-23 15:22:33 +0800` |

## 2. Evaluation Metrics

The benchmark reports four efficiency-oriented metrics. `Params` is the number of trainable and non-trainable model parameters. `Train latency` measures forward propagation, loss computation, and backward propagation; optimizer creation and optimizer step are excluded to isolate network computation. `Infer latency` measures forward propagation under `torch.no_grad()`. GPU memory is reported as the mean and standard deviation of per-iteration peak allocated memory, followed by the maximum observed per-iteration peak. The memory ratio is computed as train memory mean divided by inference memory mean.

All timings are measured with CUDA events after warmup. All models are executed with FP16 automatic mixed precision. Point Transformer V3 and LitePT are explicitly configured with flash attention enabled. The `balanced` profile widens/deepens undersized baselines to reduce capacity mismatch while preserving the project-native defaults for PTV3, DeepPLANet, and LitePT.

## 3. Main Efficiency Results

| Model | Status | Params (M) | Train Latency (ms) | Infer Latency (ms) | Train Mem Mean +/- Std / Max (MiB) | Infer Mem Mean +/- Std / Max (MiB) | Mem Ratio | Implementation | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| RandLA-Net | ok | 6.060 | 16.96 +/- 0.89 | 5.26 +/- 0.40 | 1253.70 +/- 0.00 / 1253.70 | 131.62 +/- 0.00 / 131.62 | 9.52x | Approximate RandLA-Net-style PyTorch baseline in tests/ | Balanced approximation: widened/deepened to reduce capacity gap with project-native models. |
| SPVCNN | ok | 4.458 | 13.10 +/- 0.53 | 4.23 +/- 0.26 | 1054.47 +/- 0.00 / 1054.47 | 141.04 +/- 0.00 / 141.04 | 7.48x | Approximate SPVCNN fallback in tests/ | Balanced fallback used because project SPVCNN could not be built: AssertionError: SPVCNN: Please follow `README.md` to install torchsparse.` |
| PointNeXt | ok | 7.893 | 20.53 +/- 1.08 | 6.98 +/- 0.79 | 1598.11 +/- 0.00 / 1598.11 | 143.01 +/- 0.00 / 143.01 | 11.17x | Approximate PointNeXt-style PyTorch baseline in tests/ | Balanced approximation: residual inverted-MLP style network with higher capacity. |
| Point Transformer V2 | ok | 9.960 | 657.83 +/- 6.78 | 242.23 +/- 3.15 | 11003.86 +/- 0.00 / 11003.86 | 1915.36 +/- 0.00 / 1915.36 | 5.75x | PointSpace config `configs/dales/semseg-pt-v2m4-0-base.py` (balanced profile) | Balanced profile increases PTV2 channels/depth from the small DALES default. |
| Point Transformer V3 | ok | 46.163 | 130.20 +/- 7.18 | 70.47 +/- 2.92 | 3164.03 +/- 0.00 / 3164.03 | 404.91 +/- 0.00 / 404.91 | 7.81x | PointSpace config `configs/dales/semseg-pt-v3m1-0-base.py` (balanced profile) |  |
| DeepPLANet | ok | 11.186 | 277.16 +/- 5.44 | 137.02 +/- 2.08 | 5012.19 +/- 0.00 / 5012.19 | 883.77 +/- 0.00 / 883.77 | 5.67x | PointSpace config `configs/dales/semseg-deeplanet-v2-0.py` (balanced profile) |  |
| LitePT | ok | 12.709 | 78.77 +/- 6.84 | 43.56 +/- 5.64 | 1954.18 +/- 0.00 / 1954.18 | 319.55 +/- 0.00 / 319.55 | 6.12x | PointSpace config `configs/dales/semseg-litept-v1m1-0-base.py` (balanced profile) |  |

## 4. Efficiency Analysis

- Fastest training latency: `SPVCNN` at `13.10` ms.
- Fastest inference latency: `SPVCNN` at `4.23` ms.
- Lowest mean training memory: `SPVCNN` at `1054.47` MiB.
- Parameter count should not be interpreted as a direct proxy for inference memory or latency. Runtime memory is dominated by activations, neighborhood buffers, attention workspaces, and the number of points remaining at each hierarchy stage.
- PTV3 has many more parameters than LitePT, but a large fraction of those weights operate after downsampling. This can keep activation memory and latency closer than the parameter ratio alone would suggest.
- LitePT and PTV3 have similar channel/depth schedules, but they do not allocate operators in the same way. LitePT disables decoder blocks and uses convolution-only early encoder stages; PTV3 applies attention through both encoder and decoder.
- Memory is averaged over per-iteration peaks instead of using only one global peak, reducing sensitivity to occasional allocator or kernel-workspace outliers while still preserving the maximum observed value in the table.
- Training memory excludes optimizer state allocation. This makes train/infer memory ratios reflect forward/backward activations and gradients, not AdamW moment buffers.
- Ratios above 2-3x are plausible in this protocol because inference runs under `no_grad` and does not retain intermediate activations, while training must keep backward tensors for every block. A 2-3x rule of thumb is more likely when comparing total job memory with larger inference batches, optimizer state included, or activation checkpointing enabled.
- The approximate RandLA-Net, PointNeXt, and SPVCNN fallback baselines should still be interpreted as hardware/runtime probes rather than strict architectural reproductions.
- For project-native models, LitePT shows the expected efficiency advantage over heavier attention-based or deep hierarchical baselines under the measured synthetic setting.

## 5. Architecture and Comparability

| Model | Capacity Setting | Main Operator Allocation | Comparability Note |
|---|---|---|---|
| RandLA-Net approx | balanced local surrogate | local residual point MLP | widened/deepened only in `balanced` |
| SPVCNN fallback | balanced local surrogate | local point-voxel pooling MLP | real project SPVCNN requires `torchsparse` |
| PointNeXt approx | balanced local surrogate | local residual inverted-MLP style network | widened/deepened only in `balanced` |
| Point Transformer V2 | channels (72,144,288,576), depths (3,3,3 enc; 2,2,2 dec) | local attention neighborhoods | `balanced` enlarges DALES small default |
| Point Transformer V3 | enc channels `(32,64,128,256,512)`, enc depths `(2,2,2,6,2)`, dec depths `(2,2,2,2)` | attention in encoder and decoder, flash enabled | many parameters are in coarse-resolution stages, so params do not scale linearly with activation memory |
| DeepPLANet | enc channels `(64,128,256,512)`, enc depths `(10,10,30,10)` | deep hierarchical local aggregation | much deeper than LitePT/PTV3 at local stages |
| LitePT | enc channels `(36,72,144,252,504)`, enc depths `(2,2,2,6,2)`, dec depths `(0,0,0,0)` | conv in first three encoder stages, attention only in last two encoder stages, flash enabled | channel/depth shape resembles PTV3, but active attention/decoder allocation is much lighter |

## 6. Implementation Details

- PTV2, PTV3, DeepPLANet, and LitePT are built from DALES default PointSpace configs.
- Under `balanced`, PTV2 is widened/deepened because the DALES default used here is much smaller than the later project-native networks.
- SPVCNN uses the project implementation when `torchsparse` is available; otherwise the script uses a local sparse point-voxel approximation and marks this in the note.
- RandLA-Net and PointNeXt are local approximations for comparable latency/memory probing only, not faithful accuracy reproductions.
- Each model receives the same synthetic single-scene point cloud with `coord`, `grid_coord`, `feat`, `offset`, and `segment` fields.
- The synthetic point count strongly affects attention, neighborhood search, and sparse voxel operators. For dataset-level comparison, rerun with a point count close to post-transform training samples.

## 7. Limitations

- This benchmark is an efficiency comparison only. It cannot support claims about semantic segmentation accuracy, mIoU, or convergence behavior.
- Approximate external baselines are not suitable for publication as faithful RandLA-Net, PointNeXt, or SPVCNN numbers unless replaced by validated implementations.
- A parameter-matched comparison would require manually designing separate width/depth variants for PTV3 and LitePT. The current benchmark compares practical/default-style configurations plus a balanced profile for obviously undersized baselines.
- The results are hardware-, CUDA-, and dependency-specific. Installing `torchsparse` may change the SPVCNN result from fallback to the project implementation.

## 8. External References for Approximate Baselines

- RandLA-Net PyTorch reference: https://github.com/aRI0U/RandLA-Net-pytorch
- PointNeXt / OpenPoints reference: https://github.com/guochengqian/PointNeXt
- TorchSparse / SPVCNN dependency reference: https://github.com/mit-han-lab/torchsparse
