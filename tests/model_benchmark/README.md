# Model Benchmark

This folder contains an isolated benchmark tool for comparing segmentation model
runtime on the current machine. It does not modify PointSpace training or model
code.

## Run

Use the `pointspace` conda environment:

```powershell
conda run -n pointspace python tests\model_benchmark\benchmark_models.py --num-points 32768 --warmup 5 --repeat 20
```

The default profile is `balanced`, which widens/deepens the undersized
RandLA-Net, SPVCNN fallback, PointNeXt, and PTV2 baselines to reduce capacity
mismatch in the efficiency comparison. To reproduce the original smaller
defaults, add:

```powershell
--profile default
```

For a faster smoke test:

```powershell
conda run -n pointspace python tests\model_benchmark\benchmark_models.py --num-points 2048 --warmup 1 --repeat 2
```

The script writes:

- `tests/model_benchmark/benchmark_results.json`
- `tests/model_benchmark/benchmark_report.md`

## Scope

- PTV2, PTV3, DeepPLANet, and LitePT are built from DALES default PointSpace
  configs.
- With `--profile balanced`, PTV2 is widened/deepened because its DALES default
  is much smaller than PTV3, DeepPLANet, and LitePT.
- PTV3 and LitePT force `enable_flash=True`.
- PTV3 and LitePT have similar channel/depth schedules, but they are not
  operator-matched: LitePT disables decoder blocks and uses convolution-only
  early encoder stages, while PTV3 applies attention in both encoder and
  decoder.
- RandLA-Net and PointNeXt are local approximations intended for latency and
  memory probing only.
- SPVCNN uses the project implementation when `torchsparse` is available;
  otherwise it falls back to a local sparse point-voxel approximation.
- GPU memory is reported as per-iteration peak memory statistics: mean,
  standard deviation, and maximum.
- Training memory measures forward + backward only. Optimizer state allocation
  and optimizer step are excluded so the train/infer ratio reflects network
  activation and gradient cost instead of AdamW moment buffers.

## References

- RandLA-Net PyTorch reference: https://github.com/aRI0U/RandLA-Net-pytorch
- PointNeXt / OpenPoints reference: https://github.com/guochengqian/PointNeXt
- TorchSparse / SPVCNN dependency reference: https://github.com/mit-han-lab/torchsparse
