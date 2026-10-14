# Controlled downstream transfer matrix

This directory contains 81 standalone configurations:

- datasets: `dales`, `vaihingen`, `opengf`;
- pretraining: `scratch`, `point-m2ae`, `msc`, `sonata`, `d-ditr`,
  `concerto`, `vpsd`, `rrcl`, `cvrl`;
- protocols: `lin` (strict linear probe), `dec` (frozen encoder and trainable
  native LitePT decoder), and `ft` (full fine-tuning).

Every file is generated without `_base_` inheritance and explicitly contains
the complete model, optimizer, hooks and dataset pipeline. Regenerate the
matrix with:

```powershell
& D:/app/anaconda3/envs/pointspace/python.exe utils/generate_downstream_ssl_configs.py
```

To regenerate only one dataset without touching experiments that are still
running for the others, append its name, for example:

```powershell
& D:/app/anaconda3/envs/pointspace/python.exe utils/generate_downstream_ssl_configs.py dales
```

All pretrained checkpoints referenced by this matrix use the
`*_bncal_swiss.pth` copies.  Their learnable weights are byte-identical to the
original checkpoints; only BatchNorm running buffers were recomputed from a
fixed random subset of 512 clean Swiss-100 tiles.  Scratch deliberately keeps
the original random checkpoint and receives no Swiss calibration data.  The
calibration is reproducible with:

```powershell
& D:/app/anaconda3/envs/pointspace/python.exe utils/calibrate_swiss_bn.py
```

The per-checkpoint audit is written to
`exp/bn_calibration_swiss100/calibration_report.json`.

## Shared protocol

- 10 epochs and `train.loop=5`;
- CrossEntropyLoss + LovaszLoss;
- complete LitePT decoder with `dec_depths=(1, 1, 1, 1)` and
  `dec_conv=(True, True, True, True)`;
- decoder probes keep the encoder strictly frozen and train a fresh native
  LitePT decoder with ordinary running-stat BatchNorm. Stateless BatchNorm is
  intentionally avoided because its test prediction depends on how fragments
  are grouped into batches;
- gradient accumulation 2 throughout. DALES uses micro-batch 12 (effective
  batch 24); a representative CVRL-FT batch reserved 8.35 GiB, while micro 16
  reserved 10.75 GiB but left less protection against dense-tile mixtures.
  Vaihingen and OpenGF retain micro-batch 6 (effective batch 12);
- BF16 AMP and MLP checkpointing. Dataset-specific post-voxel safety caps are
  200k for DALES, 90k for Vaihingen, and 240k for OpenGF. These values cover
  approximately the P99.5 population plus augmentation headroom instead of
  imposing the old shared 160k limit;
- no early stopping and no prediction LAS writer;
- the best validation checkpoint is evaluated after training and all metrics
  are written to the log.

DALES uses `E:\data\DALES\dales_las\tile` for both its train and held-out
test blocks. DALES and OpenGF class 0 points are remapped to `-1`; during training they are
physically removed before augmentation and voxel sampling. During validation
and testing they remain available for spatial reconstruction but are excluded
from every loss and metric. Vaihingen class 0 is the valid ISPRS Powerline
class and is therefore retained.

## Example

```powershell
& D:/app/anaconda3/envs/pointspace/python.exe tools/train.py --config-file configs/downstream/dales/semseg-cvrl-litept-v1m4-dales-ft.py
```

Run the complete 81-task matrix sequentially from the repository root (stop
immediately if any task fails):

```powershell
$py = "D:/app/anaconda3/envs/pointspace/python.exe"; Get-ChildItem "configs/downstream" -Recurse -Filter "*.py" | Sort-Object FullName | ForEach-Object { Write-Host "`n=== [$($_.FullName)] ===" -ForegroundColor Cyan; & $py "tools/train.py" --config-file $_.FullName; if ($LASTEXITCODE -ne 0) { throw "Downstream task failed (exit $LASTEXITCODE): $($_.FullName)" } }
```

The smoke-test implementation and latest JSON result are located at
`paper_cvrl/validate_downstream_ssl_matrix.py` and
`paper_cvrl/results/section_45/downstream_smoke_validation.json`.
The dense-sample FT memory benchmark and its raw measurements are located at
`paper_cvrl/benchmark_downstream_ft_memory.py` and
`paper_cvrl/results/section_45/downstream_ft_memory.json`.
