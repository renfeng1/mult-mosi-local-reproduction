# MulT MOSI Reproduction

This workspace reproduces the `CMU-MOSI` core sentiment result from the ACL 2019 MulT paper using the official repository as a base.

## Environment

- Verified local runtime during implementation: `base` conda env, Python `3.12.7`, PyTorch `2.5.0`, CUDA `12.1`.
- An isolated `mult-repro` env was created, but the actual runs use `base` because it already has a working CUDA-enabled PyTorch stack.
- Required Python packages: `numpy`, `scipy`, `scikit-learn`, `pandas`.
- CUDA runs use automatic mixed precision to reduce VRAM pressure.

## Paper Preset

`--preset mosi_paper` locks the run to the paper-oriented MOSI non-aligned setup:

- dataset: `mosi`
- batch size: `128`
- grad accumulation: `1`
- learning rate: `1e-3`
- epochs: `100`
- layers: `4`
- heads: `10`
- projection dim: `40`
- kernels `(l/a/v)`: `1/3/3`
- `attn_dropout`, `attn_dropout_a`, `attn_dropout_v`, `relu_dropout`, `res_dropout`: `0.2`
- `embed_dropout`: `0.2`
- `out_dropout`: `0.1`
- `clip`: `0.8`
- `seed`: `1111`

## Data Setup

Run:

```powershell
python .\scripts\setup_mosi_data.py
```

The script downloads the official processed archive and extracts `mosi_data_noalign.pkl` into `data/`.

## Commands

Smoke test:

```powershell
.\run_mosi.ps1 -Smoke
```

Full run:

```powershell
.\run_mosi.ps1
```

The runner automatically retries with:

1. `batch_size=128, grad_accum_steps=1`
2. `batch_size=64, grad_accum_steps=2`
3. `batch_size=32, grad_accum_steps=4`
4. `batch_size=16, grad_accum_steps=8`
5. `batch_size=8, grad_accum_steps=16`

## Outputs

- Logs: `reports/logs/`
- Checkpoints: `pre_trained_models/`
- JSON run summaries: next to each log file

## Acceptance Target

- Paper target on MOSI: `Accuracy ≈ 0.83`, `F1 ≈ 0.828`, `MAE ≈ 0.871`, `Corr ≈ 0.698`
- Local acceptance band: `Accuracy >= 0.815`, `F1 >= 0.813`, `MAE <= 0.95`, `Corr >= 0.65`
