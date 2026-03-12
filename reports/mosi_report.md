# MulT MOSI Reproduction Report

## Summary

- Paper: `MulT (ACL 2019)` on `CMU-MOSI`
- Final run: `python -u main.py --preset mosi_paper --no_prompt --batch_size 32 --grad_accum_steps 4 --name mosi_full_bs32_ga4 --log_interval 1`
- Runtime: `9298.60 s` (`2h 34m 59s`)
- Peak CUDA memory: `8956.82 MB`
- Speed diagnosis: the earlier slow run was caused by the monitoring wrapper piping every batch log through a parent Python process and updating JSON status files continuously. Reverting to direct `main.py` restored the steady-state speed to about `2.0 s/batch` in later epochs.

## Paper vs Reproduced

| Metric | Paper | Reproduced | Delta | Acceptance |
| --- | ---: | ---: | ---: | --- |
| Accuracy | 0.8300 | 0.7988 | -0.0312 | Fail (`>= 0.8150`) |
| F1 | 0.8280 | 0.7981 | -0.0299 | Fail (`>= 0.8130`) |
| Corr | 0.6980 | 0.6593 | -0.0387 | Pass (`>= 0.6500`) |
| MAE | 0.8710 | 0.9749 | +0.1039 | Fail (`<= 0.9500`) |

## Interpretation

- This run did not fully hit the target acceptance band.
- The reproduced result is directionally close but still behind the paper on `Accuracy`, `F1`, and `MAE`.
- The likely causes are the paper/code mismatch, the need to use AMP plus gradient accumulation on an 8 GB GPU, and possible preprocessing or implementation drift from the original training environment.

## Speed Comparison

| Scenario | Avg batch time |
| --- | ---: |
| High-overhead monitored runner | 3872.62 ms |
| Direct `main.py` training | 1727.32 ms |

The speed comparison plot and metric comparison plots were generated from these final numbers.

## Figures

- Metrics vs paper: `reports/figures/mosi_metrics_vs_paper.png`
- Training speed comparison: `reports/figures/training_speed_comparison.png`
