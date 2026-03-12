# AMP And Minimal Tuning Summary

## Step 1: AMP Ablation

Controlled setting:

- dataset: `CMU-MOSI`
- preset: `mosi_paper`
- physical batch size: `16`
- gradient accumulation: `8`
- effective batch size: `128`
- epochs: `6`

Results:

| Config | Best valid loss (6 ep) | Best test loss (6 ep) | MAE | Corr | F1 | Acc | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AMP on | 1.2826 | 1.2108 | 1.2200 | 0.4289 | 0.6595 | 0.6616 | 253.27 |
| AMP off | 1.2222 | 1.1453 | 1.1565 | 0.5084 | 0.7323 | 0.7317 | 333.61 |

Conclusion:

- In this controlled early-stage comparison, `AMP off` is clearly better than `AMP on`.
- `AMP` is therefore a plausible contributor to the final metric gap, not just a neutral memory optimization.
- The tradeoff is runtime: `AMP off` is about `31.7%` slower in this ablation.

## Step 2: Minimal Tuning Screening

Controlled setting:

- same as above
- AMP left on for tuning screening

Results:

| Config | Best valid loss (6 ep) | Best test loss (6 ep) | MAE | Corr | F1 | Acc | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline AMP on | 1.2826 | 1.2108 | 1.2200 | 0.4289 | 0.6595 | 0.6616 | 253.27 |
| `out_dropout=0.0` | 1.2539 | 1.2474 | 1.2474 | 0.4148 | 0.6447 | 0.6463 | 317.08 |
| `dropouts=0.1` and `out_dropout=0.0` | 1.2319 | 1.2015 | 1.2015 | 0.4124 | 0.6365 | 0.6372 | 337.71 |

Conclusion:

- These two minimal dropout tweaks did not beat the strongest 6-epoch baseline we tested.
- The best short-run performer in this round is still `AMP off` baseline.

## Practical Recommendation

Next full run to improve `MAE/Accuracy/F1` should prioritize:

1. `AMP off`
2. `batch_size=16`
3. `grad_accum_steps=8`
4. keep the paper preset otherwise unchanged

This is the lowest-risk next experiment because it changes only the numeric precision path while preserving the effective batch and paper hyperparameters.
