# MR recovery analysis

**withheld: mock or exploratory run**

Mock/exploratory estimates are software diagnostics, not model evidence.

| Arm | Checkpoint | n | Accuracy % | Forecast % | Missing | MAE |
|---|---:|---:|---:|---:|---:|---:|
| MR_A | 10 | 2 | 15.0 | 100.0 | 0 | 85.0 |
| MR_B | 10 | 2 | 0.0 | 100.0 | 0 | 100.0 |
| MR_C | 10 | 2 | 15.0 | 100.0 | 0 | 85.0 |
| MR_D | 10 | 2 | 0.0 | 100.0 | 0 | 100.0 |

HG tests, adjusted p-values, CIs, class counts, parse flags, paired interactions, calibration and input provenance are in `analysis.json`.

![accuracy](accuracy.png)
![forecasts](forecasts.png)
![hg_effects](hg_effects.png)
![secondary](secondary.png)
