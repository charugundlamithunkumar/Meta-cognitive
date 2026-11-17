# Full GPU 1 calibration pilot

**Completed: 8 calibration trajectories × 7 checkpoints × 4 arms = 224 conditions and 2,240 real task answers.**

Run identifier: `recovery_gpu1_pilot_20261117`. The user authorized completion of this pilot; no time-based early stopping or main inference occurred.

## Execution and validation

Granite 4.2 8B was loaded from existing local files on physical GPU 1. The calibration-only profile used bf16, greedy decoding, 38% allocation, eager mode, one active sequence, 512-token chunked prefill and a 6,144-token context limit. All supplied histories fit the audited 5,261-token conservative bound. No histories were truncated.

All 224 checkpoint payloads, 2,464 rows, 32 trajectory summaries and four immutable arm archives validated. All post-cue forecasts parsed. Original generation source files are retained under the run’s `source_snapshot/` and checked against the recorded hashes. Our engine exited after the pilot and released GPU memory; existing GPU jobs were left alone.

## Exploratory calibration observations

| Arm | Correct / generated totals | Accuracy | Mean post-cue forecast |
|---|---:|---:|---:|
| MR_A | 85/560 | 15.18% | 30.36% |
| MR_B | 21/560 | 3.75% | 28.57% |
| MR_C | 85/560 | 15.18% | 19.64% |
| MR_D | 23/560 | 4.11% | 23.21% |

Clean (A) and repaired (C) mean accuracy coincide in this pilot, while C improves over the unrepaired B reference. However, C’s overall forecast exceeds its realized accuracy: the proposed overall underconfidence condition is not established. Late forecasts are mostly zero even where some answers are correct. These are calibration observations on eight sequences, not an independent replication or a confirmatory result.

## Paired diagnostic estimates

Percentage points; 10,000 whole-trajectory bootstrap draws, four-test Holm family. The main eligibility gate withholds a research verdict because these are calibration ids.

| Contrast | Estimate | 95% CI | Holm p | Diagnostic criterion |
|---|---:|---|---:|---|
| HG1 | 11.43 | [5.89, 16.61] | 0.0004 | passes |
| HG2 | 0.00 | [-2.32, 2.32] | 0.0004 | passes |
| HG3 | 10.71 | [3.57, 19.64] | 0.8017 | does not pass |
| HG4 | -4.46 | [-13.57, 4.29] | 0.9993 | does not pass |

HG1 = R_C−R_B; HG2 = R_C−R_A (equivalence within ±5); HG3 = F_A−F_C (>10); HG4 = R_C−F_C (>10). HG2’s 90% equivalence interval and all raw tests are in `analysis.json`. The combined proposed dissociation does not pass the diagnostic criteria. Do not change margins or select late checkpoints to manufacture a positive primary outcome.

## Precision review and its limits

Initial simulations shifted bounded arm scores; many high-variance draws were out of range, so some scenarios had no valid repetitions. Those files are retained as development diagnostics and should not be used for power claims. The corrected method shifts joint trajectory contrasts and shares the exact HG test helper. All registered test outputs on the observed pilot were verified identical after this refactor. This changes precision simulation, not the empirical HG rule.

Current contrast-space simulations use 500 repetitions, 1,000 bootstrap draws each, candidate N=32, and calibration residuals at their observed scale and doubled scale. All scenarios retain all repetitions. This is a local contrast sensitivity analysis, not a generative model of bounded scores, a test of future data, or proof of adequate power; eight-trajectory variance and plug-in effects are uncertain. The pilot-estimate plug-in scenario has zero combined support in these simulations because the full hypothesized pattern is absent in the pilot. Stronger counterfactual scenarios test a different assumption, not the observed effect.

[Contrast precision](../recovery_gpu1_pilot_precision_contrasts_20261117.json) · [Doubled-variance sensitivity](../recovery_gpu1_pilot_precision_contrasts_conservative_20261117.json)

## Next decision

Review these outcomes before freezing HG and scheduling main inference. In particular, the repaired arm is not globally underconfident in the pilot, and the forecast-history gap is imprecise around the proposed threshold. Keep all preregistered outcomes and report null/mixed findings. M/HF still needs real calibration and can directly address the instructor’s length-versus-error question. Main GPU allocation/profile, registration commits and independent confirmation remain unresolved; this pilot does not authorize silently bypassing those gates.

[Full report and figures](report.md) · [Machine-readable analysis](analysis.json) · [Raw manifest](../../runs/recovery_gpu1_pilot_20261117/manifest.json) · [Runtime log](../../runs/recovery_gpu1_pilot_20261117.log)
