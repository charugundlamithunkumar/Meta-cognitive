# Phase 1 pilot summary (calibration data only)

**Model:** IBM Granite 4.2 8B (bf16, vLLM).
**Data:** Long-Horizon-Execution calibration samples 0–19, plus freshly generated sequences.
**Date:** 2026-11-13.

> These runs set the design (K, τ, prompts, audit settings) and catch problems. They are **not** tests of the hypotheses. The hypotheses are tested only in the main runs on test samples 20–51, after pre-registration.

## Design settings fixed by the pilot

| Setting | Value | Evidence |
|---|---|---|
| Reply format | Integer forced by regex (`-?[0-9]{1,6}`); logprobs raw | Unconstrained: 58% non-answers (Granite drifts into prose). Constrained: 0 non-answers |
| K (keys per turn) | 1 | ≥ 15 numeric slips in every position bin |
| τ (logit gate) | −0.138 (min answer-token logprob) | Youden J = 0.668 on 20 × 300 turns |
| Gate re-compute | Prompt v2 + Granite low-effort thinking (budget 1,024) | Correct 98.8% (v1 non-thinking 23.7%; v2 non-thinking 5.0%), about 20 tokens |
| Audit (A3/A4) | Full thinking, budget 8,192 (proposed) | See the audit table below |

## Baseline A0: 20 trajectories × 300 turns

| Position bin (turns) | 0–49 | 50–99 | 100–199 | 200–299 |
|---|---|---|---|---|
| New numeric-slip rate | 0.24 | 0.39 | 0.43 | 0.50 |
| Logit AUROC (slip vs. locally correct) | 0.918 | 0.896 | 0.899 | 0.893 |

- Accuracy vs. truth: 0.014. First error at turn 1.5 (median).
- Self-report AUROCs are at chance: P(Yes) 0.48; verbal confidence 0.55 (most common answer: 0).
- HA upper bound: 82% of inherited errors trace to a seed the logit gate would catch.

## Logit gate A1b vs. baseline: same 20 trajectories, turns 0–99 (paired)

| | A0 | A1b | Difference [95% CI] |
|---|---|---|---|
| Accuracy vs. truth | 0.042 | 0.164 | +0.123 [+0.051, +0.204] |
| Turns to first error | 3.8 | 16.5 | +12.7 [+5.6, +20.6] |
| New-slip rate | 0.310 | 0.060 | −0.250 [−0.282, −0.217] |
| Error chains (seed slips) per trajectory | 31.0 | 6.0 | n/a |
| Inherited-error rate | 0.649 | 0.775 | +0.127 [+0.048, +0.195] |

- The gate fired on 30% of turns and caught 79% of slips. Re-computes were correct 98.8% of the time.
- The first A1b attempt, with the v1 non-thinking re-compute, was worse than the baseline: re-computes were correct only 30% of the time and often echoed the starting total.

## Audits (A3, fresh context): 4 trajectories × 40 turns, 32 audits

| | Full thinking (8,192) | Low-effort (4,096) |
|---|---|---|
| Unfinished | 9.4% | 0% |
| TPR / FPR / J | 1.00 / 0.00 / 1.00 | 1.00 / 0.09 / 0.91 |
| Corrections equal to the truth | 20/20 | 17/22 |
| Accuracy vs. truth (A0 on the same turns: 0.231) | 0.456 | 0.456 |
| Median thinking tokens; wall time | 2,029; 1,397 s | 74; 33 s |

## Checks

- **Contamination:** 20 fresh vs. 20 published sequences, 100 turns each. Slip rate 0.292 vs. 0.310 (difference −0.017 [−0.072, +0.037]); logit AUROC 0.900 vs. 0.910. No evidence of memorization. In turns 0–19 the published samples look slightly better (+0.048 [−0.023, +0.122]); this is listed as a limitation.
- **Inline format** (4 × 30): 120/120 well-formed after the prose-format fix. Inline confidence = 100 on every reply (AUROC 0.50). Logit AUROC 0.96 (14 slips).
