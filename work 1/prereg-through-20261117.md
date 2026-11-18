# Pre-registration: Catching Silent Collapse (Granite 4.2 8B)

**Registered:** 2026-11-13, before any run on test samples 20–51.
**Repository:** `charugundlamithunkumar/Meta-cognitive`. This file's git commit is the registration timestamp.
**Design documents:** `implementation-plan.md` (§8 decision rules; §13 deviations up to this point); `results/pilot_summary.md` (calibration pilot that fixed the settings below).

Everything below is fixed. Any later change is logged in `implementation-plan.md` §13, marked **post-registration**, with its reason. Results that do not follow these rules are reported as **exploratory**.

---

## 1. Frozen materials

| Item | Value |
|---|---|
| Dataset | Long-Horizon-Execution, HF `arvindh75/Long-Horizon-Execution`, revision `7e4901fbb42b88144a65782052d0c8d87a358edc` (checksums in `data/lhe/SHA256SUMS`) |
| Test set | Samples **20–51** (32 trajectories), first **T = 300** turns, **K = 1** key per turn. Calibration 0–19 is never reused for tests; reserve 52–99 stays untouched |
| Model | `ibm-granite/granite-4.2-8b`, revision `a7042144125ce5ae64f89610e8289647fc000b9a`, bf16 |
| Software | vLLM 0.24.0, torch 2.11.0, transformers 5.12.1 (`requirements.txt`) |
| Hardware | GPU 0 (RTX 6000 Ada 48 GB), shared machine; `gpu_memory_utilization` 0.6 |
| Agent decoding | Non-thinking; greedy; reply forced by regex `-?[0-9]{1,6}` (inline: `-?[0-9]{1,6} \| [0-9]{1,3}`); top-20 **raw** logprobs |
| Prompts | `src/prompts.py`: agent prompt version 2; inline prose format v2; re-compute prompt v2 |
| Gate (A1a/A1b/A2) | Re-compute in a fresh context with Granite low-effort thinking, budget 1,024, temperature 1.0 / top-p 0.95, seeded per (trajectory, turn) |
| τ (A1b) | **−0.13805100321769714** on the minimum answer-token logprob (Youden J on calibration) |
| A2 rate | **Procedure:** after A1b finishes, the rate is A1b's realized share of turns with a fired gate on the test set. It is passed to A2 unchanged |
| Audits (A3/A4) | Every **m = 5** turns; full thinking, budget **8,192**, temperature 1.0 / top-p 0.95, seeded; a correction note on `VERDICT: WRONG, total = N`; an unfinished audit takes no action |
| HC checkpoints | Turns {10, 25, 50, 100, 150, 200, 250} in A0; window of 10 totals; placebo = count from the next trajectory in the group with a different count |
| Grouping | Trajectories run in groups of 8; rows flushed every turn; one line per finished trajectory; resume skips finished trajectories |

**SHA-256 of the frozen configs and source** (`scripts/main_runs.py` refuses to run if any differs):

```
3a0d6792d4642cdd3da1bd5cd67027d3c9d6554f02cac9628a82fa993b9056f0 configs/A0.json
b2d745d7f1b31e7faf7bd1cdcdf8ed372c5fc80b7fbc7f8779587bf56ff9a82c configs/A0_inline.json
9a7be9f362ab76c42ae21642a184d998eb95891105247b6d71c8afd1754c89b9 configs/A1a.json
ed92901af2418f18c8a1c8dd302c93b88f2186f890d8102f81050fd5b5ba3674 configs/A1b.json
59d854c53ca555ffb36d56821c101721537657997a8450d8ed3c0d457fb6451c configs/A2.json
0cc2ff2e146f5c59ae1c93deb969539360d4b642b962166181d9f3064805d08f configs/A3.json
a110b7b587ef15739c0f99c7e1aa68c2a9914c87365de9a261143c4acc911c02 configs/A4.json
8b4024da3ef351600dadd08f17b51f4586422761d7c941b0561d6a9859a06268 configs/D1.json
9438d0d058bf3c44e866af2be50d5edcec522256176ef912880e17b95b4db0b9 configs/D2.json
746992247a7118a110eafb57882269495f2ee8f6208fc76d95c9eb07c8a2d028 src/agent.py
41baddc9d9f3aaae51eff583ba6a3e07056bc3256758233a4e025d120c136baa src/labels.py
bc1f56f55e7d652afc7fb6b38f3258c6e7c077d2a5ae6eb8a7ad664072d407fb src/lhe_data.py
984961c31bd680f8b6e296816a3491c4524bd672cdcc68f5a6e67a71f8b11c4f src/metrics.py
e719e169e7bca45412223b317d5049825361c60c34d8f500d75d4d332bb117f5 src/prompts.py
80016318cffb95d8d5b88280e6014f60debdac48f828b5dc9c8cbfc017e3b2c4 src/stats.py
```

## 2. Arms and run order

Run order: A0 → A0_inline → A1a → A1b → A2 → D1 → D2 → A3 → A4. The cheap arms run first, so the most results exist if GPU time runs out. This order is a change from the plan's order, recorded here before running.

| Arm | Description |
|---|---|
| A0 | Baseline, with side-question probes (P(Yes), verbal confidence, HC forecasts) |
| A0_inline | Replies are `total \| confidence` (HE) |
| A1a | Format-only gate. **Expected to equal A0:** forced-integer replies are never malformed, so its gate cannot fire. It is kept as a reproducibility check |
| A1b | Format + logit gate (τ above) → re-compute |
| A2 | Format + random gate at A1b's realized rate → identical re-compute |
| A3 | Fresh-context audit every 5 turns |
| A4 | The same audit asked inside the full history |
| D1 | No history; state line = the agent's own last committed total |
| D2 | No history; state line = the true total (oracle control) |

## 3. Definitions

For each turn, `prev` = the total the agent should build on (its last committed total, or the latest correction). Labels (`src/labels.py`):
- **ok** = committed total equals the truth;
- **inherited** = equals `prev + v` but is wrong;
- **numeric_slip** = well-formed and ≠ `prev + v`;
- **nonanswer** = not well-formed.

| Metric | Definition (per trajectory, then averaged) |
|---|---|
| **Accuracy** (primary end-to-end metric) | Share of the 300 turns labelled ok |
| Error chains | Number of turns labelled numeric_slip or nonanswer (each starts a chain) |
| Turns to first error | Index of the first turn not labelled ok |
| Logit AUROC | Type-2 AUROC of `lp_min` (higher = more confident) for **locally correct** turns (ok + inherited) vs. numeric slips, pooled over trajectories |
| Auditor J | TPR − FPR over **finished** audits, where positive = the audited total is wrong vs. the truth and flagged = `VERDICT: WRONG` |

**Statistics:**
- Paired bootstrap over the 32 trajectories, 10,000 resamples, seed 0; 95% percentile CIs.
- Two-sided bootstrap p-values.
- **Holm correction across the five primary tests** (one per hypothesis, marked ★ below; each test's p is the largest p among its required comparisons).
- AUROC and J CIs come from bootstrapping trajectories (resampling whole trajectories).

## 4. Hypotheses and decision rules

Only finished trajectories (all 300 turns) enter the analysis. No trajectory is excluded for any other reason.

**HA: catch the seed**
- ★ Supported if accuracy(A1b) − accuracy(A2) > 0 **and** accuracy(A1b) − accuracy(A0) > 0, both with paired 95% CIs excluding 0 after Holm, **and** error chains per trajectory are lower in A1b than in A2 and A0 (paired CIs excluding 0).
- Falsified if either accuracy difference has a CI including 0, or if the live gate's logit AUROC on A1b's original replies has a 95% CI including 0.5.
- A1b − A1a is reported too; it is expected to equal A1b − A0.

**HB: audit outside the history**
- Δ = J(turns 200–299) − J(turns 0–49), with a trajectory-bootstrap CI.
- ★ Supported if A3's Δ CI includes 0 or lies above 0, **and** A4's Δ CI lies entirely below 0, **and** accuracy(A3) − accuracy(A0) > 0 (CI excludes 0).
- Falsified if A3's Δ CI lies entirely below 0, or accuracy(A3) − accuracy(A0) has a CI including 0.
- If A4's Δ CI includes 0, the result is "A3 works, but in-history decline not shown" (partial).
- The share of unfinished audits per arm and bin is reported. Unfinished audits are not counted in J.

**HC: awareness is input-starved** (A0 checkpoints where the true count ≠ the placebo count)
- MAE_c = mean |forecast/100 − realized next-10 accuracy| for condition c ∈ {none, true, placebo}, computed per trajectory.
- ★ Supported if MAE_true < MAE_none and MAE_true < MAE_placebo (paired CIs excluding 0), **and** the Spearman correlation between the true-evidence forecast and realized accuracy is > 0 (bootstrap CI excluding 0).
- Falsified if MAE_true ≥ MAE_none (forecasts don't use the evidence), or MAE_true ≈ MAE_placebo (it reacts to the mention of a checker, not to the count).

**HD: collapse is inheritance, not context length**
- ★ Supported if the 90% CI of accuracy(D1) − accuracy(A0) lies within **±0.05** (equivalence, two one-sided tests at α = 0.05), **and** D2's mean per-trajectory OLS slope of correctness on turn has a 95% CI within **±0.0005 per turn**.
- Falsified if accuracy(D1) − accuracy(A0) > 0.05 (CI excludes the margin), or D2's slope CI lies outside the band.
- Secondary: the slip rate before vs. after the first error in A0 and D1 (the length vs. self-conditioning decomposition).

**HE: logit format effect is partly a scoring artifact** (conditional)
- The condition holds if AUROC(A0_inline, tokens before `|`) is lower than AUROC(A0, bare), with the CI of the difference below 0.
- If it holds: ★ supported if AUROC(A0_inline, final-integer tokens) lies inside A0's 95% CI; falsified otherwise.
- If it does not hold: HE is reported as "**discrepancy not reproduced on this task**". It is then not counted among the Holm-corrected tests, and Holm is applied over the remaining four.

## 5. Secondary and exploratory analyses (not used for the verdicts)

- Turns to first error; run length of wrong totals.
- Gate recall, false-trigger rate and re-compute accuracy.
- Slip rate by position bin.
- AUROC of P(Yes) and verbal confidence.
- Inline confidence distribution; awareness probe (P "lower").
- Extra tokens per turn per arm.
- Exploratory `retrieval_like` slips.
- A0 vs. A1a agreement (reproducibility).
- All four position bins (0–49 / 50–99 / 100–199 / 200–299) for every per-bin metric.

## 6. Known limitations, stated in advance

- One model (Granite 4.2 8B), one synthetic task. The other group members run Qwen3-8B and Llama with the same protocol.
- Forced-integer decoding changes the output channel (as in the instructor's E8).
- Accuracy against the truth starts near the floor in the baseline, because the first error comes very early.
- Contamination check: no fresh-vs-published difference, but turns 0–19 showed a non-significant edge for published samples.
- GPU 0 is shared, so wall-clock times are not comparable across arms.

## 7. HF: metacognitive pessimism under controlled histories (2026-11-16 addendum)

**Prospective for M only.** Written before the new analysis code and before any M
model run. A0 has already been inspected; its metacognition analysis is explicitly
exploratory. This addendum does not change HA–HE, their multiplicity family, or
their frozen materials. Commit this addendum and the M implementation before a
confirmatory GPU run. Method precedent: Sinha et al. (2026),
[§3.2 / Figure 5](https://arxiv.org/html/2509.09677#S3.SS2), controlled error rates
in supplied histories; our extension measures the existing metacognition probes.

**Hypothesis.** Confidence can decline because of context length, erroneous
history, or both. Error sensitivity alone is not sufficient evidence of accurate
self-monitoring; its relation to realized performance is tested separately.

**Design fixed before M results:**
- Same test ids 20–51, T=300, K=1, model/revision, prompt version 2, constrained
  greedy decoding, and zero-based checkpoints {10,25,50,100,150,200,250} as A0.
  Checkpoint c is after answer c: history contains c+1 turns, continuation is
  c+1 through c+10. Each checkpoint is an independent fork.
- New arms M_L0, M_L25, M_L50, M_L75, M_L100 show the entire prefix, with
  0/25/50/75/100% wrong reported totals. M_S0 and M_S100 show only the last five
  user/assistant pairs of their corresponding long history, plus the unchanged
  system prompt. No checker evidence or extra oracle state is supplied. Thus
  the short control also removes earlier evidence needed to reconstruct the
  absolute sum; interpret the length contrast as history truncation, not a
  token-count-only intervention. S100 is a descriptive interaction control.
- Use the existing +1 forced numeric-slip generator in
  `tests/test_arms_mock.py:FakeLLM.respond`, as confirmed by the user. Call it
  from the true previous state to obtain truth+1 at selected positions. This
  is a controlled, small arithmetic error, not an empirically sampled error
  distribution. Consecutive injected wrong totals can be locally inherited;
  the treatment rate is **wrong totals**, not the rate of fresh numeric slips.
- At each (sample id, checkpoint), shuffle the c+1 positions using Python
  Random(seed*1000003 + sample_id*10007 + checkpoint), seed=0. Select the first
  floor(rate*(c+1)+0.5) positions. Masks are nested across doses; log actual
  counts, mask, and history. Rounding error is at most 0.5/(c+1).
  Short histories are exact truncations of the corresponding long histories.
- Ask the unchanged verbal, fore_none, and aware probes in independent side
  copies of that history. Then independently generate ten answers, appending
  only those answers to the continuation. Save answer-token features and
  forked verbal and P(Yes) probes after each generated answer for secondary
  slip localization. No probe answer enters any task history.
- Only trajectories with all seven checkpoints and all ten continuation
  answers at every checkpoint enter confirmatory inference; report missing
  ids and parsing failures. Resume is at checkpoint granularity with atomic
  checkpoint files. Never treat mock outputs as experimental measurements.

**Primary estimands (percentage points of verbal confidence):** first average
over seven checkpoints within each trajectory, then equally over trajectories.
Length effect L = conf(L0) − conf(S0) (pessimism predicts L < 0).
Endpoint error effect E100 = conf(L0) − conf(L100) is reported with a CI.
The primary error contrast D is minus the OLS confidence slope over doses
{0,.25,.50,.75}, expressed per 0→100% error-rate change, computed within each
trajectory/checkpoint and then averaged. Report L25–L75 separately and their
observed ordering; no post-hoc selection of doses or checkpoints.

**★ Decision rule:** meaningful pessimism is **10 confidence points**. Use
paired bootstrap over whole trajectories (10,000 resamples, seed 0), retaining
all arms/checkpoints together; report 95% percentile CIs. Let P = −L.
Four tests form a new, separate Holm family at alpha=.05: P>10, D>10,
equivalence of P within (−10,10), and equivalence of D within (−10,10).
For superiority require the 95% CI lower bound >10 and Holm-adjusted two-sided
bootstrap p<.05 against 10. Equivalence requires the 90% CI inside (−10,10)
and Holm-adjusted TOST p<.05, where TOST p is the larger one-sided bootstrap
tail at the two margins. Report all four adjusted p-values.
- **Length-driven:** P is meaningfully positive and D is equivalent to small.
- **Monitoring-driven:** D is meaningfully positive and P is equivalent to small
  (operational evidence of error-sensitive reports, not proof of introspection).
- **Mixed:** P and D are both meaningfully positive.
- Otherwise **inconclusive**; failure to reject is not equivalence. Report
  reversed effects explicitly. A nonmonotone dose profile qualifies interpretation.

**Floor rule:** mean next-10 accuracy <=.05 constitutes a floor. Report it by
arm/checkpoint. L100 alone is uninformative about monitoring when at this floor;
the L0–L75 dose contrast is primary regardless of whether L100 is at floor.
If every intermediate arm is also at floor, classify confidence effects but
report performance-monitoring evidence as unidentifiable at the floor.

**Secondary (descriptive, no primary verdict changes):**
- Forecast calibration: realized next-10 accuracy regressed on fore_none/100,
  with checkpoint fixed effects and trajectory-bootstrap CIs; intercept and
  slope (ideal 0 and 1). Also report slopes for verbal/100 and awareness
  optimism = P(higher)+0.5*P(same). Awareness is ordinal, not a calibrated
  probability; its slope is association only. Constant predictors/outcomes
  yield an unidentifiable slope, not evidence of successful calibration.
- Per-checkpoint, within-arm generated-step AUROC: higher verbal/P(Yes)/logit
  confidence predicts ok+inherited versus numeric_slip, excluding nonanswers,
  exactly as §3. Use lp_min, lp_mean, negative ent_max, margin_min and final
  integer lp_min; require >=15 observations in each class. Compare verbal and
  logit on the same steps, with paired trajectory-bootstrap CIs. Injected
  answers have no model logits and never enter these AUROCs. Checkpoint and
  continuation offset are held fixed for strict position-matched summaries;
  report insufficient cells instead of pooling positions silently.

For the exploratory A0 matched-history analysis, history means turns strictly
before the current answer. Compare clean versus already-erroneous history at
each exact turn 0–60; show n, trajectory-bootstrap CIs, and flag n<30. Aggregate
only over positions with both groups, giving those positions equal weight;
also report an inclusive-current-answer sensitivity definition. Next-10 windows
must contain all ten future rows (the final ten A0 turns have no such outcome).
