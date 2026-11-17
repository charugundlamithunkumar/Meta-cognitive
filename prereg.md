# Pre-registration: active M / HF study

Status: FROZEN

**Finalized:** 2026-11-18 before any M test inference. HF was drafted before its analysis code. The registration timestamp is the commit containing this frozen file; calibration used ids 0–7 only.

The completed HA–HE registration is archived in [previous work/prereg.md](previous%20work/prereg.md). The complete pre-reorganization document, including HF, is preserved in [the dated snapshot](previous%20work/prereg-through-20261117.md).

**Administrative amendment, 2026-11-17:** retired code/configs/tests moved into `previous work/`; the +1 generator is now `previous work/tests/test_arms_mock.py`. Paths changed, not the HF design. Local-slip labels referenced as §3 below mean locally correct = ok + inherited, negative = numeric_slip, nonanswers excluded, as in the archived registration.

**Calibration runtime amendment, 2026-11-18 (before M model inference):** use
calibration ids 0–7, all seven arms/checkpoints, on GPU 1. The separately logged
`gpu1-long` profile uses utilization .38, context 6,144, eager mode, one active
sequence and 512-token chunked prefill. A CPU context/resource check precedes
loading. No input truncation, prompt, mask, probe, decoding, outcome or HF
decision-rule changes. Overrides are rejected for test inference. Save all
392 conditions / 3,920 task answers, including failed/invalid probes. Main
runtime settings and registration still require a separate pre-test freeze.

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
  `previous work/tests/test_arms_mock.py:FakeLLM.respond`, as confirmed by the user. Call it
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


## Final pre-test runtime and sample decision — 2026-11-18

After the complete seven-arm calibration, retain the original **32 test ids
20–51**, seven checkpoints, seeds, probes, error masks, effect margins and
four-test Holm decision rule. Neither the uncertain error-dose pilot result
nor the MR pilot’s lack of overall underconfidence changes HF/HG thresholds.
These sequences are a prospective intervention extension, not independent
confirmation of the previously inspected A0 finding. No optional stopping.

Adopt the successfully calibrated runtime for main inference: **GPU 1, bf16,
utilization .38, maximum context 6,144, eager mode, one active sequence,
512-token chunked prefill**, greedy constrained answers and raw top-20 logits.
All eleven M/MR main-input bounds were audited on CPU; maximum 5,430 tokens.
No truncation, quantization, changed prompt, extra state, or altered task inputs
is introduced. Runtime settings are now in canonical configs; calibration-only
CLI overrides remain prohibited for test runs. This supersedes the initial
.6 / 16,384 runtime inherited from Study 1 and does not alter the estimands.

Calibration precision used joint trajectory-contrast resampling with baseline
and doubled residual scale, N=32. Pilot-variance uncertainty and the chance of
an inconclusive dose-equivalence decision remain explicit; these simulations
are not a guarantee of classification power. Retain N=32 rather than choosing
a sample size to force significance. Report all confidence/accuracy curves,
floors, invalid probes and uncertain effects. Empirical HF bootstrap p=0 is
reported as below the simulation resolution, not a mathematically zero probability.

Main compute estimate: about 56 minutes M plus about 33 minutes MR at pilot
throughput, with additional startup/CPU-analysis/shared-GPU variability. The
full long-run budget requires user approval before launch. Freeze/commit alone
does not authorize unbounded compute.
