# Do confidence reports track failure and recovery during long execution?

Working manuscript, 18 November 2026. **Both main studies complete; internal manuscript draft.**
Author list, affiliations and submission venue remain to be supplied by the
research team. This is an internal draft, not a completed or submitted paper.

## Abstract

Confidence can become pessimistic during long execution even when that
pessimism does not identify individual errors. We study whether reports respond
to supplied error history, history truncation, and subsequent state repair in
Granite 4.2 8B on a running-sum task. In a registered seven-arm controlled-history
experiment using 32 trajectories and seven checkpoints, verbal confidence was
53.57 points lower with a complete clean history than with its last-five-turn
suffix (95% trajectory-bootstrap CI [51.71, 55.36]). The intermediate error-dose
contrast was −1.23 points (95% CI [−3.86, 1.40]); its 90% interval was inside the
registered ±10-point equivalence band. The registered classification was
length-driven, interpreted as a history-truncation effect because truncation
also changes available evidence. A separately registered four-arm experiment collected 8,960 additional answers
on the same 32 sequences. State repair improved accuracy by 15.36
points (95% CI [12.05, 18.93]) and was equivalent to the low clean
reference within ±5 points. However, repaired forecasts exceeded accuracy by
14.03 points on average. The registered recovery/underconfidence
conjunction was inconclusive; it did not establish the proposed dissociation.

## 1. Motivation and research questions

An incorrect running total can propagate through subsequent locally correct
updates. Global failure, a new arithmetic slip, and awareness of failure are
therefore distinct outcomes. Existing A0 observations motivate this study but
confound position, accumulated errors and realized performance. The instructor
proposed examining whether pessimism is self-monitoring or a consequence of
the growing conversation.

We ask two questions. First, do self-reports respond to injected wrong totals
at a fixed checkpoint, or primarily to long versus truncated histories? Second,
when the current state is repaired without removing the erroneous history, do
future task accuracy and forecasts recover together? These questions concern
behavioral reports and outcomes; they do not identify an internal introspective
mechanism.

## 2. Relation to prior work

The task and controlled-history manipulation build directly on
[Sinha et al.](https://arxiv.org/abs/2509.09677). Success forecasting during
multi-step execution has been studied by
[Barkan et al.](https://arxiv.org/abs/2512.24661). Confidence-guided correction,
oracle-state recovery and retained-context repair also have substantial
precedents. Our candidate contribution is the joint, controlled measurement of
future-accuracy forecasts and execution after a numerical state repair while
the displayed erroneous history remains present. The
[updated comparison](related-work.md) documents overlap and review limits;
priority is not established by this targeted search.

## 3. Shared task and implementation

We use the locally downloaded IBM Granite 4.2 8B revision
`a7042144125ce5ae64f89610e8289647fc000b9a` and pinned LHE dataset revision
`7e4901fbb42b88144a65782052d0c8d87a358edc`. Each sequence has one key per
turn and a 300-turn horizon. The dictionary and original task/probe prompts
are reused. Test ids are 20–51. These sequences were previously examined in
A0, so the interventions are prospective extensions, not independent
replications. Calibration ids 0–7 are excluded from main estimation.

At zero-based checkpoints 10, 25, 50, 100, 150, 200 and 250, a supplied prefix
ends after answer c. Each branch independently generates answers c+1 through
c+10. Greedy constrained integer generation uses bf16, eager execution, one
active sequence, 512-token chunked prefill and context limit 6,144. M used GPU 1;
MR is assigned GPU 0 under a documented pre-inference hardware amendment.
No histories are silently truncated beyond the specified short-arm treatment.
The audited maximum input bound is 5,430 tokens. Every checkpoint is saved
atomically; raw outputs, source snapshots and archive/config hashes support
resumption and provenance. Probe questions and answers never enter task
continuations.

HF and HG were finalized before main inference in local registration commit
`1d11bc5a2a665062b3aec023f2dc18ba0c0d02a4`. This is a local, versioned
registration record, not an external registry deposit. Calibration and
technical amendments are retained in the respective protocols.

## 4. M: history length and wrong-total dose

Long arms L0, L25, L50, L75 and L100 supply the full prefix with corresponding
percentages of wrong displayed totals. Wrong values use the existing +1
generator. A fixed shuffled ordering produces nested masks across doses;
counts are rounded to the nearest possible number of turns. S0 and S100
retain exactly the last five user/assistant pairs of their long counterparts.

Independent forks measure verbal confidence about the displayed answer,
forecast percentage of globally correct next-ten answers, and ordinal
awareness. Generated answers have additional verbal and P(Yes) probes and
answer-token features. Global correctness is agreement with the original
cumulative truth. Local-slip discrimination separates ok+inherited from fresh
numeric slips, excluding nonanswers; inherited wrong totals can be locally
correct.

Primary P is S0−L0 verbal confidence. D is the negative OLS confidence slope
over doses 0, .25, .50 and .75, expressed per full dose range. We average
checkpoints within trajectory and give trajectories equal weight. We use
10,000 whole-trajectory bootstrap resamples, seed 0, and the four-test HF Holm
family. The registered meaningful-effect margin is 10 confidence points;
equivalence uses a 90% interval within ±10 as well as the adjusted test.
Endpoint L100 is descriptive when accuracy is at or below 5%.

## 5. M results

All seven arms, 32 trajectories and seven checkpoints completed: 1,568
conditions and 15,680 task answers. No primary checkpoint report was invalid.

| Primary quantity | Estimate, points | 95% CI |
|---|---:|---|
| P: S0−L0 confidence | 53.57 | [51.71, 55.36] |
| D: intermediate error-dose effect | −1.23 | [−3.86, 1.40] |
| L0−L100, descriptive | −1.05 | [−2.89, 0.78] |

The dose 90% interval was [−3.43, 0.94]. Meaningful P and small-dose equivalence
passed the registered interval/Holm criteria, yielding the operational
classification **length-driven**. The profile was not monotonically decreasing
at six of seven checkpoints, qualifying any simple dose-response account.
Empirical bootstrap p-values stored as zero are below resampling resolution,
not mathematically zero. The dose-superiority test's two-sided tail does not
establish the required direction; its directional interval criterion fails.

| Arm | Verbal, 0–100 | Forecast, % | Next-ten accuracy, % |
|---|---:|---:|---:|
| L0 | 46.38 | 35.27 | 25.40 |
| L25 | 48.03 | 34.82 | 17.37 |
| L50 | 47.78 | 37.05 | 12.95 |
| L75 | 47.48 | 35.71 | 5.94 |
| L100 | 47.43 | 35.25 | 1.65 |
| S0 | 99.95 | 99.50 | 49.02 |
| S100 | 99.94 | 98.17 | 0.80 |

The arm table averages checkpoints and trajectories equally. L100 and S100
are at the overall accuracy floor. Intermediate accuracy decreases strongly
while their confidence means remain similar; the registered dose-equivalence
finding concerns the stated practical margin rather than an exact zero effect.

Forecast regression with checkpoint effects has slope 0.100 (95% CI
[0.057, 0.141]). This pooled cross-arm regression is descriptive, not a causal
mechanism test. Verbal confidence concerns an existing answer and is not itself
a probability of future correctness; awareness is ordinal. Only 31 of 490
strict checkpoint/offset cells support minimum-logprob AUROC under the
registered minimum of 15 observations per class. Missing cells prevent broad
claims of token-score superiority.

![Confidence and realized accuracy by checkpoint](../results/meta_HF_main_20261118/figures/confidence.png)

Full estimates, all arm/checkpoint curves and secondary outcomes are in the
[registered analysis](../results/meta_HF_main_20261118/meta_experiment.md) and
[machine-readable results](../results/meta_HF_main_20261118/meta_experiment.json).

## 6. MR: registered recovery design

MR crosses clean/error history with a once-supplied true/wrong current state.
A is clean/true, B error/wrong, C error/true and D clean/wrong. A/D share the
same displayed history; B/C share the same displayed history. Error histories
have rounded 50% wrong earlier totals and a forced wrong endpoint. The neutral
cue is “Continue using this current running total: {s}.” It occurs once before
continuation; the first local label uses the supplied state, while global
truth remains unchanged.

Independent forks collect pre-cue and primary post-cue forecasts. Each of ten
new answers receives verbal/P(Yes) probes; awareness follows the tenth answer.
The registered contrasts require: R_C−R_B>0; R_C−R_A equivalent within ±5
points; F_A−F_C>10; and R_C−F_C>10. All four must pass the HG interval and
separate Holm criteria using 10,000 paired trajectory resamples. If A or C
overall accuracy is at most 5%, the recovery verdict is withheld as
floor-limited. A lower forecast than an overconfident reference alone cannot
establish underconfidence.

## 6.1 MR results

All 896 conditions, 32 paired trajectories and four arms completed on GPU 0:
8,960 task answers and four validated archives. There were no missing or
out-of-range primary forecasts. Hardware and scheduling amendments were
recorded before main inference; the failed GPU 2 initialization generated no
task outcomes. The supervisor finished collection, validation and analysis at
13:15 IST on 18 November 2026.

| Arm | History / supplied state | Accuracy % | Forecast % |
|---|---|---:|---:|
| MR_A | clean / true | 17.54 | 41.52 |
| MR_B | error / wrong | 1.16 | 37.50 |
| MR_C | error / true | 16.52 | 30.54 |
| MR_D | clean / wrong | 2.37 | 26.85 |

| Test | Paired estimate, points | 95% CI | Holm p | Supported |
|---|---:|---|---:|---|
| HG1 | 15.36 | [12.05, 18.93] | 0.000400 | Yes |
| HG2 | -1.03 | [-4.20, 2.14] | 0.026397 | Yes |
| HG3 | 10.97 | [6.13, 15.88] | 0.690931 | No |
| HG4 | -14.03 | [-19.37, -8.88] | 1.000000 | No |

HG1 and HG2 pass. The HG2 90% interval [-3.71, 1.65] lies
inside the ±5-point equivalence band. Thus the repair improves execution and
reaches a clean baseline of only 17.54% accuracy, not near-perfect
performance. A and C exceed the 5% overall floor.

HG3 does not establish a forecast gap greater than 10 points after the
registered interval/Holm criteria, although its point estimate exceeds 10.
HG4 is opposite to predicted underconfidence: repaired forecasts average
30.54% against 16.52% accuracy. The primary verdict is
**inconclusive**, and the full recovery/underconfidence pattern is not
supported. This does not establish appropriate updating or an internal
mechanism. We retain the registered margins, all checkpoints and all outcomes.

Minimum-logprob AUROC is identifiable in 0/280 strict global-correctness cells
and 22/280 local-slip cells under the common-observation minimum-class rule.
Those sparse cells cannot support universal token-score superiority. Secondary
calibration, awareness and interactions do not rescue the primary decision.

![Registered HG effects](../results/recovery_HG_gpu0_main_20261118/hg_effects.png)

See the [MR summary](../results/recovery_HG_gpu0_main_20261118/main_summary.md),
[full report](../results/recovery_HG_gpu0_main_20261118/report.md) and
[all estimates/provenance](../results/recovery_HG_gpu0_main_20261118/analysis.json).

## 7. Interpretation and limitations

M supports a large long-versus-short history contrast in reported confidence
and a practically small intermediate injected-dose contrast under this
protocol. Truncation also removes reconstruction evidence, so it does not
isolate physical token count. High short-arm confidence is not evidence of
correctness. MR supports improved execution after state repair and equivalence to the low
clean baseline, but it does not support the proposed joint underconfidence
claim. Confidence-report changes and accurate monitoring remain distinct.

Limitations include one model and arithmetic task; artificial +1 totals;
supplied rather than naturally generated histories; possible numeric-width
differences; cue trust and contradiction; reused benchmark sequences; sparse
fixed-position AUROC support; and no mechanistic activation intervention.
M and MR used separate physical GPUs of the same Ada model; the failed
Blackwell initialization produced no MR outcome data. Hardware scheduling and
technical failures must remain disclosed. Independent confirmation is not yet
allocated or collected. Additional models,
mask seeds or matched-state length controls require separate registrations.

## Reproducibility and acknowledgments

See the [execution record](../MAIN_RUNS.md), [HF protocol](../prereg.md) and
[HG protocol](../prereg_recovery.md). Preserve null, reversed and floor-limited
outcomes when completing the paper. The instructor supplied the originating
metacognition question; Sinha et al. supplied the benchmark and injection
precedent. Author contributions, assistance disclosure, licenses and the final
release package must be reviewed before submission.
