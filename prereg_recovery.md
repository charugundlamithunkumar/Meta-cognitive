# HG: recognition of recovery after state repair

Status: FROZEN

**Finalized:** 2026-11-18 before any MR test inference. The draft was written
2026-11-17 before recovery analysis code. Calibration used ids 0–7 only.
The registration timestamp is the commit containing this file. HA–HF remain
separate registrations.

## Design and materials

Granite 4.2 8B, local `models/granite-4.2-8b`, revision
`a7042144125ce5ae64f89610e8289647fc000b9a`; pinned LHE revision
`7e4901fbb42b88144a65782052d0c8d87a358edc`. K=1, T=300, prompt v2,
non-thinking greedy constrained integer answers, bf16, raw top-20 logprobs,
context limit 6,144, utilization .38, eager mode, one active sequence,
512-token chunked prefill. No silent truncation.

Initial prospective intervention extension: ids 20–51, checkpoints
10,25,50,100,150,200,250 (zero-based), ten subsequent generated answers each.
These sequences were previously inspected in A0: this is not independent
replication. Calibration uses ids 0–19 only. Reserve ids 52–99 and fresh seeds
require a separate allocation audit/registration; they are excluded here.
Retain the original sample size of 32 after calibration precision review;
variance estimates remain uncertain. No significance-based stopping.

H_clean contains truth at positions 0…c. H_error contains
floor(.5*c+.5) wrong totals among 0…c−1, plus a forced wrong endpoint at c.
Shuffle the earlier indices with Python Random(seed*1000003+id*10007+c),
seed=0. Use the unchanged existing +1 generator via `numeric_slip`; record
masks/rates, token counts, number width/sign/magnitude and transitions.

| Arm | History | Supplied state |
|---|---|---|
| MR_A | clean | truth |
| MR_B | error | truth+1 from existing generator |
| MR_C | same history as B | truth |
| MR_D | same history as A | truth+1 from existing generator |

Cue: `Continue using this current running total: {s}.` Every arm uses identical
wording; no “verified” assertion. The cue appears once, in the user message
before the first continuation key. Later messages contain only keys. Label
the first new answer with prev=s; global ground truth never changes.
Use A,B,C,D order, serial checkpoints/trajectories, one task answer per call;
the two answer probes may be batched together. Record order and batch settings.

## Forks and preprocessing

Pre-cue forecast: unchanged P_FORECAST in an independent user turn after H.
Post-cue forecast: cue + newline + P_FORECAST in a single user message after H.
Task: independently cue + newline + keys at c+1, then generate through c+10.
No probe output/question enters the task history. After each generated answer
fork verbal and P(Yes); after the tenth answer fork awareness. Reuse existing
task/probe prompts, chat template, token features and labels.

Numeric probes: accept a complete unsigned integer string, retain raw output
and parsed integer, clip to [0,100] for analysis, flag out-of-range values.
Missing/unparseable primary forecasts withhold the full verdict; never impute
zero or discard an inconvenient trajectory. Report a raw-only sensitivity
summary excluding out-of-range forecasts. Malformed task answers count wrong
and are stored, not regenerated. For probability probes record retained target
token mass; no target mass means missing rather than zero confidence.

Persist each completed checkpoint atomically, including raw invalid probes.
Resume only identical source/config/model/data/registration fingerprints.
Partial crashes may rerun the uncommitted checkpoint, never a committed one.
Reject mismatching or corrupt outputs; keep mock/calibration/test distinct.

## Primary hypothesis and four tests

R=100×fraction of ten globally correct answers; F=post-cue forecast. First
average the seven checkpoint contrasts within trajectory, then trajectories
equally. All 32 complete paired trajectories and all four arms are required
for the full main verdict. Partial results are exploratory; mock outputs never
yield an empirical verdict.

- HG1: R_C−R_B > 0 points.
- HG2: R_C−R_A equivalent within (−5,+5) points.
- HG3: F_A−F_C > 10 points.
- HG4: R_C−F_C > 10 points.

10,000 paired trajectory resamples, NumPy default_rng seed 0, same indices for
all four contrasts; retain all arms/checkpoints together. Report 95% percentile
CIs and the 90% CI for HG2. For observed d, resampled means d*, e*=d*−d:
one-sided p against margin m = (1+count(e*>=d−m))/(B+1). HG2 lower p uses
m=−5; upper p=(1+count(e*<=d−5))/(B+1); TOST p=max(lower,upper).
Inclusive counts; no zero p-values. These are approximate bootstrap tests.
Holm-adjust the four primary p-values at .05, separately from HF. Also require
95% lower CI above the relevant superiority margin, or 90% CI strictly inside
the equivalence bounds. All four tests must pass for the proposed dissociation.
Synthetic null/boundary checks precede registration freeze.

If A or C overall mean accuracy ≤5%, withhold a recovery verdict as floor-limited.
Show per-checkpoint floors but retain all checkpoints in primary averaging.
HG1 failure or unresolved HG2 does not establish recovery. Non-significance
does not establish appropriate updating. Lower F_C than F_A alone does not
establish underconfidence when A is overconfident: HG4 is required.

## Secondary reporting

Per-arm/checkpoint R, pre/post F, forecast MAE, signed gap, awareness and parse
counts; paired history×state interactions for R and F; calibration regression
of R/100 on F/100 with checkpoint effects and trajectory-bootstrap CIs. Constant
outcomes/predictors are unidentifiable. Use exact checkpoint and generated
offset cells for AUROC; compare verbal/P(Yes)/token features on common valid
observations, both global correctness and local correctness (ok+inherited vs
numeric_slip, nonanswers excluded), at least 15 per class. Prefix answers have
no model logits. Report class/support counts and missing cells. No secondary
finding changes the primary verdict.

## Freeze and deviations

Main runner requires this draft to be marked `Status: FROZEN` and relevant files
to match HEAD, plus an immutable local model/data/source manifest. Calibration
does not authorize a main run. Do not add the frozen marker until calibration
and power/precision review are complete. Any changed design gets a dated
amendment and new output directory. Positive, negative and inconclusive results
are all reportable. A single model/task and supplied errors limit generality.

**Calibration amendment, 2026-11-17 (before calibration inference):** user
requested GPU 1, with about 20.2 GiB free. The optional `gpu1-small` profile
uses utilization .38, context limit 2,048, eager mode, one active sequence,
512-token chunked prefill. The initial smoke remains ids 0–1, checkpoint 10,
all four arms, 80 continuation answers. Runtime/memory settings change only
for calibration, are logged in configs and are rejected for main inference.
No history truncation, prompt, dtype, decoding or hypothesis changes. Failure
to fit must be reported, not treated as data. Main inference has not started.

**Broader calibration amendment, 2026-11-17 (before the pilot):** following the
successful early smoke, the user authorized the planned eight-trajectory pilot
to completion on GPU 1. Use calibration ids 0–7, all seven checkpoints, all four
arms (224 conditions / 2,240 task answers). The `gpu1-long` profile retains
utilization .38, eager mode, one active sequence and 512-token chunked prefill,
with context limit 6,144 after a full local-tokenizer context check. Inputs are
never truncated. This is a runtime-only calibration change; main configs,
decoding, masks, probes and hypotheses remain unchanged. Complete the prespecified
pilot rather than stop on favorable outcomes. Save checkpoints and resume only
matching fingerprints. No main inference or automatic HG freeze is authorized
by this calibration run.


## Final pre-test decision — 2026-11-18

The full MR pilot and M calibration completed. Retain **all original HG1–HG4
contrasts, ±5/10-point margins, floor gate, ids 20–51, checkpoints and seed**.
The pilot did not establish overall repaired-arm underconfidence; it does not
justify selecting late checkpoints, removing HG4 or changing the margins. The
main experiment can produce a null, mixed or inconclusive finding.

Main runtime adopts the calibrated `.38 / 6,144 / eager / one sequence /
512-token chunked prefill` settings on GPU 1 in canonical configs, preserving
prompts, bf16, greedy decoding, masks and actual context. CPU audit found all
main M/MR input bounds ≤5,430. The MR clean/reference cue is shared by all four
MR arms; absolute accuracy should not be equated across M and MR, whose cue
protocols differ.

Precision is a local joint-contrast sensitivity analysis using eight-trajectory
calibration residuals and doubled variance. It cannot establish future model
behavior or guaranteed power; retain N=32 and report uncertainty. The corrected
simulation avoids invalid shifted arm percentages. Extracting the shared test
helper left the observed HG test outputs identical. This is a simulation fix,
not an alteration of empirical HG tests. The pilot generation source snapshot
and original simulation outputs remain preserved.

Both studies are to be collected regardless of favorable/unfavorable outcomes;
main-run compute still requires approval of the measured roughly 90-minute
combined budget. Independent confirmation needs a separate allocation and
registration, since the reserve was proposed in an older study plan.
