# Implementation plan: current metacognition and recovery study

**Project:** Catching Silent Collapse — Do LLM agents recognize failure and recovery?

**Updated:** 2026-11-18. The instructor accepted the recovery proposal. This file
contains only the current M/HF and MR/HG implementation and experiment roadmap.

**Start here:** M main is complete: 1,568 conditions / 15,680 task answers,
32 paired trajectories, seven validated archives, registered analysis and figures.
HF's classification is length-driven, interpreted as history truncation.
MR main was launched on GPU 0 at 12:26 IST under migration commit `e45f9df`. GPU 2 initialization
failed before any main answers because its service PATH omitted the installed
`ninja`; the service is inactive and its logs are preserved. GPU 0 is the same
RTX 6000 Ada model used for successful calibration. The
[GPU 0 amendment](prereg_recovery_gpu0_amendment.md) records the move and PATH
fix before inference. Original protocols were frozen in commit `1d11bc5`;
all scientific settings remain fixed. The user approved both studies to
completion. Keep the new job's source/config/resource fingerprints unchanged.

**Local resources:** Granite 4.2 8B is downloaded in `models/granite-4.2-8b/`.
Weights, tokenizer/template and pinned LHE data have CPU checksum manifests.

**Execution from here:** finish MR collection → automatic HG analysis/figures
→ validate paired cells and archives → complete the combined manuscript.
Independent confirmation is a separate conditional study requiring allocation,
registration and budget. CPU reporting deliverables are the
[partial manuscript](paper/manuscript-draft.md),
[literature update](paper/related-work.md) and
[confirmation audit](paper/confirmation-readiness.md).

**Current documents:** [accepted proposal](after-the-fix-problem-statement.pdf),
[M guide](METACOGNITION.md), [HF registration](prereg.md).
All old implementation plans are inside [previous work](previous%20work/README.md).
The completed study is background evidence; its protocols are not active run instructions.

---

## 1. Scope, questions and current readiness

Instructor acceptance is reported by the user on 2026-11-17. This authorizes
preparation of the accepted study; it is not a report of successful experiments
or a reserved GPU budget. The approved question is documented in
[`after-the-fix-problem-statement.md`](after-the-fix-problem-statement.md).

| Stage | Question | Existing evidence / remaining work |
|---|---|---|
| A0, exploratory | What changes as execution collapses? | Existing 32 × 300 results and `results/meta_analysis.{md,json}`; preserve them. |
| M, HF | How do history length and injected error dose affect reports and next-10 accuracy? | Main complete and validated; registered analysis and figures in `results/meta_HF_main_20261118/`. |
| MR, new HG | When current state is repaired, does execution recover and do forecasts reflect that recovery? | Implementation, calibration and freeze complete; main is moving to GPU 0 under the recorded hardware amendment. No main conclusion yet. |
| Confirmation | Does the finding repeat on unused sequences or another model? | Config/archive audit complete without reading outcomes; historical reservations remain. Separate manifest, runner, registration and budget required. |

**Primary scope:** Granite 4.2 8B, LHE running sum, K=1, T=300, the existing
dictionary, non-thinking greedy continuation and independent probe forks.
The core paper combines A0 motivation, M identification and MR recovery.
The completed intervention study is archived; this plan covers M and MR.

**Protection:** no changes to or re-runs of A0–D2, their registered source,
configs, archives or results. Do not delete local weights to free space. Keep
HF separate from HG; additional controls get new names and registrations.
Never send results to the instructor or publish them without an instruction.

### 1.1 M: direct test of the instructor’s question

M asks whether pessimistic self-reports respond to history length, historical
errors, or both. Reuse `scripts/run_meta.py`, its seven existing configs and
`scripts/analyze_meta_experiment.py`. The authoritative hypothesis and decision
rule are in `prereg.md` §7; do not change them through this plan.

| Arms | Supplied history |
|---|---|
| M_L0, M_L25, M_L50, M_L75, M_L100 | Entire prefix with 0/25/50/75/100% wrong totals. |
| M_S0, M_S100 | Exact last-five-turn suffix of the corresponding long history. |

Use ids 20–51, seven checkpoints, the existing +1 generator, unchanged probes
and independently generated next-ten answers. Primary confidence contrasts are
length and intermediate-dose effects; the L100 endpoint is descriptive when
performance reaches a floor. Use the existing paired trajectory bootstrap and
HF Holm family. Short history also removes reconstruction evidence, so its
effect cannot be interpreted as token length alone. Report forecasts against
actual accuracy and distinguish global correctness from local slips.

The A0 exploratory analysis in `results/meta_analysis.md` supplies motivation;
raw A0 archives reside in `previous work/results/`. M identifies responses to
controlled histories; MR then tests whether reports update after state repair.
Both studies must report their full outcomes, including null or floor-limited
results. The MR design must be frozen before examining either main study’s outcomes.

## 2. Resource and provenance gate

Use `models/granite-4.2-8b`, configured revision
`a7042144125ce5ae64f89610e8289647fc000b9a`, and local data revision
`7e4901fbb42b88144a65782052d0c8d87a358edc`. Dataset checksums and the existing
61-test CPU suite passed again on 2026-11-18. Three separate reporting-audit tests also pass. Do not equate configured revision
with independently verified downloaded weight contents.

Before inference, record an MR manifest containing model/config/tokenizer and
weight-shard SHA-256 hashes, data hashes, source/config hashes, registration
commit, Python/library versions, GPU model and driver, decoding settings,
sample ids, checkpoints, mask seed, arm order and output path. Compute weight
hashes on CPU once and retain the manifest; no network download is needed.
Load the tokenizer with `local_files_only=True` and the model from its local path.

Check GPU availability, free disk and context length immediately before a run;
check process ownership before loading a model on the shared GPU.
Retain frozen bf16, memory utilization .38 and max context 6,144, with eager
execution, one sequence and 512-token chunked prefill. These supersede the
initial .6 / 16,384 runtime after calibration and before main inference. Never
truncate history silently. Both main runs already have the user's approval;
additional studies need their own long-run budget approval.

## 3. MR history construction and four arms

At each zero-based checkpoint c in `{10,25,50,100,150,200,250}`, use task
positions 0…c as history and c+1…c+10 as the continuation. Each branch is
independent; generated answers at one checkpoint do not enter another.

1. Build H_clean with every assistant total equal to truth.
2. Build H_error with exactly `floor(0.5*c + 0.5)` wrong totals among positions
   0…c−1, plus a forced wrong answer at c. Shuffle earlier positions with
   `Random(seed*1000003 + sample_id*10007 + c)`, seed 0, and select that count.
   The earlier-error rate excludes the forced endpoint; record both rates.
3. Obtain every wrong total from the approved existing +1 generator through
   `src/meta_history.py:numeric_slip`. Verify every injected value differs from
   truth. Do not substitute random offsets or change the generator for HG.
4. Cross history with the supplied continuation state s, independently:

| Arm | Identical history shared with | History | State s supplied once |
|---|---|---|---|
| MR_A | MR_D | H_clean | True total T_c |
| MR_B | MR_C | H_error | Wrong total from the existing +1 generator at c |
| MR_C | MR_B | H_error, including wrong endpoint | True total T_c |
| MR_D | MR_A | H_clean | Wrong total from the existing +1 generator at c |

This crosses **historical error exposure** with **supplied current-state
correctness**. It does not independently cross the final history answer with
all earlier errors: that final answer is part of H_clean/H_error. Describe this
precisely when interpreting effects.

Freeze the neutral cue in new `src/recovery_prompts.py`, initially:
`Continue using this current running total: {s}.`
Use identical wording for all four arms; never describe the wrong value as
verified. Preserve the full history and system prompt in every arm. Record
input tokens, numeric magnitude/width, sign and correction-transition counts.
Same turn count does not imply exact token-count equality. No extra padding,
thinking, reset or continuous state correction belongs in the primary design.

## 4. Probe placement, continuation and correctness

Use the existing forecast text from `src/prompts.py:P_FORECAST`. The outcome is
the **percentage of ten globally correct totals**, not the probability that
all ten answers are correct. Save these distinct forks:

- **Pre-cue forecast:** append the forecast question to H alone. A/D and B/C
  can share this deterministic fork only when their exact prefix hashes match.
- **Post-cue forecast (primary):** append one user message containing the state
  cue, newline, then P_FORECAST to H. Do not add an invented assistant
  acknowledgement. This asks about future execution from the supplied state.
- **Task continuation:** independently append one user message containing the
  same cue, newline, then the actual keys at c+1. Generate ten totals. Later
  user messages contain only their keys. The cue appears exactly once.
- **Answer probes:** after each newly generated total, fork unchanged verbal
  confidence and P(Yes). Fork awareness after the tenth generated answer; its
  question about “these turns” then follows observed continuation behavior.
  Record its exact placement; it is not a probability of future success.

Never append a probe question or answer to the task continuation. Pre-cue and
post-cue forecasts are separate branches. Verbal confidence about an old,
teacher-forced answer is not the primary recovery measure.

**Critical label rule:** for the first generated step, `prev = s`, including
in MR_C and MR_D, rather than the last displayed history total. Thereafter
reuse `next_prev` on actual committed answers. Global truth remains the
original LHE cumulative sum; a cue never changes ground truth. Apply existing
`label_turn` and answer-token extraction. A malformed answer is incorrect,
not a missing or silently retried execution outcome.

Store raw numeric probes and the legacy clipped [0,100] value, with explicit
parse and out-of-range flags. HG must freeze this preprocessing before runs.
Missing primary forecasts make the full primary verdict unavailable until a
technical failure is resolved; do not silently drop trajectories or fill zero.
Record unavailable Yes/No or A/B/C probability mass rather than interpreting
zero retained top-k mass as confident evidence. This affects secondary probes,
not the numeric forecast primary outcome.

## 5. HG hypotheses and analysis contract

**First implementation deliverable:** `prereg_recovery.md`, separate from
HA–HF. Write its draft before `scripts/analyze_recovery.py`. Include all
estimands, preprocessing, exact test algorithms, seeds, sample manifests,
missingness rules, stopping rules and the floor rule. Finalize after calibration
and commit before any test inference. Calibration changes are dated amendments;
never adapt thresholds using A0 test outcomes or forthcoming M/MR test results.

For each arm, R is next-10 accuracy ×100 and F is the post-cue forecast.
Compute contrasts per trajectory by averaging all seven checkpoints equally,
then average the trajectory contrasts equally. Each trajectory is one paired
sampling unit; 224 checkpoints are not 224 independent observations.

| Hypothesis | Required effect | Reason |
|---|---|---|
| HG1 | R_C − R_B > 0 | Demonstrate actual benefit from repairing state. |
| HG2 | R_C − R_A equivalent within (−5,+5) percentage points | Demonstrate recovery near the clean reference, rather than infer equivalence from non-significance. |
| HG3 | F_A − F_C > 10 percentage points | Detect persistently lower forecasts at comparable performance. |
| HG4 | R_C − F_C > 10 percentage points | Establish actual underconfidence; a gap from an overconfident baseline is insufficient. |

Use 10,000 bootstrap resamples of complete paired trajectories, NumPy seed 0,
keeping all checkpoints/arms together and reusing resample indices across
contrasts. Report 95% percentile intervals and a 90% interval for equivalence.
Holm-adjust the four HG p-values at alpha .05, separately from HF and HA–HE.
All four adjusted tests must pass, with their practical margins, to support
the primary recovery/report dissociation. Margins are the accepted proposal's
values; a change requires an explicit pre-test amendment, not hidden tuning.

**Proposed exact p-value implementation to freeze in HG:** for observed mean
contrast d and bootstrap means d*, form e*=d*−d. A one-sided superiority test
against m uses `(1 + count(e* >= d-m))/(B+1)`. The lower equivalence test uses
m=−5; the upper test uses `(1 + count(e* <= d-5))/(B+1)`. The equivalence
p-value is their maximum. Use inclusive comparisons and B=10,000; never print
zero p-values. Validate null behavior and boundaries with synthetic simulation
before registration. These are approximate bootstrap tests, not exact
finite-sample tests. Reuse `src/stats.py:holm`; do not modify frozen Study-1
statistics or silently reuse HF's different p-value procedure for HG.

**Floor gate:** report mean accuracy by arm/checkpoint and across all
checkpoints. If either A or C has overall mean accuracy ≤5%, withhold a
recovery-recognition verdict as floor-limited. Otherwise include all checkpoints
in the primary analysis, even individually floor-limited ones. Show the late
curves explicitly; a pooled result is not proof of late-checkpoint recovery.

**Interpretation:** all criteria passing supports the specified behavioral
dissociation, not an internal mechanism. Failure of HG1 or inability to establish
HG2 leaves recovery recognition unresolved. Non-significance does not establish
appropriate updating; that claim would require a separately specified
equivalence test for calibration error. Report reversed and uncertain effects.
Do not select only trajectories that recovered or stop early for significance.

Secondary outputs: pre/post forecast change; forecast MAE and calibration slope
with checkpoint effects and trajectory intervals; per-arm/checkpoint curves;
the accuracy and forecast history×state interactions; awareness probabilities;
and generated-step AUROC for verbal/P(Yes)/token scores. Report global
correctness and local-slip targets separately, on common observations, holding
checkpoint and continuation offset fixed. Require the existing minimum of 15
observations per class; report unidentifiable cells and degenerate regressions.
Teacher-forced prefix values have no model logits and never enter AUROC.
Secondary findings do not rescue a failed primary decision.

## 6. Implementation units and reuse boundaries

The following MR paths are **implemented**. CPU/mock paths and the full real
calibration pilot have passed; main collection remains:

| New file / artifact | Responsibility and reuse |
|---|---|
| `prereg_recovery.md` | HG contract, calibration deviations, frozen allocation and registration record. |
| `src/recovery_history.py` | Build paired histories/masks using `numeric_slip`; return immutable history and state metadata. |
| `src/recovery_prompts.py` | Only the new state cue; import unchanged original task/probe prompts. |
| `src/recovery_runner.py` | Compose/subclass M's engine primitives for explicit forks and correct `prev=s`; adapt checkpoint persistence without editing frozen loops. |
| `configs/MR_{A,B,C,D}.json` | Local model, decoding, checkpoints, factors, mask rule, schema version and registration path. |
| `scripts/run_recovery.py` | Local-only preflight, mock/calibration/test CLI, manifest check, registration guard, measured timing and resume. Reuse id/data loaders. |
| `scripts/analyze_recovery.py` | HG estimates/decisions, secondary tables and figures; reuse archive readers, metrics and Holm. |
| `scripts/recovery_precision.py` | CPU simulation using synthetic or calibration data only; power/precision and null-test validation. |
| `tests/test_recovery.py` | Construction, probe isolation, state labels, resume and statistical fixtures. |
| `RECOVERY.md` | Exact implemented commands, provenance and interpretation; no claims from mock results. |

Keep M artifacts unchanged unless a real bug is found; document and register any
necessary M fix before its inference. Existing `run_meta_arm` explicitly rejects
MR names, and its validator fixes the M row structure. Reuse safe low-level
helpers such as atomic writes, chat formatting and archiving; give MR its own
schema validator and arm dispatcher. Do not bypass M guards or copy its entire
runner/analysis just to rename arms.

## 7. Persistence, reproducibility and failure handling

Maintain the familiar per-arm `config.json`, `rows.jsonl`,
`trajectories.jsonl`, `run.log`, atomic `checkpoints/<id>_<c>.json`, and
`MR_<arm>.tar.gz` with one root directory. Add a top-level manifest and a schema
version. The row key is `(study, arm, sample_id, checkpoint, record_type, offset)`.
Each completed checkpoint has one metadata row and ten continuation rows;
attach forecast/awareness forks as explicitly named fields, not fictitious task
turns. Record shown histories, mask, true/supplied state, prompt hashes, raw
probes, token features and next-10 truth labels.

Resume must match model/data/source/config/registration fingerprints before
allocating a GPU. A crash before atomic checkpoint commit reruns only that
checkpoint; a crash after commit rebuilds JSONL without duplicates. Store raw
completed outputs even if parsing fails, so resume cannot generate a new answer
until it gets a preferable one. Reject corrupt/mismatched checkpoints and
archives instead of overwriting them. Never mix mock and real results.

Deterministic arm order and generation batch size must be recorded and frozen.
Timing and transient GPU memory are runtime logs, not resume fingerprint inputs.
A resume of fully complete arms should validate/export without model loading.
Analysis accepts archive inputs read-only and emits results to a new MR-owned
directory; it must refuse an accidental existing output overwrite.

## 8. Validation and sample-size gates

**CPU acceptance tests:** exact earlier-error count and forced endpoint;
A/D and B/C byte-identical histories; paired keys/checkpoints/cues; wrong totals
never truth; both cue values differ; first-step labels use s; no probe leakage;
one-time cue; exact ten-step horizon; correct raw/clipped/missing parsing;
context-limit rejection; crash/resume equivalence; config/model mismatch
rejection; immutable archives; and test runs rejected without registration.

Synthetic analysis fixtures must cover all HG criteria, no performance
recovery, equivalence boundaries, a lower forecast than an overconfident A
without actual underconfidence, all-zero floors, missing arms, single-class
AUROC and the distinction between trajectory and row resampling. A mock may
demonstrate plumbing but must never receive an empirical HG verdict.
Use a new mock adapter if it needs to recognize the neutral cue; preserve the
original test generator and tests. All existing tests must continue passing.

**Calibration sequence after GPU approval:** first ids 0–1 at checkpoint 10
in all four MR arms (8 conditions, 80 continuation answers). Then a separately
budgeted feasibility pilot on ids 0–7 at all seven checkpoints (224 conditions,
2,240 answers). Measure successful cue handling, clean/repaired accuracy,
probe validity, context fit and runtime. Do not treat pilot significance as the
criterion for choosing a design. A technical or floor failure triggers a dated
calibration-only revision and new output directory, not a quiet switch to
thinking or continual oracle corrections in test runs.

Use pilot variance and declared effect-size scenarios for prospective joint
power and CI-width simulations. Report power for all HG criteria jointly,
including equivalence, rather than only the easiest contrast. Small pilots
give uncertain variance estimates; include a conservative variance sensitivity.
Do not promise that 32 trajectories can establish ±5-point equivalence.

The initial prospective MR extension uses ids 20–51, as accepted, with all seven
checkpoints. These task sequences have already been examined in A0, so call it
a prospective intervention extension, not independent replication. Do not
reuse calibration ids for confirmatory tests or count mask seeds as independent
trajectories. All 32 complete paired trajectories are needed for the full
primary verdict; partial data are diagnostic/exploratory.

For independent confirmation, first audit uses/reservations of ids 52–99 and
fresh seeds in the archived plan, run configs and available archives, without inspecting
outcomes to choose a favorable subset. Freeze a new disjoint manifest or use
documented fresh LHE sequences from unused seeds via `generate_fresh`. Add
explicit manifest support to the new runner; do not change `SPLITS` or claim
the existing M CLI accepts reserve/fresh inputs. The required N and budget are
chosen before that confirmation. Coordinate cross-model replication later;
Granite-only evidence remains Granite-specific.

## 9. Milestones and completion criteria

| Milestone | Deliverable | Exit condition |
|---|---|---|
| P0 — readiness (done 2026-11-17) | Local model/data and full content manifest | All indexed shards and tokenizer/template files hashed; pinned data verified; CPU tests pass. |
| P1 — design (done) | HF/HG registrations | Drafted before analysis code; frozen before main inference. |
| P2 — CPU implementation (done) | Runners, analyses, configs and tests | All 61 existing tests pass; mock/resume and statistical fixtures exercised. |
| P3 — calibration and precision (done) | Pilots, amendments and precision reports | M 392 conditions / 3,920 answers; MR 224 / 2,240. Uncertain precision retained, thresholds unchanged. |
| P4 — freeze and authorize (done) | Commit `1d11bc5` and resource manifests | Both registrations frozen; both main studies explicitly approved. |
| P5 — collect (M done; MR pending) | Seven M and four MR archives | M 1,568 / 15,680 complete; MR requires 896 / 8,960 on GPU 0 under the hardware amendment. |
| P6 — analyze (M done; MR pending) | HF/HG tests, estimates and figures | M reported with limits; automatic HG analysis awaits MR data. |
| P7 — confirm and write (partial) | Draft, literature check and allocation audit | CPU documents saved. MR synthesis and final review remain. Independent confirmation has no allocation or budget yet. |

No fixed calendar completion or GPU-hour claim is made before P3 measurements.
Implement both protocols before looking at their main outcomes; M results are
not a gate that selects whether to report MR. Main inference can run M then MR
for scheduling convenience, but the HG specification must already be frozen.

## 10. Commands and compute accounting

**CPU verification and live status** (from the repository root):

```bash
python3 -c "from src.lhe_data import verify; verify(); print('Dataset checksums OK')"
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s paper/tools -v
python3 scripts/run_registered_meta.py --describe --gpu 1 --tag main_20261118
systemctl --user status mithun-recovery-gpu0-main-20261118
cat runs/main_recovery_gpu0_main_20261118/status.json
```

**Read-only MR preflight**, if a new technical concern warrants it:

```bash
python3 scripts/run_recovery.py --preflight --split test --gpu 0 --resource-manifest results/recovery_resources_complete_20261117.json
```

Existing main collection and analysis are handled by the approved queue; do not
rerun calibration, M or completed analyses. See §12 for the guarded manual
resume command, which must not run alongside the active service. For synthetic
software checks use a new output path:

```bash
python3 scripts/run_recovery.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/recovery_new_mock_check
python3 scripts/run_recovery.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/recovery_new_mock_check --resume
python3 scripts/analyze_recovery.py runs/recovery_new_mock_check --n-boot 100
```

Mock measurements never enter empirical conclusions. Existing smoke results
need not be regenerated solely because the plan's status changed.

`--preflight` must be read-only and CPU-only: resource/hash/registration/context
checks, counts and planned paths; no weight loading or generation. `--dry-run`
is synthetic and needs neither model nor dataset. Default MR arms are all four;
main test overrides of checkpoints, masks or prompts must be rejected.

M requires 7×32×7 = **1,568 checkpoint conditions / 15,680 continuation
answers**. MR requires 4×32×7 = **896 conditions / 8,960 answers**. Combined:
**2,464 conditions / 24,640 answers**, excluding all side probes and calibration.
For MR also budget pre/post forecasts, 20 per-answer verbal/Yes–No forks and
one post-continuation awareness fork per condition; exact prefix deduplication
can save repeated pre-cue forecasts. Batch calls are not equivalent to generated
sequences. Measure prefill/generated tokens and wall time across early/late
checkpoints to estimate the full budget. Ask for that concrete budget approval
before a long run; do not infer hours from answer count alone.

## 11. Research outputs and claim boundaries

Write new MR results under `results/recovery_<run-id>/`: machine-readable
estimates, a report, provenance and figures. Required figures are (1) accuracy
by arm/checkpoint, (2) pre/post forecasts versus realized accuracy, (3) paired
HG effects with practical margins and intervals, and (4) secondary calibration
and discrimination with support counts. Keep A0 exploratory figures unchanged.

The new candidate contribution is the separation of **execution recovery**
from **forecast recovery** under retained erroneous history. State repair and
confidence trajectories already have precedents; credit the eight papers in
the accepted proposal and the instructor's originating question. The current
literature record was refreshed on 2026-11-18 in `paper/related-work.md`,
including rollback and conversational-repair precedents. This targeted review
does not establish priority.

Explicit limitations: artificial +1 patterns; a teacher-forced past; one model
and task; numerical-width effects; cue consistency/trust (repair contradicts
the last displayed answer); no mechanistic activation intervention; limited
precision; and no independent replication yet. Matched-state long/short
controls, additional mask seeds, naturally generated failure histories and
other models are separately registered robustness studies, not silently added
primary arms. A floor-limited or null result remains reportable. Neither an
accepted proposal nor a completed implementation guarantees publication.

## 12. Current execution status and next checks

M is complete and preserved: 1,568 conditions / 15,680 task answers, 32 paired
trajectories, seven validated archives, registered analysis and figures.

The original GPU 1 queue stopped at its capacity guard. GPU 2 later attempted
model initialization, but failed on a service PATH missing `ninja`; no MR
output root or checkpoint was created. Both obsolete supervisors are inactive.
Their original logs, status and source snapshots are preserved.

The user instructed continuation on free GPU 0. The new GPU 0 amendment keeps
the frozen .38 / 6,144 / eager / bf16 / one-sequence / 512-token prefill runtime,
all four arms, ids, prompts, masks and HG statistical rules. The shared
supervisor now supplies the current Python environment's executable directory
to child PATH and records the existing ninja version/hash. No package download
or experiment-runner change is needed.

The GPU 0 entry point reuses the MR-only supervisor and original runner:

```bash
python3 scripts/run_recovery_gpu0.py --describe
systemctl --user status mithun-recovery-gpu0-main-20261118
cat runs/main_recovery_gpu0_main_20261118/status.json
```

Raw root: `runs/recovery_HG_gpu0_main_20261118/`.
Analysis root: `results/recovery_HG_gpu0_main_20261118/`.
All source/config/model/data/registration guards must pass before loading.
The memory guard is 19,185.2 MiB free on this 49,140 MiB GPU. Do not restart
obsolete GPU 1/GPU 2 jobs or launch duplicate supervisors. Resume only after
confirming that no GPU 0 supervisor/child is active:

```bash
/home/viswanath/miniconda/bin/python3 -u scripts/run_recovery_gpu0.py --resume
```

After collection the supervisor validates 896 conditions / 8,960 task answers,
four archives, 32 complete paired trajectories and source snapshots, then runs
the original HG analysis/figures. Review every test, missingness/floor rule and
secondary support count before completing the manuscript. A failed primary
criterion must remain visible; calibration cannot substitute for main results.

The partial manuscript, literature review and metadata-only allocation audit
are saved. Independent confirmation remains conditional and unallocated.
Earlier plans are archived only inside `previous work/`.

**GPU 0 launch record, 2026-11-18 12:26 IST:** commit
`e45f9df02567cab48613f96c7af9799bfcc53cbe` passed the HG and supervisor guards.
All 61 existing tests and seven reporting/migration tests passed. Full resource
checks and CPU context preflight passed (bound 5,430). GPU 0 had 48,462 MiB
free at admission. The service `mithun-recovery-gpu0-main-20261118.service` is
active; status is `inference` (includes model initialization). Inspect actual
checkpoint counts before claiming generated-answer progress or completion.

**Initialization confirmed:** vLLM completed model loading, cache creation and
warmup on GPU 0 at 12:27 IST. Real checkpoints are now being saved and their
checksums/labels pass the existing loader. See
`results/recovery_gpu0_launch_20261118.md`; the run remains in progress and no
interim HG conclusion is issued.
