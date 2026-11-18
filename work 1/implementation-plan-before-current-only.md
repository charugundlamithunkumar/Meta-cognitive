# Implementation plan: Catching Silent Collapse

**Project:** When a long-horizon agent becomes pessimistic, does it recognize its errors—and which signals can help stop its decline?
**Working paper title:** *Catching Silent Collapse: Do LLM Agents Recognize Failure and Recovery?* This is a research question; neither the explanation of pessimism nor a recovery dissociation has been established.
**Companion docs (in this folder):** `after-the-fix-problem-statement.pdf` (accepted recovery proposal), `METACOGNITION.md` (M guide), and `prereg.md` (HF). Earlier documents are under `previous work/`.
**Group setup:** three members, one model each, same dataset, splits, prompts, arms and metrics:
- Qwen3-8B (teammate)
- Llama (teammate)
- **IBM Granite 4.2 8B (this plan)**

**Status, updated 2026-11-17:** The user reports that the instructor accepted the [After the Fix proposal](after-the-fix-problem-statement.pdf). The active implementation and experiment roadmap is **[§18](#18-approved-metacognition-and-recovery-program-2026-11-17)**. Study 1 (A0–D2, HA–HE) and exploratory A0 analysis are complete. M/HF is implemented and mock-tested, but real M inference is still pending. The recovery experiment (MR/HG) is accepted for development; its runner, configs, analysis and separate preregistration are **not yet implemented**. HF still needs its registration commit. The existing **43 tests passed again on 2026-11-17**. No GPU inference was launched for this plan update.

**Local resources verified:** `models/granite-4.2-8b/` contains all four shards named by its weight index (17,583,228,032 bytes total), model config and tokenizer files. `data/lhe/test.jsonl` exists and the pinned dataset checksums pass. Reuse these local files; downloading again is unnecessary. Shard presence is not a weight-content checksum or an inference smoke test.

**Execution order:** write HG registration draft → implement and CPU-test MR → approved calibration smoke/pilot → freeze HF/HG and materials → separately approved full M and MR runs → paired analysis → independent confirmation if feasible. M answers the instructor's length-versus-errors question; MR tests recognition of recovery. Developing MR does not depend on obtaining significant M results.

**Reading this plan:** Only the active metacognition/recovery roadmap is retained here. Historical sections 0–14 and the complete pre-reorganization plan are preserved in `previous work/implementation-plan-through-20261117.md`. Follow §18 for current execution; references to the old §14 refer to that archived snapshot.

---

## 15. Current priority: M, metacognition under controlled histories

### 15.1 Research question and the evidence already available

**Question:** As execution deteriorates, does the model's expressed pessimism
respond to the length of its history, to errors in that history, or to both—and
how much information does it provide about actual correctness?

Keep four quantities distinct throughout the paper:

| Quantity | Observable | What it can establish |
|---|---|---|
| Confidence level | Mean verbal confidence, no-evidence forecast, awareness probabilities | Whether reports become more pessimistic |
| Sensitivity to controlled history | Differences between M conditions at the same trajectory/checkpoint | Whether supplied-history treatments change reports |
| Error discrimination | AUROC for a clearly specified correctness target | Whether scores rank correct and incorrect events |
| Calibration / usefulness | Forecast versus realized next-10 accuracy; held-out predictive or intervention benefit | Whether expressed uncertainty is accurate or useful for decisions |

A falling mean is not evidence of improved discrimination. Error-sensitive
reports do not by themselves prove introspection, nor does low forecast error
when all outcomes are zero establish useful monitoring. Here, “silent collapse”
means sustained execution failure without dependable step-specific detection or
repair; it does not require the model to remain verbally confident forever.

The existing exploratory analysis reads the A0 archive directly: 32 test
trajectories × 300 turns. These are **A0 test results**, not a newly run pilot.

| Observation | Recomputed A0 result |
|---|---|
| Verbal confidence, bins 0–49 and 200–299 | 96.72 → 1.20 |
| No-evidence forecast, checkpoints 10,25,50,100,150,200,250 | 90.625, 62.4375, 62.46875, 0, 3.125, 0, 0 percent |
| Pooled verbal / P(Yes) / minimum-logprob local-slip AUROC | 0.613 / 0.489 / 0.792 |
| Verbal AUROC within the four position bins | 0.539 / 0.562 / 0.549 / 0.524 |
| Exact-position early history comparison | Error-history minus clean-history confidence −0.48 points, CI [−0.76, −0.21]; sparse overlap |

Source: [`results/meta_analysis.md`](results/meta_analysis.md) and its JSON.
The matched contrast covers 30 overlapping exact positions; only 6,359 of 10,000
trajectory resamples retain both groups at every included position. Report that
support loss and the n<30 flags. The small early difference is not evidence of a
large monitoring effect or proof that context length causes later pessimism.
The 224 all-checkpoint forecast observations here differ from the 56 eligible
checkpoints in the original HC comparison.

Use “weaker local-slip discrimination” rather than “no signal”: the verbal score
has a pooled AUROC above chance, and some within-bin CIs also exclude .5. Coarse
bins do not fully control position. Do not infer a statistically significant
verbal-versus-logit difference merely from their point estimates; use a paired
comparison on common observations for that claim.

### 15.2 Implemented experiment and registration

`prereg.md` §7 is the authoritative HF specification. The following is a summary,
not a replacement registration.

| New arm | Supplied history through checkpoint c | Intended wrong-total rate |
|---|---|---:|
| M_L0 | Full prefix | 0% |
| M_L25 | Full prefix | 25% |
| M_L50 | Full prefix | 50% |
| M_L75 | Full prefix | 75% |
| M_L100 | Full prefix | 100% |
| M_S0 | Last five pairs of the L0 history | 0% |
| M_S100 | Last five pairs of the L100 history | 100% |

All arms use ids 20–51, T=300, K=1 and A0 checkpoints
{10,25,50,100,150,200,250}, indexed from zero. Checkpoint c follows c+1 supplied
answers; its generated continuation covers c+1,…,c+10. Each fork starts afresh
from its designated history. Generated branches never contaminate later
teacher-forced checkpoints.

The selected wrong values come directly from the existing +1 generator in
`previous work/tests/test_arms_mock.py:FakeLLM.respond`, called from the true preceding state.
At each checkpoint, nested masks select the nearest integer to rate×(c+1), with
the fixed seeded procedure in HF. Log the mask and actual rate. This manipulates
**wrong totals**, not necessarily new arithmetic slips: repeated truth+1 answers
can be locally inherited. It does not reproduce an empirical error distribution.

Verbal, fore_none and aware use the original prompts in independent side forks.
The ten continuation answers are generated without probe responses in their
history. Each generated answer supplies token features and separate verbal and
P(Yes) probes. Teacher-forced answers have no generated logits.

The implementation reuses the frozen chat, prompt, parsing, probability,
labeling and statistical primitives. New checkpoint files are committed
atomically, exported to `rows.jsonl` / `trajectories.jsonl`, and packaged into
one-root-directory tar.gz archives. Resume checks configuration/source hashes
and skips completed checkpoints. Full design counts are 1,568 checkpoint forks
and 15,680 generated continuation answers; these are planned counts, not results.

### 15.3 HF estimands and decision rules

Average over checkpoints within each trajectory, then equally over trajectories.
All primary effects are measured in verbal-confidence percentage points.

- **Length contrast L:** conf(L0) − conf(S0); negative values mean more pessimism
  in the long condition. Define P=−L for the decision rule.
- **Endpoint error contrast E100:** conf(L0) − conf(L100), with a CI; descriptive.
- **Primary error-dose contrast D:** minus the confidence slope across
  L0/L25/L50/L75 at nominal rates 0/.25/.50/.75, expressed per 0→100% change.
  Report the arm means and departures from monotonicity as well.
- Meaningful effect: **10 points**. This is a prespecified practical margin,
  not an empirically established universal boundary for metacognition.
- Paired whole-trajectory bootstrap: 10,000 resamples, seed 0. Four tests have
  their own Holm family: P>10, D>10, equivalence of P within ±10, and equivalence
  of D within ±10. Superiority requires a 95% CI lower bound >10 and adjusted
  two-sided p<.05. Equivalence requires a 90% CI inside ±10 and adjusted TOST
  p<.05. Follow the exact bootstrap tails in HF/code.

| Decision | Required evidence |
|---|---|
| Length-driven | Meaningful P and equivalently small D |
| Monitoring-driven (operational label) | Meaningful D and equivalently small P |
| Mixed | Both P and D meaningful |
| Inconclusive | None of the above, including insufficient precision |

In prose, describe “monitoring-driven” as **error-sensitive reports** until
calibration or additional controls support a stronger interpretation. Similarly,
“length-driven” is currently a **history-truncation contrast**, not proof of a
pure token-count mechanism.

A mean next-10 accuracy ≤.05 is the registered floor. Intermediate doses remain
primary even if L100 is at floor. If all intermediates are also at floor, report
confidence effects while marking performance-monitoring evidence unidentifiable.
HF secondary analyses include checkpoint-adjusted forecast/accuracy associations
and within-arm AUROCs at fixed checkpoint **and** continuation offset. Preserve
the ≥15-per-class rule and report unidentifiable cells. All 32 complete paired
trajectories are required for a confirmatory verdict; missing/partial/mock runs
cannot supply one. HA–HE are not retested in the HF family.

### 15.4 Runnable sequence and remaining work

| Stage | Status / exit condition |
|---|---|
| A0 analysis | Done; report, JSON, three figures and archive provenance available |
| M implementation | Done; seven configs, generator reuse, checkpoint forks, continuation logging and resume |
| CPU validation | Latest validation: 43 passing tests, including 13 new tests; seven-arm mock and resume exercised |
| Registration | HF text written before analysis code; commit relevant materials before test inference |
| GPU smoke | Pending approval; calibration ids only; inspect prompt lengths, nonanswers, probe parsing and saved histories |
| M test run | Pending approval; all seven arms on the fixed 32 ids; do not change settings after examining outcomes |
| HF analysis | Implemented; empirical outputs pending; all verdicts, floors, invalid values and missing cells must be reported |
| Publication extensions | Proposed in §16; neither implemented nor registered by this plan update |

CPU commands:

```bash
python3 scripts/analyze_meta.py
python3 -m unittest discover -s tests -v
python3 scripts/run_meta.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/meta_cpu_smoke
python3 scripts/analyze_meta_experiment.py runs/meta_cpu_smoke --n-boot 100
```

After review, registration commit and GPU approval:

```bash
python3 scripts/run_meta.py --split calibration --ids 0-1 --arms L0 L100 S0 S100 --checkpoints 10 --gpu 0 --out runs/meta_gpu_smoke
python3 scripts/run_meta.py --split test --gpu 0 --out runs/meta_HF_20261116
# Only if interrupted:
python3 scripts/run_meta.py --split test --gpu 0 --out runs/meta_HF_20261116 --resume
# CPU analysis of completed outputs:
python3 scripts/analyze_meta_experiment.py runs/meta_HF_20261116
```

Use new output roots if these already exist. `METACOGNITION.md` has the explicit
registration staging command and file schema. The GPU path is not yet validated
by real M inference. A smoke failure warrants a documented fix and fresh freeze,
not tuning on test outcomes.

---

## 16. Publication plan: evidence needed for defensible claims

**Scope of this update:** the steps below strengthen the research plan. They are
proposed work, not changes to the implemented HF protocol. Give supplemental
conditions new arm names and a separate dated registration before collecting
their outcomes. Publication depends on the resulting evidence and review; this
plan cannot guarantee acceptance.

### 16.1 Address the main identification problems

| Risk in the current design | Consequence | Proposed follow-up / reporting requirement |
|---|---|---|
| S0 drops the earlier keys and absolute-state evidence, while the system still says the total starts at zero | L0−S0 changes information as well as context length | **Priority:** add matched-state long/short controls. Give both the same truthful state at the boundary before the retained five turns, in the same wording and position relative to that suffix. Only the older prefix differs. This reduces the state-information confound; it still does not isolate physical token count from all contextual content. |
| Error dose also changes whether the last supplied total is correct | Continuation failure can be ordinary inheritance, and current-total confidence may respond just to that final answer | **Priority:** independently cross earlier-history error dose with final-total status. Hold the endpoint at truth across a dose series, and at truth+1 across a second series; define dose over the earlier positions only. Compare within endpoint status. Keep these results distinct from the original seven M arms. |
| A constant +1 injection creates repeated offsets and correction transitions | An effect could reflect this artificial error pattern | Preserve +1 as the registered M manipulation. Propose separately frozen calibration-derived slip perturbations and multiple mask seeds as robustness checks; never fit the generator to test outcomes. Log numeric/token lengths and correction-transition counts. |
| Verbal/P(Yes) asks about exact global correctness, but the registered AUROC labels ok+inherited as positive | Weak local-slip AUROC does not establish absence of information about the question actually asked | Report both targets in a supplemental analysis: global correctness versus all errors, and the original local-slip taxonomy. Evaluate all signals on common observations; mark global discrimination unidentifiable at floor. |
| Teacher-forced answers were supplied rather than generated by the agent | Confidence elicited after reading a supplied answer need not reflect the same computation as confidence after producing it | Keep checkpoint and generated-continuation results separate. Validate transfer to naturally generated histories using held-out LHE sequences; do not claim access to the model's internal monitoring mechanism from input interventions alone. |
| Pooled calibration may be explained by checkpoint or arm averages | Reports can look predictive without adding useful information within a condition | Add held-out prediction against position-only and position-plus-arm baselines; split by whole trajectory. Assess report value within matched conditions where outcomes vary. Keep prospective forecast accuracy separate from current-answer confidence. |

For verbal confidence, its relation to **future** accuracy is an association,
not calibration of the event asked in the verbal prompt. Actual verbal
calibration uses current global correctness. Awareness is ordinal; the chosen
optimism score is not a probability of correctness. Raw token logprob is also
not a calibrated probability of arithmetic correctness. Keep these distinctions
in figure labels, methods and conclusions.

For the constant-endpoint control, prespecify how endpoint correctness is
presented without adding a checker verdict. For matched-state length controls,
log exact token counts: equal turn count is not necessarily equal token count,
particularly when corrupted numbers change tokenization. A pure length claim
would need further content/length controls, not just a changed arm label.

### 16.2 Precision, replication and practical value

1. **Power and coverage before expensive follow-ups.** Use synthetic simulations
   or calibration trajectories to assess precision for the 10-point effect and
   the equivalence rules under clustered outcomes, ceilings and floors. Check
   the bootstrap decision procedure under null scenarios. Do not claim that
   32 trajectories guarantee adequate power. With 32 items at an exact
   checkpoint/offset, ≥15 per class requires a nearly balanced cell; preserve
   missing AUROCs rather than lowering the threshold after seeing results.
   Any expanded sample or alternative stratified estimand needs its own freeze.
2. **Independent LHE replication.** Check which reserve ids/fresh seeds have
   actually been used or allocated to §14. Select and register disjoint unused
   sequences before accessing their outcomes. Reusing ids 20–51 for M offers
   paired interventions, but not an independent confirmation of the originally
   discovered A0 pattern. Avoid optional stopping when significance appears.
3. **Replication across model families.** Coordinate the same protocol with the
   Qwen and Llama teammates, with exact model/revision/tokenizer metadata and
   matched decoding and probe placement. Report effects separately per model;
   cross-model agreement is a result to test, not an assumption. A Granite-only
   paper must make a Granite-specific claim.
4. **Within-dataset generalization.** Consider fresh LHE sequences and a bounded
   total variant to check number-magnitude effects. The bounded variant changes
   the task distribution and must be separately registered and reported. Another
   task domain would strengthen external validity but is outside the current
   one-dataset scope; it requires a separately agreed extension.
5. **Signal utility, if claimed.** Existing HA gives evidence about a logit gate
   versus a random compute control; it does not compare verbal-triggered and
   logit-triggered repairs. A new comparison would need matched intervention
   budgets, calibration-only threshold selection, and held-out trajectories.
   Keep such a utility experiment optional unless the paper's claim requires it.

### 16.3 Paper structure and release criteria

A focused paper should lead with the metacognition question. Use the original
intervention results as context and supporting evidence rather than making every
proposed §14 extension a prerequisite for the M study.

| Paper component | Evidence / artifact |
|---|---|
| Motivation and behavioral phenomenon | A0 confidence/forecast/awareness versus accuracy, explicitly exploratory |
| Measurement definitions | Global correctness, local slips, forecast target, teacher-forced versus generated answers |
| Main controlled experiment | M arm × checkpoint report/accuracy curves; L, D and E100 CIs; all four Holm tests; floor reporting |
| Identification checks | Matched-state length and fixed-endpoint error controls, or an explicit restriction of causal claims if absent |
| Discrimination and calibration | Paired signal contrasts, per-class counts, exact-position coverage, future forecast calibration and prediction baselines |
| Replication and limitations | Held-out LHE evidence and other models if available; otherwise a narrow case-study conclusion |
| Reproducibility | Registration commit, source/config hashes, archived raw rows, input-history masks, mock tests and exact analysis commands |

Before submission: reconcile every numerical claim with a saved artifact; state
which results were exploratory or prospective; report all seven M arms including
null, reversed and inconclusive findings; disclose floors and parsing failures;
freeze the literature-search date; include the instructor's contribution and
agree attribution for the originating hypothesis. Public releases must exclude
unpublished instructor material unless permission is given. Do not use a claim
of novelty to appropriate the instructor's prior observation.

**Claim ladder:** existing A0 supports a descriptive Granite/LHE dissociation;
completed M can establish sensitivity to the specified history manipulations;
additional controls can strengthen causal separation; held-out and cross-model
replication can support broader generality. None of those results is established
merely by writing the code. A well-controlled null or mixed result can still be
scientifically useful if it resolves the competing explanations with precision.

---

## 17. Related work and novelty assessment (search checked 2026-11-16)

This is a targeted literature check, not an exhaustive systematic review or a
proof of priority. Searches combined “LLM metacognition,” “long-horizon confidence,”
“context length,” “error history,” “self-conditioning,” “pessimism,” and
“metacognitive decay.” Primary paper pages and the closest experiment sections
were checked; repeat the search before submission and examine citations of the
closest papers. Older and newer versions can differ, so record the versions
actually used in the manuscript bibliography.

| Related work | Established overlap | Boundary for this project's contribution |
|---|---|---|
| [Sinha et al., *The Illusion of Diminishing Returns*, §3.2 / Fig. 5](https://arxiv.org/html/2509.09677#S3.SS2) | Manipulates error rates in supplied histories to study execution degradation and self-conditioning | Credit the manipulation and LHE task to this work. The candidate extension is measuring self-reports, their correctness targets and calibration under those interventions. |
| [Barkan, Black and Sourbut, *Do Large Language Models Know What They Are Capable Of?*, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1a96349bbc03432c5ec8c6c502d102e7-Abstract-Conference.html) | Studies success forecasts, learning from in-context outcomes, and confidence updates during multi-step agentic tasks | Confidence over agent trajectories is already studied. Our proposed distinction is controlled history-dose/length comparisons with exact step ground truth, not the first trajectory-confidence evaluation. |
| [Lin et al., *Uncertainty Quantification for LLM Agents*, 2026](https://arxiv.org/abs/2609.07395) | Examines step/trajectory calibration, horizon-specific evaluation and confidence baselines on agent traces | Do not claim to introduce trajectory-aware calibration or the need for horizon baselines. Demonstrate what the controlled intervention adds. |
| [Tian et al., *Just Ask for Calibration*, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.330/) | Finds settings where verbalized confidence is better calibrated than conditional token probabilities | The current local-slip AUROC comparison is task- and metric-specific; it does not show universal superiority of token probabilities or contradict a different calibration result. |
| [Kumaran et al., *Causal Evidence that Language Models use Confidence to Drive Behavior*, 2026](https://arxiv.org/abs/2603.22161) | Uses confidence measures and activation interventions to study abstention; verbal confidence can matter despite weaker correctness discrimination | Do not equate weak local-slip discrimination with absence of metacognitive control. Our study manipulates input histories, not internal confidence representations. |
| [Kumaran et al., *How do LLMs Compute Verbal Confidence*, 2026](https://arxiv.org/abs/2603.17839) | Investigates representations and retrieval of confidence around answer generation | Teacher-forced checkpoint confidence is a different measurement setting; generated-answer validation matters before making a mechanistic claim. |

**Assessment:** the broad topic—LLM metacognition, confidence/calibration, or
self-conditioning—is not new. A Granite replication alone is a limited extension.
The potentially distinctive contribution is the **joint test of history-dependent
pessimism, local error discrimination and forecast validity under controlled
history interventions**, with the state/endpoint and target-definition problems
addressed. The reviewed sources do not establish that this exact combination has
already been tested, but this search is insufficient to assert a “first” claim.

**Safe proposed contribution statement:** “We evaluate whether pessimistic
self-reports during long-horizon execution reflect sensitivity to supplied
history length and errors, and whether that sensitivity yields accurate forecasts
or identifies local slips. We extend controlled-history experiments with paired
metacognitive measurements and explicit checks for inheritance, floor effects
and correctness-target mismatch.” This describes the intended study; change
“evaluate” to a concrete empirical finding only after the relevant runs and tests.

A credible publication case requires the causal question to be answered rather
than just new labels on existing curves. The strongest current path is a focused,
reproducible empirical study with carefully bounded claims; stronger controls and
independent replication would make that case more persuasive. Venue suitability
and acceptance remain judgments for the completed evidence and peer review.

---

## 18. Approved metacognition and recovery program (2026-11-17)

### 18.1 Scope, questions and current readiness

Instructor acceptance is reported by the user on 2026-11-17. This authorizes
preparation of the accepted study; it is not a report of successful experiments
or a reserved GPU budget. The approved question is documented in
[`after-the-fix-problem-statement.md`](after-the-fix-problem-statement.md).

| Stage | Question | Existing evidence / remaining work |
|---|---|---|
| A0, exploratory | What changes as execution collapses? | Existing 32 × 300 results and `results/meta_analysis.{md,json}`; preserve them. |
| M, HF | How do history length and injected error dose affect reports and next-10 accuracy? | Seven configs, runner, analysis and tests exist. Registration commit and real runs remain. Follow §15 and `prereg.md` §7 unchanged. |
| MR, new HG | When current state is repaired, does execution recover and do forecasts reflect that recovery? | Four-arm protocol below. Write registration before new analysis code, then implement, validate and run. |
| Confirmation | Does the finding repeat on unused sequences or another model? | Audit sample allocations first. Separate manifest, registration and budget; not a prerequisite for reporting an inconclusive first study. |

**Primary scope:** Granite 4.2 8B, LHE running sum, K=1, T=300, the existing
dictionary, non-thinking greedy continuation and independent probe forks.
The core paper combines A0 motivation, M identification and MR recovery.
Do not expand into the older §14 intervention project automatically.

**Protection:** no changes to or re-runs of A0–D2, their registered source,
configs, archives or results. Do not delete local weights to free space. Keep
HF separate from HG; additional controls get new names and registrations.
Never send results to the instructor or publish them without an instruction.

### 18.2 Resource and provenance gate

Use `models/granite-4.2-8b`, configured revision
`a7042144125ce5ae64f89610e8289647fc000b9a`, and local data revision
`7e4901fbb42b88144a65782052d0c8d87a358edc`. Dataset checksums and the existing
43-test CPU suite passed for this update. Do not equate configured revision
with independently verified downloaded weight contents.

Before inference, record an MR manifest containing model/config/tokenizer and
weight-shard SHA-256 hashes, data hashes, source/config hashes, registration
commit, Python/library versions, GPU model and driver, decoding settings,
sample ids, checkpoints, mask seed, arm order and output path. Compute weight
hashes on CPU once and retain the manifest; no network download is needed.
Load the tokenizer with `local_files_only=True` and the model from its local path.

Check GPU availability, free disk and context length immediately before a run;
old GPU availability and disk figures elsewhere in this plan are historical.
Retain bf16, configured memory utilization 0.6 and max context 16,384 unless a
calibration-only amendment is necessary. Never truncate history silently.
Instructor acceptance does not waive the user's requirement to approve GPU runs.

### 18.3 MR history construction and four arms

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

### 18.4 Probe placement, continuation and correctness

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

### 18.5 HG hypotheses and analysis contract

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

### 18.6 Implementation units and reuse boundaries

The following MR paths are **planned, not currently runnable**:

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

### 18.7 Persistence, reproducibility and failure handling

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

### 18.8 Validation and sample-size gates

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
fresh seeds in §14, run configs and available archives, without inspecting
outcomes to choose a favorable subset. Freeze a new disjoint manifest or use
documented fresh LHE sequences from unused seeds via `generate_fresh`. Add
explicit manifest support to the new runner; do not change `SPLITS` or claim
the existing M CLI accepts reserve/fresh inputs. The required N and budget are
chosen before that confirmation. Coordinate cross-model replication later;
Granite-only evidence remains Granite-specific.

### 18.9 Milestones and completion criteria

| Milestone | Deliverable | Exit condition |
|---|---|---|
| P0 — readiness (checked 2026-11-17) | Local model/data presence, dataset hashes, existing tests | Four indexed shards present; data checksums pass; 43 existing tests pass. Weight-content manifest still to create. |
| P1 — register the design | Draft HG, sample/seed rules, analysis contract | Written before MR analysis code; outstanding assumptions explicit. |
| P2 — implement and validate on CPU | New modules/configs/CLI, tests, mock+resume+analysis | Existing and new tests pass; synthetic statistical checks pass; mock verdict withheld. |
| P3 — calibration and precision | Approved GPU smoke/pilot, timing and variance report | Cue behavior validated, floors assessed, cost/precision reported, amendments logged. |
| P4 — freeze and authorize | Registration commits and source/model/data manifests | Relevant files committed; test guards pass; actual GPU budget approved. |
| P5 — collect | All seven M arms and four MR arms in separate roots | Complete expected paired cells, resumable archives, no test-driven protocol changes. |
| P6 — analyze | HF and HG tables, intervals, adjusted tests, figures | Full rules applied, missingness/floors disclosed, every number traceable to rows. |
| P7 — confirm and write | Disjoint confirmation if feasible, bounded manuscript claims | Literature update and attribution review; all null/reversed results included. |

No fixed calendar completion or GPU-hour claim is made before P3 measurements.
Implement both protocols before looking at their main outcomes; M results are
not a gate that selects whether to report MR. Main inference can run M then MR
for scheduling convenience, but the HG specification must already be frozen.

### 18.10 Commands and compute accounting

**Runnable now, CPU only** (from the repository root):

```bash
python3 -c "from src.lhe_data import verify; verify(); print('Dataset checksums OK')"
python3 -m unittest discover -s tests -v
python3 scripts/run_meta.py --help
python3 scripts/run_meta.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/meta_cpu_20261117
python3 scripts/run_meta.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/meta_cpu_20261117 --resume
python3 scripts/analyze_meta_experiment.py runs/meta_cpu_20261117 --n-boot 100
```

Choose a new output path if that example already exists. Existing CPU smoke
results need not be regenerated merely because this plan changed.

**Existing M GPU commands, only after the corresponding approval**; test runs
also require the HF registration/materials to match their commit:

```bash
python3 scripts/run_meta.py --split calibration --ids 0-1 --arms L0 L100 S0 S100 --checkpoints 10 --gpu 0 --out runs/meta_cal_20261117
python3 scripts/run_meta.py --split test --gpu 0 --out runs/meta_HF_20261117
python3 scripts/run_meta.py --split test --gpu 0 --out runs/meta_HF_20261117 --resume
python3 scripts/analyze_meta_experiment.py runs/meta_HF_20261117
```

**MR target CLI contract — not runnable until P2 implements these files:**

```bash
python3 scripts/run_recovery.py --preflight --split calibration --ids 0-1 --checkpoints 10
python3 scripts/run_recovery.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/recovery_mock_20261117
python3 scripts/run_recovery.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/recovery_mock_20261117 --resume
python3 scripts/analyze_recovery.py runs/recovery_mock_20261117 --n-boot 100
# GPU: only after approval; test requires the finalized HG registration.
python3 scripts/run_recovery.py --split calibration --ids 0-1 --checkpoints 10 --gpu 0 --out runs/recovery_cal_20261117
python3 scripts/run_recovery.py --split test --gpu 0 --out runs/recovery_HG_20261117
python3 scripts/run_recovery.py --split test --gpu 0 --out runs/recovery_HG_20261117 --resume
python3 scripts/analyze_recovery.py runs/recovery_HG_20261117 --n-boot 10000
```

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

### 18.11 Research outputs and claim boundaries

Write new MR results under `results/recovery_<run-id>/`: machine-readable
estimates, a report, provenance and figures. Required figures are (1) accuracy
by arm/checkpoint, (2) pre/post forecasts versus realized accuracy, (3) paired
HG effects with practical margins and intervals, and (4) secondary calibration
and discrimination with support counts. Keep A0 exploratory figures unchanged.

The new candidate contribution is the separation of **execution recovery**
from **forecast recovery** under retained erroneous history. State repair and
confidence trajectories already have precedents; credit the eight papers in
the accepted proposal and the instructor's originating question. The current
literature record was checked on 2026-11-16, not refreshed by this plan edit.

Explicit limitations: artificial +1 patterns; a teacher-forced past; one model
and task; numerical-width effects; cue consistency/trust (repair contradicts
the last displayed answer); no mechanistic activation intervention; limited
precision; and no independent replication yet. Matched-state long/short
controls, additional mask seeds, naturally generated failure histories and
other models are separately registered robustness studies, not silently added
primary arms. A floor-limited or null result remains reportable. Neither an
accepted proposal nor a completed implementation guarantees publication.

**Change record, 2026-11-17:** accepted recovery proposal incorporated as the
active roadmap; local model/data verified; baseline CPU suite rerun. Only this
plan changed. No HG registration, MR code or real inference is claimed complete.


**Administrative reorganization, 2026-11-17:** Unused Study-1 files and all old-work shortcuts now reside only in `previous work/`. Active `prereg.md` retains HF; complete historical registrations/plans are archived. Shared primitives remain in `src/`. The id parser and bin definitions are in `src/study_common.py`; current tools read the unchanged archived +1 generator directly. Archived regression tests run via `tests/test_legacy_regression.py`. This changes file locations/imports, not the experimental design. Existing mock outputs retain their original source fingerprints and must not be resumed with changed code; use a new output root.
