# After the Fix
## Do LLM Agents Recognize Recovery from Silent Collapse?

**Problem statement for instructor review · 16 November 2026**

Project: Catching Silent Collapse · IBM Granite 4.2 8B · Long-Horizon-Execution (LHE)

**Status:** Proposed extension. The recovery experiment described here has not been implemented, registered or run. Existing A0 observations motivate it; they are not evidence that the proposed effect exists.

### The problem

An LLM agent can become pessimistic as a long task deteriorates. However, a lower confidence report does not establish that the agent recognizes individual mistakes or can recognize a successful repair. Conversation length, accumulated errors and current execution accuracy change together, making ordinary trajectories insufficient to distinguish these explanations.

**Research question:** If we correct the current running total while retaining the same erroneous conversation history, does the agent’s subsequent execution recover—and do its forecasts recover appropriately?

This extends the instructor’s metacognition question from recognizing deterioration to recognizing recovery. Here, metacognition means measurable correspondence between the model’s reports and its performance; it does not imply consciousness or establish an internal psychological mechanism.

### What the existing experiment establishes

The exploratory A0 analysis contains 32 trajectories, 300 turns each (9,600 rows). All turn indices below are zero-based.

| Measure | Recomputed A0 observation |
| --- | --- |
| Verbal confidence | 96.72 in turns 0–49; 1.20 in turns 200–299, on a 0–100 scale. |
| Forecast of next-10 accuracy | At checkpoints 10, 25, 50, 100, 150, 200, 250: 90.625%, 62.4375%, 62.46875%, 0%, 3.125%, 0%, 0%. |
| Local-slip discrimination | Pooled AUROC: verbal 0.613; P(Yes) 0.489; minimum token log-probability 0.792. Verbal AUROC within position bins: 0.539–0.562 for the first three bins, 0.524 for the last. |

Source: results/meta_analysis.md and results/meta_analysis.json, computed from the existing A0 archive. “Local slip” excludes inherited errors: a locally correct step can still have the wrong global total. Verbal confidence asks about global correctness, so the comparison also involves different prediction targets. The pooled verbal result should not be described simply as chance performance.

### Candidate contribution

The proposed contribution is a controlled test of **whether forecast recovery follows execution recovery**, with historical error exposure and supplied current state varied independently at a fixed checkpoint. The central possible finding is restored accuracy accompanied by persistently low forecasts. Aligned recovery or a failure to restore execution are equally valid outcomes.

**Novelty assessment:** Plausibly distinctive in this specific formulation, but not established as a first. History injection, state repair, confidence dynamics and intervention evaluation all have precedents. Page 4 identifies the closest overlaps. Publication would require convincing results, careful controls and independent confirmation.

<!-- PAGEBREAK -->

# Proposed experiment
## Separate historical errors from the state used to continue

### Foundation: the existing M design

M compares long histories with 0%, 25%, 50%, 75% and 100% injected errors, plus short five-turn histories with 0% or 100% errors. Its purpose is to separate history length and error exposure while measuring probes and next-10 accuracy. Its code and mock checks exist; no real M measurements are claimed here. The proposed recovery extension needs a separate registration and new arm names.

### A paired 2 × 2 recovery experiment

At each checkpoint, construct two full histories: a clean history and an error-containing history. In the latter, inject truth + 1 at approximately 50% of earlier positions using the existing numeric-slip generator; force the checkpoint answer to be wrong. Freeze the rounding and mask seed in the new registration. Independently supply a correct or incorrect current total in an identical neutral instruction template.

| Proposed arm | History read by the model | Supplied current total | Purpose |
| --- | --- | --- | --- |
| MR_A | Clean | Correct | Clean reference |
| MR_B | Error-containing | Wrong: truth + 1 | Unrepaired reference |
| MR_C | Identical to MR_B | Correct | State repair; past errors remain |
| MR_D | Identical to MR_A | Wrong: truth + 1 | Current-state corruption control |

The earlier corruption rate excludes the forced checkpoint error. A and D share exactly the same history; B and C share exactly the same history. All arms use the same dictionary, keys, checkpoint, future inputs, decoding settings and instruction wording. The state cue is supplied once, before continuation; it does not continuously correct subsequent answers.

### Procedure and measurements

- Use the same seven checkpoints: 10, 25, 50, 100, 150, 200 and 250. The initial feasibility extension uses the existing 32 task sequences, ids 20–51.
- Supply a neutral cue such as “Continue using this current running total: [value].” Do not falsely describe the wrong state as verified. Audit token counts and numeric-width differences; equal turn count alone is not exact token-length matching.
- On a separate fork, append the same state cue and the existing no-evidence forecast question: the expected percentage of exactly correct totals over the next ten turns. Separately generate those ten answers from the identical task state. No probe response enters execution history.
- Record pre-cue forecasts on separate forks as a secondary change measure. After generated answers, reuse verbal confidence, P(Yes), awareness and token features. Verbal confidence concerns the answer just generated; it is not a forecast of future recovery.

**Primary comparison:** MR_C versus MR_A, after establishing that MR_C improves over MR_B. Accuracy is the fraction of ten globally correct totals; forecast is the predicted percentage on the same horizon. Merely inserting the true state is not evidence of recovery.

**Planned scale:** 4 arms × 32 trajectories × 7 checkpoints = 896 checkpoint conditions and 8,960 continuation answers, plus forked probes. These are design counts, not measured compute costs.

<!-- PAGEBREAK -->

# Hypotheses, decisions and approval
## Proposed criteria to freeze before new analysis code

### Primary hypothesis: recovery can outpace self-reports

Let R denote next-10 accuracy and F its forecast, both on a 0–100 scale. Weight checkpoints and trajectories equally. The proposed signature requires all four conditions below.

| Test | Proposed practical criterion |
| --- | --- |
| Execution improves after repair | R_C − R_B > 0. |
| Execution returns near the clean reference | R_C − R_A is equivalent within ±5 percentage points. Failure to reject a difference is not equivalence. |
| Forecast remains depressed relative to the reference | F_A − F_C > 10 percentage points. |
| The repaired arm is actually underconfident | R_C − F_C > 10 percentage points. A lower forecast than an overconfident baseline alone is insufficient. |

Use 10,000 paired whole-trajectory bootstrap resamples, retaining all arms and checkpoints together. Apply Holm correction to the four primary tests; equivalence uses the larger of its two one-sided p-values. Report effects, intervals and adjusted decisions. Freeze the precise testing procedure before analysis code. These are proposed thresholds, not a completed registration or power guarantee.

### Interpretation and safeguards

- All four criteria support a behavioral dissociation, without identifying an internal mechanism. Appropriate updating instead requires positive evidence of small forecast error; non-significance alone is insufficient.
- Without improved execution, recovery remains unresolved. If clean or repaired mean accuracy is at or below 5%, the comparison is floor-limited. Report all checkpoints, not only successful repairs.
- Secondary outcomes: calibration, awareness shifts, and verbal/P(Yes)/token-feature AUROC at fixed checkpoints. Separate global correctness from local numeric slips; AUROC is undefined with only one class.

### Scope, deliverables and limitations

The initial scope is one model and one arithmetic task. Supplied errors may differ from naturally generated failures; state cues change available evidence. Confirmation needs disjoint sequences and, if approved, a second model. Audit sample reservations and determine sample size through calibration-based power or precision analysis. Reused A0 sequences are not independent replication.

Deliverables: separate preregistration, tested runner, immutable outputs, paired estimates, calibration/recovery figures and a paper draft reporting all outcomes. Reuse existing prompts, forked probes, injection and archives. Preserve A0–D2 and HF.

### Permission requested from the instructor

Approve the question and design, implementation, and a small calibration pilot with an agreed budget. Freeze registration before the full experiment. Long GPU runs require separate approval after measuring feasibility and cost.

**Decision:** Approve / Revise / Defer     **Instructor/date:** ____________________

**Requested revisions or compute limit:** ______________________________________

<!-- PAGEBREAK -->

# Related research and novelty boundary
## Targeted literature review · checked 16 November 2026

This is a targeted review, not an exhaustive novelty certificate. The defensible candidate contribution is the joint, controlled measurement of future execution accuracy and self-forecasting after state repair while erroneous history remains visible. The following precedents narrow that claim.

### 1. History injection and long execution

Sinha et al., **The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs** (ICLR 2026), especially §3.2/Fig. 5. Manipulated error rates in supplied histories separate self-conditioning from long-context effects. Our M manipulation builds directly on this precedent; the proposed extension adds recovery-sensitive forecasts. [Paper: arXiv 2509.09677](https://arxiv.org/abs/2509.09677).

### 2. Self-forecasts during agent trajectories

Barkan, Black and Sourbut, **Do Large Language Models Know What They Are Capable Of?** (arXiv, 2025). Studies predicted success during multi-step tasks and learning from failure experience. Confidence changing over a trajectory is therefore not a new research question. Our proposed contrast holds the supplied continuation state fixed while varying prior error exposure. [Paper: arXiv 2512.24661](https://arxiv.org/abs/2512.24661).

### 3. Confidence updating and error correction

Kumaran et al., **Competing Biases underlie Overconfidence and Underconfidence in LLMs** (Nature Machine Intelligence, 2026), studies answer-consistency effects and asymmetric responses to advice. This motivates controlling cue wording and evidence. Their **How LLMs Detect and Correct Their Own Errors: The Role of Internal Confidence Signals** (arXiv, 2026) links confidence to detection and correction; verbal confidence can outperform token probabilities in those settings. Our claim must remain task-specific. [Confidence updating](https://www.nature.com/articles/s42256-026-01217-9); [error correction: arXiv 2604.22271](https://arxiv.org/abs/2604.22271).

### 4. Recovery with misleading history retained

Shnaidman, **Reset Is Not Recovery: Evaluating Recoverability from False Conversational Context via Sycophancy Hysteresis** (arXiv, 2026). Measures residual wrong-answer preference after conversational pressure and tests several repairs, including trusted evidence. Persistent effects after repair are not new. Our narrower target is future-accuracy forecasts after numerical state repair, compared with measured execution. [Paper: arXiv 2609.33672](https://arxiv.org/abs/2609.33672).

### 5. Oracle state and execution recovery

Jia et al., **Causal Decomposition of LLM Agent Failure via Oracle-State Intervention** (REALM 2026). Tests recovery by replacing failing agents’ context with ground-truth state. Oracle state intervention is an established method. Our proposed experiment retains erroneous history and asks whether self-reports align with any restored performance. [ACL Anthology paper](https://aclanthology.org/2026.realm-1.6/).

### 6. Uncertainty versus useful intervention

Zhang et al., **Calibration Is Not Control: Intervention Value for LLM-Agent Oversight** (arXiv, 2026), and Li et al., **From Uncertainty to Action: Learning to Steer LLM Agents** (arXiv, 2026), evaluate intervention benefit beyond failure prediction. Our proposal concerns reports after repair, rather than choosing the best intervention point. It should not claim to introduce the connection between confidence and intervention. [arXiv 2606.21399](https://arxiv.org/abs/2606.21399); [arXiv 2610.09115](https://arxiv.org/abs/2610.09115).

**Assessment for approval:** A focused, testable extension with substantial prior-work overlap. Proceed as a candidate contribution, subject to instructor review, fuller related-work checking and empirical confirmation. Neither novelty nor publication acceptance is guaranteed.
