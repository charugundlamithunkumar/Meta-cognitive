# Implementation plan: Catching Silent Collapse

**Project:** When a long-horizon agent becomes pessimistic, does it recognize its errors—and which signals can help stop its decline?
**Working paper title:** *Catching Silent Collapse: Do LLM Agents Recognize Failure and Recovery?* This is a research question; neither the explanation of pessimism nor a recovery dissociation has been established.
**Companion docs (in this folder):** `research-proposal.pdf` (problem statement, hypotheses HA–HE), `metacognitive-decay-report.md` and `findings-and-new-claim.md` (instructor's prior work, used **only as cited findings**).
**Group setup:** three members, one model each, same dataset, splits, prompts, arms and metrics:
- Qwen3-8B (teammate)
- Llama (teammate)
- **IBM Granite 4.2 8B (this plan)**

**Status, updated 2026-11-17:** The user reports that the instructor accepted the [After the Fix proposal](after-the-fix-problem-statement.pdf). The active implementation and experiment roadmap is **[§18](#18-approved-metacognition-and-recovery-program-2026-11-17)**. Study 1 (A0–D2, HA–HE) and exploratory A0 analysis are complete. M/HF is implemented and mock-tested, but real M inference is still pending. The recovery experiment (MR/HG) is accepted for development; its runner, configs, analysis and separate preregistration are **not yet implemented**. HF still needs its registration commit. The existing **43 tests passed again on 2026-11-17**. No GPU inference was launched for this plan update.

**Local resources verified:** `models/granite-4.2-8b/` contains all four shards named by its weight index (17,583,228,032 bytes total), model config and tokenizer files. `data/lhe/test.jsonl` exists and the pinned dataset checksums pass. Reuse these local files; downloading again is unnecessary. Shard presence is not a weight-content checksum or an inference smoke test.

**Execution order:** write HG registration draft → implement and CPU-test MR → approved calibration smoke/pilot → freeze HF/HG and materials → separately approved full M and MR runs → paired analysis → independent confirmation if feasible. M answers the instructor's length-versus-errors question; MR tests recognition of recovery. Developing MR does not depend on obtaining significant M results.

**Reading this plan:** Start with §18 for the accepted work. §§2–12 retain historical Study-1 planning, including obsolete setup/download/cleanup instructions; they are not current action requests. Original registrations and deviations remain authoritative for their original studies. §§15–17 document M and earlier publication recommendations. §18 operationalizes the accepted recovery proposal without amending HA–HF. Preserve the accepted proposal PDF as reviewed; record later protocol refinements in the separate HG registration and change log.

---

## 0. Ground rules

1. **Self-contained.** All code, data, model weights, runs and results live under `/home/viswanath/mithun`. We do **not** read, import, copy or run anything from any other folder (including the instructor's experiment directory). We cite his two markdown files only as prior findings.
2. **One dataset:** Long-Horizon-Execution (LHE), §2. No other datasets.
3. **Honest numbers.** Every number we report comes from our own runs. Numbers quoted from the instructor's files are labelled as his.
4. **Pre-register before main runs.** We fix hypotheses, metrics, thresholds, seeds and sample ids in `prereg.md` and commit them to git before Phase 4.
5. **Never overwrite a run.** Each run writes to a new `runs/<arm>_<timestamp>/` folder with its config.
6. **Separate evidence from plans.** Preserve A0–D2 code, configs, raw archives and registered verdicts. Use new files and arm names for follow-ups. Existing A0 metacognition findings are exploratory; prospective M outcomes are not an independent replication on unseen task sequences, because ids 20–51 have already been examined in A0.
7. **Registration and compute.** Keep HA–HE and HF in separate multiplicity families. Freeze any additional controls, outcomes and stopping rules before their runs. Obtain user approval before GPU inference in this workflow, particularly long runs, or any destructive action. Mock outputs validate software and never support empirical claims.

---

## 1. Current implementation and artifacts

```
mithun/
├── implementation-plan.md      ← this file
├── research-proposal.pdf
├── prereg.md                   ← original HA–HE registration + prospective HF addendum
├── METACOGNITION.md            ← M commands, output schema, resume and caveats
├── requirements.txt            ← pinned versions
├── data/lhe/                   ← vendored dataset + SHA256SUMS + SOURCE.md
├── models/                     ← Granite 4.2 8B weights (git-ignored)
├── src/
│   ├── lhe_data.py             ← load, verify checksums, split, group K keys/turn
│   ├── prompts.py              ← every prompt string, in one place
│   ├── agent.py                ← frozen arm loop, gates, audits, compaction, forked probes
│   ├── meta_history.py         ← M injection masks, existing +1 generator adapter, mock tokenizer
│   ├── meta_runner.py          ← M checkpoint probes, ten-step branches, atomic resume, archives
│   ├── labels.py               ← error taxonomy (§4)
│   ├── metrics.py              ← AUROC, J, accuracy, run lengths
│   └── stats.py                ← paired bootstrap, Holm correction
├── scripts/                    ← original run_arm/main_runs/analyze_main/figures scripts;
│                                  analyze_meta.py, run_meta.py, analyze_meta_experiment.py
├── configs/                    ← original arm configs + seven M_*.json configs
├── tests/                      ← test_core.py, test_arms_mock.py, test_meta.py (no model inference)
├── runs/                       ← raw outputs (one folder per run)
└── results/                    ← original results + meta_analysis.{md,json}, figures/meta_*.png
```

---

## 2. Dataset: Long-Horizon-Execution (the one dataset)

| | |
|---|---|
| Source | Sinha, Arun, Goel, Staab, Geiping, *The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs*, ICLR 2026 ([arXiv 2509.09677](https://arxiv.org/abs/2509.09677)) |
| Hugging Face | `arvindh75/Long-Horizon-Execution` · **pinned revision `7e4901fbb42b88144a65782052d0c8d87a358edc`** |
| Licence | **Apache-2.0** (free use and redistribution, with citation) |
| Task | Running-sum execution: a dictionary maps 100 five-letter words to integers; each turn gives a key; the model adds its value to a running total that starts at 0 and replies with the new total |
| Contents | `dictionary.json` (100 keys, values −99…97, 76 distinct values), `test.jsonl` (100 samples × 50,000 turns; fields `id`, `input`, `values`, `output`), `config.json` |
| Ground truth | Exact running total at every turn |

**Integrity check we ran (2026-11-13):**
- 0 of 100 samples have a value that disagrees with the dictionary.
- 0 of 100 samples have an `output` that differs from the cumulative sum of `values` starting at 0.
- The maximum |total| over the first 300 turns is 3,528, so totals are 1–4-digit numbers.

**SHA-256 checksums (for `data/lhe/SHA256SUMS`):**
```
33d783c31d664d791a6ff08671b86775cac681768d2853987d528c30457c017e  dictionary.json
111ef6e3850368efd84175576c279aeba36ddfb20da3582c01a078bfa77620c6  test.jsonl
0cc5df39a45c71d1044e197f660045de09922d74011efa0c688a658c72aa4656  config.json
```

**Why this dataset will not cause problems later:**
- **Licence:** Apache-2.0, so there are no restrictions on use or redistribution for coursework or a paper.
- **Availability:**
  - We pin the exact revision and vendor a copy in `data/lhe/` with checksums.
  - The loader refuses to run if a checksum differs.
  - If Hugging Face ever removes the dataset, our copy and the paper's description are enough to regenerate it.
- **Contamination:** the data is synthetic (random key sequences). However, the samples were published (Sept 2025) **before** Granite 4.2 (Aug 2026), so we cannot rule out by date alone that they were in its training data. **Check in Phase 1:** run A0 on freshly generated sequences (same dictionary, keys drawn uniformly at random with our own seeds) and compare the per-turn slip rate with the published calibration samples. A clearly lower slip rate on the published samples would indicate memorization. In that case we switch all runs to fresh sequences (`data/lhe/` stays the source of the dictionary).
- **Ground truth:** exact and machine-checkable. No human labels, no annotator disagreement.
- **Size:** 50,000 turns per sample is far more than we need, so we never run out of horizon.
- **Peer reviewed and directly relevant:** its paper is the source of the "self-conditioning" finding that the instructor's report builds on.

**Splits (fixed now, never changed):**

| Split | Sample ids | Use |
|---|---|---|
| Calibration | 0–19 | Pilot: choose K, T, τ, m. Never used for hypothesis tests |
| Test | 20–51 | All main runs (32 trajectories per arm, the same ids in every arm, so comparisons are paired) |
| Reserve | 52–99 | Untouched; replication or extra power if needed |

We use only the first T turns of each sample (T is chosen in the pilot; planned T = 300). Turn complexity K is also chosen in the pilot. K > 1 groups K consecutive keys per turn, and the target is the last running total of the group, as the dataset card specifies.

---

## 3. Model and environment

| Item | Choice |
|---|---|
| Model | **IBM Granite 4.2 8B** (`ibm-granite/granite-4.2-8b`; Apache-2.0; no download approval needed; 8.79 B parameters; dense; released Aug 2026), pinned revision `a7042144125ce5ae64f89610e8289647fc000b9a`, bf16 |
| Why this model | A different family from the group's other two models (Qwen3-8B, Llama) at the same size class (about 8B), so differences between members come from the family, not size. Thinking switches on/off **in the same model**, so A3/A4 audits are still the model checking itself. Small models are a poor fit: the instructor's files show 4B is at the floor. |
| Mode | Non-thinking for the agent (`enable_thinking=False`). Thinking only inside A3/A4 audits (`enable_thinking=True`) |
| Decoding | Agent: greedy (temperature 0), maximum reply 16 tokens, top-20 logprobs saved. Thinking audits: the model card's recommended sampling (temperature 1.0, top-p 0.95), seeded per (trajectory, turn) |
| Chat format | Derived automatically from the model's official chat template (`src/agent.py: Chat`); a unit test checks it matches `apply_chat_template` exactly in both modes |
| Tokenization note | Granite splits numbers into multi-digit chunks (e.g. `-1523` → `-`, `15`, `23`), unlike Qwen's one digit per token. Logit statistics are computed over the tokens of the reported integer, whatever the chunking |
| Inference | vLLM 0.24.0 with prefix caching; torch 2.11.0; transformers 5.12.1 (already installed on this machine; versions pinned in `requirements.txt`). `GraniteForCausalLM` is supported by this vLLM |
| Weights location | `mithun/models/granite-4.2-8b` (about 17.6 GB) |
| GPU | **GPU 0** (RTX 6000 Ada, 48 GB). Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` because the GPUs on this machine are of different models. |

---

## 4. Error taxonomy (implemented in `src/labels.py`)

For every turn t:
- `prev` = the total the agent should build on: its last committed total, or the latest correction it was given (A3/A4).
- `v` = the true value of the key(s).
- `truth` = the true running total.

| Label | Rule |
|---|---|
| `ok` | The committed total equals `truth` |
| `nonanswer` | The reply is not a well-formed integer |
| `inherited` | Well-formed, equals `prev + v`, but `prev` was already wrong (so ≠ `truth`) |
| `numeric_slip` | Well-formed and ≠ `prev + v`. This is a fresh mistake made at this step |
| ↳ `retrieval_like` *(exploratory)* | A numeric slip that equals `prev + v′` for some other dictionary value v′ ≠ v |

Caveat for `retrieval_like`: 76 distinct dictionary values cover much of −99…99, so some arithmetic slips will match by chance. We report the chance rate alongside it and **do not** use it for any primary test.

Every `inherited` error has a **seed**: the most recent `numeric_slip` or `nonanswer` that started the chain of wrong totals. `labels.py` records the seed turn for each inherited error, which is what HA's upper-bound analysis needs.

**Key property of this dataset:** there is one running total, so after a seed error *every* later turn is wrong compared with the truth until the error is repaired. Accuracy over the last 50 turns would therefore sit at the floor in every arm. **The primary end-to-end metric is accuracy over all T turns** (see §8). This is a deliberate change from the "last-50" wording in the proposal PDF.

---

## 5. Prompts (in `src/prompts.py`; final wording frozen in `prereg.md`)

- **Agent system prompt:** "You are an execution agent. Keep a running total that starts at 0. Dictionary (key: value): {100 lines}. Each turn I send one key. Look up its value, add it to the running total, and reply with ONLY the new running total as a single integer. Example: if the total is 120 and the key's value is −45, reply: 75."
- **Agent turn:** `Key: {w}` (or `Keys: w1, w2, …` for K > 1).
- **In-line format (HE only):** the reply must be `<total> | <confidence 0–100>`, with **no numeric confidence example** in the prompt. The instructor's E8 shows that models copy example values.
- **Forked probes**, asked in a side copy of the context so the main history is unchanged:
  - P(Yes): "Is the running total you just reported exactly correct? Answer Yes or No."
  - Verbal: "How confident are you, from 0 to 100, …? Reply with only a number."
  - Forecast: "Over your next 10 turns, what percentage of your running totals do you expect to be exactly correct?"
  - Awareness: "Compared with the start, is your accuracy now A) higher, B) about the same, C) lower?"
- **Re-compute (A1a/A1b/A2), in a fresh context:** dictionary + "Current total: {prev}. Key: {w}. Reply with ONLY the new total."
- **Audit (A3 fresh / A4 in-context), thinking on:** "Verified total after turn {t0}: {V}. Keys since then: {w…}. The reported total after turn {t} is {R}. Recompute from {V}. End with `VERDICT: CORRECT` or `VERDICT: WRONG, total = N`." A3 asks this in a fresh context with the dictionary; A4 appends the identical text to the full history.
- **Correction message (A3/A4):** prepended to the next turn: "Note: an independent check found the running total after turn {t} should be {N}. Continue from that total. Key: {w}."
- **HC evidence line**, prepended to the forecast probe:
  - *true:* "An independent checker recomputed your last 10 totals and found {N} wrong."
  - *placebo:* the same sentence with N taken from a different trajectory at the same checkpoint.
  - *none:* the line is omitted.

---

## 6. Arms

All arms use the test ids 20–51, the same K and T, and the same prompts except for what the arm changes.

| Arm | What the code does | Hypothesis |
|---|---|---|
| **A0** | Baseline loop. Logs logits, P(Yes), verbal confidence, and HC probes at checkpoints {10, 25, 50, 100, 150, 200, 250} | Reference, HC |
| **A0-inline** | A0 with `total \| confidence` replies; saves **per-token** logprobs | HE |
| **A1a** | If the reply is malformed → re-compute; the re-computed total is committed in place of the reply | HA control |
| **A1b** | Malformed **or** `min answer-token logprob < τ` → re-compute | **HA** |
| **A2** | Malformed **or** a random trigger with probability = A1b's realized trigger rate (so A2 runs after A1b) | HA compute control |
| **A3** | Every m turns: fresh-context audit of the last m keys from the last verified total V. If the verdict is WRONG → correction message, and V updates to the auditor's total | **HB** |
| **A4** | Same audit and correction, but asked inside the full history | HB contrast |
| **D1** | No history: each turn's context is the system prompt + "Your current total: {agent's own last total}" + the key | **HD** |
| **D2** | Same as D1 with the **true** total (oracle control) | HD control |

HC uses the probes in A0. A3 adds a second evidence source (auditor flags). HE uses A0-inline; the bare-number logits come from A0.

**What we log per turn** (one JSON row):
- Sample id, turn, key(s), true value, truth, prev.
- Raw reply, parsed reply, well-formed flag, committed total, label, seed turn.
- Logit statistics: min/mean logprob, max entropy, min margin, and per-token logprobs.
- Probe outputs.
- Gate fired / reason / re-compute reply.
- Audit verdict / auditor total / thinking tokens / unfinished flag.
- Tokens used.

---

## 7. Original Study-1 phases and timeline (historical)

### Phase 0: Setup (days 1–2)
1. Run `git init` and add `.gitignore` (models/, runs/ raw).
2. Vendor the LHE files into `data/lhe/` at the pinned revision; write `SHA256SUMS` and `SOURCE.md` (citation, licence, revision).
3. Download Granite 4.2 8B at the pinned revision into `models/granite-4.2-8b` *(approved)*.
4. Write `lhe_data.py`, `labels.py`, `metrics.py` and `stats.py` with **unit tests that need no GPU**:
   - The loader rejects a corrupted file.
   - The labeller gets hand-built cases right (a planted slip followed by inherited errors, a non-answer, a correction mid-chain).
   - AUROC, J and bootstrap match `sklearn` / known values.
5. Write `agent.py` (A0) and run a smoke test on 2 trajectories × 20 turns.
6. **Prompt check (calibration ids only):** confirm the model replies with the running total, not the key's value. This was added after the Phase 0 smoke test exposed the problem (§13). Freeze the chosen prompt version.

**Gate G0:** the tests pass, and the smoke run produces parsed replies and logged logits, with the model doing the actual task (replying with the running total).

### Phase 1: Pilot on calibration ids 0–19 (days 3–4)
1. **Choose K and T.** Run A0 at K ∈ {1, 2, 5}, T = 300. Choose the smallest K for which the calibration set has **at least 15 numeric slips in each position bin** (needed for a stable AUROC) and the model is not at 100% local accuracy. If K=1 already qualifies, use K=1.
2. **Choose τ.** On the calibration A0 run, τ = the `min logprob` threshold that maximizes J (TPR − FPR) for numeric slips vs. ok. Record the expected trigger rate.
3. **Choose m.** Use m = 5 (chain length 5; the instructor's E4b gives J = 0.86 at k=5, N=30 on his task). Confirm on calibration that A3 audits finish within the thinking budget (2,048 tokens). If more than 10% are unfinished, raise the budget.
4. **Contamination check (§2):** run A0 on 20 freshly generated sequences (same dictionary, uniformly random keys, our own seeds) and compare the per-turn slip rate with calibration ids 0–19.
5. **HA upper bound, offline on pilot rows:**
   - What share of inherited errors trace back to a seed that A1a would catch? What share would A1b catch?
   - If A1b adds almost nothing over A1a, we record that HA is expected to fail on the "logits add value" criterion, and still run it (a null result is informative).

**Gate G1:** K, T, τ and m are fixed, and enough slips exist. If Granite 4.2 8B makes almost no slips even at K=5, raise T to 600 before changing anything else.

### Phase 2: Build the intervention arms (days 5–7)
- Write `gates.py`, `audit.py`, `compaction.py` and `probes.py`.
- **Forced tests:**
  - A1: inject a malformed reply → the gate fires and the committed total is correct.
  - A3: inject a wrong total → the auditor gets the planted error and the correction message appears in the next turn.
  - D1/D2: the state line shows the expected total.
- Run each arm as a smoke test on calibration ids only.

### Phase 3: Pre-register (day 8)
- Write `prereg.md`: hypotheses, primary and secondary metrics, every threshold below, K/T/τ/m, test ids, exclusion rules, and the analysis script, already tested on pilot data.
- **Commit to git before any test-id run.**

### Phase 4: Main runs on test ids 20–51 (days 9–11)
- Order: A0 → A0-inline → A1a → A1b → A2 (needs A1b's rate) → A3 → A4 → D1 → D2.
- **Quality checks after each run:**
  - 32 trajectories × T turns present.
  - Share of non-answers.
  - Gate trigger rate.
  - Share of unfinished audits.
  - Spot-check 20 random rows by hand.
- If a run crashes, rerun the **whole** arm into a new folder; never stitch partial runs together.

### Phase 5: Analysis (days 12–15)
- Write `analyze.py` and `make_figs.py`, using the pre-registered tests only. Anything else is labelled *exploratory*.
- Figures:
  1. Accuracy vs. turn for each arm.
  2. Share of each error type vs. turn for each arm.
  3. Auditor J by position bin, A3 vs. A4.
  4. Forecast vs. realized accuracy for each HC evidence condition.
  5. Logit AUROC by format and scoring rule (HE).

### Phase 6: Write-up and instructor review (days 16–18, plus buffer)
- A short report: results per hypothesis (supported / falsified / inconclusive), limitations, and the next step.

---

## 8. Original Study-1 metrics and decision-rule sketch

This section records the initial sketch. The operational HA–HE rules and reported verdicts use `prereg.md` §§3–4, including the changes recorded in §13. The HF family is separate and summarized in §15 below.

**Unit of analysis:** the trajectory.
- Paired bootstrap over the 32 test ids: 10,000 resamples, 95% CI.
- Holm correction across the five primary tests.
- Position bins: turns 0–49, 50–99, 100–199, 200–299 (rescaled if T ≠ 300).

**Primary metric per hypothesis**

| | Primary metric | Supported if | Falsified if |
|---|---|---|---|
| **HA** | Accuracy vs. truth over all T turns | A1b − A2 > 0 **and** A1b − A1a > 0 (paired CI excludes 0), and the **number of error chains (seed slips) per trajectory** is lower in A1b than in A2 and A1a | Either difference's CI includes 0, or the live gate's numeric-slip AUROC is near 0.5 |
| **HB** | Auditor J per position bin; accuracy over all T | A3's J in the last bin is within the CI of its first bin, A4's J declines (last-bin CI below first-bin CI), and A3 > A0 on accuracy | A3's J also declines, or A3 − A0 has a CI that includes 0 |
| **HC** | Mean absolute forecast error and the forecast–realized correlation, at checkpoints where true ≠ placebo count | With true evidence: lower error and positive correlation vs. both *none* and *placebo* | Forecast stays ≈ 100 with true evidence, or true ≈ placebo |
| **HD** | Accuracy over all T; slip rate before vs. after the first error | D1 within ±0.05 of A0 on accuracy (pre-registered margin; equivalence test), and D2's accuracy slope over turns ≈ 0 | D1 − A0 > 0.05 (CI excludes the margin): length or self-conditioning then matter for the collapse |
| **HE** | Logit AUROC for numeric slips vs. ok: A0 (bare), A0-inline all tokens before `\|`, A0-inline final-integer tokens only | *Conditional:* if A0-inline (all tokens) is lower than A0, final-integer scoring brings it back inside A0's CI | Final-integer AUROC stays below A0's CI. If A0-inline is **not** lower than A0, HE is recorded as "discrepancy not reproduced on this task" |

**Secondary metrics:**
- Turns until the first error.
- Mean length of a run of wrong totals (turns until repair).
- Gate recall and false-trigger rate.
- Auditor TPR/FPR.
- Extra tokens per turn (cost).
- AUROC of P(Yes) and verbal confidence (does the instructor's "verbal confidence is uninformative" finding hold on a second task?).

**HD decomposition (secondary, on this dataset):**
- A0 vs. D1 slip rate **before the first error** isolates pure context length (no errors in context yet).
- **After the first error**, the difference also includes self-conditioning (Sinha et al.).
- The carried wrong total isolates inheritance.

---

## 9. Compute budget

- Plain loops are short: 32 trajectories × 300 turns, with prompts of a few thousand tokens and prefix caching.
- The costly parts are the thinking audits (A3/A4) and forked probes. We measure wall time in the pilot and record it in `prereg.md`.
- If one arm takes more than 2 hours, we drop A0's per-turn verbal probe first (it is the least important signal).

---

## 10. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Granite 4.2 8B almost never slips at K=1 | Medium | Pilot grid over K, then T (Gate G1) |
| Granite misreads the task (replies with the key's value) | Seen with the first prompt on Qwen3-8B | Prompt check on calibration ids (Phase 0 step 6) |
| Published samples memorized (Granite released after the dataset) | Low | Fresh-sequence check in Phase 1 (§2) |
| Granite is primarily a reasoning model; its non-thinking answers may be weaker | Medium | That is a property of the model to report, not a flaw. The pilot shows whether the slip rate is in a usable range |
| Logit signal weak on this task | Medium | It's a valid finding (HA falsified on that criterion). The A1a/A2 controls keep the comparison fair |
| Re-compute repeats the slip | Medium | Fresh context; log re-compute accuracy |
| Auditor wrongly corrects a right total, which starts a new chain of wrong totals | Medium | Report auditor FPR. V only updates from audits, so the damage is measurable |
| Thinking audits run out of budget | Medium | Pilot check; raise the budget; report the unfinished share and exclude nothing silently |
| HC: the model copies the number given | Medium | Give a count, not an accuracy; placebo condition |
| Floor effect: one chain means everything after a slip is wrong | Certain | Primary metric is accuracy over all T turns, not the last 50 |
| `retrieval_like` labels are ambiguous | Certain | Exploratory only; report the chance rate |
| GPU shared with other users | High | Confirm GPU (§12); keep memory utilisation ≤ 0.6; runs restart cleanly |
| Disk nearly full (`/home` 96% used) | Medium | Weights about 16 GB; raw rows compressed; delete weights at the end |
| Results differ across vLLM batch sizes | Low | Fixed batch size per arm; same ids in every arm |
| Scooping (fast-moving field) | Medium | Pre-register early; ask the instructor about posting a short write-up |

---

## 11. Deliverables

1. `prereg.md` (git-committed before main runs).
2. Code in `src/` and `scripts/` with passing tests; one command reproduces each arm from its config.
3. `runs/` raw rows for all arms; `results/` tables and five figures.
4. Short report: per-hypothesis verdicts with CIs, limitations, and the comparison with the instructor's ledger findings (a second task family).

---

## 12. Decisions needed before Phase 0

1. **Approve downloading** the model into `mithun/models/`.
2. **Which GPU** to use on this shared machine (at the time of writing, GPU 0, an RTX 6000 Ada with 48 GB, was almost idle; GPU 2 was busy).
3. **Correction style:** A1 replaces the reply before it enters the history, and A3/A4 add a correction note to the next turn (the default above). Keep this, or use one style everywhere?
4. **Scope:** all nine arms, or the core (A0, A1a, A1b, A2, A3, A4) plus A0-inline, with D1/D2 only if time allows?
5. Confirm with the instructor that the switch from "last-50 accuracy" to "accuracy over all T turns" (forced by the single running total in this dataset) is acceptable.

**Resolved 2026-11-13:**
- GPU 0 (RTX 6000 Ada).
- Model: **Granite 4.2 8B**, download approved. The instructor assigned one model per group member; Qwen3-8B and Llama are taken by teammates.
- Correction style and scope: defaults above (A1 replaces the reply; A3/A4 add a correction note to the next turn; all arms, with D1/D2 last).

---

## 13. Deviations log (changes after this plan was written, with reasons)

| Date | Change | Reason |
|---|---|---|
| 2026-11-13 | The logit AUROC and τ for **numeric slips** compare them with **locally correct** steps (ok + inherited), not with ok steps only | With one running total, almost no turn after the first slip is ok compared with the truth, so "slip vs. ok" would have no ok items in later bins. A logit signal can only judge local correctness anyway. Inherited vs. ok is reported only where both exist (early bins). |
| 2026-11-13 | **Model changed from Qwen3-8B to IBM Granite 4.2 8B** | The instructor assigned each of the 3 group members a different model; Qwen3-8B and Llama are taken. Granite 4.2 8B: Apache-2.0, no download approval needed, same size class, thinking switch in the same model. |
| 2026-11-13 | Thinking audits (A3/A4) use the model card's recommended sampling (Granite 4.2: temperature 1.0, top-p 0.95), with a fixed seed per (trajectory, turn) | The model card requires these settings for thinking mode. The fixed seeds keep runs reproducible. |
| 2026-11-13 | The agent's own answers stay **greedy** (temperature 0), even though the Granite model card recommends temperature 1.0 for all modes | Greedy answers are deterministic and standard for evaluation, and keep the three group members comparable. Logprobs are recorded either way. |
| 2026-11-13 | The chat format is **derived from each model's official chat template** (`Chat` in `src/agent.py`) instead of a hard-coded format | Works for any model with an `enable_thinking` switch. For Granite, past assistant turns are `<think></think>answer` and the non-thinking prompt ends with `<think></think>`. A unit test checks that our tokens equal `apply_chat_template` in both modes. |
| 2026-11-13 | **Agent prompt revised** (prompt v2/v3 with a worked example on a separate toy dictionary) | In the Phase 0 smoke test (on Qwen3-8B, before the model switch), prompt v1 made the model reply with the key's value instead of the running total. The version is chosen on calibration ids only and frozen in `prereg.md`. |
| 2026-11-13 | Added a **contamination check** with freshly generated sequences (§2, Phase 1 step 4) | Granite 4.2 was released after the dataset, so training-data contamination cannot be ruled out by date alone. |
| 2026-11-13 | **Prompt v2 frozen** for the bare-number arms (all configs: `prompt_version: 2`) | Prompt check on Granite (calibration ids 0–3, 40 turns): v1 → the model writes its working and is cut off (158/160 non-answers); v3 → degenerates into repeating the start of the history (152/160 non-answers); v2 → answers with the number first. |
| 2026-11-13 | **Replies end at the first line break** (`stop=["\n"]` for answers, re-computes and numeric probes); logit statistics exclude the line-break token | With v2, Granite writes the number on line 1 and then adds "Let me verify…", which made correct answers count as malformed and polluted the history. With the stop: on ids 0–3, 100 turns, replies are clean bare numbers (31 non-answers out of 400, all the string `-?`). |
| 2026-11-13 | Inline prompt (A0_inline) v2: the format is **described in prose** ("the new running total, then a space, a vertical bar '\|', a space, and your confidence…"), with no placeholders and no numeric confidence example | Placeholders were copied literally: `<88> \| 100` with angle brackets; `? \| ?` with letters N/C. The `scripts/diag_inline.py` check of first-token probabilities on calibration samples 0–7 showed `?`/`N` as top tokens under the letter wording, and 8/8 well-formed replies under the prose wording. |
| 2026-11-13 | Added Granite's **low-effort thinking** as an audit option (`audit_low_effort`), implemented exactly as the official template does (a unit test checks it) | Gives a cheaper audit mode to compare against full thinking in the Phase 1 budget check. |
| 2026-11-13 | `generate_fresh()` builds fresh sequences: same dictionary, i.i.d. uniform keys | Needed for the contamination check. The published calibration samples match this distribution: all 100 keys at 0.73–1.33% frequency; consecutive-repeat rate 0.75%. |
| 2026-11-13 | **Constrained decoding** (`constrain: true`): agent replies are forced by regex to be an integer (`-?[0-9]{1,6}`; inline `N \| C`; numeric probes `[0-9]{1,3}`; re-computes likewise). Logprobs stay **raw** (vLLM default `raw_logprobs`, i.e. before the constraint); a number-renormalized `lp_fin_renorm_min` is logged alongside. The first unconstrained pilot (20 × 250 turns) was stopped and is kept only as a record | In the unconstrained pilot, **58% of replies were non-answers**, growing with position: Granite in non-thinking mode drifts into prose ("We need to compute the running total…"), the first line is cut, that line enters the history and is copied. That measured format failure rather than the slips the hypotheses are about. Check with the constraint (calibration ids 0–7, 200 turns): 0 non-answers; logit AUROC on numeric slips 0.88–0.91 per bin (raw = renormalized = 0.903); HA upper bound 0.78. This follows the instructor's E8 precedent of constrained output; **group members should use the same constraint** so the three models stay comparable. |
| 2026-11-13 | **τ fixed at −0.138** (Youden J = 0.668 on numeric slips vs. locally correct steps) and **K = 1** confirmed, from pilot job 1 (A0, calibration ids 0–19 × 300 turns, constrained) | K = 1 already gives ≥ 15 numeric slips in every position bin (235 / 385 / 866 / 1,002) and is far from 100% local accuracy. |
| 2026-11-13 | **Pilot trimmed** (user decision, GPU 0 shared and slow): treatment A1b on the same 20 calibration trajectories × 100 turns (paired with job 1's first 100 turns); contamination check 20 fresh × 100 turns with probes off; inline check 4 × 30; audit checks 4 × 40 per mode (`scripts/pilot_trimmed.py`) | Keeps a baseline-vs-treatment comparison with enough trajectories while cutting GPU time from many hours to about 1–2. |
| 2026-11-13 | **Per-trajectory saving and resume** in `agent.run`: rows flushed every turn; one line per finished trajectory in `trajectories.jsonl`; `resume=True` skips finished trajectories and discards rows of unfinished ones; trajectories run in groups of `group_size` | User requirement: results must be saved after every trajectory, not only at the end. A unit test simulates a crash mid-trajectory and checks that the resumed output equals an uninterrupted run. |
| 2026-11-13 | **Gate re-compute = prompt v2 + Granite low-effort thinking** (`recompute_think`, budget 1,024, model-card sampling, seeded) for A1a/A1b/A2 | Pilot: with the v1 non-thinking re-compute, A1b was *worse* than the baseline (re-computes 30% correct; they often echoed the starting total). Isolation check on 400 fired cases: v1 non-thinking 23.7%, v2 non-thinking 5.0%, **v2 + low-effort thinking 98.8%** (about 20 tokens). A1b re-run (20 × 100, calibration): accuracy 0.042 → 0.164, turns to first error 3.8 → 16.5. A2 uses the identical re-compute, so the compute control stays fair. |
| 2026-11-13 | **HA criterion changed** from "the inherited share drops" to "**error chains (seed slips) per trajectory** fall" (accuracy over all turns stays primary) | With one running total, every slip the gate misses still corrupts all later turns, so the inherited *share* of the remaining errors rises mechanically even when the gate works (pilot: rate 0.649 → 0.775 while chains fell 31 → 6 per trajectory). Chains per trajectory measures what "catching the seed" predicts. Decided on calibration data, before pre-registration. |
| 2026-11-13 | **A3/A4 audits: full thinking, budget 8,192, m = 5** | Pilot (4 × 40, 32 audits): full thinking J = 1.00 and 20/20 corrections exact (9.4% unfinished); low-effort J = 0.91 but only 17/22 corrections exact (wrong corrections start new error chains). HB should test auditing outside the history, not a sloppy auditor. Cost: about 42× the GPU time of low-effort. |
| 2026-11-13 | **Pre-registration** (`prereg.md`): run order changed to A0 → A0_inline → A1a → A1b → A2 → D1 → D2 → A3 → A4 (cheap arms first); A1a is expected to equal A0, because forced-integer replies are never malformed, so its gate cannot fire (it is kept as a reproducibility check, and HA's control comparison is A1b vs. A0 and A2); HB/HC/HD/HE rules made fully operational (bin Δ for J, MAE + Spearman for HC, TOST ±0.05 and a slope band ±0.0005/turn for HD, the conditional for HE) | So that partial results exist if GPU time runs out, and so that every verdict is mechanical. Real-model smoke tests of A1a, A2, A4, D1, D2 with the final configs all ran (A4: 6/6 audits finished). |
| 2026-11-13 | Phase 2 forced tests run **on CPU with a scripted fake model** (`tests/test_arms_mock.py`, 9 tests) while GPU 0 was busy; a short GPU smoke run of each arm remains | The fake reads the real prompts (real tokenizer) and answers by rule, so gate firing, commits, labels, audit corrections, D1/D2 state lines and HC evidence can be checked exactly. All pass. |
| 2026-11-13 | Repository: the instructor's two files and the 102 MB `test.jsonl` are excluded from git (`.gitignore`); `scripts/fetch_data.py` fetches the dataset at the pinned revision and verifies the checksums | The target GitHub repo is public and the instructor's material is unpublished; GitHub rejects files over 100 MB. |
| 2026-11-13 | GPU memory utilization lowered from 0.7 to 0.6 | Another user started a 13 GB job on GPU 0; 0.6 × 48 GB leaves room for it. |
| 2026-11-13 | Thinking-audit budget of 2,048 tokens is **too small for Granite** (smoke test: 5/6 audits unfinished in A3, 4/6 in A4) | To be fixed in Phase 1 step 3 (raise the budget toward the model card's 8,192, or use Granite's low-effort thinking mode). |
| 2026-11-13 | **Post-registration:** `scripts/analyze_main.py` committed (`4a605de`) 10 minutes after the main runs started, before any test output existed; it is not in the prereg checksum list | It implements prereg §3–§4 as written. It was written after the freeze, so it is recorded here. |
| 2026-11-14 | **Post-registration (reporting):** HD's Holm p is the TOST p of its D1 equivalence part. The D2 slope part has a CI but no p-value. The verdict uses both parts | The prereg defines one p per hypothesis, but the slope band is a CI rule. This choice does not change any verdict: HD is "not supported" on the slope part either way. |
| 2026-11-14 | **Post-registration (exploratory):** D2 accuracy split by the size of the true running total (`results/figures/fig7_d2_by_magnitude.png`) | D2 failed the slope band. Inspection showed that the running total grows with position (mean \|total\| about 230 → 985), and accuracy depends strongly on the digit count (3-digit about 0.66, 4-digit about 0.38) and only weakly on position within a size class. Post hoc, so reported only as exploratory. The verdict is unchanged. |
| 2026-11-14 | **Post-registration (note):** HB's A4 ΔJ CI upper bound is exactly 0.000 (percentile bootstrap), so "entirely below 0" is not met, and HB is "partial" as the rule states | Recorded so the borderline is visible. Late bins have few correct audited totals (near-floor accuracy), so the FPR side of J is thin there. |
| 2026-11-14 | **Post-registration (note):** the §5 contrast "slip rate before vs. after the first error" is degenerate: before the first error the slip rate is 0 by definition. We report the after-first-error rate per arm and the across-arm slip rates (A0 vs. D1 vs. D2) instead | Definitional, found when computing it. Not used for any verdict. |
| 2026-11-14 | Figures use trajectory-bootstrap CIs with 2,000 resamples; verdict CIs use the registered 10,000. Raw main-run outputs are published as compressed archives in `results/raw_main_20261113-173033/` | Figures are illustrative; the verdicts come from `main_results.json`. The archives make every number re-computable without the GPU. |
| 2026-11-16 | **Post-Study-1 exploratory analysis:** added `scripts/analyze_meta.py`, report/JSON and three new figures; corrected the no-evidence forecast wording in `results/main_results.md` | Analyze all 224 A0 checkpoints, with exact-position history comparisons, trajectory-bootstrap CIs and the existing local-slip AUROC definition. Do not recast these inspected outcomes as a prospective test. |
| 2026-11-16 | **Prospective HF materials:** added `prereg.md` §7 before analysis implementation; seven M arms with controlled histories, existing +1 generator, independent probes and ten-step continuations; added CPU/mock and resume tests | Addresses the instructor's metacognition question. Implementation is complete, registration commit and GPU validation/results are pending. Original registered source/config hashes remain unchanged. |
| 2026-11-16 | **Documentation/publication planning:** revised current status and architecture; added §§15–17, a literature comparison, claim boundaries and proposed robustness controls | These recommendations do not alter HF's existing primary tests or authorize a model run. New supplemental designs require separate prospective registration. |

---

## 14. Study 2: confirmatory follow-up for publication (added 2026-11-14, after Study 1)

**Status clarification, 2026-11-16:** this earlier intervention-extension proposal is preserved. Its referenced commit `4025761` exists, but `prereg2.md` and `scripts/study2_runs.py` are not present in the current checkout. Treat its implementation/readiness statements as historical notes requiring reconciliation, not as commands ready to run here. M (§15) is the immediate metacognition follow-up. The reserve ids and fresh seeds proposed below must not also be called an untouched replication set for M without an allocation and usage check.

**Why.** Study 1 left three questions a reviewer would ask:
1. Would simply letting the agent think on every turn do as well?
2. Can the two working fixes be combined to approach the oracle ceiling (D2 = 0.60)?
3. Is D2's drift caused by position or by the growing size of the total?

Study 2 answers them with a second pre-registration (`prereg2.md`) on data never used before.

**Data.**
- Reserve samples **52–99** (48 trajectories, untouched until now) for the agent arms.
- **Fresh sequences** (seeds 1000–1047, same dictionary) for the D2 controls. These are unbounded (`D2u`), or with every key drawn uniformly from those that keep |total| ≤ 300 (`D2b`).
- T = 300, K = 1, the same model, prompts and decoding as Study 1.

| Arm | Definition | Answers |
|---|---|---|
| R_A0, R_A1b, R_A3 | Study 1's A0 (probes off), A1b and A3, unchanged | Replication on new data; paired baselines |
| A1all | A1b's re-compute fired on **every** turn | Gate upper bound; compute outside the history on every step |
| T1 | The agent itself answers each turn with Granite **low-effort thinking**, with the full history (budget set by the pilot) | The "just let it think" baseline |
| A5 | A1b's logit gate **+** A3's fresh audit (m = 5) | Combination of the two working fixes |
| D2u / D2b | D2 (true total, no history) on fresh unbounded / bounded sequences | Position vs. number size |

**Planned primary tests** (paired bootstrap over trajectories, 10,000 resamples, 95% CIs, Holm over S1–S4; final wording in `prereg2.md`):
- **S1 (replication of HA):** accuracy(R_A1b) − accuracy(R_A0) > 0.
- **S2 (combination):** accuracy(A5) − accuracy(R_A3) > 0 **and** accuracy(A5) − accuracy(R_A1b) > 0.
- **S3 (compute outside the history):** accuracy(A1all) − accuracy(T1) > 0. The prediction parallels HB: the same thinking is more reliable in a fresh context than inside the history.
- **S4 (position vs. size):** D2b's mean per-trajectory slope has a 95% CI within ±0.0005 per turn, **and** D2u's slope CI lies below 0.

**Secondary (descriptive):** accuracy per generated token (gate vs. always vs. T1 vs. audit); gate recall; T1's unfinished-thinking rate; slip rate by bin; magnitude regression as in `scripts/magnitude_analysis.py`.

**Steps:**
1. Code and mock tests: done (commit 4025761). The 9 Study-1 arms give byte-identical prompts and rows under the registered code and the new code.
2. Pilot on calibration ids 0–3 × 40 turns for T1, A1all and A5 (fix T1's thinking budget; check that A5 runs).
3. Commit `prereg2.md` with checksums.
4. Runs on GPU 0 (`scripts/study2_runs.py`, groups of 16, cheap arms first).
5. Analysis script, figures, and an update of the report.

The deviations for Study 2 are logged in §13 as usual.

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
`tests/test_arms_mock.py:FakeLLM.respond`, called from the true preceding state.
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
