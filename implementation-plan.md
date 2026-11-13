# Implementation plan: Catching Silent Collapse

**Project:** Can a long-horizon agent's working signals be used to stop its own decline?
**Companion docs (in this folder):** `research-proposal.pdf` (problem statement, hypotheses HA–HE), `metacognitive-decay-report.md` and `findings-and-new-claim.md` (instructor's prior work, used **only as cited findings**).
**Group setup:** three members, one model each, same dataset, splits, prompts, arms and metrics:
- Qwen3-8B (teammate)
- Llama (teammate)
- **IBM Granite 4.2 8B (this plan)**

**Status:** Phases 0–1 complete (pilot summary: `results/pilot_summary.md`). Pre-registered (`prereg.md`). Next: main runs on test samples 20–51. Last updated 2026-11-13.

---

## 0. Ground rules

1. **Self-contained.** All code, data, model weights, runs and results live under `/home/viswanath/mithun`. We do **not** read, import, copy or run anything from any other folder (including the instructor's experiment directory). We cite his two markdown files only as prior findings.
2. **One dataset:** Long-Horizon-Execution (LHE), §2. No other datasets.
3. **Honest numbers.** Every number we report comes from our own runs. Numbers quoted from the instructor's files are labelled as his.
4. **Pre-register before main runs.** We fix hypotheses, metrics, thresholds, seeds and sample ids in `prereg.md` and commit them to git before Phase 4.
5. **Never overwrite a run.** Each run writes to a new `runs/<arm>_<timestamp>/` folder with its config.

---

## 1. Folder layout (to be created in Phase 0)

```
mithun/
├── implementation-plan.md      ← this file
├── research-proposal.pdf
├── prereg.md                   ← Phase 3
├── requirements.txt            ← pinned versions
├── data/lhe/                   ← vendored dataset + SHA256SUMS + SOURCE.md
├── models/                     ← Granite 4.2 8B weights (git-ignored)
├── src/
│   ├── lhe_data.py             ← load, verify checksums, split, group K keys/turn
│   ├── prompts.py              ← every prompt string, in one place
│   ├── agent.py                ← the turn loop (vLLM), history handling, logging
│   ├── gates.py                ← A1a / A1b / A2 re-compute gates
│   ├── audit.py                ← A3 / A4 auditors
│   ├── compaction.py           ← D1 / D2 state-table contexts
│   ├── probes.py               ← forked probes: P(Yes), verbal, forecast (HC)
│   ├── labels.py               ← error taxonomy (§4)
│   ├── metrics.py              ← AUROC, J, accuracy, run lengths
│   └── stats.py                ← paired bootstrap, Holm correction
├── scripts/                    ← run_arm.py, pilot.py, analyze.py, make_figs.py
├── configs/                    ← one JSON per arm
├── tests/                      ← unit tests (no GPU)
├── runs/                       ← raw outputs (one folder per run)
└── results/                    ← tables and figures
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

## 7. Phases and timeline (about 3 weeks)

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

## 8. Metrics and pre-registered decision rules

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
