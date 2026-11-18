# Catching Silent Collapse

**Can a long-horizon agent's working signals be used to stop its own decline?**

Long-horizon LLM agents often fail silently. One early mistake is carried forward (an *inherited* error), and the agent's own confidence reports do not notice it. This project tests whether signals that *do* work can be used **inside the agent loop** to stop the collapse:

- the model's own token probabilities (logits), to catch a mistake at the step where it happens;
- self-audits run in a fresh, short context, to find errors that have already been carried forward.

It also asks what this reveals about why the collapse happens and why agents don't notice it.

This is a student group project. Each of the three members runs the same study on a different model: Qwen3-8B, Llama, and **IBM Granite 4.2 8B (this repository)**. It builds on our instructor's unpublished analysis of metacognitive decay in long-horizon agents, which is not included here.

## Hypotheses

| | Hypothesis |
|---|---|
| **HA** | *Catch the seed.* Re-computing a step when its answer-token probability is low raises end-to-end accuracy more than a random gate firing on the same share of steps, and reduces inherited errors. |
| **HB** | *Audit outside the history.* An audit run in a fresh context stays accurate at every position, while the same audit asked inside the full history degrades. |
| **HC** | *Awareness is input-starved.* Giving the agent a checker's error count makes its self-forecast track its real accuracy; a placebo count does not. |
| **HD** | *Collapse is inheritance, not context length.* Replacing the history with a state line holding the agent's own total does not prevent the collapse; the true total does. |
| **HE** | *Logit format effect.* Scoring only the final-integer tokens removes any loss of logit signal caused by a `total \| confidence` reply format. |

## Results (Granite 4.2 8B, 32 test trajectories × 300 turns, pre-registered)

| | Verdict | Key number |
|---|---|---|
| HA | **Supported** | Logit-gated re-compute beats the same re-compute fired at random: accuracy +0.045 [+0.024, +0.068]; first error at turn 20 instead of 4–6 |
| HB | **Partial** | The fresh-context audit raises accuracy 0.014 → 0.130 and stays accurate late in the run. The in-history audit did *not* degrade for Granite |
| HC | **Supported** | The checker's true error count cuts forecast error 0.71 → 0.10; a placebo count gives 0.35 |
| HD | **Not supported** | Removing the history (own total) matches the baseline's accuracy, but even the true-total control still drifts downward (number size explains part of this drift, exploratory; Study 2 tests it directly) |
| HE | Not reproduced | The inline format did not lower logit AUROC on this task |

The final report (PDF, for instructor review) is [`final-report.pdf`](final-report.pdf); the full write-up, with figures and limitations, is in [`results/main_results.md`](results/main_results.md). Machine-readable verdicts are in `results/main_results.json`.

The pass/fail criteria, statistics (paired bootstrap over trajectories, Holm correction) and every design decision are in [`implementation-plan.md`](implementation-plan.md); the proposal is in [`research-proposal.pdf`](research-proposal.pdf). Changes made after the plan was written are listed in its deviations log (§13).

## Dataset

[Long-Horizon-Execution](https://huggingface.co/datasets/arvindh75/Long-Horizon-Execution) (Sinha et al., ICLR 2026; Apache-2.0), pinned to revision `7e4901fb…`.
- **Task:** running sum over a 100-key dictionary, with the exact total known at every turn.
- **Splits:** samples 0–19 calibration, 20–51 test, 52–99 reserve.
- **Contamination check:** the same dictionary with freshly generated key sequences.

`data/lhe/test.jsonl` (102 MB) is not stored in git. Fetch it with:

```bash
python3 scripts/fetch_data.py      # downloads the pinned files and verifies SHA-256 checksums
```

## Model

`ibm-granite/granite-4.2-8b`, revision `a7042144125ce5ae64f89610e8289647fc000b9a` (Apache-2.0), bf16, served with vLLM.
- **Agent:** non-thinking, greedy decoding, reply = first line.
- **Audits (A3/A4):** Granite's thinking mode, with the model card's sampling settings.

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-4.2-8b', \
  revision='a7042144125ce5ae64f89610e8289647fc000b9a', local_dir='models/granite-4.2-8b')"
```

The prompt format is derived from the model's own chat template, so the harness works with any model that has an `enable_thinking` switch.

## Arms

| Arm | What changes |
|---|---|
| A0 | Baseline, plus side-question probes (P(Yes), verbal confidence, forecasts with none/true/placebo evidence) |
| A0_inline | Replies are `total \| confidence` (HE) |
| A1a / A1b / A2 | Re-compute gate: format only / format + logit threshold τ / format + random at A1b's rate |
| A3 / A4 | Thinking audit every m turns, in a fresh context / inside the full history, with a correction note |
| D1 / D2 | No history: state line holds the agent's own total / the true total |

## Usage

```bash
python3 -m unittest discover -s tests -v                      # no GPU needed (template tests use the tokenizer)
python3 scripts/smoke_all.py --ids 0-1 --T 12                 # all arms, one model load, calibration ids
python3 scripts/pilot_batch.py                                # Phase 1 pilot jobs (calibration + fresh sequences)
python3 scripts/run_arm.py --config configs/A0.json --split calibration
python3 scripts/run_arm.py --config configs/A0.json --split test   # main runs, only after pre-registration
python3 scripts/pilot_report.py runs/<run_dir>
python3 scripts/main_runs.py --gpu 0                          # all 9 arms on the test set (guards: prereg checksums)
python3 scripts/analyze_main.py --stamp <STAMP>               # pre-registered verdicts -> results/main_results.json
python3 scripts/figures.py --stamp <STAMP>                    # figures -> results/figures/
python3 scripts/secondary.py --stamp <STAMP>                  # secondary analyses -> results/secondary.json
# without a GPU: unpack results/raw_main_20261113-173033/*.tar.gz into runs/ and run the three analysis scripts
```

`run_arm.py` refuses test ids unless `--split test` is passed explicitly.

## Layout

```
src/       lhe_data.py (pinned loader, splits, fresh sequences) · labels.py (error taxonomy) · prompts.py
           agent.py (turn loop for all arms) · metrics.py · stats.py
scripts/   main_runs.py · analyze_main.py · figures.py · secondary.py · compare_arms.py · run_arm.py · smoke_all.py · pilot_batch.py · pilot_report.py · prompt_check.py · diag_inline.py · fetch_data.py
configs/   one JSON per arm
tests/     unit tests
data/lhe/  dictionary, config, checksums, source note (test.jsonl fetched separately)
```

## Status

- **Phase 0** (setup, harness, tests, prompt check): done.
- **Phase 1** (pilot on calibration data): in progress.
- **Main runs:** only after pre-registration (`prereg.md`).

## Citation

If you use the dataset, cite Sinha, Arun, Goel, Staab, Geiping. *The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs.* ICLR 2026. arXiv:2509.09677.
