# Catching Silent Collapse — metacognition and recovery

Current question: **Do an LLM agent’s self-reports track failure and recovery,
or mainly reflect its conversation history?**

This repository uses the locally downloaded **IBM Granite 4.2 8B** model on the
Long-Horizon-Execution running-sum task. The instructor accepted the recovery
proposal; implementation and experiment stages are in the current plan.

## Current work

- [Implementation plan](implementation-plan.md): accepted design, implementation units, tests and GPU run stages.
- [Accepted recovery proposal (PDF)](after-the-fix-problem-statement.pdf) and [editable source](after-the-fix-problem-statement.md).
- [M experiment guide](METACOGNITION.md): controlled history length/error-dose experiment; full calibration completed on GPU 1.
- [Current preregistration](prereg.md): active HF; completed HA–HE is archived. HF was frozen before main inference in commit `1d11bc5`.
- [Exploratory A0 metacognition analysis](results/meta_analysis.md).

**All approved experiments are complete.** Original M/MR plus independent
fresh-sequence C saved 49,280 core task answers across 4,928 conditions and
22 validated archives. Raw checkpoints, source snapshots and technical logs
remain saved. All registered analyses and figures are complete; GPU 0's model
allocation is released.

The requested deliverable is a **results PDF only**:
[Download all experimental results](results/all_experiments_20261118/all_experimental_results.pdf)
(71 pages, including original, fresh confirmation, calibration and A0 exploration).
See the [completion and integrity record](results/all_experiments_completion_20261118.json).

HF is length-driven under the history-truncation contrast in both cohorts.
HG is inconclusive in both: execution repair and clean-reference equivalence
pass, but the forecast-margin and underconfidence criteria do not. All null,
reversed, floor-limited and sparse-support results are retained.

- [Original M results](results/meta_HF_main_20261118/main_summary.md).
- [Original MR results](results/recovery_HG_gpu0_main_20261118/main_summary.md).
- [Fresh confirmation results](results/confirmation_C_main_20261118/report.md).
- [Completed implementation plan](implementation-plan.md).

Earlier publication drafts are archived under `work 1/publication drafts/`.
No further experiment, paper or submission remains in the approved task.

All 22 main/confirmation data archives are included in
[`results/raw_core_20261118/`](results/raw_core_20261118/README.md), with SHA-256
checksums. Model weights, the large dataset and full runtime directories remain
local and are excluded from Git.

## Previous work

The completed A0–D2 study is organized in **[work 1](work%201/README.md)**.
That folder holds its proposal, final report, results, original figures and raw
runs, plus historical planning documents and references to reused code.

Old reports, results, runs, scripts, configs and tests are located only in that
folder. Active shared primitives remain in `src/`; current tools directly read
the archived +1 generator and A0 data when needed. The archived regression
tests are exercised through a small loader in `tests/`.

## Local resources and validation

The model is already in `models/granite-4.2-8b/`; the dataset is in `data/lhe/`.
All four indexed weight shards and tokenizer files were found, dataset
checksums passed, and the 61 CPU tests passed again on 2026-11-18.
No download is needed for the current workspace.

From the repository root:

```bash
python3 -c "from src.lhe_data import verify; verify(); print('Dataset checksums OK')"
python3 -m unittest discover -s tests -v
python3 scripts/run_recovery.py --preflight --split calibration --ids 0-1 --checkpoints 10
```

Consult `METACOGNITION.md` and the active plan for mock, resume and approved GPU
commands. Do not execute the old Study-1 run instructions for this new study.

## Layout

- `src/`, `scripts/`, `configs/`, `tests/`: shared study infrastructure and M/MR code.
- `results/`: active metacognition analysis outputs.
- `runs/`: completed M/MR experiments, technical logs and mock checks.
- `work 1/`: organized completed-study artifacts and historical documents.
- `previous work`: compatibility symlink to `work 1/`, preserving frozen source paths.
- `models/`, `data/`: shared local resources, kept in place.

The task and history-injection precedent come from Sinha et al.,
[The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](https://arxiv.org/abs/2509.09677),
ICLR 2026. The instructor’s originating metacognition question is acknowledged
in the research plan. Novelty remains a qualified research claim, not an
established result.
