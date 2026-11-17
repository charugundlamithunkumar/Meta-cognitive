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
- [Current preregistration](prereg.md): active HF; completed HA–HE is archived. HF registration commit remains pending.
- [Exploratory A0 metacognition analysis](results/meta_analysis.md).

The MR recovery runner, four configs, HG draft, analysis, precision simulator
and tests are implemented. See [RECOVERY.md](RECOVERY.md) for CPU commands,
validation artifacts and pending GPU stages. HF/HG are finalized for the pre-test registration commit.
MR real calibration is complete (224 conditions / 2,240 task answers); main inference is pending. M calibration is complete (392 conditions / 3,920 task answers). Main runs await registration and long-run budget approval. Mock outputs remain software diagnostics.

## Previous work

The completed A0–D2 study is organized in **[previous work](previous%20work/README.md)**.
That folder holds its proposal, final report, results, original figures and raw
runs, plus historical planning documents and references to reused code.

Old reports, results, runs, scripts, configs and tests are located only in that
folder. Active shared primitives remain in `src/`; current tools directly read
the archived +1 generator and A0 data when needed. The archived regression
tests are exercised through a small loader in `tests/`.

## Local resources and validation

The model is already in `models/granite-4.2-8b/`; the dataset is in `data/lhe/`.
All four indexed weight shards and tokenizer files were found, dataset
checksums passed, and the 61 pre-freeze CPU tests passed on 2026-11-17.
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
- `runs/`: active M/MR mock and future experiment runs.
- `previous work/`: organized completed-study artifacts and historical documents.
- `models/`, `data/`: shared local resources, kept in place.

The task and history-injection precedent come from Sinha et al.,
[The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](https://arxiv.org/abs/2509.09677),
ICLR 2026. The instructor’s originating metacognition question is acknowledged
in the research plan. Novelty remains a qualified research claim, not an
established result.
