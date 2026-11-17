# Metacognition follow-up (M)

HF is in `prereg.md` §7, written before the new analysis implementation. A0
analysis is exploratory because its outcomes were already inspected. Commit HF
and the M materials before a confirmatory model run. No GPU run was performed
while implementing this follow-up.

The M modules reuse `src/agent.py`'s Chat, answer-token extraction and probability
normalization, plus the existing prompts, labels, metrics and bootstrap routines.
The original arm loop contains its probe block as a nested implementation; the new
checkpoint loop assembles the same forks without modifying that frozen file.
`run_meta.py` uses the shared id parser in `src/study_common.py` and the existing dataset loader.

The injected values come directly from `previous work/tests/test_arms_mock.py:FakeLLM.respond`:
the existing `slip` branch adds +1, as confirmed by the user. The test module is
loaded as an adapter; its test suite and GPU initialization are not executed.
This is a controlled small-error manipulation, not a claim that +1 reproduces
the empirical distribution of Granite's mistakes.

## CPU analysis and validation

```bash
python3 scripts/analyze_meta.py
python3 -m unittest discover -s tests -v
python3 scripts/run_meta.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/meta_cpu_smoke
python3 scripts/run_meta.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/meta_cpu_smoke --resume
python3 scripts/analyze_meta_experiment.py runs/meta_cpu_smoke --n-boot 100
```

`--mock` is an alias for `--dry-run`. It exercises the real M orchestration with a
character tokenizer, synthetic trajectories, and the existing scripted model;
it requires neither model files, the large dataset, vLLM, nor a GPU. All output
rows/configs are marked mock. The analysis never issues a study verdict for mock
inputs. Choose a new output directory if the example already exists.

A0 output: `results/meta_analysis.md`, `results/meta_analysis.json`, and
`results/figures/meta_{position,calibration,matched}.png`. The script reads
`previous work/results/raw_main_*/A0.tar.gz` directly without extracting or modifying it. If
multiple A0 archives exist, pass `--source` explicitly. Default: 10,000 bootstrap
resamples, seed 0. `--out` redirects only the new analysis artifacts.

## Registration and GPU commands (run only after approval)

The real model path and `data/lhe/test.jsonl` were present in this checkout.
The GPU path is implemented but has not been validated with model inference.
The runner reports missing resources and offers the mock path if moved elsewhere.

First review and commit the new registration/code/configs. Stage explicit paths
so unrelated changes to `implementation-plan.md` remain yours:

```bash
git add prereg.md METACOGNITION.md requirements.txt configs/M_*.json src/meta_history.py src/meta_runner.py src/study_common.py scripts/run_meta.py scripts/analyze_meta.py scripts/analyze_meta_experiment.py tests/test_meta.py tests/test_legacy_regression.py "previous work/tests/test_arms_mock.py" "previous work/configs/A0.json" results/meta_analysis.md results/meta_analysis.json results/figures/meta_*.png
git commit -m "Register HF and add controlled-history metacognition study"
```

Optional short GPU smoke on **calibration** ids, with its own output directory:

```bash
python3 scripts/run_meta.py --split calibration --ids 0-1 --arms L0 L100 S0 S100 --checkpoints 10 --gpu 0 --out runs/meta_gpu_smoke
```

Confirmatory run (all seven arms, all 32 test trajectories, all seven checkpoints):

```bash
python3 scripts/run_meta.py --split test --gpu 0 --out runs/meta_HF_20261116
```

Resume the identical run, then analyze on CPU:

```bash
python3 scripts/run_meta.py --split test --gpu 0 --out runs/meta_HF_20261116 --resume
python3 scripts/analyze_meta_experiment.py runs/meta_HF_20261116
```

To run a single arm, add `--arms L25` or `--config configs/M_L25.json`. Use a new
root for a new run; `--resume` can also add unfinished/unstarted M arms to an
existing M root. Config/source changes cause resume to fail before loading the
model. The test runner requires the relevant materials to match HEAD. It never
runs A0–D2. A full M experiment has 1,568 checkpoint forks and 15,680 generated
continuation answers, with extra side probes; teacher-forced prefix answers do
not require inference. These are design counts, not measured results or runtime.

## Output and interpretation

Each `runs/<root>/M_<arm>/` contains the existing `config.json`, `rows.jsonl`,
`trajectories.jsonl`, and `run.log` layout, plus atomic `checkpoints/<id>_<c>.json`
files for resume. Completed arms also have `M_<arm>.tar.gz` archives with one root
directory, like the original archives. `rows.jsonl` contains a checkpoint row
and ten continuation rows per checkpoint. Use `(arm, sample_id, checkpoint,
turn, record_type)` as the key: branches can revisit absolute turn indices.
Injected checkpoint rows have no model logits and never enter slip AUROC.

A checkpoint stores the shown user/assistant history, full injection mask,
requested and realized error rate, checkpoint probes, and next-10 accuracy.
Continuations store ordinary answer fields/logits/labels and independent verbal
and P(Yes) probes. Completion summaries count generated answers rather than
claiming the model generated all 300 teacher-forced turns.

A crash before the atomic checkpoint commit reruns only that checkpoint.
A crash after commit reconstructs JSONL from the completed checkpoint files.
No completed checkpoint is regenerated. Existing archives are checked instead
of overwritten. Archive-only directories can be analyzed without extraction.

Primary M inference follows HF, including an inconclusive outcome and a separate
four-test Holm family. Full HF decisions require all 32 paired ids, the seven
registered checkpoints, non-mock test outputs, and 10,000 bootstrap resamples.
Partial runs are reported as exploratory. Secondary calibration uses checkpoint
fixed effects; constant predictors/outcomes are unidentifiable. Step AUROCs hold
both checkpoint and continuation offset fixed, so the 15-per-class rule often
leaves cells unidentifiable with only 32 trajectories; these are shown as missing.

Short controls show only the last five task turns, retaining the system prompt
but adding no oracle state. They also lose earlier reconstruction evidence.
The +1 injection changes wrong-total frequency; consecutive wrong totals may be
locally inherited. Error-sensitive confidence is an operational effect and does
not by itself establish accurate introspection. L100 at the accuracy floor is
not sufficient evidence; the intermediate-dose contrast remains primary.

## GPU 1 calibration — 2026-11-18

The current M runner supports the shared calibration-only `gpu1-long` profile,
full resource hashing, read-only context preflight and GPU free-memory checks.
The profile changes runtime limits only, not prompts/masks/probes/decoding.
Test inference rejects the override. New source/config fingerprints require a
new run root; earlier mocks are retained.

```bash
python3 scripts/run_meta.py --preflight --split calibration --ids 0-7 --calibration-profile gpu1-long --resource-manifest results/recovery_resources_complete_20261117.json
python3 scripts/run_meta.py --split calibration --ids 0-7 --gpu 1 --calibration-profile gpu1-long --resource-manifest results/recovery_resources_complete_20261117.json --out runs/meta_gpu1_calibration_20261118
python3 scripts/run_meta.py --split calibration --ids 0-7 --gpu 1 --calibration-profile gpu1-long --resource-manifest results/recovery_resources_complete_20261117.json --out runs/meta_gpu1_calibration_20261118 --resume
```

Do not rerun these commands into an existing root without `--resume`, and only
resume with identical source/config/model/data fingerprints. The user authorized
continuing the implementation plan. Main experiments remain behind their
registration and full-compute budget gates.

## Main registration/runtime decision — 2026-11-18

M calibration completed all 392 conditions and 3,920 task answers. See
[the calibration summary](results/meta_gpu1_calibration_20261118/calibration_summary.md).
HF is finalized without changing thresholds, masks, prompts or N=32. All canonical
main configs now use the calibrated .38 / 6,144 eager runtime; the old .6 / 16,384
commands above are historical. Real runs require the resource manifest.

The sequential main queue is prepared as:

```bash
python3 scripts/run_registered_meta.py --describe --gpu 1 --tag main_20261118
# Only after registration commit and approval of the roughly 90-minute budget:
python3 scripts/run_registered_meta.py --gpu 1 --tag main_20261118
```

It runs all M arms then MR, automatically analyzes both and saves figures.
Status is `runs/main_meta_main_20261118/status.json`. Source/config/commit
fingerprints must stay fixed during collection; both studies continue regardless
of positive/null research outcomes.
