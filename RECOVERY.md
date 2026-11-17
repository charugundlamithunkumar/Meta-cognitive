# Recovery experiment (MR / HG)

This implements the accepted “After the Fix” design on CPU/mock paths and
prepares a separately guarded local GPU path. **MR calibration is complete; main MR inference is pending.** `prereg_recovery.md` was drafted before the analysis code. It is
not frozen or committed: final review and registration are the next gates.

## What exists

- Four configs: MR_A (clean/true state), MR_B (errors/wrong state), MR_C
  (same errors/true state), MR_D (same clean history/wrong state).
- Exactly rounded earlier-error masks, forced wrong endpoint, existing +1
  generator, shared history hashes and explicit supplied state.
- Independent pre/post forecasts, ten task answers, per-answer verbal/P(Yes),
  and awareness after the continuation. Probe responses never enter execution.
- First generated label uses supplied state; global correctness uses LHE truth.
- Atomic checkpoint files with checksums, canonical JSONL, trajectory summaries,
  source/config/resource fingerprints and immutable tar.gz archives.
- Paired HG bootstrap/TOST/Holm decisions, missing/floor/eligibility gates,
  calibration, signed errors, interactions, strict-position AUROC and figures.
- Synthetic precision and boundary/null simulations, with optional real
  calibration-only residuals. They are not model measurements or a power guarantee.

The model and dataset are already local. `scripts/recovery_resources.py` hashes
the weight shards, tokenizer/config, pinned data and installed software without
loading weights. `results/recovery_resources_complete_20261117.json` contains the initial
CPU inventory. Hashes record current local content; the configured remote
revision is not an independent verification of downloaded content.

## Run CPU validation

From the repository root; use fresh output names if an example already exists:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_recovery.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/recovery_cpu_example
python3 scripts/run_recovery.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/recovery_cpu_example --resume
python3 scripts/analyze_recovery.py runs/recovery_cpu_example --n-boot 100 --out results/recovery_cpu_example
python3 scripts/recovery_precision.py --n 32 --reps 200 --n-boot 1000 --sigma 2 --out results/recovery_precision_example.json
python3 scripts/recovery_precision.py --n 32 --reps 200 --n-boot 1000 --sigma 2 --variance-multiplier 2 --out results/recovery_precision_example_conservative.json
```

`--dry-run`/`--mock` uses a character tokenizer, synthetic keys and scripted
arithmetic; it needs neither weights, data, vLLM nor a GPU. Mock analysis always
withholds empirical HG verdicts, even if a statistical fixture satisfies tests.
The mock deliberately supplies generic forecasts; it is not an implementation
of a hypothesized recovery effect. Real model behavior remains to measure.
The character-token mock uses a 100,000-token fixture limit for the long
histories; the real tokenizer and model retain the registered 16,384 limit.

## Read-only local preflight

```bash
python3 scripts/run_recovery.py --preflight --split calibration --ids 0-1 --checkpoints 10
# Optional full content validation (reads all local weights on CPU):
python3 scripts/run_recovery.py --preflight --split calibration --ids 0-1 --checkpoints 10 --resource-manifest results/recovery_resources_complete_20261117.json
```

Without a resource manifest, preflight checks local presence, pinned data and
context bounds, explicitly reporting `resource_hashes_verified: false`.
With one, it recomputes and checks all content/software hashes. Both paths load
only the local tokenizer, never model weights, and write no run files.

To inventory a changed environment, create a new file rather than overwrite:

```bash
python3 scripts/recovery_resources.py --out results/recovery_resources_new.json
```

## Registration and main steps still pending

After GPU approval and availability, the first calibration smoke is 8 conditions
and 80 continuation answers, plus probes:

```bash
python3 scripts/run_recovery.py --split calibration --ids 0-1 --checkpoints 10 --gpu 0 --resource-manifest results/recovery_resources_complete_20261117.json --out runs/recovery_gpu_smoke
```

Then a separately budgeted pilot on ids 0–7 at all seven checkpoints, and CPU
analysis/precision using `--pilot runs/<calibration-root>`. Pilot precision only
accepts real complete calibration arms; it refuses mock or test inputs. The
simulation uses bounded uniform synthetic noise (SD `--sigma`) or resampled
calibration residuals and reports any rejected out-of-range draws; inspect these and variance
sensitivity rather than treating simulated rejection frequencies as validated
sample-size estimates. No calibration results were used during implementation.

Finalize sample/precision review and dated calibration deviations. Add a standalone
`Status: FROZEN` line to the HG registration only after that review, then commit
the scoped materials listed by `scripts/run_recovery.py:source_files`. The runner
refuses test inference while the draft is unfrozen or relevant files differ
from HEAD. It does not commit, weaken checks or substitute mock results.

Only after this gate and approval of the measured full budget:

```bash
python3 scripts/run_recovery.py --split test --gpu 0 --resource-manifest results/recovery_resources_complete_20261117.json --out runs/recovery_HG_main
python3 scripts/run_recovery.py --split test --gpu 0 --resource-manifest results/recovery_resources_complete_20261117.json --out runs/recovery_HG_main --resume
python3 scripts/analyze_recovery.py runs/recovery_HG_main --n-boot 10000 --out results/recovery_HG_main
```

Main inference requires all four arms, ids 20–51 and the seven fixed checkpoints.
It checks free GPU memory before allocation; it never stops another process.
The real GPU path completed the smoke and full calibration pilot on GPU 1.
See the pilot summary for timing and limits; full-run estimates require final review. M/HF has its separate commands in
`METACOGNITION.md` and is not rerun as a side effect of MR.

## Reading outputs and resuming

Root `manifest.json` stores all selected configs and resource fingerprints.
Each arm has `config.json`, `rows.jsonl`, `trajectories.jsonl`, `run.log`, atomic
`checkpoints/<id>_<c>.json` and a completed archive at the run root. A checkpoint
envelope stores its content SHA-256 and eleven rows. `record_type` distinguishes
supplied history metadata from ten generated steps; offsets 0–10 are unique
within `(arm, sample_id, checkpoint)`. Forecast raw text, parsed/clipped values,
out-of-range flags, probability retained mass and prompt hashes are recorded.

Only incomplete checkpoints rerun after a crash; completed invalid forecasts
are retained. `--resume` repairs JSONL from checkpoint files, refuses changed
source/config/resources, and validates all saved data before model allocation.
When nothing is pending, it does not load a model. New code requires a new root;
do not resume earlier mock runs with changed fingerprints. Final archives cannot
be overwritten with different contents. Resume after a technical interruption
may recompute an uncommitted checkpoint, never a committed response.

Analysis accepts arm folders or tar.gz archives, without extraction. A new output
directory is mandatory; no report overwrite occurs. `analysis.json` contains all
tests, provenance, class/support counts and missingness. `report.md` and four
PNG figures summarize accuracy, forecasts, HG contrasts and secondary measures.
Incomplete, calibration and mock runs remain diagnostic/exploratory. Complete
paired trajectories, original source provenance and registration are necessary
for main inference; the 224 checkpoints do not constitute 224 independent units.

No sample allocation for independent confirmation has been made. Reserve ids and
fresh seeds overlap prior plans and need an allocation audit before registration.

## CPU artifacts from 2026-11-17

- Full-checkpoint mock report: [results/recovery_cpu_final_20261117/report.md](results/recovery_cpu_final_20261117/report.md). These are scripted software diagnostics only.
- [Complete local resource hashes](results/recovery_resources_complete_20261117.json), including the external chat template and auxiliary tokenizer files.
- [Synthetic precision/null checks](results/recovery_precision_bounded_20261117.json) and [larger-variance sensitivity](results/recovery_precision_bounded_conservative_20261117.json). These use 200 synthetic repetitions and 1,000 bootstrap draws per repetition, not empirical Granite variance; review Monte Carlo uncertainty before freezing HG.
- [Allocation metadata audit](results/recovery_allocation_audit_20261117.json): records local run/config use without reading outcomes. Historical reservations still prevent automatically declaring the reserve independent and available.

Earlier development mocks/manifests/simulations are retained with their original
fingerprints. The complete resource manifest and final mock linked above are
the current validation artifacts. Do not resume development mocks after source changes.

## GPU 1 calibration-only memory profile (2026-11-17)

With about 20.2 GiB free on GPU 1, `--calibration-profile gpu1-small` uses
utilization .38, context 2,048, eager execution, one active sequence and
512-token chunked prefill. This profile is rejected for test inference. It
changes runtime settings only; input histories must pass the context audit.
The calibration amendment is recorded in `prereg_recovery.md`.

```bash
python3 scripts/run_recovery.py --preflight --split calibration --ids 0-1 --checkpoints 10 --calibration-profile gpu1-small --resource-manifest results/recovery_resources_complete_20261117.json
python3 scripts/run_recovery.py --split calibration --ids 0-1 --checkpoints 10 --gpu 1 --calibration-profile gpu1-small --resource-manifest results/recovery_resources_complete_20261117.json --out runs/recovery_gpu1_cal_20261117
```

Choose a new root if that output already exists. Fit is established only by a
successful real smoke run. Main configs remain utilization .6/context 16,384.

**Observed smoke result:** The GPU 1 run above completed all eight calibration
conditions and 80 real task answers. All four archives validated and the GPU
engine exited. See [the smoke summary](results/recovery_gpu1_cal_20261117/smoke_summary.md).
This validates the small profile only; it is not evidence for HG. A wider pilot
needs a larger context profile and an agreed compute budget.

## Full calibration pilot on GPU 1

The user authorized the planned pilot to completion: ids 0–7, seven
checkpoints, four arms (224 conditions, 2,240 task answers). Use `gpu1-long`,
a calibration-only profile with the tested memory/batching settings and a
6,144-token context limit. The complete local-tokenizer context check is
required before launch. The larger limit does not truncate or alter histories.

```bash
python3 scripts/run_recovery.py --preflight --split calibration --ids 0-7 --calibration-profile gpu1-long --resource-manifest results/recovery_resources_complete_20261117.json
python3 scripts/run_recovery.py --split calibration --ids 0-7 --gpu 1 --calibration-profile gpu1-long --resource-manifest results/recovery_resources_complete_20261117.json --out runs/recovery_gpu1_pilot_20261117
# Resume only the same source/config/model/data fingerprints:
python3 scripts/run_recovery.py --split calibration --ids 0-7 --gpu 1 --calibration-profile gpu1-long --resource-manifest results/recovery_resources_complete_20261117.json --out runs/recovery_gpu1_pilot_20261117 --resume
python3 scripts/analyze_recovery.py runs/recovery_gpu1_pilot_20261117 --n-boot 10000 --out results/recovery_gpu1_pilot_20261117
```

Completion of the pilot does not freeze HG or constitute a confirmatory
result. Review real-variance precision, floors and prompt/probe behavior
before main-registration decisions. Existing earlier run fingerprints remain
preserved; do not resume the early smoke with the new code.

## Pilot completion and precision review

The full GPU 1 pilot completed all 224 conditions and 2,240 answers. Results
and caveats are in [the pilot summary](results/recovery_gpu1_pilot_20261117/pilot_summary.md).
This remains calibration evidence; no HG main verdict is available.

The first pilot-residual power implementation rejected many arm-score draws
outside 0–100, making some scenarios unidentifiable. Those outputs are retained
for transparency. Precision simulation now shifts the joint **trajectory-level
contrasts**, sharing the exact HG test helper with main analysis, instead of
clipping or rejecting artificial arm percentages. This is a local contrast
sensitivity calculation, not a generative model of bounded scores or a guarantee
of power. Variance and plug-in effects estimated from eight trajectories remain
uncertain. The registered pilot test outputs were checked identical before and
after extracting the helper.

```bash
python3 scripts/recovery_precision.py --pilot runs/recovery_gpu1_pilot_20261117 --n 32 --reps 500 --n-boot 1000 --out results/new_pilot_precision.json
python3 scripts/recovery_precision.py --pilot runs/recovery_gpu1_pilot_20261117 --n 32 --reps 500 --n-boot 1000 --variance-multiplier 2 --out results/new_pilot_precision_conservative.json
```

Pilot generation source files are preserved at
`runs/recovery_gpu1_pilot_20261117/source_snapshot/`, matching recorded hashes.
Subsequent precision/reporting changes require a new root for future inference;
do not resume the completed pilot with changed source fingerprints.
