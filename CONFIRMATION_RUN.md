# Prepared additional experiment: independent-sequence study C

**No confirmation model answers have been generated. GPU budget approval is
the remaining launch gate.** The completed M/MR studies are preserved.

The implementation plan calls for independent confirmation if feasible. The
new explicit allocation avoids the historically reserved ids/seeds: 32 fresh
seeds 2000–2031, ids 12000–12031, same dictionary and generator. The
186-record config/archive audit found no recorded overlap. All 300-turn
content hashes also differ from the 100 published sequences and previously
recorded fresh seeds 0–19. This is independent sequence generation within the
recorded project scope, not proof about all external or model-training uses.

## Concrete job

- GPU 0, same Ada model and frozen bf16/.38/6,144/eager/one-sequence/512-token
  chunked-prefill settings. No model download or package installation.
- M replication then MR replication, separate model processes. All seven M
  conditions and all four MR conditions; C labels identify the new instance.
- 2,464 conditions / 24,640 task answers, plus all unchanged probes.
- Identical HF/HG tests and margins, all seven checkpoints, separate Holm
  families, 10,000 whole-trajectory bootstrap samples, seed 0. No tuning to
  the main study's null HG outcome; no significance-based stopping.
- Approximate budget: 111 minutes from the observed original main durations,
  plus shared-machine/input variability. This is an estimate, not a hard cap.

The [C registration](prereg_confirmation.md) preceded its analysis code.
The [manifest](data/lhe/confirmation_20261118_manifest.json) freezes sample
content. Original `SPLITS`, old runners and original results stay unchanged.
The new runner directly reuses their history, engine, checkpoint, archive and
statistical functions, with a distinct C eligibility contract.

## Read-only / CPU commands

```bash
python3 scripts/run_confirmation.py --describe
python3 scripts/run_confirmation.py --preflight
python3 -m unittest discover -s paper/tools -p test_confirmation.py -v
# Optional new mock directory only; never interpreted as model evidence:
python3 scripts/run_confirmation.py --dry-run --ids 12000-12001 --checkpoints 10,25 --out runs/C_mock_new_check
python3 scripts/run_confirmation.py --dry-run --ids 12000-12001 --checkpoints 10,25 --out runs/C_mock_new_check --resume
```

The real local-tokenizer preflight passed all fresh ids/checkpoints with upper
bounds 5,452 tokens for M and 5,473 for MR, below 6,144. The complete-family
mock/resume fixture passes, with corrupted checkpoints rejected and all mock
scientific verdicts withheld. These are software checks, not C findings.

## Exact launch after additional long-run approval

Use the existing Python environment. The new supervisor verifies the frozen
registration and source/config/sample/resource fingerprints before inference,
records the installed ninja tool, saves atomic checkpoints, releases each
model process, validates all archives and automatically runs C analysis.

```bash
/home/viswanath/miniconda/bin/python3 -u scripts/run_confirmation.py --out runs/confirmation_C_main_20261118
# Same fixed source/config/manifest/commit, if technically interrupted:
/home/viswanath/miniconda/bin/python3 -u scripts/run_confirmation.py --out runs/confirmation_C_main_20261118 --resume
```

A persistent user service should host this exact command once approved; do
not launch duplicates. Parent status is `runs/confirmation_C_main_20261118/status.json`.
Outputs and archives are under its `M/` and `MR/` subdirectories; independent
C analysis/figures are under `analysis/`. Partial or mock outputs cannot be
promoted to a scientific verdict. Do not overwrite the completed main studies.

If the instructor chooses to report the completed main studies without this
optional extension, keep C marked uncollected and retain that limitation.
No submission or message to the instructor is sent automatically.
