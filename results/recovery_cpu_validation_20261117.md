# Recovery study: CPU implementation validation

Validated 2026-11-17. **No GPU/model inference performed.**

- Preregistration draft: `prereg_recovery.md`, written before analysis code; not frozen or committed.
- Existing plus recovery regression tests: **55 passed**.
- Full four-arm mock: **56 checkpoint conditions / 560 scripted continuation answers**, two synthetic trajectories and all seven checkpoints.
- Resume completed without regenerating committed checkpoints; analysis explicitly withholds a research verdict.
- Real local-tokenizer preflight: calibration ids 0–1, all seven checkpoints/four arms, **5,261-token conservative bound**, below 16,384. Complete local content/software hashes verified; no weights loaded.
- Resource inventory includes all four shards, external chat template, tokenizer/config/vocabulary files and pinned data.
- Paired statistics fixtures cover margins, equivalence boundaries, floors, missing forecasts, underconfidence versus baseline overconfidence and trajectory resampling.
- Synthetic precision/null/boundary diagnostics saved with a larger-variance sensitivity. These do not establish empirical power; pilot variance and Monte Carlo uncertainty still need review.
- Metadata-only allocation audit saved; reserve task usage/reservations still require review before independent confirmation.
- All **209** inventoried archived artifact/code/config/test file hashes unchanged.

## Outputs

- [Mock report and figures](recovery_cpu_final_20261117/report.md)
- [Mock machine-readable analysis](recovery_cpu_final_20261117/analysis.json)
- [Complete resource manifest](recovery_resources_complete_20261117.json)
- [Synthetic precision checks](recovery_precision_bounded_20261117.json)
- [Larger-variance sensitivity](recovery_precision_bounded_conservative_20261117.json)
- [Allocation audit](recovery_allocation_audit_20261117.json)
- [Commands and limitations](../RECOVERY.md)

## Remaining gates

1. Approved GPU 0 calibration smoke and pilot; validate real vLLM inference and measure runtime, probe validity, accuracy and floors.
2. Calibration-based power/precision review; record amendments, finalize sample allocation, freeze and commit HG/HF materials.
3. Approve the measured full budget and collect real M/MR data before empirical conclusions or independent confirmation.

The mocked character tokenizer uses a larger fixture-only context limit. The real model retains 16,384 tokens. Development mocks and initial simulation/resource files are retained separately; the artifacts linked above are the current CPU validation outputs.
