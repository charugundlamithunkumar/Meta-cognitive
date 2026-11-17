# Main M/HF and MR/HG execution — 2026-11-18

Both calibration pilots and the final code/precision review are complete.
HF/HG are finalized in `prereg.md` and `prereg_recovery.md`, with their
registration timestamp supplied by the commit containing those files. Original
32 ids, seven checkpoints, thresholds and separate Holm families are retained.
The registered runners reject changed materials before test inference.

## Concrete job awaiting compute approval

- GPU 1, canonical .38 memory utilization, context 6,144, eager bf16,
  one active sequence and 512-token chunked prefill; no quantization/truncation.
- M first: seven arms × 32 ids × seven checkpoints = 1,568 conditions and
  15,680 task answers, plus probes.
- MR second: four arms × 32 ids × seven checkpoints = 896 conditions and
  8,960 task answers, plus probes.
- Combined: 2,464 conditions and 24,640 task answers; both studies are collected
  regardless of favorable, null or inconclusive results.
- Pilot-scaled collection time: approximately 56.1 + 32.8 = 88.9 minutes,
  plus startup, CPU analysis and shared-GPU variability. This is an estimate,
  not a guaranteed finish time or permission to stop at significance.
- The runner checks available GPU memory and never stops another user's job.

```bash
# Read-only description:
python3 scripts/run_registered_meta.py --describe --gpu 1 --tag main_20261118
# After registration commit and user approval of the long-run budget:
python3 -u scripts/run_registered_meta.py --gpu 1 --tag main_20261118
# Same fixed source/config/commit if technically interrupted:
python3 -u scripts/run_registered_meta.py --gpu 1 --tag main_20261118 --resume
```

The actual resource manifest is `results/recovery_resources_complete_20261117.json`.
A different environment/model/data manifest requires a new run rather than a
silent resume. Leave registered source/config files and the registration commit
unchanged during the job; the guards deliberately fail on changed fingerprints.

## Saved status and outputs

- Persistent queue status: `runs/main_meta_main_20261118/status.json`.
- Per-study inference and analysis logs: `runs/main_meta_main_20261118/`.
- M raw outputs: `runs/meta_HF_main_20261118/`; report/figures in
  `results/meta_HF_main_20261118/`.
- MR raw outputs: `runs/recovery_HG_main_20261118/`; report/figures in
  `results/recovery_HG_main_20261118/`.
- Every checkpoint is saved, final archives are immutable, and generation source
  snapshots accompany the runs. Calibration outputs remain in their own roots.

Analysis automatically follows each completed study. A scientific null does
not cancel the other study; technical failures record `failed` status and preserve
all committed checkpoints. Independent confirmation and paper submission are
separate steps; reserve allocation is not silently consumed here.

## Review evidence

- [M calibration](results/meta_gpu1_calibration_20261118/calibration_summary.md)
- [MR calibration](results/recovery_gpu1_pilot_20261117/pilot_summary.md)
- [HF baseline precision](results/meta_precision_20261118.json)
- [HF doubled-variance sensitivity](results/meta_precision_conservative_20261118.json)
- [HG contrast precision](results/recovery_gpu1_pilot_precision_contrasts_20261117.json)
- [HG doubled-variance sensitivity](results/recovery_gpu1_pilot_precision_contrasts_conservative_20261117.json)
- [CPU main-input context audit](results/main_context_audit_20261118.json)

The M pilot classification is inconclusive, and the MR pilot did not establish
overall underconfidence. Those findings do not justify tuning the preregistered
thresholds or selecting checkpoints. Sample size remains 32; precision limits
and null/mixed outcomes will be reported. These are prospective interventions on
previously inspected A0 task sequences, not independent replication.
