# Main M/HF and MR/HG execution — 2026-11-18

Both calibration pilots and the final code/precision review are complete.
HF/HG are finalized in `prereg.md` and `prereg_recovery.md`, with their
registration timestamp supplied by the commit containing those files. Original
32 ids, seven checkpoints, thresholds and separate Holm families are retained.
The registered runners reject changed materials before test inference.

## All approved experiments finished

Independent C also completed at 17:08:53 IST on 18 November 2026. Original main
plus C totals are 4,928 conditions / 49,280 task answers and 22 validated
archives. All analyses/figures and the requested results-only PDF are complete:
`results/all_experiments_20261118/all_experimental_results.pdf`. See
`results/all_experiments_completion_20261118.json` for counts/digests.
No watcher or model process is waiting; do not restart completed jobs.

## Completed main studies

M and MR are complete. M: 1,568 conditions / 15,680 task answers, seven validated
archives. MR: 896 / 8,960, four validated archives and all 32 paired trajectories.
MR completed validation/analysis at 13:15:26 IST on 18 November 2026 and released
GPU 0. Its final status is `complete`; no watcher is waiting. The registered
HG verdict is inconclusive, with HG1/HG2 supported and HG3/HG4 unsupported.
See `results/recovery_HG_gpu0_main_20261118/main_summary.md` and its
`completion_record.json`. No further main-study GPU run is needed.

The launch/resume commands below are historical operational records; do not
rerun completed or obsolete jobs. Submission and any new confirmation study
remain separate work.

## Historical approved job: GPU 0 continuation

The user approved both main studies to completion and instructed continuation
on GPU 0. M is complete and preserved. GPU 1 stopped at its memory guard;
GPU 2 later failed vLLM initialization because its service PATH could not find
the installed `ninja`. No MR main output/checkpoint was generated in either
attempt. Both obsolete services are inactive and their logs remain preserved.

See [the GPU 0 amendment](prereg_recovery_gpu0_amendment.md), recorded before
any main MR answer. GPU 0 is the same RTX 6000 Ada model used for successful
calibration. The original runner/configs/protocols remain unchanged. A shared
supervisor correction makes the existing ninja executable available to children.
It does not install packages or change inference/statistical settings.

```bash
python3 scripts/run_recovery_gpu0.py --describe
# The persistent service handles collection; do not launch a duplicate.
systemctl --user status mithun-recovery-gpu0-main-20261118
cat runs/main_recovery_gpu0_main_20261118/status.json
# Only after confirming no active supervisor/child, if interrupted:
python3 -u scripts/run_recovery_gpu0.py --resume
```

Status/logs: `runs/main_recovery_gpu0_main_20261118/`.
Raw outputs: `runs/recovery_HG_gpu0_main_20261118/`.
Analysis: `results/recovery_HG_gpu0_main_20261118/`.
The MR-only supervisor skips M, collects all four arms, validates archives and
source snapshots, then runs the original HG analysis and figures.

- M used GPU 1; MR uses GPU 0 under the recorded amendment. Canonical .38
  memory utilization, context 6,144, eager bf16,
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
# Historical initial launch; the existing tag now requires --resume:
python3 -u scripts/run_registered_meta.py --gpu 1 --tag main_20261118
# Historical GPU 1 queue only; do not restart after GPU 2 migration:
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
- MR raw outputs: `runs/recovery_HG_gpu0_main_20261118/`; report/figures in
  `results/recovery_HG_gpu0_main_20261118/`.
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

## Historical GPU 2 launch status — superseded by failed initialization

Migration commit: `739613ef8d7d56f940c88e30692bd14dc5051b20`.
Service: `mithun-recovery-gpu2-main-20261118.service` (active at launch).
The initial status is `waiting_for_gpu`: only 13,077 MiB free when the service
checked, versus the unchanged .38 allocation's 37,709.06 MiB requirement.
It polls every 30 seconds; the first model load/inference has not yet occurred.
Check current progress rather than assuming this historical snapshot is live:

```bash
systemctl --user status mithun-recovery-gpu2-main-20261118
cat runs/main_recovery_gpu2_main_20261118/status.json
tail -n 10 runs/recovery_gpu2_main_20261118_service.log
```

GPU 2's capacity wait ended, but initialization failed at 11:41 IST with
`FileNotFoundError: ninja`. No MR answers were generated. Preserve
`runs/main_recovery_gpu2_main_20261118/HG_inference.log`; do not resume that job
under a new fingerprint. Current continuation is on GPU 0 as described above.

## GPU 0 launch record — 2026-11-18 12:26 IST

Migration commit `e45f9df02567cab48613f96c7af9799bfcc53cbe`; source/registration
checks and the resource/context preflight passed. All 61 existing tests plus
seven reporting/migration tests pass. Service
`mithun-recovery-gpu0-main-20261118.service` admitted GPU 0 with 48,462 MiB free
and entered inference (including initialization). Child PATH now resolves
`/home/viswanath/miniconda/bin/ninja`, with its version/hash saved in the job
signature. Read live status/checkpoint counts; this note does not assert that
collection or analysis is complete.
