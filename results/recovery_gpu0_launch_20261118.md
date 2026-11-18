# HG main successfully started on GPU 0 — 18 November 2026

Migration commit: `e45f9df02567cab48613f96c7af9799bfcc53cbe`.
Original scientific registration: `1d11bc5a2a665062b3aec023f2dc18ba0c0d02a4`.

The user authorized continuation on GPU 0. The obsolete GPU 1/GPU 2 supervisors
are inactive. GPU 2's missing-ninja initialization failure is preserved;
prepending the current Python executable directory to child PATH resolved the
service environment issue without installing packages or changing model settings.

The GPU 0 job was admitted at 12:26 IST with 48,462 MiB free. vLLM successfully
loaded the model (16.42 GiB weights), created its cache and finished warmup at
12:27 IST. Real task checkpoint generation is now confirmed, not merely queued.
At this check 24 conditions / 240 task answers were saved and their
checkpoint checksums/labels validated with the existing loader. Counts are a
launch snapshot; use live status for current progress. No interim HG decision
was calculated.

All 61 existing tests and seven reporting/migration tests pass. Model/data
resource hashes and context preflight pass (bound 5,430 < 6,144). Their logs
are copied under `runs/main_recovery_gpu0_main_20261118/launch_validation/`.

The service `mithun-recovery-gpu0-main-20261118.service` runs all 896 conditions
/ 8,960 task answers, then validates exports/archives/source snapshots and runs
the unchanged 10,000-bootstrap HG analysis and figures, regardless of outcome.
M's completed data and analysis are preserved. Do not commit or change registered
job inputs/HEAD during collection. No claim of complete MR data is made here.

```bash
systemctl --user status mithun-recovery-gpu0-main-20261118
cat runs/main_recovery_gpu0_main_20261118/status.json
tail -n 10 runs/main_recovery_gpu0_main_20261118/HG_inference.log
```
