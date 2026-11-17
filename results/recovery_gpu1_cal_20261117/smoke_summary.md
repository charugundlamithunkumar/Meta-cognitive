# GPU 1 calibration smoke — 2026-11-17

**Completed successfully. Real Granite inference; exploratory calibration only.**

- GPU: NVIDIA RTX 6000 Ada, physical index 1.
- Model: existing local Granite 4.2 8B, bf16, greedy; no re-download.
- Profile: utilization .38, 2,048-token limit, eager mode, one active sequence, 512-token chunked prefill.
- Scope: calibration ids 0–1, checkpoint 10, all four arms; eight conditions, 80 generated task answers.
- All eight checkpoint files validated and four archives completed. No missing post-cue forecasts.
- Existing jobs remained running; our engine shut down and released its GPU memory.

| Arm | Globally correct / answers | Next-10 accuracy | Post-cue forecasts |
|---|---:|---:|---|
| MR_A | 3/20 | 15.0% | 100, 100 |
| MR_B | 0/20 | 0.0% | 100, 100 |
| MR_C | 3/20 | 15.0% | 100, 100 |
| MR_D | 0/20 | 0.0% | 100, 100 |

These two trajectories at one early checkpoint validate the pipeline and memory fit. They do not establish recovery recognition, pessimism, statistical equivalence, or a main HG finding. The forecasts in this smoke are uniformly optimistic. Main inference remains blocked pending broader calibration, precision review and registration freeze.

Runtime evidence from the engine log: weight loading 3.28 s; engine profile/cache/warmup 71.37 s; 1.42 GiB available KV cache (9,280 token slots). Per-arm generation telemetry is saved in the raw arm logs. These measurements are specific to this shared-GPU smoke and do not predict late-checkpoint runtime.

For calibration at all seven checkpoints, the 2,048 limit must be replaced by a separately recorded larger-context calibration profile that passes the exact prompt bound and memory checks. Do not truncate later histories or change main configs to reuse this small profile.

[Full exploratory analysis](report.md) · [Machine-readable analysis](analysis.json) · [Raw run manifest](../../runs/recovery_gpu1_cal_20261117/manifest.json) · [Startup log](../../runs/recovery_gpu1_cal_20261117.log)
