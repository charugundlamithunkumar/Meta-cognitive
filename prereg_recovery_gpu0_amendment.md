# HG execution amendment: collect unstarted MR main on GPU 0

Recorded 18 November 2026 after 12:23 IST and before any MR main answer.
The user instructed: “now gpu 0 is free,we can continue our work there”.
Original HG registration: `1d11bc5a2a665062b3aec023f2dc18ba0c0d02a4`.
This supersedes the GPU 2 scheduling amendment for collection only.

## Prior attempt and reason

M completed on GPU 1; its results are known. GPU 2 initially waited for memory,
then attempted vLLM initialization at 11:41 IST and failed because its service
PATH could not resolve the already installed `ninja` executable. The service
exited; no MR output root or checkpoint exists. The earlier CUDA matrix smoke
did not validate full vLLM initialization. Preserve those failure logs and the
GPU 2 source snapshot. Both obsolete supervisors are inactive.

GPU 0 is now free. It is an NVIDIA RTX 6000 Ada Generation, UUID
`GPU-783ea485-96e8-57f6-226f-81f33d43e630`, the same model as the successful
GPU 1 calibration/M runs. Observed total memory is 49,140 MiB and driver
590.48.01. Use the frozen utilization .38, requiring 19,185.2 MiB free including
the original 512 MiB guard. No other user's processes are stopped.

## Unchanged scientific and inference settings

Keep all four arms, ids 20–51, seven checkpoints, ten-step continuations, masks,
+1 generator, prompts, forks, labels, seeds, bf16, greedy constrained decoding,
context 6,144, eager mode, one sequence, 512-token chunked prefill and prefix
caching. Keep model/data/software checksums, all estimands, effect margins,
bootstrap procedures, Holm families, floors, missingness and stopping rules.
Collect all 896 conditions / 8,960 task answers regardless of result. No M rerun.

## Technical correction and provenance

Reuse the MR-only supervisor through `scripts/run_recovery_gpu0.py`; parameterize
its scheduling GPU rather than copy the experiment runner. Prepend the current
Python environment's executable directory to the service-child PATH and verify
the installed `ninja` before collection. Record its path, version and SHA-256.
No dependency installation, quantization, new kernel flag or model change.
The original frozen runner, configs, prompts and analyzer remain unchanged.

Commit this amendment, entry point, shared-supervisor fix and tests before MR
inference; save the commit and source hashes with new tag `gpu0_main_20261118`.
Never resume or overwrite the failed GPU 2 job under the new fingerprint.
If loading fails, preserve the technical failure and completed checkpoints;
do not silently change inference settings to obtain results.

Report MR as the HG main study with disclosed hardware/scheduling amendments
and technical failure history. The GPU model matches calibration, but different
physical GPUs do not guarantee bit-identical greedy outputs. All MR arms use
GPU 0. This is not an independent replication, and changing hardware does not
justify retuning the registered decision rules.
