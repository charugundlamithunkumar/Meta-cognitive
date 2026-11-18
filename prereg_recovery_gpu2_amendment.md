# HG execution amendment: move unstarted MR main collection to GPU 2

Recorded 18 November 2026 after 11:12 IST, before any MR main answer generation.
The user instructed: “we will continue our work in gpu 2”. This supplements,
and does not replace, the original frozen HG protocol in commit
`1d11bc5a2a665062b3aec023f2dc18ba0c0d02a4`.

## Reason and scope

M main is already complete on GPU 1 and its results have been inspected.
MR main has not started: the original queue stopped at its GPU-memory guard,
and no original MR main output directory exists. The GPU 1 capacity watcher
was stopped before preparing this migration. The scheduling change is made
because of available shared-GPU memory, not to change an observed MR effect.

Use physical GPU 2, UUID `GPU-99fccbd6-a9c3-880c-ff83-dd0de9699ef7`, an NVIDIA
RTX PRO 6000 Blackwell Max-Q Workstation Edition, instead of the planned RTX
6000 Ada GPU 1. Driver 590.48.01, PyTorch 2.11.0+cu130 and CUDA 13.0 were
observed. A tiny bf16 CUDA matrix multiplication passed; full vLLM model
initialization must still succeed before collection. Do not change software
versions silently if initialization fails.

Keep the exact existing model/data files and checksums, bf16 precision,
utilization .38, maximum context 6,144, eager execution, one active sequence,
512-token chunked prefill, prefix caching, seed and decoding settings.
The fractional allocation is unchanged but its absolute size is larger on
GPU 2: memory guard = .38 × reported total MiB + 512 MiB. At the observed
97,887 MiB total this requires 37,709.06 MiB free. Shared utilization can change;
check capacity immediately before model loading and never stop others' jobs.

No changes to arms, masks, ids 20–51, seven checkpoints, prompts, probes,
ten-answer horizon, labels, estimands, statistical algorithms, margins,
multiplicity, missingness, floors or stopping rules. All 896 conditions and
8,960 task answers remain required regardless of outcomes. No M rerun.

## Provenance and reporting

Start a new MR-only job and output root tagged `gpu2_main_20261118`; do not
resume or relabel the old GPU 1 queue. Reuse the original runner and analyzer.
The migration supervisor and this amendment are committed before MR inference;
save their hashes, the new commit and original registration commit together.
Keep old queue logs and completed M archives unchanged. Pause on a source,
resource, archive or resume mismatch instead of bypassing the guard.

Report the result as the HG main study with a disclosed hardware amendment.
The analyzer's eligibility result does not erase this deviation. Different GPU
architectures can change floating-point kernels and exact generated outputs;
greedy decoding is not a guarantee of identical answers across hardware.
Within MR, every arm uses GPU 2. Cross-study M/MR differences cannot be
attributed solely to study treatments because hardware also differs.
This is not independent confirmation or an additional replicate.
