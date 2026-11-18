# Remaining-plan CPU deliverables — 18 November 2026

Completed the available reporting work while the approved MR main run waits
for GPU 1 capacity. This record does not mark the complete study finished.

## Changes

- Added `paper/manuscript-draft.md`: shared methods, registered M results,
  recovery design, limitations and an explicit pending MR results section.
- Added `paper/related-work.md`: checked the eight proposal sources and three
  additional primary papers; narrowed the candidate contribution and recorded
  source-access and novelty-review limits.
- Added `paper/confirmation-readiness.md` and a metadata-only allocation tool.
  The new audit includes compressed archive configs: 178 config records, no
  parsing errors, no recorded possible-real uses of ids 52–99 or seeds
  1000–1047. Historical reservations remain; no confirmation sample allocated.
- Updated the current plan, README and main execution guide to distinguish
  completed M from pending MR. Preserved the prior plan only in `previous work/`.

## Validation

- All 61 existing tests passed (`python3 -m unittest discover -s tests -v`;
  captured in `/tmp/remaining-plan-validation-20261118.log`).
- All three new audit tests passed, including archive config parsing without
  parsing an invalid outcome payload, mock exclusion and invalid metadata.
- All seven manuscript arm means match the saved registered JSON to the shown
  precision. Primary estimates/intervals were checked against that JSON.
- All relative links in the updated README, plan and paper documents resolve.
- The registration guard passes and HEAD remains
  `1d11bc5a2a665062b3aec023f2dc18ba0c0d02a4`.
- `git diff --check` passes. Registered source/configs/preregistrations were
  not edited, and no new commit was made during the frozen queue's wait.

## Commands

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s paper/tools -v
# Use a new output filename; the tool refuses overwrite:
python3 paper/tools/audit_allocations.py --out results/confirmation_allocation_audit_next.json
systemctl --user status mithun-meta-main-resume-20261118
cat runs/main_meta_main_20261118/gpu_wait_status.json
```

## Still pending

MR requires 896 conditions / 8,960 task answers on GPU 1, then its automatic
analysis and figures. The active capacity watcher resumes the exact approved
queue once the frozen .38 allocation plus guard fits (19,185.2 MiB free).
Do not launch a duplicate queue. After collection, validate all four archives,
32 paired trajectories and registration eligibility, then complete the MR
results and synthesis in the manuscript, including null or inconclusive tests.

Independent confirmation remains conditional: allocation reconciliation,
explicit manifest support in a separate runner, registration, precision and
new compute approval precede inference. Fuller literature review and author/
release review remain necessary before submission. Nothing was sent or published.
