# Previous work — completed Study 1

Organized on **2026-11-17**. This folder contains the completed A0–D2 study,
its proposal/report, results, figures, and calibration/main-run outputs.
The current metacognition and recovery project remains in the repository root.

## Start here

- [Final report](final-report.pdf)
- [Original research proposal](research-proposal.pdf)
- [Main results](results/main_results.md)
- [Original project README](README-study1.md)
- [Historical implementation plan](implementation-plan.md)
- [Historical HA–HE preregistration](prereg.md)
- [Archive inventory and checksums](archive-manifest.json)

## Contents

| Path | Contents |
|---|---|
| `results/` | Study-1 summaries, secondary/magnitude analyses and pilot summary. |
| `results/figures/` | Original Study-1 figures (`fig*.png`). |
| `results/raw_main_20261113-173033/` | Nine original compressed arm archives, including A0. |
| `runs/` | Previous main/pilot/smoke/prompt-check runs and logs; remains git-ignored. |
| `scripts/`, `configs/`, `tests/` | Original completed-study scripts, configs and tests, moved here byte-for-byte. |
| `src/` | References to shared primitives still actively used by the new experiments. |
| `implementation-plan.md`, `prereg.md` | Copies from commit `02fe4b5d478735ee4c178e763a501104b783cf70`, before the current M/HF additions. |
| `requirements.txt` | Link to the shared, corrected dependency list. |

The two instructor-provided Markdown documents were moved here unchanged and
remain git-ignored as unpublished material. They are not included in the public
project documentation links above.

## Preservation and use

All moved file contents were checked against SHA-256 hashes. The manifest
records each original and archived path. Old-work shortcuts outside this folder
have been removed at the user's request. Current tools explicitly read any
archived input they still need.

Retired scripts and configs live here. They retain their historical content and
are not current execution instructions. The archived tests still run through
`tests/test_legacy_regression.py` from the repository root. Historical commands
in `README-study1.md` and the saved plans are records, not current instructions.
The dated `*-through-20261117.md` snapshots preserve the complete documents
before their active versions were reduced to current studies.

The active [implementation plan](../implementation-plan.md),
[M guide](../METACOGNITION.md), [current registration](../prereg.md),
[accepted recovery proposal](../after-the-fix-problem-statement.pdf),
`results/meta_*`, `results/figures/meta_*`, M mock runs, downloaded model and
dataset stay in the main project. New work should use new arm names and output
directories there.
