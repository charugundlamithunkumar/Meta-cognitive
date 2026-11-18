# Saved main and confirmation data

These 22 archives are byte-identical copies of the completed real experiments,
validated against `../all_experiments_completion_20261118.json`.

- `main_M/`: original seven metacognition conditions.
- `main_MR/`: original four recovery conditions.
- `C_M/`: seven conditions on 32 fresh confirmation sequences.
- `C_MR/`: four recovery conditions on those fresh sequences.

The archives contain the exported per-arm records, including `rows.jsonl`.
Original checkpoints, runtime logs, and source snapshots remain saved locally
under `runs/`, which is excluded from Git. Model weights and the large dataset
are also excluded. Calibration and exploratory data are separate from these
49,280 main/confirmation task answers.

Verify from this directory:

```bash
sha256sum -c SHA256SUMS
```
