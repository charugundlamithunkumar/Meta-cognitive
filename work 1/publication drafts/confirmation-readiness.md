# Independent confirmation readiness

18 November 2026. **Audit complete; no confirmation allocation or inference.**

The metadata-only audit read 178 config records from loose runs and compressed
archives under `runs/`, `previous work/runs/` and `previous work/results/`.
It parsed no outcome payloads and reported no metadata errors. Recorded possible
real exposures include published ids 0–51 and fresh ids 10000–10019, corresponding
to fresh seeds 0–19 under the existing generator convention.

No configs in the audited artifacts recorded reserve ids 52–99 or fresh seeds
1000–1047. Those ranges were already reserved in the historical extension plan;
absence from local config metadata does not release that reservation or prove
absence of deleted, external, or human-inspected uses. Do not call them an
untouched confirmation sample without reconciling that allocation.

Machine-readable evidence:
[`confirmation_allocation_audit_20261118.json`](../results/confirmation_allocation_audit_20261118.json).
Reproduce in a new output file:

```bash
python3 paper/tools/audit_allocations.py --out results/confirmation_allocation_audit_next.json
python3 -m unittest discover -s paper/tools -v
```

## Gates for an independent study

1. Reconcile historical reservations and any external uses without selecting
   samples from their outcomes. Alternatively choose documented fresh seeds
   outside known reservations and uses.
2. Freeze the sample manifest, generation seed mapping, checkpoint set,
   hypotheses, missingness rules, multiplicity and N before confirmation
   inference. Use calibration-based precision with conservative variance;
   main M results must not determine a favorable subset or relaxed margin.
3. Give confirmation a new registration and new arm/run names. The current
   M and MR CLIs deliberately reject reserve/fresh main inputs. Explicit
   manifest support belongs in a separately reviewed runner after the frozen
   queue completes; it has not been implemented here.
4. Obtain the required new long-run GPU budget approval, then collect and
   report all allocated conditions regardless of outcome.

The current main studies remain reportable without an independent replication,
with scope clearly limited. No new GPU run or changed sample split is
authorized by this audit alone.
