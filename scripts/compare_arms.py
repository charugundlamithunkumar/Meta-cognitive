"""Paired comparison of two arms on the same trajectories (first T turns of each).

usage: python3 scripts/compare_arms.py BASE_RUN TREAT_RUN [--T 100]
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import turns_to_first_error, wrong_run_lengths, auroc  # noqa: E402
from src.stats import paired_bootstrap_diff, bootstrap_mean  # noqa: E402


def load(run, T):
    by = {}
    with open(Path(run) / "rows.jsonl") as f:
        for l in f:
            r = json.loads(l)
            if r["turn"] < T:
                by.setdefault(r["sample_id"], []).append(r)
    return {k: sorted(v, key=lambda r: r["turn"]) for k, v in by.items()}


def per_traj(rows):
    k = [r["kind"] for r in rows]
    fired = [r for r in rows if r.get("gate", {}).get("fired")]
    return dict(acc=np.mean([x == "ok" for x in k]), ttfe=turns_to_first_error(k),
                slip=np.mean([x == "numeric_slip" for x in k]), inh=np.mean([x == "inherited" for x in k]),
                fire=len(fired) / len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("treat"); ap.add_argument("--T", type=int, default=100)
    a = ap.parse_args()
    A, B = load(a.base, a.T), load(a.treat, a.T)
    ids = sorted(set(A) & set(B))
    assert all(len(A[i]) == a.T and len(B[i]) == a.T for i in ids), "incomplete trajectories"
    print(f"paired trajectories: {len(ids)}, turns 0-{a.T - 1}")
    pa, pb = [per_traj(A[i]) for i in ids], [per_traj(B[i]) for i in ids]
    for m, name in [("acc", "accuracy vs truth (all turns)"), ("ttfe", "turns to first error"),
                    ("slip", "fresh numeric-slip rate (committed)"), ("inh", "inherited-error rate")]:
        x, y = [p[m] for p in pa], [p[m] for p in pb]
        d = paired_bootstrap_diff(y, x)
        print(f"{name:38s} base {np.mean(x):7.3f}  treat {np.mean(y):7.3f}  diff {d['diff']:+.3f} "
              f"[95% CI {d['lo']:+.3f}, {d['hi']:+.3f}]")
    print(f"{'gate fire rate (treat)':38s} {np.mean([p['fire'] for p in pb]):.3f}")
    fired = [r for i in ids for r in B[i] if r.get("gate", {}).get("fired")]
    if fired:
        rec_ok = np.mean([r["gate"].get("rec_value") == r["prev"] + r["v"] for r in fired])
        orig_slip = np.mean([r["value"] is not None and r["value"] != r["prev"] + r["v"] for r in fired])
        print(f"{'fired turns: original reply was a slip':38s} {orig_slip:.3f}")
        print(f"{'fired turns: re-compute locally correct':38s} {rec_ok:.3f}")
        print(f"{'fired reasons':38s} {dict(Counter(r['gate']['reason'] for r in fired))}")
    allB = [r for i in ids for r in B[i]]
    slips_orig = [r for r in allB if r["value"] is not None and r["value"] != r["prev"] + r["v"]]
    caught = [r for r in slips_orig if r.get("gate", {}).get("fired")]
    if slips_orig:
        print(f"{'gate recall on original slips':38s} {len(caught) / len(slips_orig):.3f} ({len(caught)}/{len(slips_orig)})")
