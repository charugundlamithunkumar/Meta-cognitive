"""Pilot report on calibration runs: choose K, tau; HA upper bound; audit health.

usage: python3 scripts/pilot_report.py runs/<run_dir> [more run dirs]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.labels import assign_seeds  # noqa: E402
from src.metrics import auroc, youden_threshold, turns_to_first_error  # noqa: E402

BINS = [-1, 49, 99, 199, 10**9]
BIN_LABELS = ["0-49", "50-99", "100-199", "200+"]


def load(run_dir):
    rows = [json.loads(l) for l in open(Path(run_dir) / "rows.jsonl")]
    d = pd.DataFrame(rows)
    d["bin"] = pd.cut(d.turn, BINS, labels=BIN_LABELS)
    d["lp_min"] = d.lp.apply(lambda x: x["lp_min"])
    d["local_ok"] = d.kind.isin(["ok", "inherited"]).astype(int)
    return d


def report(run_dir):
    d = load(run_dir)
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    print(f"\n=== {run_dir}  arm={cfg['arm']} K={cfg['K']} T={cfg['T']} n_traj={d.sample_id.nunique()}")
    print("accuracy vs truth, all turns:", round((d.kind == "ok").mean(), 3))
    ttf = d.sort_values("turn").groupby("sample_id").kind.apply(lambda k: turns_to_first_error(list(k)))
    print("turns to first error: median", ttf.median(), "min", ttf.min(), "max", ttf.max())
    print("\nkind counts by position bin:")
    print(pd.crosstab(d.bin, d.kind).to_string())

    # logit signal: numeric slips vs locally-correct steps
    print("\nAUROC (lp_min) numeric_slip vs locally-correct, by bin:")
    for b, g in d.groupby("bin", observed=True):
        g = g[g.kind.isin(["ok", "inherited", "numeric_slip"])]
        print(f"  {b}: n_slip={int((g.kind == 'numeric_slip').sum())} auroc={auroc(g.local_ok, g.lp_min):.3f}")
    g = d[d.kind.isin(["ok", "inherited", "numeric_slip"])]
    tau, j = youden_threshold(g.local_ok, g.lp_min)
    trig = (d.lp_min < tau).mean() if tau is not None else float("nan")
    print(f"\nYouden tau on lp_min = {tau}  (J={j:.3f}); trigger rate on all turns = {trig:.3f}")

    # HA upper bound: which inherited errors trace to a seed the gates would catch
    caught_fmt = caught_logit = total_inh = 0
    seed_kinds = Counter()
    for sid, g in d.sort_values("turn").groupby("sample_id"):
        kinds = list(g.kind)
        seeds = assign_seeds(kinds)
        lps = list(g.lp_min)
        for i, (k, s) in enumerate(zip(kinds, seeds)):
            if k == "inherited" and s is not None:
                total_inh += 1
                if kinds[s] == "nonanswer":
                    caught_fmt += 1
                    caught_logit += 1
                elif tau is not None and lps[s] < tau:
                    caught_logit += 1
            if k in ("numeric_slip", "nonanswer"):
                seed_kinds[k] += 1
    if total_inh:
        print(f"HA upper bound: inherited errors={total_inh}; seeded by a format-catchable error: "
              f"{caught_fmt / total_inh:.3f}; by format-or-logit-catchable: {caught_logit / total_inh:.3f}")
    print("seed-type counts:", dict(seed_kinds))

    if "probes" in d:
        p = d.probes.dropna()
        if len(p):
            pt = d.probes.apply(lambda x: x.get("ptrue") if isinstance(x, dict) else np.nan)
            vb = d.probes.apply(lambda x: x.get("verbal") if isinstance(x, dict) else np.nan)
            print(f"\nP(Yes) AUROC slip vs local-correct: {auroc(g.local_ok, pt[g.index]):.3f};  "
                  f"verbal AUROC: {auroc(g.local_ok, vb[g.index].astype(float)):.3f}; verbal values: "
                  f"{Counter(vb.dropna().astype(int)).most_common(4)}")
    if "audit" in d:
        a = pd.DataFrame([dict(x, turn=t) for x, t in zip(d.audit, d.turn) if isinstance(x, dict)])
        if len(a):
            print(f"\naudits: n={len(a)} unfinished={1 - a.finished.mean():.3f} "
                  f"think_tokens median={a.think_tokens.median():.0f} max={a.think_tokens.max():.0f} "
                  f"actions={dict(Counter(a.action))}")
            f = a[a.finished]
            if len(f):
                from src.metrics import rates
                r = rates(f.actually_wrong, f.verdict == "WRONG")
                print(f"  finished audits: TPR={r['tpr']:.3f} FPR={r['fpr']:.3f} J={r['j']:.3f} "
                      f"(n_wrong={r['n_wrong']}, n_ok={r['n_ok']})")
                c = f[f.action == "correct"]
                if len(c):
                    print(f"  corrections: n={len(c)}, auditor total equals truth: "
                          f"{c.auditor_total_correct.mean():.3f}")
    if "gate" in d:
        gg = pd.DataFrame([x for x in d.gate if isinstance(x, dict)])
        print(f"\ngate fired rate={gg.fired.mean():.3f} reasons={dict(Counter(gg.reason.dropna()))}")
    print(f"non-answer share={((d.kind == 'nonanswer').mean()):.3f}  wellformed share={d.wellformed.mean():.3f}")
    print(f"elapsed: {(Path(run_dir) / 'run.log').read_text().strip().splitlines()[-1]}")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        report(r)
