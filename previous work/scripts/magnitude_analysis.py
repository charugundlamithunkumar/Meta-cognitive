"""Exploratory (post hoc, Study 1): does turn position predict a new slip once the size of the total is
controlled? Logistic regression of slip (committed != prev + v) on turn/100 and digits(|prev + v|),
fitted per arm on Study-1 rows; 95% CIs by bootstrapping whole trajectories (1,000 resamples, seed 0).

usage: python3 scripts/magnitude_analysis.py --stamp 20261113-173033  -> results/magnitude_analysis.json
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import statsmodels.api as sm
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from analyze_main import load, IDS  # noqa: E402
warnings.filterwarnings("ignore")


def design(R, arm, ids):
    X, y = [], []
    for i in ids:
        for r in R[arm][i]:
            target = r["prev"] + r["v"]
            X.append([r["turn"] / 100, len(str(abs(target)))])
            y.append(int(r["committed"] != target))
    return np.asarray(X, float), np.asarray(y)


def fit(X, y):
    m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    return m.params[1:]          # [turn/100, digits]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stamp", required=True); a = ap.parse_args()
    out = {}
    for arm in ["D2", "A0", "D1"]:
        R = {arm: load(a.stamp, arm)}
        X, y = design(R, arm, IDS)
        pt = fit(X, y)
        # turn-only model for comparison
        pt_turn_only = sm.Logit(y, sm.add_constant(X[:, :1])).fit(disp=0).params[1]
        rng = np.random.default_rng(0)
        bs, bs1 = [], []
        for _ in range(1000):
            ids = list(rng.choice(IDS, len(IDS)))
            Xb, yb = design(R, arm, ids)
            try:
                bs.append(fit(Xb, yb))
                bs1.append(sm.Logit(yb, sm.add_constant(Xb[:, :1])).fit(disp=0).params[1])
            except Exception:
                pass
        bs = np.array(bs)
        q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
        out[arm] = dict(n_rows=int(len(y)), slip_rate=float(y.mean()),
                        turn_only_log_odds_per_100_turns=dict(point=float(pt_turn_only), ci=q(bs1)),
                        adjusted_log_odds_per_100_turns=dict(point=float(pt[0]), ci=q(bs[:, 0])),
                        log_odds_per_extra_digit=dict(point=float(pt[1]), ci=q(bs[:, 1])),
                        n_boot=int(len(bs)))
        print(arm, json.dumps(out[arm]))
    (ROOT / "results" / "magnitude_analysis.json").write_text(json.dumps(out, indent=2))
