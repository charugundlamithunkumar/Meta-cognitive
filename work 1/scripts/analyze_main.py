"""Pre-registered analysis of the main runs (prereg.md §3-§4). Writes results/main_results.{md,json}.

usage: python3 scripts/analyze_main.py --stamp 20261113-173033
Arms that are missing or unfinished are skipped; their hypotheses are reported as 'pending'.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import auroc, rates  # noqa: E402
from src.stats import paired_bootstrap_diff, holm, N_BOOT  # noqa: E402
from src.lhe_data import SPLITS  # noqa: E402

IDS = SPLITS["test"]
BINS = [(0, 49), (50, 99), (100, 199), (200, 299)]
RNG_SEED = 0


def load(stamp, arm):
    d = ROOT / "runs" / f"main_{arm}_{stamp}"
    if not (d / "trajectories.jsonl").exists():
        return None
    with open(d / "trajectories.jsonl") as f:
        done = {json.loads(l)["sample_id"] for l in f}
    if set(IDS) - done:
        return None
    by = {}
    with open(d / "rows.jsonl") as f:
        for l in f:
            r = json.loads(l)
            by.setdefault(r["sample_id"], []).append(r)
    by = {k: sorted(v, key=lambda r: r["turn"]) for k, v in by.items()}
    assert all(len(by[i]) == 300 for i in IDS), f"{arm}: incomplete trajectories"
    return by


def acc(rows):
    return float(np.mean([r["kind"] == "ok" for r in rows]))


def chains(rows):
    return sum(r["kind"] in ("numeric_slip", "nonanswer") for r in rows)


def boot_stat(fn, n=N_BOOT, seed=RNG_SEED):
    """fn(list_of_ids) -> statistic; bootstrap over trajectories. Returns (point, lo, hi, samples)."""
    rng = np.random.default_rng(seed)
    point = fn(IDS)
    s = []
    for _ in range(n):
        v = fn(list(rng.choice(IDS, len(IDS))))
        if v is not None and not np.isnan(v):
            s.append(v)
    s = np.asarray(s)
    return point, float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975)), s


def p_two_sided(samples, null=0.0):
    return float(min(1.0, 2 * min((samples <= null).mean(), (samples >= null).mean())))


def fmt(d):
    return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def ha(R):
    A0, A1b, A2 = R.get("A0"), R.get("A1b"), R.get("A2")
    if not (A0 and A1b and A2):
        return dict(status="pending")
    out = {}
    for name, B in [("A2", A2), ("A0", A0)]:
        out[f"acc_A1b_minus_{name}"] = paired_bootstrap_diff([acc(A1b[i]) for i in IDS], [acc(B[i]) for i in IDS])
        out[f"chains_A1b_minus_{name}"] = paired_bootstrap_diff([chains(A1b[i]) for i in IDS], [chains(B[i]) for i in IDS])
    if R.get("A1a"):
        out["acc_A1b_minus_A1a"] = paired_bootstrap_diff([acc(A1b[i]) for i in IDS], [acc(R["A1a"][i]) for i in IDS])

    def gate_auc(ids):  # live gate AUROC on A1b's ORIGINAL replies (before any re-compute)
        rows = [r for i in ids for r in A1b[i] if r["value"] is not None]
        return auroc([int(r["value"] == r["prev"] + r["v"]) for r in rows], [r["lp"]["lp_min"] for r in rows])
    pt, lo, hi, s = boot_stat(gate_auc)
    out["gate_auroc"] = dict(point=pt, lo=lo, hi=hi)
    ok = (out["acc_A1b_minus_A2"]["lo"] > 0 and out["acc_A1b_minus_A0"]["lo"] > 0
          and out["chains_A1b_minus_A2"]["hi"] < 0 and out["chains_A1b_minus_A0"]["hi"] < 0 and lo > 0.5)
    out["p"] = max(out[k]["p"] for k in ["acc_A1b_minus_A2", "acc_A1b_minus_A0", "chains_A1b_minus_A2", "chains_A1b_minus_A0"])
    out["verdict_uncorrected"] = "supported" if ok else "not supported"
    return out


def j_by_bin(arm_rows, ids, b):
    lo, hi = BINS[b]
    a = [r["audit"] for i in ids for r in arm_rows[i] if "audit" in r and lo <= r["turn"] <= hi and r["audit"]["finished"]]
    if not a or all(x["actually_wrong"] for x in a) or not any(x["actually_wrong"] for x in a):
        return np.nan
    return rates([x["actually_wrong"] for x in a], [x["verdict"] == "WRONG" for x in a])["j"]


def hb(R):
    A0, A3, A4 = R.get("A0"), R.get("A3"), R.get("A4")
    if not (A0 and A3 and A4):
        return dict(status="pending")
    out = {}
    for name, X in [("A3", A3), ("A4", A4)]:
        out[f"J_bins_{name}"] = [j_by_bin(X, IDS, b) for b in range(4)]
        pt, lo, hi, s = boot_stat(lambda ids, X=X: j_by_bin(X, ids, 3) - j_by_bin(X, ids, 0))
        out[f"delta_{name}"] = dict(point=pt, lo=lo, hi=hi, p=p_two_sided(s))
        aud = [r["audit"] for i in IDS for r in X[i] if "audit" in r]
        out[f"unfinished_{name}"] = float(np.mean([not x["finished"] for x in aud]))
    out["acc_A3_minus_A0"] = paired_bootstrap_diff([acc(A3[i]) for i in IDS], [acc(A0[i]) for i in IDS])
    a3_ok = out["delta_A3"]["lo"] <= 0 <= out["delta_A3"]["hi"] or out["delta_A3"]["lo"] > 0
    a4_decl = out["delta_A4"]["hi"] < 0
    acc_ok = out["acc_A3_minus_A0"]["lo"] > 0
    out["p"] = max(out["acc_A3_minus_A0"]["p"], out["delta_A4"]["p"])
    if a3_ok and a4_decl and acc_ok:
        out["verdict_uncorrected"] = "supported"
    elif a3_ok and acc_ok:
        out["verdict_uncorrected"] = "partial: A3 works, in-history decline not shown"
    else:
        out["verdict_uncorrected"] = "not supported"
    return out


def hc(R):
    A0 = R.get("A0")
    if not A0:
        return dict(status="pending")
    pts = {}
    for i in IDS:
        rows = A0[i]
        for r in rows:
            h = r.get("hc")
            if not h or h["n_true"] == h["n_placebo"]:
                continue
            pr = r["probes"]
            c = r["turn"]
            realized = float(np.mean([x["kind"] == "ok" for x in rows[c + 1:c + 11]]))
            f = {k: min(pr[f"fore_{k}"], 100) / 100 for k in ("none", "true", "placebo") if pr.get(f"fore_{k}") is not None}
            if len(f) == 3:
                pts.setdefault(i, []).append(dict(realized=realized, **f))
    ids = [i for i in IDS if i in pts]
    mae = {k: [float(np.mean([abs(p[k] - p["realized"]) for p in pts[i]])) for i in ids] for k in ("none", "true", "placebo")}
    out = dict(n_traj=len(ids), n_checkpoints=sum(len(v) for v in pts.values()),
               mae={k: float(np.mean(v)) for k, v in mae.items()},
               true_minus_none=paired_bootstrap_diff(mae["true"], mae["none"]),
               true_minus_placebo=paired_bootstrap_diff(mae["true"], mae["placebo"]))
    rng = np.random.default_rng(RNG_SEED)

    def rho(sel):
        x = [p["true"] for i in sel for p in pts[i]]
        y = [p["realized"] for i in sel for p in pts[i]]
        if len(set(x)) < 2 or len(set(y)) < 2:
            return np.nan
        return spearmanr(x, y).statistic
    s = np.array([rho(list(rng.choice(ids, len(ids)))) for _ in range(N_BOOT)])
    s = s[~np.isnan(s)]
    out["spearman_true"] = dict(point=float(rho(ids)), lo=float(np.quantile(s, 0.025)), hi=float(np.quantile(s, 0.975)),
                                p=p_two_sided(s))
    ok = (out["true_minus_none"]["hi"] < 0 and out["true_minus_placebo"]["hi"] < 0 and out["spearman_true"]["lo"] > 0)
    out["p"] = max(out["true_minus_none"]["p"], out["true_minus_placebo"]["p"], out["spearman_true"]["p"])
    out["verdict_uncorrected"] = "supported" if ok else "not supported"
    return out


def hd(R):
    A0, D1, D2 = R.get("A0"), R.get("D1"), R.get("D2")
    if not (A0 and D1 and D2):
        return dict(status="pending")
    d = [acc(D1[i]) - acc(A0[i]) for i in IDS]
    rng = np.random.default_rng(RNG_SEED)
    bs = np.array([np.mean(rng.choice(d, len(d))) for _ in range(N_BOOT)])
    lo90, hi90 = float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))
    slopes = [float(np.polyfit(np.arange(300), [int(r["kind"] == "ok") for r in D2[i]], 1)[0]) for i in IDS]
    bs2 = np.array([np.mean(rng.choice(slopes, len(slopes))) for _ in range(N_BOOT)])
    slo, shi = float(np.quantile(bs2, 0.025)), float(np.quantile(bs2, 0.975))
    out = dict(acc_D1_minus_A0=dict(diff=float(np.mean(d)), lo90=lo90, hi90=hi90),
               D2_slope=dict(mean=float(np.mean(slopes)), lo=slo, hi=shi),
               acc=dict(A0=float(np.mean([acc(A0[i]) for i in IDS])), D1=float(np.mean([acc(D1[i]) for i in IDS])),
                        D2=float(np.mean([acc(D2[i]) for i in IDS]))))
    equiv = -0.05 < lo90 and hi90 < 0.05
    flat = -0.0005 < slo and shi < 0.0005
    # TOST p = max of the two one-sided bootstrap p-values
    out["p"] = float(max((bs <= -0.05).mean(), (bs >= 0.05).mean()) * 2) if equiv else 1.0
    out["verdict_uncorrected"] = "supported" if (equiv and flat) else "not supported"
    return out


def he(R):
    A0, AI = R.get("A0"), R.get("A0_inline")
    if not (A0 and AI):
        return dict(status="pending")

    def au(arm_rows, key, ids):
        rows = [r for i in ids for r in arm_rows[i] if r["kind"] in ("ok", "inherited", "numeric_slip")]
        return auroc([int(r["kind"] != "numeric_slip") for r in rows], [r[key]["lp_min"] for r in rows])
    p0, lo0, hi0, _ = boot_stat(lambda ids: au(A0, "lp", ids))
    pt, lo, hi, s = boot_stat(lambda ids: au(AI, "lp_pre", ids) - au(A0, "lp", ids))
    out = dict(auroc_A0=dict(point=p0, lo=lo0, hi=hi0), auroc_inline_pre=au(AI, "lp_pre", IDS),
               auroc_inline_fin=au(AI, "lp_fin", IDS), diff_inline_pre_minus_A0=dict(point=pt, lo=lo, hi=hi))
    if hi < 0:
        inside = lo0 <= out["auroc_inline_fin"] <= hi0
        out["condition"] = True
        out["p"] = p_two_sided(s)
        out["verdict_uncorrected"] = "supported" if inside else "not supported"
    else:
        out["condition"] = False
        out["verdict_uncorrected"] = "discrepancy not reproduced on this task"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    a = ap.parse_args()
    R = {arm: load(a.stamp, arm) for arm in ["A0", "A0_inline", "A1a", "A1b", "A2", "D1", "D2", "A3", "A4"]}
    avail = [k for k, v in R.items() if v]
    res = dict(stamp=a.stamp, arms_complete=avail, HA=ha(R), HB=hb(R), HC=hc(R), HD=hd(R), HE=he(R))
    tested = [h for h in ["HA", "HB", "HC", "HD", "HE"] if "p" in res[h]]
    complete = all("status" not in res[h] for h in ["HA", "HB", "HC", "HD", "HE"])
    if tested:
        adj = holm([res[h]["p"] for h in tested])
        for h, pa in zip(tested, adj):
            res[h]["p_holm"] = pa
            if not complete:
                res[h]["note"] = "Holm adjustment is provisional until all hypotheses are complete"
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "main_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({h: {k: v for k, v in res[h].items() if k in ("status", "verdict_uncorrected", "p", "p_holm")}
                      for h in ["HA", "HB", "HC", "HD", "HE"]}, indent=1))
    print("arms complete:", avail)
