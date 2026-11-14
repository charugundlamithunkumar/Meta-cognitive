"""Secondary / exploratory analyses (prereg.md §5). Not used for any verdict.

usage: python3 scripts/secondary.py --stamp 20261113-173033  -> results/secondary.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from analyze_main import load, IDS, BINS  # noqa: E402
from src.metrics import auroc  # noqa: E402

ARMS = ["A0", "A0_inline", "A1a", "A1b", "A2", "D1", "D2", "A3", "A4"]


def first_err(rows):
    return next((r["turn"] for r in rows if r["kind"] != "ok"), 300)


def gate_stats(R, arm):
    rows = [r for i in IDS for r in R[arm][i] if r.get("gate")]
    slip = [r for r in rows if r["value"] != r["prev"] + r["v"]]          # original reply locally wrong
    good = [r for r in rows if r["value"] == r["prev"] + r["v"]]
    fired = [r for r in rows if r["gate"]["fired"]]
    rec_ok = [r for r in fired if r["gate"].get("rec_value") == r["prev"] + r["v"]]
    return dict(n_turns=len(rows), fire_rate=len(fired) / len(rows),
                recall_on_local_slips=float(np.mean([r["gate"]["fired"] for r in slip])) if slip else None,
                false_trigger_rate=float(np.mean([r["gate"]["fired"] for r in good])) if good else None,
                recompute_locally_correct=len(rec_ok) / len(fired) if fired else None,
                recompute_unfinished=float(np.mean([not r["gate"].get("rec_finished", True) for r in fired])) if fired else None)


def slip_before_after(R, arm):
    """Local slip rate (reply != prev + v) before vs after the first error."""
    pre, post = [], []
    for i in IDS:
        rows = R[arm][i]; f = first_err(rows)
        for r in rows:
            (pre if r["turn"] < f else post).append(r["kind"] == "numeric_slip")
    return dict(before=float(np.mean(pre)) if pre else None, n_before=len(pre),
                after=float(np.mean(post)), n_after=len(post))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stamp", required=True); a = ap.parse_args()
    R = {arm: load(a.stamp, arm) for arm in ARMS}
    out = {}
    out["turns_to_first_error"] = {arm: float(np.mean([first_err(R[arm][i]) for i in IDS])) for arm in ARMS}
    out["slip_rate_by_bin"] = {arm: [float(np.mean([r["kind"] == "numeric_slip" for i in IDS for r in R[arm][i]
                                                    if lo <= r["turn"] <= hi])) for lo, hi in BINS] for arm in ARMS}
    out["gate"] = {arm: gate_stats(R, arm) for arm in ["A1a", "A1b", "A2"]}
    out["slip_before_after_first_error"] = {arm: slip_before_after(R, arm) for arm in ["A0", "D1", "D2"]}
    same = [R["A0"][i][t]["value"] == R["A1a"][i][t]["value"] for i in IDS for t in range(300)]
    out["A0_vs_A1a_identical_reply_share"] = float(np.mean(same))
    out["A0_vs_A1a_identical_first_10_turns"] = float(np.mean([R["A0"][i][t]["value"] == R["A1a"][i][t]["value"]
                                                              for i in IDS for t in range(10)]))
    # P(Yes) and verbal-confidence AUROC (A0 probes), locally correct vs numeric slip
    rows = [r for i in IDS for r in R["A0"][i] if r["kind"] in ("ok", "inherited", "numeric_slip") and r.get("probes")]
    y = [int(r["kind"] != "numeric_slip") for r in rows]
    out["probe_auroc"] = {k: auroc([yy for yy, r in zip(y, rows) if r["probes"].get(k) is not None],
                                   [r["probes"][k] for r in rows if r["probes"].get(k) is not None])
                          for k in ("ptrue", "verbal")}
    out["probe_n"] = len(rows)
    ci = [r["conf_inline"] for i in IDS for r in R["A0_inline"][i] if r.get("conf_inline") is not None]
    out["inline_confidence"] = dict(n=len(ci), mean=float(np.mean(ci)), share_100=float(np.mean([c == 100 for c in ci])),
                                    quantiles=[float(q) for q in np.quantile(ci, [0.05, 0.25, 0.5, 0.75, 0.95])])
    out["mean_generated_tokens_per_turn"] = {}
    for arm in ARMS:
        tot = []
        for i in IDS:
            for r in R[arm][i]:
                t = r.get("gen_tokens") or 0
                t += (r.get("gate") or {}).get("rec_tokens") or 0
                t += (r.get("audit") or {}).get("gen_tokens") or 0
                tot.append(t)
        out["mean_generated_tokens_per_turn"][arm] = float(np.mean(tot))
    out["retrieval_like_share_of_slips"] = {arm: float(np.mean([bool(r.get("retrieval_like")) for i in IDS for r in R[arm][i]
                                                                if r["kind"] == "numeric_slip"])) for arm in ["A0", "D2"]}
    (ROOT / "results" / "secondary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
