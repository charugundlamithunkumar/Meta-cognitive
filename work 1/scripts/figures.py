"""Figures for the main runs (Phase 5). Reads runs/main_<arm>_<stamp>/ and results/main_results.json.

usage: python3 scripts/figures.py --stamp 20261113-173033
Writes results/figures/*.png. Figure CIs are trajectory-bootstrap 95% percentile intervals with 2,000
resamples (seed 0); the verdict CIs in results/main_results.json use 10,000 (prereg.md §3).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_main import load, acc, IDS, BINS, j_by_bin  # noqa: E402
from src.metrics import auroc  # noqa: E402

NB = 2000
ARMS = ["A0", "A0_inline", "A1a", "A1b", "A2", "D1", "D2", "A3", "A4"]
# Colour follows the arm in every figure. Baseline = neutral grey; treatments take categorical slots in fixed order
# (validated: scripts in the dataviz reference palette, light mode, adjacent pairs pass; low-contrast hues get direct labels).
COL = {"A0": "#7d7c78", "A1a": "#b4b3ad", "A1b": "#2a78d6", "A2": "#eb6834", "A3": "#1baf7a", "A4": "#eda100",
       "D1": "#e87ba4", "D2": "#008300", "A0_inline": "#6250d6"}
NAME = {"A0": "A0 baseline", "A0_inline": "A0_inline (total | conf)", "A1a": "A1a format gate", "A1b": "A1b logit gate",
        "A2": "A2 random gate", "D1": "D1 own-total compaction", "D2": "D2 true-total compaction",
        "A3": "A3 fresh audit", "A4": "A4 in-history audit"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "lines.linewidth": 2, "legend.frameon": False})


def boot(fn, n=NB, seed=0):
    rng = np.random.default_rng(seed)
    s = [fn(list(rng.choice(IDS, len(IDS)))) for _ in range(n)]
    s = np.array([x for x in s if x is not None and not np.isnan(x)])
    return float(fn(IDS)), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def rolling(x, w=10):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def per_turn_acc(R, arm):
    return np.mean([[r["kind"] == "ok" for r in R[arm][i]] for i in IDS], axis=0)


def label_end(ax, y_by_arm, x_end, min_gap):
    """Direct labels at the right end, nudged apart so they never overlap."""
    items = sorted(y_by_arm.items(), key=lambda kv: kv[1])
    placed = []
    for arm, y in items:
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(arm, (x_end, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)


def fig_accuracy_by_turn(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, arms, title in [(axs[0], ["A0", "A1b", "A2", "A3", "A4"], "(a) In-loop interventions"),
                            (axs[1], ["A0", "D1", "D2"], "(b) Compaction controls (no history)")]:
        ends = {}
        for arm in arms:
            y = rolling(per_turn_acc(R, arm))
            ax.plot(np.arange(300), y, color=COL[arm], label=NAME[arm], lw=2 if arm != "A0" else 1.6)
            ends[arm] = y[-1]
        label_end(ax, ends, 299, 0.035)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Turn")
        ax.set_xlim(0, 330)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8.5)
    axs[0].set_ylabel("Accuracy vs truth\n(mean of 32 trajectories, 10-turn rolling)")
    fig.tight_layout()
    fig.savefig(out / "fig1_accuracy_by_turn.png", dpi=200)
    plt.close(fig)


def fig_accuracy_per_arm(R, out):
    rows = []
    for arm in ARMS:
        pt, lo, hi = boot(lambda ids, a=arm: float(np.mean([acc(R[a][i]) for i in ids])))
        rows.append((arm, pt, lo, hi))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (arm, pt, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [k, k], color=COL[arm], lw=2, solid_capstyle="round")
        ax.plot(pt, k, "o", ms=8, color=COL[arm], mec=SURF, mew=2)
        ax.annotate(f"{pt:.3f}", (hi, k), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [NAME[a] for a, *_ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy vs truth over 300 turns (95% CI over trajectories)")
    ax.set_xlim(0, 0.72)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_accuracy_per_arm.png", dpi=200)
    plt.close(fig)
    return rows


def fig_error_types(R, out):
    kinds = [("ok", "ok (correct)", "#2a78d6"), ("inherited", "inherited (= prev + v, wrong)", "#eb6834"),
             ("numeric_slip", "numeric slip (new error)", "#1baf7a"), ("nonanswer", "non-answer", "#6250d6")]
    share = {arm: {k: np.mean([r["kind"] == k for i in IDS for r in R[arm][i]]) for k, *_ in kinds} for arm in ARMS}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k_i, arm in enumerate(ARMS):
        left = 0.0
        for k, lab, c in kinds:
            w = share[arm][k]
            if w > 0:
                ax.barh(k_i, w, left=left, height=0.62, color=c, edgecolor=SURF, linewidth=2,
                        label=lab if k_i == 0 else None)
                if w >= 0.06:
                    ax.text(left + w / 2, k_i, f"{w:.2f}", ha="center", va="center", fontsize=8, color="white")
            left += w
    ax.set_yticks(range(len(ARMS)), [NAME[a] for a in ARMS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of turns (32 trajectories x 300 turns)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.4, 1.17), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig3_error_types.png", dpi=200)
    plt.close(fig)
    return share


def fig_error_types_by_turn(R, out, w=10):
    """Plan §7 Phase 5 figure 2: share of each error type vs. turn, one small panel per arm (10-turn windows)."""
    kinds = [("ok", "ok", "#2a78d6"), ("inherited", "inherited", "#eb6834"), ("numeric_slip", "numeric slip", "#1baf7a")]
    fig, axs = plt.subplots(3, 3, figsize=(11, 8), sharex=True, sharey=True)
    x = np.arange(0, 300, w) + w / 2
    for ax, arm in zip(axs.flat, ARMS):
        ys = [[np.mean([r["kind"] == k for i in IDS for r in R[arm][i][t:t + w]]) for t in range(0, 300, w)]
              for k, *_ in kinds]
        ax.stackplot(x, ys, colors=[c for *_, c in kinds], labels=[l for _, l, _ in kinds], edgecolor=SURF, linewidth=0.8)
        ax.set_title(NAME[arm], loc="left", fontsize=10)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 1)
        ax.grid(False)
    for ax in axs[-1]:
        ax.set_xlabel("Turn")
    for ax in axs[:, 0]:
        ax.set_ylabel("Share of turns")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig3b_error_types_by_turn.png", dpi=200)
    plt.close(fig)


def fig_auditor(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.arange(4)
    lab = [f"{a}-{b}" for a, b in BINS]
    res = {}
    for off, arm in [(-0.09, "A3"), (0.09, "A4")]:
        js, unf = [], []
        for b in range(4):
            js.append(boot(lambda ids, a=arm, b=b: j_by_bin(R[a], ids, b)))
            lo, hi = BINS[b]
            au = [r["audit"] for i in IDS for r in R[arm][i] if "audit" in r and lo <= r["turn"] <= hi]
            unf.append(np.mean([not x["finished"] for x in au]))
        res[arm] = dict(J=js, unfinished=unf)
        pts = np.array(js)
        axs[0].errorbar(xs + off, pts[:, 0], yerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]], fmt="o-",
                        color=COL[arm], ms=7, capsize=0, mec=SURF, mew=1.5, label=NAME[arm])
        axs[1].plot(xs + off, unf, "o-", color=COL[arm], ms=7, mec=SURF, mew=1.5, label=NAME[arm])
        axs[0].annotate(arm, (xs[-1] + off, pts[-1, 0]), xytext=(10, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
        axs[1].annotate(arm, (xs[-1] + off, unf[-1]), xytext=(10, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
    axs[0].set_title("(a) Auditor J = TPR - FPR (finished audits)", loc="left", fontsize=11)
    axs[0].set_ylim(0.6, 1.02)
    axs[1].set_title("(b) Share of audits that did not finish (8,192-token budget)", loc="left", fontsize=11)
    axs[1].set_ylim(0, 0.25)
    for ax in axs:
        ax.set_xticks(xs, lab)
        ax.set_xlabel("Turn bin")
        ax.set_xlim(-0.4, 3.6)
        ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig4_auditor_by_bin.png", dpi=200)
    plt.close(fig)
    return res


def hc_points(A0):
    pts = {}
    for i in IDS:
        rows = A0[i]
        for r in rows:
            h = r.get("hc")
            if not h or h["n_true"] == h["n_placebo"]:
                continue
            pr = r["probes"]
            realized = float(np.mean([x["kind"] == "ok" for x in rows[r["turn"] + 1:r["turn"] + 11]]))
            f = {k: min(pr[f"fore_{k}"], 100) / 100 for k in ("none", "true", "placebo") if pr.get(f"fore_{k}") is not None}
            if len(f) == 3:
                pts.setdefault(i, []).append(dict(realized=realized, **f))
    return pts


def fig_hc(R, out):
    pts = hc_points(R["A0"])
    ids = [i for i in IDS if i in pts]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    res = {}
    rng = np.random.default_rng(0)
    for k, cond in enumerate(["none", "true", "placebo"]):
        per = np.array([np.mean([abs(p[cond] - p["realized"]) for p in pts[i]]) for i in ids])
        bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(NB)]
        pt, lo, hi = per.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)
        res[cond] = (float(pt), float(lo), float(hi))
        axs[0].plot([lo, hi], [k, k], color="#2a78d6", lw=2, solid_capstyle="round")
        axs[0].plot(pt, k, "o", ms=8, color="#2a78d6", mec=SURF, mew=2)
        axs[0].annotate(f"{pt:.2f}", (hi, k), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    axs[0].set_yticks(range(3), ["no evidence", "true checker count", "placebo count"])
    axs[0].invert_yaxis()
    axs[0].set_xlim(0, 1)
    axs[0].grid(axis="y", visible=False)
    axs[0].set_xlabel("MAE |forecast - realized next-10 accuracy|")
    axs[0].set_title(f"(a) Forecast error ({sum(len(v) for v in pts.values())} checkpoints, {len(ids)} trajectories)",
                     loc="left", fontsize=11)
    allp = [p for i in ids for p in pts[i]]
    g = np.random.default_rng(1)
    jx, jy = g.uniform(-0.015, 0.015, len(allp)), g.uniform(-0.015, 0.015, len(allp))
    names = {"none": "no evidence", "true": "true count", "placebo": "placebo count"}
    for cond, c, mk in [("placebo", "#eb6834", "^"), ("true", "#2a78d6", "o")]:
        axs[1].scatter([p["realized"] + j for p, j in zip(allp, jx)], [p[cond] + j for p, j in zip(allp, jy)], s=34,
                       marker=mk, color=c, edgecolor=SURF, linewidth=1, label=names[cond], zorder=3)
    axs[1].scatter([p["realized"] + j for p, j in zip(allp, jx)], [p["none"] - j for p, j in zip(allp, jy)], s=60,
                   marker="s", facecolor="none", edgecolor="#52514e", linewidth=1.4, label=names["none"], zorder=4)
    axs[1].plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", zorder=1)
    axs[1].set_xlim(-0.05, 1.05)
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].set_xlabel("Realized accuracy over the next 10 turns")
    axs[1].set_ylabel("Forecast / 100")
    axs[1].set_title("(b) Each checkpoint (dashed = perfect)", loc="left", fontsize=11)
    axs[1].legend(loc="center right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig5_hc_forecasts.png", dpi=200)
    plt.close(fig)
    return res


def fig_auroc(R, out, main):
    def au(arm, key, ids):
        rows = [r for i in ids for r in R[arm][i] if r["kind"] in ("ok", "inherited", "numeric_slip")]
        return auroc([int(r["kind"] != "numeric_slip") for r in rows], [r[key]["lp_min"] for r in rows])
    items = [("A0 bare integer", "A0", boot(lambda ids: au("A0", "lp", ids))),
             ("A0_inline, tokens before '|'", "A0_inline", boot(lambda ids: au("A0_inline", "lp_pre", ids))),
             ("A0_inline, final-integer tokens", "A0_inline", boot(lambda ids: au("A0_inline", "lp_fin", ids)))]
    g = main["HA"]["gate_auroc"]
    items.append(("A1b live gate (original replies)", "A1b", (g["point"], g["lo"], g["hi"])))
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for k, (lab, arm, (pt, lo, hi)) in enumerate(items):
        ax.plot([lo, hi], [k, k], color=COL[arm], lw=2, solid_capstyle="round")
        ax.plot(pt, k, "o", ms=8, color=COL[arm], mec=SURF, mew=2)
        ax.annotate(f"{pt:.3f}", (hi, k), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(items)), [x[0] for x in items])
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Type-2 AUROC of min answer-token logprob (95% CI; dashed = chance)")
    fig.tight_layout()
    fig.savefig(out / "fig6_logit_auroc.png", dpi=200)
    plt.close(fig)
    return {lab: v for lab, _, v in items}


def fig_d2_magnitude(R, out):
    """Exploratory (not pre-registered): D2 accuracy by turn bin, split by the size of the true running total."""
    groups = [("|total| < 1000", lambda m: m < 1000, "#2a78d6"), ("|total| >= 1000", lambda m: m >= 1000, "#eb6834")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    res = {}
    xs = np.arange(4)
    for off, (lab, sel, c) in zip([-0.08, 0.08], groups):
        ys, ns = [], []
        for lo, hi in BINS:
            r = [x["kind"] == "ok" for i in IDS for x in R["D2"][i] if lo <= x["turn"] <= hi and sel(abs(x["prev"]))]
            ys.append(np.mean(r) if len(r) >= 20 else np.nan)
            ns.append(len(r))
        res[lab] = dict(acc=ys, n=ns)
        ax.plot(xs + off, ys, "o-", color=c, ms=7, mec=SURF, mew=1.5, label=lab)
        for x, y, n in zip(xs + off, ys, ns):
            if not np.isnan(y):
                ax.annotate(f"n={n}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8,
                            color=INK2)
    allacc = [np.mean([x["kind"] == "ok" for i in IDS for x in R["D2"][i] if lo <= x["turn"] <= hi]) for lo, hi in BINS]
    ax.plot(xs, allacc, "--", color="#7d7c78", lw=1.4, label="all turns")
    res["all"] = allacc
    ax.set_xticks(xs, [f"{a}-{b}" for a, b in BINS])
    ax.set_xlabel("Turn bin")
    ax.set_ylabel("D2 accuracy (true total supplied)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("Exploratory: D2 accuracy by size of the total", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig7_d2_by_magnitude.png", dpi=200)
    plt.close(fig)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    a = ap.parse_args()
    R = {arm: load(a.stamp, arm) for arm in ARMS}
    missing = [k for k, v in R.items() if not v]
    if missing:
        sys.exit(f"missing or unfinished arms: {missing}")
    main = json.loads((ROOT / "results" / "main_results.json").read_text())
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_accuracy_by_turn(R, out)
    summary = dict(
        accuracy_per_arm={a: dict(point=p, lo=l, hi=h) for a, p, l, h in fig_accuracy_per_arm(R, out)},
        error_type_shares=fig_error_types(R, out),
        _error_types_by_turn=fig_error_types_by_turn(R, out),
        auditor=fig_auditor(R, out),
        hc_mae=fig_hc(R, out),
        logit_auroc=fig_auroc(R, out, main),
        d2_by_magnitude_exploratory=fig_d2_magnitude(R, out),
    )
    (ROOT / "results" / "figure_data.json").write_text(json.dumps(summary, indent=2, default=float))
    print("wrote", sorted(p.name for p in out.glob("*.png")))
