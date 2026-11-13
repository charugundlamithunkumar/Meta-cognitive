"""Metrics (implementation-plan.md §8)."""
import math
import numpy as np
from sklearn.metrics import roc_auc_score

MIN_PER_CLASS = 15  # minimum items per class for a reported AUROC (pre-registered choice)


def auroc(is_ok, score, min_per_class=MIN_PER_CLASS):
    """Type-2 AUROC: P(score of a correct item > score of a wrong item). NaN if too few."""
    y = np.asarray(is_ok, int)
    s = np.asarray(score, float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if len(y) == 0 or min(y.sum(), (1 - y).sum()) < min_per_class:
        return float("nan")
    return float(roc_auc_score(y, s))


def rates(is_wrong, flagged):
    """TPR = flagged among wrong; FPR = flagged among correct; J = TPR - FPR."""
    w = np.asarray(is_wrong, bool)
    f = np.asarray(flagged, bool)
    tpr = f[w].mean() if w.any() else float("nan")
    fpr = f[~w].mean() if (~w).any() else float("nan")
    return dict(tpr=float(tpr), fpr=float(fpr), j=float(tpr - fpr), n_wrong=int(w.sum()), n_ok=int((~w).sum()))


def youden_threshold(is_ok, score):
    """Threshold t maximizing J when flagging items with score < t."""
    y = np.asarray(is_ok, int)
    s = np.asarray(score, float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    best = (-1.0, None)
    for t in np.unique(s):
        flagged = s < t
        r = rates(y == 0, flagged)
        if not math.isnan(r["j"]) and r["j"] > best[0]:
            best = (r["j"], float(t))
    return best[1], best[0]


def turns_to_first_error(kinds):
    for i, k in enumerate(kinds):
        if k != "ok":
            return i
    return len(kinds)


def wrong_run_lengths(kinds):
    runs, cur = [], 0
    for k in kinds:
        if k != "ok":
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def token_stats(chosen_lps, top_lps_list):
    """min/mean logprob of chosen tokens; max entropy and min top1-top2 margin over tokens."""
    if not chosen_lps:
        nan = float("nan")
        return dict(lp_min=nan, lp_mean=nan, ent_max=nan, margin_min=nan)
    ents, margins = [], []
    for top in top_lps_list:
        lp = np.sort(np.asarray(top, float))[::-1]
        p = np.exp(lp)
        pn = p / p.sum()
        ents.append(float(-(pn * np.log(pn + 1e-12)).sum()))
        margins.append(float(lp[0] - lp[1]) if len(lp) > 1 else 20.0)
    return dict(lp_min=float(min(chosen_lps)), lp_mean=float(np.mean(chosen_lps)),
                ent_max=float(max(ents)), margin_min=float(min(margins)))
