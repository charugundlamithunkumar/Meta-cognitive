"""Paired bootstrap over trajectories and Holm correction (implementation-plan.md §8)."""
import numpy as np

N_BOOT = 10_000


def paired_bootstrap_diff(a, b, n_boot=N_BOOT, seed=0, alpha=0.05):
    """a, b: per-trajectory metric for the same ids in two arms. Returns mean(a-b) and CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return dict(diff=float(d.mean()), lo=float(lo), hi=float(hi), p=float(min(p, 1.0)))


def bootstrap_mean(a, n_boot=N_BOOT, seed=0, alpha=0.05):
    a = np.asarray(a, float)
    rng = np.random.default_rng(seed)
    boots = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return dict(mean=float(a.mean()), lo=float(lo), hi=float(hi))


def holm(pvals):
    """Holm-adjusted p-values, same order as input."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()
