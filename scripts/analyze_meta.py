"""Exploratory A0 metacognition analysis; reads archives without extracting them.

python3 scripts/analyze_meta.py [--source "previous work/results/raw_main_<stamp>/A0.tar.gz"]
All reported CIs resample whole trajectories. No model or GPU is loaded.
"""
import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.study_common import BINS
from src.metrics import auroc, MIN_PER_CLASS
from src.stats import bootstrap_mean
from src.lhe_data import sha256, SPLITS

CHECKPOINTS = [10, 25, 50, 100, 150, 200, 250]
FEATURES = ['verbal', 'ptrue', 'lp_min', 'lp_mean', 'neg_ent_max', 'margin_min', 'fin_lp_min']


def read_bundle(source):
    """Read the established config/rows/trajectories layout, including tar.gz."""
    source = Path(source)
    names = ['config.json', 'rows.jsonl', 'trajectories.jsonl']
    if source.is_dir():
        contents = {n: (source / n).read_text() for n in names}
    else:
        with tarfile.open(source, 'r:gz') as tf:
            contents = {}
            for n in names:
                members = [m for m in tf.getmembers() if m.isfile() and Path(m.name).name == n]
                if len(members) != 1:
                    raise ValueError(f'{source}: expected exactly one {n}')
                contents[n] = tf.extractfile(members[0]).read().decode()
    return (json.loads(contents['config.json']),
            [json.loads(s) for s in contents['rows.jsonl'].splitlines() if s.strip()],
            [json.loads(s) for s in contents['trajectories.jsonl'].splitlines() if s.strip()])


def complete_a0(source):
    cfg, rows, summaries = read_bundle(source)
    if cfg['arm'] != 'A0':
        raise ValueError('A0 input required')
    done = {s['sample_id'] for s in summaries if s['T'] == 300}
    by = {}
    for r in rows:
        by.setdefault(r['sample_id'], []).append(r)
    keep = {}
    for sid in SPLITS['test']:
        rr = sorted(by.get(sid, []), key=lambda r: r['turn'])
        if sid in done and [r['turn'] for r in rr] == list(range(300)):
            keep[sid] = rr
    if not keep:
        raise ValueError('No complete A0 test trajectories')
    return keep, sorted(set(SPLITS['test']) - keep.keys())


def next_accuracy(rows, turn, window=10):
    future = {r['turn']: r for r in rows if turn < r['turn'] <= turn + window}
    if set(future) != set(range(turn + 1, turn + window + 1)):
        return np.nan
    return float(np.mean([r['kind'] == 'ok' for r in future.values()]))


def records(by):
    out = []
    for sid, rr in sorted(by.items()):
        prior_error = False
        for r in sorted(rr, key=lambda r: r['turn']):
            p = r.get('probes', {})
            x = dict(r, prior_error=prior_error,
                     inclusive_error=prior_error or r['kind'] != 'ok',
                     realized=next_accuracy(rr, r['turn']))
            for k in ['verbal', 'ptrue', 'fore_none', 'fore_true', 'fore_placebo']:
                x[k] = p.get(k) if p.get(k) is not None else np.nan
                if k.startswith('fore_'):
                    x[k] = np.clip(x[k], 0, 100)  # existing HC percentage normalization
            aware = p.get('aware')
            for j, k in enumerate(['higher', 'same', 'lower']):
                x[k] = aware[j] if aware is not None and len(aware) == 3 else np.nan
            out.append(x)
            prior_error = x['inclusive_error']
    return out


def summary(rr, key, n_boot=10000):
    by = {}
    for r in rr:
        val = r.get(key, np.nan)
        if val is not None and np.isfinite(val):
            by.setdefault(r['sample_id'], []).append(val)
    vals = [np.mean(v) for v in by.values()]
    result = bootstrap_mean(vals, n_boot=n_boot) if vals else dict(mean=np.nan, lo=np.nan, hi=np.nan)
    return dict(result, n=sum(map(len, by.values())), n_trajectories=len(by))


def matched_history(rr, n_boot=10000, inclusive=False):
    """Exact-turn matching; the current outcome is excluded from history by default."""
    ids = sorted({r['sample_id'] for r in rr})
    index = {sid: i for i, sid in enumerate(ids)}
    rng = np.random.default_rng(0)
    weights = rng.multinomial(len(ids), [1 / len(ids)] * len(ids), size=n_boot)
    groups = np.full((61, 2, len(ids)), np.nan)
    field = 'inclusive_error' if inclusive else 'prior_error'
    for r in rr:
        if 0 <= r['turn'] <= 60:
            groups[r['turn'], int(r[field]), index[r['sample_id']]] = r['verbal']
    cells, diffs, boot_diffs = [], [], []
    for t in range(61):
        bs = []
        for g, name in enumerate(['clean', 'error']):
            vals = groups[t, g]
            valid = np.isfinite(vals)
            n = int(valid.sum())
            denom = weights @ valid.astype(float)
            boot = np.divide(weights @ np.nan_to_num(vals), denom,
                             out=np.full(n_boot, np.nan), where=denom > 0)
            finite = boot[np.isfinite(boot)]
            lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            cells.append(dict(turn=t, history=name, n=n, mean=float(np.nanmean(vals)) if n else np.nan,
                              lo=float(lo), hi=float(hi), sparse=n < 30))
            bs.append(boot)
        if all(np.isfinite(groups[t, g]).any() for g in [0, 1]):
            diffs.append(float(np.nanmean(groups[t, 1]) - np.nanmean(groups[t, 0])))
            boot_diffs.append(bs[1] - bs[0])
    # Require the SAME matched positions in each replicate; report support loss.
    boots = np.mean(boot_diffs, axis=0) if boot_diffs else np.array([])
    boots = boots[np.isfinite(boots)]
    ci = np.quantile(boots, [.025, .975]) if len(boots) else [np.nan, np.nan]
    return dict(definition=field, cells=cells, matched_turns=len(diffs),
                error_minus_clean=float(np.mean(diffs)) if diffs else np.nan,
                lo=float(ci[0]), hi=float(ci[1]), valid_bootstraps=len(boots),
                warning='Sparse overlap; descriptive association, not causal identification.')


def feature(r, name):
    if name in ['verbal', 'ptrue']:
        return r.get('probes', {}).get(name)
    if name == 'fin_lp_min':
        return r.get('lp_fin', {}).get('lp_min')
    if name == 'neg_ent_max':
        v = r.get('lp', {}).get('ent_max')
        return -v if v is not None else None
    return r.get('lp', {}).get(name)


def auc_summary(rows, name, n_boot=10000, min_per_class=MIN_PER_CLASS, return_samples=False):
    """Exact pooled AUROC bootstrap via trajectory-pair concordance counts.

    Precomputing concordances avoids sorting all rows for every resample. Ties
    receive half credit, exactly as src.metrics.auroc / sklearn.
    """
    ids = sorted({r['sample_id'] for r in rows})
    by = {i: [[], []] for i in ids}
    for r in rows:
        s = feature(r, name)
        if r['kind'] in ('ok', 'inherited', 'numeric_slip') and s is not None and np.isfinite(s):
            by[r['sample_id']][int(r['kind'] != 'numeric_slip')].append(float(s))
    neg = [np.sort(by[i][0]) for i in ids]
    pos = [np.asarray(by[i][1]) for i in ids]
    nneg, npos = np.array(list(map(len, neg))), np.array(list(map(len, pos)))
    y = [y for i in ids for y in [0, 1] for _ in by[i][y]]
    s = [s for i in ids for y in [0, 1] for s in by[i][y]]
    point = auroc(y, s, min_per_class=min_per_class)
    if not np.isfinite(point):
        report = dict(point=np.nan, lo=np.nan, hi=np.nan, n_local=int(npos.sum()), n_slip=int(nneg.sum()),
                      valid_bootstraps=0)
        return (report, np.full(n_boot, np.nan)) if return_samples else report
    concordance = np.array([[np.sum((np.searchsorted(n, p, side='left') +
                                     np.searchsorted(n, p, side='right')) / 2)
                             for n in neg] for p in pos])
    w = np.random.default_rng(0).multinomial(len(ids), [1 / len(ids)] * len(ids), size=n_boot)
    np_, nn_ = w @ npos, w @ nneg
    mask = (np_ >= min_per_class) & (nn_ >= min_per_class)
    samples = np.einsum('bi,ij,bj->b', w[mask], concordance, w[mask]) / (np_[mask] * nn_[mask])
    ci = np.quantile(samples, [.025, .975]) if len(samples) else [np.nan, np.nan]
    report = dict(point=point, lo=float(ci[0]), hi=float(ci[1]), n_local=int(npos.sum()),
                  n_slip=int(nneg.sum()), valid_bootstraps=len(samples))
    aligned = np.full(n_boot, np.nan)
    aligned[mask] = samples
    return (report, aligned) if return_samples else report


def analyze(by, n_boot=10000):
    rr = records(by)
    keys = ['verbal', 'ptrue', 'fore_none', 'fore_true', 'fore_placebo', 'higher', 'same', 'lower', 'realized']
    bins = []
    for lo, hi in BINS:
        selected = [r for r in rr if lo <= r['turn'] <= hi]
        bins.append(dict(bin=f'{lo}-{hi}', **{k: summary(selected, k, n_boot) for k in keys},
                         matched_realized={k: summary([r for r in selected if np.isfinite(r[k])], 'realized', n_boot)
                                           for k in keys if k != 'realized'}))
    cps = [dict(turn=c, **{k: summary([r for r in rr if r['turn'] == c], k, n_boot) for k in keys})
           for c in CHECKPOINTS]
    aucs = {label: {k: auc_summary([r for r in rr if lo <= r['turn'] <= hi], k, n_boot) for k in FEATURES}
            for label, lo, hi in [('all', 0, 299)] + [(f'{lo}-{hi}', lo, hi) for lo, hi in BINS]}
    return dict(n_trajectories=len(by), n_rows=len(rr), n_boot=n_boot, bins=bins, checkpoints=cps,
                matched=matched_history(rr, n_boot), matched_inclusive=matched_history(rr, n_boot, True), auc=aucs)


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def fmt(x, digits=2):
    return f'{x:.{digits}f}' if x is not None and np.isfinite(x) else 'NA'


def write_report(res, out):
    lines = ['# A0 metacognition (exploratory)', '',
             f"Source: `{res['source']}`; SHA-256: `{res['sha256']}`.",
             f"{res['n_trajectories']} complete trajectories; {res['n_rows']} rows. Missing ids: {res['missing_ids']}.",
             f"Whole-trajectory bootstrap: {res['n_boot']} resamples, seed 0; 95% percentile intervals.", '',
             'Turns are zero-based. Next-10 means t+1…t+10; incomplete windows are missing, never zero-filled.',
             'Means weight trajectories equally. P(Yes) is normalized over Yes/No first-token mass; awareness is normalized A/B/C mass.',
             'Forecasts are clipped to [0,100] as in HC. All checkpoints are included here, including equal true/placebo counts; HC used a restricted subset.', '',
             '| Bin | Verbal | P(Yes) | Forecast none | True | Placebo | Higher | Same | Lower | Next-10 accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ['verbal', 'ptrue', 'fore_none', 'fore_true', 'fore_placebo', 'higher', 'same', 'lower', 'realized']
    for b in res['bins']:
        lines.append('| ' + b['bin'] + ' | ' + ' | '.join(fmt(b[k]['mean']) for k in keys) + ' |')
    lines += ['', 'Bin CIs and per-metric counts are in `meta_analysis.json`. Forecast/awareness use only checkpoint rows;',
              'verbal/P(Yes) use all turns; next-10 accuracy excludes the final ten turns.',
              'The JSON also gives realized accuracy on the rows available for each signal (`matched_realized`).', '',
              '| Checkpoint | n forecast | Verbal | None | True | Placebo | Next-10 accuracy |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for c in res['checkpoints']:
        lines.append(f"| {c['turn']} | {c['fore_none']['n']} | " +
                     ' | '.join(fmt(c[k]['mean']) for k in ['verbal', 'fore_none', 'fore_true', 'fore_placebo', 'realized']) + ' |')
    lines += ['', '![Position](figures/meta_position.png)', '![Calibration](figures/meta_calibration.png)', '',
              '## Slip-step localization', '',
              'Positive = locally correct (ok + inherited); negative = numeric_slip; nonanswers excluded.',
              'Higher scores always indicate confidence; entropy is negated. Minimum 15 rows per class.', '',
              '| Position | Score | AUROC [95% CI] | n local / slip |', '|---|---|---:|---:|']
    for b, features in res['auc'].items():
        for k, v in features.items():
            lines.append(f"| {b} | {k} | {fmt(v['point'],3)} [{fmt(v['lo'],3)}, {fmt(v['hi'],3)}] | {v['n_local']} / {v['n_slip']} |")
    lines += ['', '## Exact-position matched histories, turns 0–60', '',
              'Clean = all answers strictly before this turn are correct; error = at least one earlier wrong answer.',
              'Each cell counts trajectories at exactly one turn. `*` flags n<30. Empty cells are NA.', '']
    for key, label in [('matched', 'Primary history definition'), ('matched_inclusive', 'Including current answer (sensitivity)')]:
        m = res[key]
        lines.append(f"{label}: error − clean = {fmt(m['error_minus_clean'])} [{fmt(m['lo'])}, {fmt(m['hi'])}], "
                     f"equal-weighted over {m['matched_turns']} overlapping turns; {m['valid_bootstraps']} valid bootstrap replicates.")
    lines += ['', '| Turn | Clean n | Clean mean [CI] | Error n | Error mean [CI] |', '|---|---:|---:|---:|---:|']
    cells = res['matched']['cells']
    for a, b in zip(cells[::2], cells[1::2]):
        def cell(c):
            return f"{c['n']}{'*' if c['sparse'] else ''} | {fmt(c['mean'])} [{fmt(c['lo'])}, {fmt(c['hi'])}]"
        lines.append(f"| {a['turn']} | {cell(a)} | {cell(b)} |")
    lines += ['', '![Matched histories](figures/meta_matched.png)', '',
              'Position, accumulated errors, and realized accuracy are confounded in A0. Sparse clean-history overlap',
              'prevents a causal interpretation. Falling confidence is not evidence that the model can locate a slip.',
              'M is needed to separate controlled history length and error dose; no M model measurements are reported here.', '']
    lines += ['Numerical interpretation: pooled verbal AUROC is ' + fmt(res['auc']['all']['verbal']['point'], 3)
              + ', while the four within-bin values are '
              + ', '.join(fmt(res['auc'][f'{lo}-{hi}']['verbal']['point'], 3) for lo, hi in BINS)
              + '. Coarse bins still permit position confounding; neither pooled nor binned AUROC establishes causal self-monitoring.', '']
    out.write_text('\n'.join(lines))


def plots(res, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for k in ['verbal', 'ptrue', 'realized']:
        scale = 100 if k in ['ptrue', 'realized'] else 1
        axs[0].plot([b['bin'] for b in res['bins']], [b[k]['mean'] * scale for b in res['bins']], 'o-', label=k)
    cp = res['checkpoints']
    for k in ['fore_none', 'fore_true', 'fore_placebo', 'realized']:
        axs[1].plot([c['turn'] for c in cp], [c[k]['mean'] * (100 if k == 'realized' else 1) for c in cp], 'o-', label=k)
    for k in ['higher', 'same', 'lower']:
        axs[2].plot([c['turn'] for c in cp], [100 * c[k]['mean'] for c in cp], 'o-', label=k)
    axs[2].plot([c['turn'] for c in cp], [100*c['realized']['mean'] for c in cp], '--', color='black', label='realized')
    for ax in axs:
        ax.set_ylim(-3, 103); ax.set_xlabel('Position (zero-based)'); ax.legend(fontsize=8)
        ax.set_ylabel('Percent / confidence points'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'meta_position.png', dpi=180); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, k in zip(axs, ['fore_none', 'fore_true', 'fore_placebo']):
        ax.plot([0, 100], [0, 100], '--', color='grey')
        grouped = {}
        for c in cp:
            xy = (c[k]['mean'], 100 * c['realized']['mean'])
            grouped.setdefault(xy, []).append(str(c['turn']))
        for j, ((x, y), labels) in enumerate(grouped.items()):
            ax.scatter(x, y)
            ax.annotate(', '.join(labels), (x, y), xytext=(4, 5 + 12 * j),
                        textcoords='offset points', fontsize=8,
                        arrowprops=dict(arrowstyle='-', color='grey', lw=.5))
        ax.set(xlabel='Mean forecast (%)', title=k, xlim=(-5, 105), ylim=(-5, 105))
    axs[0].set_ylabel('Mean realized next-10 accuracy (%)')
    fig.tight_layout(); fig.savefig(out / 'meta_calibration.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for g in ['clean', 'error']:
        cc = [c for c in res['matched']['cells'] if c['history'] == g and c['n']]
        x = [c['turn'] for c in cc]; y = [c['mean'] for c in cc]
        line, = ax.plot(x, y, label=g)
        ax.fill_between(x, [c['lo'] for c in cc], [c['hi'] for c in cc], color=line.get_color(), alpha=.15)
        sparse = [c for c in cc if c['sparse']]
        ax.scatter([c['turn'] for c in sparse], [c['mean'] for c in sparse], marker='x', color=line.get_color(), s=18)
    ax.set(xlabel='Exact turn (zero-based; x marks n<30)', ylabel='Verbal confidence', ylim=(-3, 103))
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'meta_matched.png', dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--out', type=Path, default=ROOT / 'results')
    ap.add_argument('--n-boot', type=int, default=10000)
    a = ap.parse_args()
    if a.n_boot < 1:
        ap.error('--n-boot must be positive')
    if a.source is None:
        sources = sorted((ROOT / 'previous work/results').glob('raw_main_*/A0.tar.gz'))
        if len(sources) != 1:
            ap.error('Specify --source: expected exactly one A0 archive')
        a.source = sources[0]
    by, missing = complete_a0(a.source)
    result = analyze(by, a.n_boot)
    result.update(source=str(a.source), sha256=sha256(a.source if a.source.is_file() else a.source / 'rows.jsonl'), missing_ids=missing)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'meta_analysis.json').write_text(json.dumps(clean_json(result), indent=2, allow_nan=False) + '\n')
    write_report(result, a.out / 'meta_analysis.md')
    plots(result, a.out / 'figures')
    print('Wrote', a.out / 'meta_analysis.md')
    print('Checkpoint fore_none:', [(c['turn'], c['fore_none']['mean']) for c in result['checkpoints']])


if __name__ == '__main__':
    main()
