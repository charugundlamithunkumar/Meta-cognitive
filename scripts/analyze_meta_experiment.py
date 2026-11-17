"""HF analysis for M outputs. Mock inputs are labelled and never yield a study verdict.

python3 scripts/analyze_meta_experiment.py runs/meta_confirmatory
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze_meta import read_bundle, clean_json, CHECKPOINTS, FEATURES, auc_summary, feature, fmt
from src.lhe_data import SPLITS
from src.meta_history import ARMS
from src.meta_runner import validate_checkpoint
from src.stats import paired_bootstrap_diff, holm


def decision(length, dose, endpoint, n_boot=10000):
    """Input vectors contain paired trajectory means (percentage points)."""
    length, dose, endpoint = map(lambda x: np.asarray(x, float), [length, dose, endpoint])
    if not len(length) or not (length.shape == dose.shape == endpoint.shape):
        raise ValueError('Expected nonempty paired trajectory vectors')
    if not all(np.isfinite(x).all() for x in [length, dose, endpoint]):
        return dict(status='unidentifiable: invalid primary probes', verdict='inconclusive')
    indices = np.random.default_rng(0).integers(0, len(length), size=(n_boot, len(length)))
    tests, contrasts = {}, {}
    for name, values in [('length', length), ('pessimism', -length), ('dose', dose), ('endpoint', endpoint)]:
        contrasts[name] = paired_bootstrap_diff(values, np.zeros_like(values), n_boot=n_boot)
    for name, values in [('pessimism', -length), ('dose', dose)]:
        bs = values[indices].mean(axis=1)
        ci95 = np.quantile(bs, [.025, .975]); ci90 = np.quantile(bs, [.05, .95])
        superiority = min(1., 2 * min(np.mean(bs <= 10), np.mean(bs >= 10)))
        equivalence = max(np.mean(bs <= -10), np.mean(bs >= 10))
        tests[name + '_gt10'] = dict(p=float(superiority), ci_rule=bool(ci95[0] > 10))
        tests[name + '_equiv'] = dict(p=float(equivalence), ci_rule=bool(-10 < ci90[0] and ci90[1] < 10),
                                    lo90=float(ci90[0]), hi90=float(ci90[1]))
    adjusted = holm([v['p'] for v in tests.values()])
    for test, adj in zip(tests.values(), adjusted):
        test.update(p_holm=adj, supported=bool(test['ci_rule'] and adj < .05))
    yes = lambda k: tests[k]['supported']
    if yes('pessimism_gt10') and yes('dose_gt10'):
        verdict = 'mixed'
    elif yes('pessimism_gt10') and yes('dose_equiv'):
        verdict = 'length-driven'
    elif yes('dose_gt10') and yes('pessimism_equiv'):
        verdict = 'monitoring-driven (error-sensitive reports)'
    else:
        verdict = 'inconclusive'
    return dict(verdict=verdict, contrasts=contrasts, tests=tests)


def calibration(points, signal, n_boot=10000):
    """Realized accuracy ~ report + checkpoint fixed effects; trajectory bootstrap."""
    points = [p for p in points if np.isfinite(p[signal]) and np.isfinite(p['realized'])]
    ids = sorted({p['sample_id'] for p in points})
    cps = sorted({p['checkpoint'] for p in points})
    if not points:
        return dict(status='unidentifiable: no valid observations', n=0)
    width = len(cps) + 1
    matrices, vectors, ysum, ysq, counts = [], [], [], [], []
    for sid in ids:
        pp = [p for p in points if p['sample_id'] == sid]
        x = np.array([[1., p[signal]] + [float(p['checkpoint'] == c) for c in cps[1:]] for p in pp])
        y = np.array([p['realized'] for p in pp])
        matrices.append(x.T @ x); vectors.append(x.T @ y)
        ysum.append(y.sum()); ysq.append(y @ y); counts.append(len(pp))
    matrix, vector = np.sum(matrices, axis=0), np.sum(vectors, axis=0)
    if np.linalg.matrix_rank(matrix) < width or np.ptp([p['realized'] for p in points]) == 0:
        return dict(status='unidentifiable: constant outcome or no within-checkpoint predictor variation', n=len(points))
    coef = np.linalg.solve(matrix, vector)
    weights = np.random.default_rng(0).multinomial(len(ids), [1 / len(ids)] * len(ids), size=n_boot)
    mats = np.einsum('bi,ijk->bjk', weights, np.array(matrices))
    vecs = weights @ np.array(vectors)
    varies = weights @ np.array(ysq) - (weights @ np.array(ysum))**2 / (weights @ np.array(counts)) > 1e-12
    good = (np.linalg.matrix_rank(mats) == width) & varies
    bs = np.linalg.solve(mats[good], vecs[good, :, None])[:, :, 0]
    ci = np.quantile(bs, [.025, .975], axis=0) if len(bs) else np.full((2, width), np.nan)
    return dict(status='estimated', n=len(points), n_trajectories=len(ids), valid_bootstraps=len(bs),
                slope=float(coef[1]), lo=float(ci[0, 1]), hi=float(ci[1, 1]),
                intercept=float(coef[0]), intercept_lo=float(ci[0, 0]), intercept_hi=float(ci[1, 0]),
                reference_checkpoint=cps[0], checkpoint_effects=dict(zip(map(str, cps[1:]), coef[2:].tolist())))


def probe_value(row, key):
    v = row['probes'].get(key)
    if key == 'aware':
        if v is None or len(v) != 3 or not np.isfinite(v).all() or not np.isclose(sum(v), 1):
            return np.nan
        return float(v[0] + .5*v[1])
    return float(v) if v is not None and np.isfinite(v) and 0 <= v <= 100 else np.nan


def analyze_suite(root, n_boot=10000):
    root = Path(root)
    bundles, missing = {}, []
    for arm in ARMS:
        name = 'M_' + arm
        source = root / name if (root / name).is_dir() else root / f'{name}.tar.gz'
        if not source.exists():
            missing.append(name); continue
        cfg, rows, summaries = read_bundle(source)
        if cfg['arm'] != name:
            raise ValueError('Arm mismatch')
        grouped = {}
        for r in rows:
            grouped.setdefault((r['sample_id'], r['checkpoint']), []).append(r)
        done_ids = {s['sample_id'] for s in summaries if s['n_checkpoints'] == len(cfg['checkpoints'])}
        complete = {}
        for sid in cfg['ids']:
            if sid not in done_ids:
                continue
            if not all((sid, c) in grouped for c in cfg['checkpoints']):
                continue
            for c in cfg['checkpoints']:
                validate_checkpoint(grouped[sid, c], sid, c, name)
            complete[sid] = {c: grouped[sid, c] for c in cfg['checkpoints']}
        bundles[arm] = (cfg, complete)
    if missing:
        return dict(status='pending: missing arms', missing_arms=missing)
    cfg0 = bundles['L0'][0]
    for cfg, _ in bundles.values():
        for key in ['ids', 'checkpoints', 'mock', 'model_revision', 'source_sha256']:
            if cfg[key] != cfg0[key]:
                raise ValueError(f'Incompatible paired runs: {key}')
    ids = sorted(set.intersection(*(set(b[1]) for b in bundles.values())))
    result = dict(mock=cfg0['mock'], ids=ids, missing_ids=sorted(set(cfg0['ids']) - set(ids)),
                  n_boot=n_boot, checkpoints=cfg0['checkpoints'])
    if not ids:
        return dict(result, status='pending: no complete paired trajectories')
    cp = cfg0['checkpoints']
    confidence = {a: np.array([[probe_value(bundles[a][1][i][c][0], 'verbal') for c in cp] for i in ids]) for a in ARMS}
    dose_x = np.array([0, .25, .5, .75]); centered = dose_x - dose_x.mean()
    slope = np.einsum('a,aij->ij', centered, np.array([confidence[a] for a in ['L0', 'L25', 'L50', 'L75']])) / (centered @ centered)
    numeric = decision((confidence['L0'] - confidence['S0']).mean(axis=1), -slope.mean(axis=1),
                       (confidence['L0'] - confidence['L100']).mean(axis=1), n_boot)
    confirmatory = (not cfg0['mock'] and all(b[0].get('confirmatory') for b in bundles.values())
                    and ids == SPLITS['test'] and cp == CHECKPOINTS and n_boot == 10000)
    result['status'] = 'confirmatory' if confirmatory else ('MOCK VALIDATION ONLY' if cfg0['mock'] else 'incomplete / exploratory')
    result['HF'] = numeric if confirmatory else dict(numeric, verdict='not issued: ' + result['status'], numerical_rule_output=numeric.get('verdict'))
    means, points, aucs = [], [], []
    for arm in ARMS:
        for c in cp:
            rows = [bundles[arm][1][i][c][0] for i in ids]
            acc = float(np.mean([r['realized_next10'] for r in rows]))
            values = {k: [probe_value(r, k) for r in rows] for k in ['verbal', 'fore_none', 'aware']}
            means.append(dict(arm=arm, checkpoint=c, n=len(rows), realized=acc, floor=acc <= .05,
                              **{k: float(np.mean(v)) for k, v in values.items()},
                              invalid={k: int(np.sum(~np.isfinite(v))) for k, v in values.items()}))
            for r in rows:
                points.append(dict(sample_id=r['sample_id'], checkpoint=c, arm=arm, realized=r['realized_next10'],
                                   verbal=probe_value(r, 'verbal') / 100, fore_none=probe_value(r, 'fore_none') / 100,
                                   aware=probe_value(r, 'aware')))
            # Exact checkpoint AND generated offset: no silent pooling across positions.
            for offset in range(1, 11):
                rr = [bundles[arm][1][i][c][offset] for i in ids]
                both = [r for r in rr if all(feature(r, k) is not None and np.isfinite(feature(r, k)) for k in ['verbal', 'lp_min'])]
                features = {k: auc_summary(rr, k, n_boot) for k in FEATURES if k not in ['verbal', 'lp_min']}
                verbal, vb = auc_summary(both, 'verbal', n_boot, return_samples=True)
                logit, lb = auc_summary(both, 'lp_min', n_boot, return_samples=True)
                features.update(verbal=verbal, lp_min=logit)
                delta = vb - lb
                delta = delta[np.isfinite(delta)]
                ci = np.quantile(delta, [.025, .975]) if len(delta) else [np.nan, np.nan]
                aucs.append(dict(arm=arm, checkpoint=c, offset=offset, features=features,
                                 verbal_minus_logit=dict(point=verbal['point'] - logit['point'],
                                                        lo=float(ci[0]), hi=float(ci[1]), valid_bootstraps=len(delta))))
    result['means'] = means
    result['dose_monotone'] = {str(c): bool(all(confidence[a][:, j].mean() >= confidence[b][:, j].mean()
                                                    for a, b in zip(['L0', 'L25', 'L50'], ['L25', 'L50', 'L75'])))
                               for j, c in enumerate(cp)}
    result['intermediates_all_at_floor'] = all(m['floor'] for m in means if m['arm'] in ['L25', 'L50', 'L75'])
    result['calibration'] = {k: calibration(points, k, n_boot) for k in ['verbal', 'fore_none', 'aware']}
    result['fixed_position_auc'] = aucs
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_root', type=Path)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--out', type=Path, help='Defaults to run_root/analysis (never overwrites A0 results)')
    a = ap.parse_args()
    if a.n_boot < 1:
        ap.error('--n-boot must be positive')
    out = a.out or a.run_root / 'analysis'
    if out.exists():
        ap.error('Output exists; choose a new --out (no overwrite)')
    result = analyze_suite(a.run_root, a.n_boot)
    out.mkdir(parents=True)
    (out / 'meta_experiment.json').write_text(json.dumps(clean_json(result), indent=2, allow_nan=False) + '\n')
    lines = ['# M metacognition', '', f"Status: **{result['status']}**.", '',
             f"HF: {result.get('HF', {}).get('verdict', 'pending')}.",
             f"Complete paired ids: {result.get('ids', [])}; missing ids: {result.get('missing_ids', [])}.",
             'Full contrasts, Holm tests, calibration slopes, and fixed-position AUROCs are in `meta_experiment.json`.', '',
             '| Arm | Checkpoint | n | Verbal | Forecast | Awareness optimism | Realized | Floor |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for m in result.get('means', []):
        lines.append(f"| {m['arm']} | {m['checkpoint']} | {m['n']} | {fmt(m['verbal'])} | {fmt(m['fore_none'])} | "
                     f"{fmt(m['aware'])} | {fmt(m['realized'])} | {m['floor']} |")
    lines += ['', 'Floor = realized accuracy <=.05. Intermediate-dose arms are primary. Error-sensitive reports alone do not establish introspection.',
              'Short histories remove earlier reconstruction evidence as well as tokens; see prereg.md §7.', '']
    (out / 'meta_experiment.md').write_text('\n'.join(lines))
    print('Wrote', out / 'meta_experiment.md', result['status'])


if __name__ == '__main__':
    main()
