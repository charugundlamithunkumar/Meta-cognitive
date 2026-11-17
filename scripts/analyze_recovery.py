"""CPU HG analysis of MR folders or archived arms. Refuses output overwrites."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze_meta import read_bundle, clean_json, auc_summary, feature, FEATURES
from scripts.analyze_meta_experiment import calibration
from src.stats import holm
from src.lhe_data import SPLITS
from src.recovery_history import ARMS, CHECKPOINTS, fingerprint
from src.recovery_runner import validate_checkpoint


def mean_ci(values, indices):
    values = np.asarray(values, float)
    boots = values[indices].mean(axis=1)
    return dict(mean=float(values.mean()), lo=float(np.quantile(boots, .025)),
                hi=float(np.quantile(boots, .975))), boots


def test_contrasts(contrasts, n_boot=10000, seed=0):
    """The registered tests on paired trajectory-level HG1…HG4 contrasts.

    Also reused for precision simulations in contrast space, avoiding invalid
    fabricated arm percentages. This helper issues no empirical study verdict.
    """
    contrasts = np.asarray(contrasts, float)
    if contrasts.ndim != 2 or contrasts.shape[1] != 4 or not len(contrasts) or n_boot < 1:
        raise ValueError('Expected trajectories × four finite contrasts and positive n_boot')
    if not np.isfinite(contrasts).all():
        raise ValueError('Contrasts must be finite')
    indices = np.random.default_rng(seed).integers(len(contrasts), size=(n_boot, len(contrasts)))
    tests = {}
    for i, values in enumerate(contrasts.T, 1):
        report, boots = mean_ci(values, indices)
        observed = report['mean']; errors = boots-observed
        if i == 2:
            lower_p = (1+np.count_nonzero(errors >= observed+5))/(n_boot+1)
            upper_p = (1+np.count_nonzero(errors <= observed-5))/(n_boot+1)
            report.update(p=max(lower_p, upper_p), lo90=float(np.quantile(boots, .05)),
                          hi90=float(np.quantile(boots, .95)), margin=[-5, 5])
            report['ci_pass'] = report['lo90'] > -5 and report['hi90'] < 5
        else:
            margin = 0 if i == 1 else 10
            report.update(p=(1+np.count_nonzero(errors >= observed-margin))/(n_boot+1), margin=margin)
            report['ci_pass'] = report['lo'] > margin
        tests[f'HG{i}'] = report
    for test, adj in zip(tests.values(), holm([t['p'] for t in tests.values()])):
        test.update(p_holm=adj, supported=bool(adj < .05 and test['ci_pass']))
    return tests


def decision(r, f, n_boot=10000, seed=0):
    """r/f: paired trajectory × checkpoint × arm percentages; eligibility separate."""
    r, f = np.asarray(r, float), np.asarray(f, float)
    if r.shape != f.shape or r.ndim != 3 or r.shape[2] != 4 or not r.shape[0] or not r.shape[1]:
        raise ValueError('Expected nonempty trajectory × checkpoint × four-arm arrays')
    if n_boot < 1:
        raise ValueError('n_boot must be positive')
    if not np.isfinite(r).all() or not np.isfinite(f).all():
        return dict(verdict='unavailable: missing primary observations', tests={})
    if (r < 0).any() or (r > 100).any() or (f < 0).any() or (f > 100).any():
        raise ValueError('Primary inputs must be percentages')
    rr, ff = r.mean(axis=1), f.mean(axis=1)
    contrasts = np.column_stack([rr[:, 2]-rr[:, 1], rr[:, 2]-rr[:, 0], ff[:, 0]-ff[:, 2], rr[:, 2]-ff[:, 2]])
    tests = test_contrasts(contrasts, n_boot, seed)
    floor = bool(rr[:, [0, 2]].mean(axis=0).min() <= 5)
    verdict = 'floor-limited' if floor else ('recovery/report dissociation supported' if
               all(t['supported'] for t in tests.values()) else 'inconclusive')
    return dict(verdict=verdict, tests=tests, floor=floor, n_trajectories=len(rr),
                n_checkpoints=r.shape[1], n_boot=n_boot,
                arm_accuracy=dict(zip(ARMS, rr.mean(axis=0).tolist())))


def bundle_suite(root):
    root = Path(root)
    bundles, missing = {}, []
    for arm in ARMS:
        source = root/arm if (root/arm).is_dir() else root/f'{arm}.tar.gz'
        if not source.exists():
            missing.append(arm); continue
        cfg, rows, summaries = read_bundle(source)
        if cfg.get('arm') != arm or cfg.get('study') != 'MR' or cfg.get('schema_version') != 1:
            raise ValueError('Unexpected arm/schema')
        grouped = {}
        expected = {(sid, cp) for sid in cfg['ids'] for cp in cfg['checkpoints']}
        for row in rows:
            key = (row['sample_id'], row['checkpoint'])
            if key not in expected:
                raise ValueError('Unexpected trajectory/checkpoint')
            grouped.setdefault(key, []).append(row)
        for (sid, cp), rr in grouped.items():
            validate_checkpoint(rr, sid, cp, arm, cfg['mock'])
        summary_ids = [s['sample_id'] for s in summaries]
        if len(summary_ids) != len(set(summary_ids)) or not set(summary_ids) <= set(cfg['ids']):
            raise ValueError('Duplicate/unknown trajectory summaries')
        for s in summaries:
            sid = s['sample_id']
            if any((sid, cp) not in grouped for cp in cfg['checkpoints']) or s['n_checkpoints'] != len(cfg['checkpoints']) or s['mock'] != cfg['mock']:
                raise ValueError('Invalid completion summary')
        bundles[arm] = dict(config=cfg, rows=rows, grouped=grouped, source=str(source), complete_ids=sorted(summary_ids),
                            source_sha256=fingerprint(dict(config=cfg, rows=rows, summaries=summaries)))
    if bundles:
        ref = next(iter(bundles.values()))['config']
        keys = ['ids', 'checkpoints', 'mock', 'split', 'source_sha256', 'resource_sha256',
                'registration_commit', 'model_revision', 'seed']
        if any(any(b['config'].get(k) != ref.get(k) for k in keys) for b in bundles.values()):
            raise ValueError('Mixed provenance/configs across arms')
    return bundles, missing


def secondary_auroc(bundles, n_boot):
    cells = []
    for arm, bundle in bundles.items():
        cfg = bundle['config']
        for cp in cfg['checkpoints']:
            for offset in range(1, 11):
                original = [r for r in bundle['rows'] if r['record_type']=='continuation'
                            and r['checkpoint']==cp and r['offset']==offset]
                for target in ['global', 'local']:
                    common = [r for r in original if all(feature(r, k) is not None and
                              np.isfinite(feature(r, k)) for k in FEATURES)
                              and (target == 'global' or r['kind'] != 'nonanswer')]
                    if target == 'global':
                        common = [dict(r, kind='ok' if r['kind']=='ok' else 'numeric_slip') for r in common]
                    scores, samples = {}, {}
                    for key in FEATURES:
                        scores[key], samples[key] = auc_summary(common, key, n_boot, return_samples=True)
                    good = np.isfinite(samples['verbal']) & np.isfinite(samples['lp_min'])
                    delta = samples['lp_min'][good]-samples['verbal'][good]
                    cells.append(dict(arm=arm, checkpoint=cp, offset=offset, target=target,
                                      n_common=len(common), scores=scores,
                                      lp_minus_verbal=dict(point=scores['lp_min']['point']-scores['verbal']['point'],
                                      lo=float(np.quantile(delta, .025)) if len(delta) else None,
                                      hi=float(np.quantile(delta, .975)) if len(delta) else None,
                                      valid_bootstraps=int(good.sum()))))
    return cells


def analyze_suite(root, n_boot=10000):
    bundles, missing = bundle_suite(root)
    report = dict(study='MR', n_boot=n_boot, missing_arms=missing,
                  provenance={a:dict(source=b['source'], sha256=b['source_sha256'],
                                    config=b['config']) for a,b in bundles.items()},
                  checkpoint_summary=[], calibration={}, interactions={})
    points = {}
    for arm, b in bundles.items():
        points[arm] = []
        for cp in b['config']['checkpoints']:
            heads = [rr[0] for (sid,c),rr in b['grouped'].items() if c==cp]
            r = [100*h['realized_next10'] for h in heads]
            f = [h['post_forecast']['value'] for h in heads]
            pre = [h['pre_forecast']['value'] for h in heads]
            valid = [i for i,x in enumerate(f) if x is not None]
            aware = [h['post_continuation_awareness']['probabilities'] for h in heads]
            awareness = [a for a in aware if a is not None]
            report['checkpoint_summary'].append(dict(arm=arm, checkpoint=cp, n=len(heads),
                accuracy=float(np.mean(r)) if r else None, floor=bool(np.mean(r)<=5) if r else None,
                forecast=float(np.mean([f[i] for i in valid])) if valid else None,
                pre_forecast=float(np.mean([p for p in pre if p is not None])) if any(p is not None for p in pre) else None,
                missing_forecasts=len(heads)-len(valid),
                out_of_range=sum(h['post_forecast']['out_of_range'] for h in heads),
                mae=float(np.mean([abs(f[i]-r[i]) for i in valid])) if valid else None,
                signed_gap=float(np.mean([r[i]-f[i] for i in valid])) if valid else None,
                awareness=np.mean(awareness, axis=0).tolist() if awareness else None,
                awareness_n=len(awareness),
                raw_only_forecast=float(np.mean([h['post_forecast']['parsed'] for h in heads if
                      h['post_forecast']['valid'] and not h['post_forecast']['out_of_range']])) if any(
                      h['post_forecast']['valid'] and not h['post_forecast']['out_of_range'] for h in heads) else None,
                forecast_change=float(np.mean([f[i]-pre[i] for i in valid if pre[i] is not None])) if any(
                      pre[i] is not None for i in valid) else None))
            for h in heads:
                points[arm].append(dict(sample_id=h['sample_id'], checkpoint=cp,
                                       forecast=h['post_forecast']['value']/100 if h['post_forecast']['value'] is not None else np.nan,
                                       realized=h['realized_next10']))
        report['calibration'][arm] = calibration(points[arm], 'forecast', n_boot)
    if not bundles or missing:
        report['primary'] = dict(verdict='unavailable: missing arms')
    else:
        cfg = bundles[ARMS[0]]['config']; ids, cps = cfg['ids'], cfg['checkpoints']
        absent = [(a,s,c) for a in ARMS for s in ids for c in cps if (s,c) not in bundles[a]['grouped']]
        if absent:
            report['primary'] = dict(verdict='unavailable: incomplete paired checkpoints', missing=absent)
        else:
            r = np.array([[[100*bundles[a]['grouped'][s,c][0]['realized_next10'] for a in ARMS] for c in cps] for s in ids])
            f = np.array([[[bundles[a]['grouped'][s,c][0]['post_forecast']['value'] if
                            bundles[a]['grouped'][s,c][0]['post_forecast']['value'] is not None else np.nan
                           for a in ARMS] for c in cps] for s in ids])
            stats = decision(r, f, n_boot)
            eligible = (not cfg['mock'] and cfg['split']=='test' and ids==SPLITS['test'] and cps==CHECKPOINTS
                        and n_boot==10000 and bool(cfg.get('registration_commit'))
                        and cfg.get('confirmatory') and cfg.get('resource_sha256') is not None
                        and all(b['complete_ids']==sorted(ids) for b in bundles.values()))
            report['primary'] = stats if eligible else dict(verdict='withheld: mock or exploratory run',
                          diagnostic_estimates=stats.get('tests', {}))
            indices = np.random.default_rng(0).integers(len(ids), size=(n_boot, len(ids)))
            for key, arr in [('accuracy',r), ('forecast',f)]:
                if np.isfinite(arr).all():
                    # Difference in the benefit of true state between erroneous and clean history.
                    values = (arr[:,:,2]-arr[:,:,1] - (arr[:,:,0]-arr[:,:,3])).mean(axis=1)
                    report['interactions'][key] = mean_ci(values, indices)[0]
    report['auroc'] = secondary_auroc(bundles, n_boot)
    return clean_json(report)


def figures(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries = report['checkpoint_summary']
    for metric, filename, label in [('accuracy','accuracy.png','Next-10 accuracy (%)'),
                                     ('forecast','forecasts.png','Forecast / actual (%)')]:
        fig, ax = plt.subplots(figsize=(7,4))
        for arm in ARMS:
            cells = sorted([r for r in summaries if r['arm']==arm], key=lambda r:r['checkpoint'])
            ax.plot([r['checkpoint'] for r in cells], [r[metric] for r in cells], marker='o', label=arm)
            if metric=='forecast':
                ax.plot([r['checkpoint'] for r in cells],[r['accuracy'] for r in cells],linestyle=':',alpha=.6)
        ax.set(xlabel='Checkpoint (zero-based)', ylabel=label, ylim=(-3,103),
               title='MOCK / exploratory' if report['primary']['verdict'].startswith('withheld') else 'MR recovery')
        ax.legend(); fig.tight_layout(); fig.savefig(out/filename,dpi=160); plt.close(fig)
    tests = report['primary'].get('tests', report['primary'].get('diagnostic_estimates',{}))
    fig, ax = plt.subplots(figsize=(7,4))
    for i,(name,t) in enumerate(tests.items()):
        ax.errorbar(t['mean'], i, xerr=[[max(0,t['mean']-t['lo'])],[max(0,t['hi']-t['mean'])]], fmt='o')
        margins = t['margin'] if isinstance(t['margin'],list) else [t['margin']]
        for margin in margins: ax.plot(margin,i,marker='|',markersize=16,color='gray')
    ax.set_yticks(range(len(tests)), list(tests)); ax.set_xlabel('Paired effect (percentage points), 95% CI')
    ax.set_title('HG diagnostics — '+report['primary']['verdict'])
    fig.tight_layout(); fig.savefig(out/'hg_effects.png',dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,4))
    for arm in ARMS:
        cells=[r for r in summaries if r['arm']==arm and r['forecast'] is not None]
        axes[0].scatter([r['forecast'] for r in cells],[r['accuracy'] for r in cells],label=arm)
    axes[0].plot([0,100],[0,100],':',color='gray'); axes[0].legend(fontsize=7)
    axes[0].set(xlabel='Forecast (%)',ylabel='Accuracy (%)')
    cells=[r for r in report['auroc'] if r['scores']['lp_min']['point'] is not None]
    axes[1].scatter([r['scores']['verbal']['point'] for r in cells], [r['scores']['lp_min']['point'] for r in cells])
    axes[1].set(xlabel='Verbal AUROC',ylabel='Minimum-logprob AUROC',xlim=(0,1),ylim=(0,1))
    if not cells: axes[1].text(.5,.5,'No cells with ≥15 per class',ha='center',fontsize=8)
    fig.suptitle(report['primary']['verdict'])
    fig.tight_layout(); fig.savefig(out/'secondary.png',dpi=160); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path); ap.add_argument('--out',type=Path)
    ap.add_argument('--n-boot',type=int,default=10000)
    a=ap.parse_args(); out=a.out or ROOT/'results'/f'recovery_{a.root.name}'
    if out.exists(): ap.error('Output exists; choose a new --out (no overwrite)')
    if a.n_boot<1: ap.error('n-boot must be positive')
    report=analyze_suite(a.root,a.n_boot)
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    text='# MR recovery analysis\n\n**'+report['primary']['verdict']+'**\n\n'
    configs=[v['config'] for v in report['provenance'].values()]
    if any(c['mock'] for c in configs):
        text+='Synthetic mock outputs are software diagnostics, not model evidence.\n\n'
    elif configs and all(c['split']=='calibration' for c in configs):
        text+='Real calibration observations; exploratory only, without a confirmatory HG verdict.\n\n'
    else:
        text+='Real inference outputs; the eligibility checks above determine whether a main verdict is available.\n\n'
    text+='| Arm | Checkpoint | n | Accuracy % | Forecast % | Missing | MAE |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in report['checkpoint_summary']:
        text+='| '+ ' | '.join(str(r[k]) for k in ['arm','checkpoint','n','accuracy','forecast','missing_forecasts','mae'])+' |\n'
    text+='\nHG tests, adjusted p-values, CIs, class counts, parse flags, paired interactions, calibration and input provenance are in `analysis.json`.\n\n'
    text+='\n'.join(f'![{n}]({n}.png)' for n in ['accuracy','forecasts','hg_effects','secondary'])+'\n'
    (out/'report.md').write_text(text)
    figures(report,out)
    print('Wrote',out,report['primary']['verdict'])


if __name__=='__main__':
    main()
