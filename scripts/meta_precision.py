"""Calibration-only HF trajectory-contrast precision scenarios; no model loading."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.analyze_meta import read_bundle
from scripts.analyze_meta_experiment import decision,probe_value
from src.meta_history import ARMS


def contrasts_from_confidence(confidence):
    shape=np.asarray(confidence['L0']).shape
    if len(shape)!=2 or any(np.asarray(confidence[a]).shape!=shape for a in ARMS):
        raise ValueError('Expected paired trajectory × checkpoint confidence arrays')
    x=np.array([0,.25,.5,.75]);x-=x.mean()
    slope=np.einsum('a,aij->ij',x,np.array([confidence[a] for a in ['L0','L25','L50','L75']]))/(x@x)
    return np.column_stack([(confidence['S0']-confidence['L0']).mean(axis=1),
                            -slope.mean(axis=1),(confidence['L0']-confidence['L100']).mean(axis=1)])


def pilot_vectors(root):
    confidence={};reference=None
    for arm in ARMS:
        cfg,rows,summaries=read_bundle(Path(root)/('M_'+arm))
        if cfg['mock'] or cfg['split']!='calibration':
            raise ValueError('Real calibration data only; no test/mock inputs')
        signature=(cfg['ids'],cfg['checkpoints'],cfg['source_sha256'])
        if reference is not None and signature!=reference:raise ValueError('Unpaired calibration arms')
        reference=signature
        heads={(r['sample_id'],r['checkpoint']):r for r in rows if r['record_type']=='checkpoint'}
        if len(heads)!=len(cfg['ids'])*len(cfg['checkpoints']) or len(summaries)!=len(cfg['ids']):
            raise ValueError('Incomplete calibration data')
        confidence[arm]=np.array([[probe_value(heads[s,c],'verbal') for c in cfg['checkpoints']] for s in cfg['ids']])
    values=contrasts_from_confidence(confidence)
    if not np.isfinite(values).all():raise ValueError('Missing calibration primary probes')
    return values


def simulate(values,n=32,reps=500,n_boot=1000,multiplier=1.,seed=0):
    values=np.asarray(values,float)
    if values.ndim!=2 or values.shape[1]!=3 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Expected finite paired P,D,E trajectory vectors')
    if n<2 or reps<1 or n_boot<1 or multiplier<=0:raise ValueError('Invalid simulation settings')
    residuals=values-values.mean(axis=0);rng=np.random.default_rng(seed)
    scenarios={'length-driven':[20,0,0],'error-sensitive':[0,20,20],
               'mixed':[20,20,20],'small':[0,0,0],'pilot_estimate_diagnostic':values.mean(axis=0)}
    result={}
    for name,mean in scenarios.items():
        counts={};widths=[]
        for _ in range(reps):
            draw=np.asarray(mean)+multiplier*residuals[rng.integers(len(values),size=n)]
            report=decision(-draw[:,0],draw[:,1],draw[:,2],n_boot)
            label=report['verdict'];counts[label]=counts.get(label,0)+1
            widths.append(report['contrasts']['dose']['hi']-report['contrasts']['dose']['lo'])
        result[name]=dict(classification_rates={k:v/reps for k,v in counts.items()},
                          mean_dose_CI_width=float(np.mean(widths)))
    return dict(method='Resampled joint trajectory-contrast residuals; hypothetical effects',
                pilot_n=len(values),n=n,reps=reps,n_boot=n_boot,multiplier=multiplier,seed=seed,
                scenarios=result,warning='Small calibration sample; plug-in effects/variance are uncertain. This is not proof of power or an empirical HF verdict. Floors are reviewed separately.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pilot',type=Path);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--n',type=int,default=32);ap.add_argument('--reps',type=int,default=500)
    ap.add_argument('--n-boot',type=int,default=1000);ap.add_argument('--multiplier',type=float,default=1.)
    a=ap.parse_args()
    if a.out.exists():ap.error('Output exists; choose a new path')
    result=simulate(pilot_vectors(a.pilot),a.n,a.reps,a.n_boot,a.multiplier)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Saved HF calibration precision:',a.out)


if __name__=='__main__':main()
