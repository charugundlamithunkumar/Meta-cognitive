"""CPU synthetic precision/joint-power scenarios; optional calibration-only residuals.

All reported frequencies are simulation outputs, never observed model findings.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.analyze_recovery import test_contrasts, bundle_suite
from src.recovery_history import ARMS


def pilot_residuals(root, return_means=False):
    bundles,missing=bundle_suite(root)
    if missing or any(b['config']['split']!='calibration' or b['config']['mock'] for b in bundles.values()):
        raise ValueError('Pilot input must be real, complete calibration arms only')
    cfg=bundles['MR_A']['config']; ids=cfg['ids']; cps=cfg['checkpoints']
    values=[]
    for sid in ids:
        trajectory=[]
        for arm in ARMS:
            heads=[bundles[arm]['grouped'][sid,c][0] for c in cps]
            forecasts=[h['post_forecast']['value'] for h in heads]
            if any(x is None for x in forecasts): raise ValueError('Pilot has missing forecasts')
            trajectory.append([np.mean([100*h['realized_next10'] for h in heads]),np.mean(forecasts)])
        values.append(trajectory)
    values=np.asarray(values).transpose(0,2,1)
    residuals = values-values.mean(axis=0)
    return (residuals, values.mean(axis=0)) if return_means else residuals


def simulate(n=32,reps=100,n_boot=1000,seed=0,sigma=10.,variance_multiplier=1.,pilot=None,pilot_means=None):
    if n<2 or reps<1 or n_boot<1 or sigma<0 or variance_multiplier<=0:
        raise ValueError('Invalid simulation settings')
    rng=np.random.default_rng(seed)
    scenarios={
        'dissociation':([80,30,80,30],[80,30,50,30]),
        'no_recovery':([80,30,30,30],[80,30,20,30]),
        'HG1_null':([80,80,80,30],[80,30,50,30]),
        'HG2_boundary':([80,30,85,30],[90,30,50,30]),
        'HG3_boundary':([80,30,80,30],[70,30,60,30]),
        'HG4_boundary':([80,30,80,30],[90,30,70,30]),
    }
    if pilot_means is not None:
        scenarios['pilot_estimate_diagnostic'] = pilot_means
    report={}
    for label,(rmean,fmean) in scenarios.items():
        joint=0; widths=[]; raw={f'HG{i}':0 for i in range(1,5)}; adjusted=dict(raw)
        for _ in range(reps):
            if pilot is None:
                # Draw trajectory means, already aggregated over checkpoints.
                errors=rng.uniform(-np.sqrt(3)*sigma,np.sqrt(3)*sigma,size=(n,2,4))*variance_multiplier
            else:
                errors=pilot[rng.integers(len(pilot),size=n)]*variance_multiplier
            # Shift the joint trajectory-contrast residual distribution directly.
            # These are hypothetical contrast estimates, not generated arm scores.
            er,ef=errors[:,0,:],errors[:,1,:]
            noise=np.column_stack([er[:,2]-er[:,1],er[:,2]-er[:,0],ef[:,0]-ef[:,2],er[:,2]-ef[:,2]])
            r,f=np.asarray(rmean),np.asarray(fmean)
            means=np.array([r[2]-r[1],r[2]-r[0],f[0]-f[2],r[2]-f[2]])
            tests=test_contrasts(means+noise,n_boot,seed=int(rng.integers(2**31)))
            joint+=all(t['supported'] for t in tests.values()) and min(r[0],r[2])>5
            widths.append(tests['HG2']['hi90']-tests['HG2']['lo90'])
            for key,test in tests.items():
                raw[key]+=test['p']<.05
                adjusted[key]+=test['supported']
        valid=[x for x in widths if x is not None]
        report[label]=dict(valid_replicates=len(valid),rejected_out_of_bounds=reps-len(valid),
             joint_support_rate=joint/len(valid) if valid else None,
             mean_equivalence_CI_width=float(np.mean(valid)) if valid else None,
             raw_rejection_rate={k:v/len(valid) if valid else None for k,v in raw.items()},
             holm_support_rate={k:v/len(valid) if valid else None for k,v in adjusted.items()})
    return dict(kind='CONTRAST-SPACE SIMULATION ONLY',n=n,reps=reps,n_boot=n_boot,sigma=sigma,
                method='Shifted joint trajectory contrasts; no arm-score clipping or rejection',
                noise='bounded uniform with SD sigma' if pilot is None else 'resampled calibration residuals',
                variance_multiplier=variance_multiplier,seed=seed,
                warning='Not a sample-size guarantee or a generative model of bounded scores. Joint contrast covariance is preserved; eight-trajectory calibration variance and plug-in effect estimates are uncertain. Floor probability is not modeled beyond the scenario mean gate.',
                scenarios=report)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--n',type=int,default=32);ap.add_argument('--reps',type=int,default=100)
    ap.add_argument('--n-boot',type=int,default=1000);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--sigma',type=float,default=5.)
    ap.add_argument('--variance-multiplier',type=float,default=1.)
    ap.add_argument('--pilot',type=Path)
    a=ap.parse_args()
    if a.out.exists():ap.error('Output exists; choose a new path')
    residuals,means=pilot_residuals(a.pilot,return_means=True) if a.pilot else (None,None)
    report=simulate(a.n,a.reps,a.n_boot,a.seed,a.sigma,a.variance_multiplier,residuals,means)
    report['pilot_source']=str(a.pilot) if a.pilot else None
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('Saved CPU simulation:',a.out)


if __name__=='__main__':main()
