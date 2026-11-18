"""C-specific eligibility with unchanged HF/HG statistical functions. CPU only."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.confirmation_common import settings
from scripts.analyze_meta import read_bundle,clean_json
from scripts.analyze_meta_experiment import analyze_suite as analyze_m
from scripts.analyze_recovery import analyze_suite as analyze_mr,bundle_suite,decision,figures
from src.lhe_data import sha256
from src.meta_history import ARMS as META_ARMS
from src.recovery_history import ARMS as RECOVERY_ARMS
from scripts.run_confirmation import validate_family


def eligibility(root,n_boot):
    spec=settings();parent=json.loads((root/'status.json').read_text())['signature']
    ids=[10000+s for s in spec['fresh_seeds']];cps=spec['checkpoints']
    reasons=[]
    if parent['mock']:reasons.append('mock')
    if parent['ids']!=ids or parent['checkpoints']!=cps:reasons.append('partial allocation')
    if n_boot!=10000:reasons.append('nonregistered bootstrap count')
    if not parent.get('registration_commit'):reasons.append('no registration commit')
    if not parent.get('resource_manifest_sha256'):reasons.append('no resource fingerprint')
    for family,names in [('M',['M_'+a for a in META_ARMS]),('MR',list(RECOVERY_ARMS))]:
        manifest=json.loads((root/family/'manifest.json').read_text())
        if manifest['sample_manifest_sha256']!=sha256(ROOT/spec['manifest']):reasons.append('sample manifest mismatch')
        if not parent['mock'] and (manifest['resources']!=json.loads((ROOT/spec['resource_manifest']).read_text())
            or sha256(ROOT/spec['resource_manifest'])!=parent['resource_manifest_sha256']):
            raise ValueError('Confirmation resource provenance mismatch')
        if not parent['mock'] and json.loads((root/family/'hardware.json').read_text())['index']!=spec['gpu']:
            reasons.append('wrong GPU')
        for name in names:
            cfg,_,summaries=read_bundle(root/family/name)
            if (cfg['study_instance']!=spec['study_instance'] or cfg['confirmation_label']!='C_'+name
                or cfg['source_sha256']!=parent['source_sha256']
                or cfg['registration_commit']!=parent['registration_commit']
                or cfg['sample_manifest_sha256']!=manifest['sample_manifest_sha256']
                or cfg['mock']!=parent['mock'] or cfg['split']!='confirmation'
                or cfg['ids']!=parent['ids'] or cfg['checkpoints']!=parent['checkpoints']):
                raise ValueError('Mixed C provenance')
            if not parent['mock'] and (not cfg.get('confirmatory') or cfg.get('resource_sha256')!=manifest['resources']['sha256']):
                raise ValueError('Unregistered or mismatched C arm')
            if sorted(s['sample_id'] for s in summaries)!=sorted(parent['ids']):reasons.append('incomplete trajectories')
        validate_family(root/family,family,parent['ids'],parent['checkpoints'])
    return dict(eligible=not reasons,reasons=sorted(set(reasons)),signature=parent)


def analyze(root,n_boot=10000):
    root=Path(root);gate=eligibility(root,n_boot)
    m=analyze_m(root/'M',n_boot);mr=analyze_mr(root/'MR',n_boot)
    m_primary=copy.deepcopy(m['HF'])
    numerical=m_primary.pop('numerical_rule_output',m_primary['verdict'])
    m_primary['verdict']=numerical if gate['eligible'] else 'withheld: '+', '.join(gate['reasons'])
    bundles,missing=bundle_suite(root/'MR')
    if missing:raise ValueError('Missing C recovery arms')
    cfg=bundles[RECOVERY_ARMS[0]]['config'];ids,cps=cfg['ids'],cfg['checkpoints']
    rr=np.array([[[100*bundles[a]['grouped'][sid,c][0]['realized_next10'] for a in RECOVERY_ARMS] for c in cps] for sid in ids])
    ff=np.array([[[bundles[a]['grouped'][sid,c][0]['post_forecast']['value'] if bundles[a]['grouped'][sid,c][0]['post_forecast']['value'] is not None else np.nan for a in RECOVERY_ARMS] for c in cps] for sid in ids])
    mr_primary=decision(rr,ff,n_boot)
    if not gate['eligible']:mr_primary=dict(verdict='withheld: '+', '.join(gate['reasons']))
    return clean_json(dict(study='C20261118',status='independent confirmation' if gate['eligible'] else 'MOCK / diagnostic only',
                           eligibility=gate,C_HF=m_primary,C_HG=mr_primary,
                           M_base_analysis=m,MR_base_analysis=mr))


def plot_m(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,key,label in zip(axes.flat,['verbal','fore_none','aware','realized'],
                           ['Verbal confidence (%)','Forecast (%)','Awareness optimism (%)','Accuracy (%)']):
        for arm in META_ARMS:
            cells=sorted([c for c in data['means'] if c['arm']==arm],key=lambda c:c['checkpoint'])
            scale=100 if key in ['aware','realized'] else 1
            ax.plot([c['checkpoint'] for c in cells],[scale*c[key] if c[key] is not None else np.nan for c in cells],label='C_M_'+arm)
        ax.set(xlabel='Checkpoint',ylabel=label,ylim=(-3,103))
    axes[0,0].legend(fontsize=6,ncol=2);fig.suptitle('C fresh-sequence M replication; '+data.get('status',''))
    fig.tight_layout();fig.savefig(out/'M_curves.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--n-boot',type=int,default=10000)
    a=ap.parse_args()
    if a.out.exists():ap.error('Output exists; refusing overwrite')
    if a.n_boot<1:ap.error('Positive bootstrap count required')
    result=analyze(a.root,a.n_boot);a.out.mkdir(parents=True)
    (a.out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot_m(result['M_base_analysis'],a.out)
    mr=copy.deepcopy(result['MR_base_analysis']);mr['primary']=result['C_HG']
    figures(mr,a.out)
    text=f"# Independent-sequence study C\n\nStatus: **{result['status']}**.\n\nC-HF: {result['C_HF']['verdict']}.\n\nC-HG: {result['C_HG']['verdict']}.\n\n"
    text+='Figures retain canonical core arm fields within the new C run; no original observations are reused. All tests, intervals, missingness, floors, class support and source/sample provenance are in `analysis.json`. Mock/partial outputs never receive an empirical verdict.\n\n'
    text+='\n'.join(f'![{n}]({n}.png)' for n in ['M_curves','accuracy','forecasts','hg_effects','secondary'])+'\n'
    (a.out/'report.md').write_text(text)
    print('Saved C analysis:',a.out,result['status'])


if __name__=='__main__':main()
