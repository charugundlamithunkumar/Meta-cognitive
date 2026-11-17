"""Static M figures from saved analysis JSON; never overwrites an output folder."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('analysis',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():ap.error('Output exists; choose a new folder')
    data=json.loads(a.analysis.read_text())
    if 'means' not in data:ap.error('No complete M arm summaries')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.out.mkdir(parents=True)
    means=data['means'];arms=['L0','L25','L50','L75','L100','S0','S100']
    title='Main M results' if data['status']=='confirmatory' else 'M calibration / exploratory'
    for key,label,name in [('verbal','Verbal confidence (%)','confidence'),
                           ('fore_none','Next-10 forecast (%)','forecast'),
                           ('aware','Awareness optimism index (0–100)','awareness')]:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for arm in arms:
            cells=sorted([r for r in means if r['arm']==arm],key=lambda r:r['checkpoint'])
            x=[r['checkpoint'] for r in cells]
            values=[r[key]*100 if key=='aware' and r[key] is not None else r[key] for r in cells]
            axes[0].plot(x,values,marker='o',label=arm)
            axes[1].plot(x,[100*r['realized'] for r in cells],marker='o',label=arm)
        axes[0].set(xlabel='Checkpoint',ylabel=label,ylim=(-3,103))
        axes[1].set(xlabel='Checkpoint',ylabel='Realized next-10 accuracy (%)',ylim=(-3,103))
        axes[0].legend(fontsize=7,ncol=2);fig.suptitle(title);fig.tight_layout()
        fig.savefig(a.out/f'{name}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for cp in data['checkpoints']:
        cells=[next(r for r in means if r['arm']==arm and r['checkpoint']==cp)
               for arm in ['L0','L25','L50','L75','L100']]
        ax.plot([0,25,50,75,100],[r['verbal'] for r in cells],marker='o',label=f'c={cp}')
    ax.set(xlabel='Injected wrong totals (%)',ylabel='Verbal confidence (%)',ylim=(-3,103),
           title=title+'; 100% endpoint descriptive')
    ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(a.out/'dose_response.png',dpi=160);plt.close(fig)
    print('Wrote M figures:',a.out)


if __name__=='__main__':main()
