"""A dated, explicitly partial plot; only immutable completed run summaries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'论文投稿/manuscript_v3/figures'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runs=ROOT/'outputs/phase5/runs'
    ce=[json.loads((runs/f'wt2_s42_screen_ce{i}'/'summary.json').read_text()) for i in range(6)]
    kd=json.loads((runs/'wt2_s42_screen_kd0'/'summary.json').read_text())
    assert all(r['status']=='COMPLETE' and r['steps']==294 for r in ce+[kd])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    fig,ax=plt.subplots(figsize=(3.3,2.3),layout='constrained')
    ax.plot([r['config']['lr'] for r in ce],[r['final_dev_nll']-r['s0_dev_nll'] for r in ce],
        color='#4E7797',marker='o',linewidth=1.2,label='CE (6 settings)')
    ax.scatter([kd['config']['lr']],[kd['final_dev_nll']-kd['s0_dev_nll']],color='#53806F',marker='D',s=38,
        label=r'KD ($\lambda=0.25$, $T=1$)',zorder=4)
    ax.axhline(0,color='#777777',linestyle='--',linewidth=.8)
    ax.set_xscale('log');ax.set_xlabel('Peak learning rate');ax.set_ylabel('Trained dev NLL − S0')
    ax.legend(loc='upper left',fontsize=7,frameon=False)
    for ext in ['pdf','svg']:fig.savefig(OUT/f'partial_wt2_screen.{ext}')
    fig.savefig(ROOT/'outputs/phase5/partial_wt2_screen.png',dpi=300)
    plt.close(fig)
    (OUT/'partial_wt2_screen_sources.json').write_text(json.dumps(dict(
        date='2026-11-25',scope='six CE and first KD short screens only, not full method results',
        runs=[r['config']['name'] for r in ce+[kd]],independent_student_seeds=1,
        remaining_methods_pending=True),indent=2)+'\n')


if __name__=='__main__':main()
