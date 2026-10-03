"""Reproducible seed-level gain and continuation-regime plots, PDF + PNG."""
import csv
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ART=Path(__file__).resolve().parents[1]
FIG=ART/'figures'
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
                     'font.size':10,'axes.labelsize':10,'legend.fontsize':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'legend.frameon':False,'savefig.bbox':'tight','savefig.dpi':300,
                     'pdf.fonttype':42,'ps.fonttype':42})
rows=list(csv.DictReader((ART/'results_multiseed.csv').open()))
seeds=[r['seed'] for r in rows]
colors=['#0072B2','#D55E00']


def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'))
    fig.savefig(FIG/(name+'.png'),dpi=300)
    plt.close(fig)


fig,ax=plt.subplots(figsize=(6.75,3.0))
for i,k in enumerate(['Gain_1','Gain_5']):
    values=[float(r[k]) for r in rows]
    x=np.arange(3)+(i-.5)*.16
    mean,sd=statistics.mean(values),statistics.stdev(values)
    ax.plot(x,values,marker=['o','s'][i],linestyle='none',color=colors[i],markersize=7,
            label=f'lambda = {1 if i==0 else 5}; mean {mean:+.4f}, SD {sd:.4f}')
    for a,b in zip(x,values):
        ax.annotate(f'{b:+.4f}',(a,b),xytext=(0,8 if i==0 else -14),
                    textcoords='offset points',ha='center',fontsize=8)
ax.axhline(0,color='#555555',linewidth=.9)
ax.set_xticks(np.arange(3),seeds)
ax.set_xlabel('Independent student preparation seed')
ax.set_ylabel('Test NLL gain: CE minus KD')
ax.set_title('FineWeb-Edu: frozen SmolLM2 warm-start replication',fontsize=11)
ax.grid(axis='y',alpha=.15)
ax.margins(y=.25)
ax.legend(loc='lower left',bbox_to_anchor=(0,1.04),ncol=1)
save(fig,'fig_primary_gains')

fig,axs=plt.subplots(1,2,figsize=(6.75,2.8),sharey=True)
for ax,split in zip(axs,['validation','test']):
    for i,arm in enumerate(['ce','kd1','kd5']):
        vals=[float(r[f'{arm}_{split}_nll'])-float(r[f's0_{split}_nll']) for r in rows]
        ax.plot(np.arange(3)+(i-1)*.13,vals,marker=['o','s','^'][i],linestyle='none',
                color=['#777777',*colors][i],label=['CE','KD, lambda = 1','KD, lambda = 5'][i])
    ax.axhline(0,color='#555555',linewidth=.9)
    ax.set_xticks(np.arange(3),seeds)
    ax.set_xlabel('Student seed')
    ax.set_title(split.capitalize()+' endpoints',fontsize=11)
    ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Primary continuation NLL minus S0\nPositive = degradation')
axs[1].legend(loc='upper left',bbox_to_anchor=(1.02,1))
fig.tight_layout()
save(fig,'fig_continuation_regime')
print('TWO_FIGURE_PAIRS_COMPLETE')
