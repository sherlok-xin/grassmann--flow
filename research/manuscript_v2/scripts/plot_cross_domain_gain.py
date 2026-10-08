#!/usr/bin/env python3
"""Figure 2: archived paired gains only; no inference or new endpoints."""
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'research/manuscript_v2/tables/table_endpoint_inventory.csv'
OUT=ROOT/'论文投稿/manuscript_v2/figures/fig2_cross_domain_gain'


def main():
    with SOURCE.open() as f: rows=list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','Arial','Helvetica'], 'font.size':10,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':0.6,'savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(6.8,2.65),layout='constrained')
    families=['PTB_fused','WT2_J_fused_TG','TinyStories_fused']
    audit={}
    for x,family in enumerate(families):
        vals={int(r['student_seed']):float(r['test_gain']) for r in rows if r['family']==family}
        assert set(vals)=={42,123,456}
        for offset,seed,color,marker in zip([-0.15,0,0.15],[42,123,456],['#608dad','#c89660','#8c7ab4'],['o','s','^']):
            ax.scatter(x+offset,vals[seed],s=36,color=color,marker=marker,zorder=3,edgecolors='white',linewidths=.45)
        mean=statistics.mean(vals.values());sd=statistics.stdev(vals.values())
        ax.errorbar(x+.30,mean,yerr=sd,fmt='D',markersize=4,color='#172a3a',capsize=4,linewidth=1.05,zorder=4)
        ax.annotate(f'{mean:+.6f}',(x,mean),xytext=(0,14 if mean>0 else -21),textcoords='offset points',ha='center',fontsize=9)
        audit[family]={'per_seed':vals,'mean':mean,'sample_sd':sd}
    ax.axhline(0,color='#78818a',linewidth=.8,linestyle=(0,(4,3)),zorder=1)
    ax.set_xticks(range(3),['PTB','WikiText-2','TinyStories'])
    ax.set_ylabel(r'Paired gain: NLL$_{\rm CE}$ − NLL$_{\rm KD}$')
    ax.set_ylim(-.16,.22);ax.set_xlim(-.5,2.7)
    ax.set_yticks([-.1,0,.1,.2]);ax.grid(axis='y',color='#e9ecf0',linewidth=.55);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    handles=[Line2D([],[],color=c,marker=m,linestyle='',markersize=5,label=f'Seed {s}') for s,c,m in zip([42,123,456],['#608dad','#c89660','#8c7ab4'],['o','s','^'])]
    handles.append(Line2D([],[],color='#172a3a',marker='D',markersize=4,linewidth=1,label='Mean ± sample SD'))
    ax.legend(handles=handles,frameon=False,ncol=4,loc='lower center',bbox_to_anchor=(0.5,1.02),fontsize=8,handlelength=1,columnspacing=1.5)
    for suffix in ['.svg','.pdf','_600dpi.png']:
        fig.savefig(str(OUT)+suffix,dpi=600)
        if suffix=='.svg':
            subprocess.run(['sed','-i',r's/[ \t]*$//',str(OUT)+suffix],check=True)
    plt.close(fig)
    print(json.dumps({'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'data':audit,'canvas_inches':[6.8,2.65],'error_bars':'sample SD, not CI','model_access':False},indent=2))


if __name__=='__main__':main()
