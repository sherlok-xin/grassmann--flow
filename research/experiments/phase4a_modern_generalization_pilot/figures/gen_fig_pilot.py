"""Plot real pilot endpoints and validation curves; no single-seed error bars."""
import csv
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ART = Path(__file__).resolve().parents[1]
OUT = ART / 'figures'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.dpi': 300})
colors = ['#0072B2', '#D55E00']
rows = list(csv.DictReader((ART / 'results_seed42.csv').open()))
fig, ax = plt.subplots(figsize=(5.5, 2.8))
for i, (arm, label) in enumerate([('kd1', 'KD, lambda = 1'), ('kd5', 'KD, lambda = 5')]):
    values = [float(next(r for r in rows if r['dataset'] == ds and r['arm'] == arm)['gain_test_nll']) for ds in ['tinystories', 'fineweb']]
    xx = np.arange(2) + (i - 0.5) * 0.24
    ax.scatter(xx, values, s=45, color=colors[i], marker=['o', 's'][i], label=label, zorder=3)
    for x, value in zip(xx, values):
        ax.annotate(f'{value:+.4f}', (x, value), xytext=(0, 7 if value >= 0 else -14), textcoords='offset points', ha='center', fontsize=8)
ax.axhline(0, color='0.3', lw=0.9)
for level in [-0.02, 0.02]:
    ax.axhline(level, color='0.65', lw=0.8, ls='--')
ax.set_xticks([0, 1], ['TinyStories', 'FineWeb-Edu'])
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel('Test NLL gain (CE minus KD)')
ax.set_title('SmolLM2 warm-start KD pilot, seed 42')
ax.margins(y=0.3)
ax.legend(loc='best', fontsize=8)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(OUT / ('fig_transfer_gain.' + ext))
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.7), sharey=False)
for ax, dataset, title in zip(axes, ['tinystories', 'fineweb'], ['TinyStories', 'FineWeb-Edu']):
    for arm, label, color, marker in [('ce', 'CE', '#666666', 'o'), ('kd1', 'KD, lambda = 1', colors[0], 's'), ('kd5', 'KD, lambda = 5', colors[1], '^')]:
        with gzip.open(ART / 'raw' / (dataset + '_' + arm + '_metrics.jsonl.gz'), 'rt') as f:
            data = [x for line in f if 'validation_nll' in (x := json.loads(line))]
        ax.plot([r['tokens'] / 1e6 for r in data], [r['validation_nll'] for r in data], color=color, marker=marker, markersize=3, lw=1.2, label=label)
        endpoint = next(r for r in rows if r['dataset'] == dataset and r['arm'] == arm)
        ax.scatter(float(endpoint['best_validation_tokens']) / 1e6, float(endpoint['validation_nll']), facecolor='none', edgecolor=color, s=65, lw=1)
    ax.set_title(title)
    ax.set_xlabel('Continuation target tokens (M)')
    ax.grid(alpha=0.15)
axes[0].set_ylabel('Validation NLL')
axes[1].legend(fontsize=7)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(OUT / ('fig_validation_curves.' + ext))
plt.close(fig)
