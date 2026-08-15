"""Tufte pass over the four evaluation charts.

Data is copied verbatim from the recovered eval_charts.ipynb, which sources it
from ur5/docs/final_evaluations.md. No value is changed here; only the display.

What changes, and why:
  - decorative per-bar palettes -> gray for context, one accent for the focal bar
  - dual y-axis on the ID/OOD chart -> two stacked panels sharing an x axis
  - hatched fills (moire) -> flat tints
  - boxed legends away from the data -> labels sitting on the data
  - heavy olive gridlines -> faint gray, behind everything
  - bar outlines -> none
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT = '/home/lps/openpi/ur5/docs/figures'

INK      = '#1a1a1a'
MUTED    = '#6b6b6b'
QUIET    = '#9a9a9a'
GRID     = '#e2e2e2'
CONTEXT  = '#c4c4c4'   # bars that are not the point
ACCENT   = '#2f6b5c'   # the article's single accent
ACCENT_L = '#a8c5bc'   # accent, de-emphasised (second series)

plt.rcParams.update({
    'font.family': 'monospace',
    'font.monospace': ['DejaVu Sans Mono', 'Courier New'],
    'font.size': 18, 'axes.labelsize': 21,
    'xtick.labelsize': 18, 'ytick.labelsize': 18,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.edgecolor': '#cfcfcf', 'axes.linewidth': 0.9,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

STAGES = ['Reach', 'Grasp', 'Transport', 'Release']


def frame(ax, gridlines=True):
    """Range frame: keep the baseline, drop the box."""
    for s in ('top', 'left', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=0)
    if gridlines:
        ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)


def stage_bands(ax, ys=(0.22, 0.47, 0.72, 0.97)):
    t = ax.get_yaxis_transform()
    for name, y in zip(STAGES, ys):
        ax.text(0.005, y, name, ha='left', va='top', transform=t,
                fontsize=15, color=QUIET, fontstyle='italic', zorder=0)


def bars(ax, x, totals, lo, hi, focal, width=0.32, labels=True):
    for i, t in enumerate(totals):
        c = ACCENT if focal[i] else CONTEXT
        ax.bar(x[i], t if t > 0 else 0.012, width, color=c, linewidth=0, zorder=2)
    ax.errorbar(x, totals, yerr=[np.array(totals) - np.array(lo),
                                 np.array(hi) - np.array(totals)],
                fmt='none', ecolor='#8a8a8a', capsize=4, lw=1.1, zorder=5)
    if labels:
        for xi, t, l, hh in zip(x, totals, lo, hi):
            off = width * 0.1 if hh - l > 0 else 0
            ax.text(xi + off, t + 0.012, f'{int(round(t*100))}%',
                    ha='left' if hh - l > 0 else 'center', va='bottom',
                    fontsize=16, color=INK if t == max(totals) else MUTED)


# ---------------------------------------------------------------- horizon
horizon = [3, 6, 9, 12, 15]
h_tot = [0.25, 0.80, 0.50, 0.25, 0.25]
h_min = [0.25, 0.50, 0.25, 0.25, 0.25]
h_max = [0.25, 1.00, 1.00, 0.25, 0.25]

fig, ax = plt.subplots(figsize=(8.5, 5.4))
x = np.arange(len(horizon)) * 0.65
bars(ax, x, h_tot, h_min, h_max, [h == 6 for h in horizon])
stage_bands(ax)
ax.set_xlabel('Actions executed per inference call', labelpad=10)
ax.set_ylabel('Average task progress', labelpad=12)
ax.set_xticks(x); ax.set_xticklabels(horizon)
ax.set_xlim(-0.62, x[-1] + 0.36); ax.set_ylim(0, 1.08)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
frame(ax)
plt.tight_layout(); plt.savefig(f'{OUT}/eval_horizon.png', bbox_inches='tight'); plt.close()

# ----------------------------------------------------------- dataset size
sizes = [0, 1, 10, 20, 30, 50]
d_tot = [0.00, 0.00, 0.80, 0.25, 0.25, 0.25]
d_min = [0.00, 0.00, 0.50, 0.25, 0.25, 0.25]
d_max = [0.00, 0.00, 1.00, 0.25, 0.25, 0.25]

fig, ax = plt.subplots(figsize=(8.5, 5.4))
x = np.arange(len(sizes)) * 0.65
bars(ax, x, d_tot, d_min, d_max, [s == 10 for s in sizes])
stage_bands(ax)
ax.set_xlabel('Dataset size (episodes)', labelpad=10)
ax.set_ylabel('Average task progress', labelpad=12)
ax.set_xticks(x); ax.set_xticklabels(sizes)
ax.set_xlim(-0.42, x[-1] + 0.36); ax.set_ylim(0, 1.08)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
frame(ax)
plt.tight_layout(); plt.savefig(f'{OUT}/eval_dataset_size.png', bbox_inches='tight'); plt.close()

# ------------------------------------------------------ ID vs OOD + loss
# Was a dual-axis chart. Split into two panels on a shared x axis: the whole
# point of the section is that the loss does NOT track the score, and a dual
# axis invites exactly the correlation the text is arguing against.
steps = [120, 150, 180, 210]
id_s  = [0.00, 0.80, 0.65, 0.75]
ood_s = [0.00, 0.70, 0.65, 0.25]
loss  = [0.0105, 0.0085, 0.0071, 0.0084]
id_lo, id_hi   = [0.00, 0.50, 0.25, 0.75], [0.00, 1.00, 0.75, 0.75]
ood_lo, ood_hi = [0.00, 0.50, 0.25, 0.25], [0.00, 0.75, 0.75, 0.25]

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.6), sharex=True,
                              gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.16})
x = np.arange(len(steps)); w = 0.30
ax.bar(x - w/2, [max(v, 0.012) for v in id_s],  w, color=ACCENT,   linewidth=0, zorder=2)
ax.bar(x + w/2, [max(v, 0.012) for v in ood_s], w, color=ACCENT_L, linewidth=0, zorder=2)
ax.errorbar(x - w/2, id_s, yerr=[np.array(id_s)-np.array(id_lo), np.array(id_hi)-np.array(id_s)],
            fmt='none', ecolor='#8a8a8a', capsize=4, lw=1.1, zorder=5)
ax.errorbar(x + w/2, ood_s, yerr=[np.array(ood_s)-np.array(ood_lo), np.array(ood_hi)-np.array(ood_s)],
            fmt='none', ecolor='#8a8a8a', capsize=4, lw=1.1, zorder=5)
# key sitting just above the data, not a boxed legend off to the side
ax.add_patch(plt.Rectangle((-0.55, 1.115), 0.10, 0.045, color=ACCENT, lw=0))
ax.text(-0.40, 1.113, 'in distribution', fontsize=14.5, color=INK, va='bottom')
ax.add_patch(plt.Rectangle((0.72, 1.115), 0.10, 0.045, color=ACCENT_L, lw=0))
ax.text(0.87, 1.113, 'out of distribution', fontsize=14.5, color=MUTED, va='bottom')
stage_bands(ax)
ax.set_ylabel('Task progress', labelpad=10, fontsize=18)
ax.set_ylim(0, 1.22)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
frame(ax)

ax2.plot(x, loss, color=MUTED, lw=1.8, marker='o', ms=5, zorder=3)
ax2.set_ylabel('Training loss', labelpad=10, fontsize=18)
ax2.set_ylim(0, 0.0132)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.006))
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax2.set_xticks(x); ax2.set_xticklabels(steps)
ax2.set_xlabel('Training step', labelpad=10)
ax2.set_xlim(-0.62, x[-1] + 0.62)
frame(ax2)
ax2.annotate('flat while the scores diverge', xy=(2.0, 0.0071), xytext=(0.55, 0.0026),
             fontsize=14, color=MUTED,
             arrowprops=dict(arrowstyle='-', color=QUIET, lw=0.9))
plt.savefig(f'{OUT}/eval_id_vs_ood.png', bbox_inches='tight'); plt.close()

print('regenerated 3 charts')
