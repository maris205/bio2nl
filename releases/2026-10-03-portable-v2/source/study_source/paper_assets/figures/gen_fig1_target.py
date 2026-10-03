from plot_common import COLORS, load, np, plt, save

d = load('transfer_summary.json')
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), constrained_layout=True)
for ax, metric, ylabel, limits in zip(axes,
        ['auroc', 'balanced_accuracy', 'accuracy'],
        ['QQP AUC', 'QQP balanced accuracy (%)', 'QQP accuracy (%)'],
        [(0.43, 0.76), (45, 56), (29, 75)]):
    factor = 1 if metric == 'auroc' else 100
    for i, condition in enumerate(['EP', 'ES', 'EE']):
        row = next(x for x in d['nested_metrics'] if x['condition'] == condition
                   and x['role'] == 'target' and x['metric'] == metric)
        vals = np.array([row[f'pt{j}_mean'] for j in range(3)]) * factor
        ax.scatter(i + np.array([-.10, 0, .10]), vals, s=29,
                   facecolors='white', edgecolors=COLORS[condition], linewidths=1.3, zorder=3)
        ax.errorbar(i, row['mean'] * factor, yerr=row['pretraining_sample_sd'] * factor,
                    color=COLORS[condition], fmt='_', ms=15, lw=1.5, capsize=4, zorder=2)
    for i, name in enumerate(['surface', 'constant0', 'constant1'], 3):
        row = next(x for x in d['reference_cells'] if x['reference'] == name and x['role'] == 'target')
        ax.scatter(i, row['metrics'][metric] * factor, marker='D', color='#555555', s=32, zorder=3)
    if metric != 'accuracy':
        ax.axhline(.5 * factor, color='#999999', linestyle='--', linewidth=.8)
    ax.set_xticks(range(6), ['EP', 'ES', 'EE', 'Surface', 'C0', 'C1'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*limits)
for i, ax in enumerate(axes):
    ax.text(-.17, 1.03, chr(97 + i), transform=ax.transAxes, fontweight='bold', fontsize=11)
save(fig, 'fig1_target')
