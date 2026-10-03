from plot_common import COLORS, load, np, plt, save

s, t = load('source_summary.json'), load('transfer_summary.json')
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), constrained_layout=True, sharey=True)
for ax, role, label in zip(axes, ['train', 'validation', 'source_test'],
                          ['Source train', 'Source validation', 'Source test']):
    for i, condition in enumerate(['EP', 'ES', 'EE']):
        if role == 'source_test':
            r = next(x for x in t['nested_metrics'] if x['condition'] == condition
                     and x['role'] == role and x['metric'] == 'accuracy')
            vals = [r[f'pt{j}_mean'] for j in range(3)]
        else:
            r = next(x for x in s['nested_metrics'] if x['condition'] == condition
                     and x['role'] == role and x['metric'] == 'accuracy' and x['stage'] == 'best')
            vals = r['pretraining_seed_values']
        ax.scatter(i + np.array([-.10, 0, .10]), np.array(vals) * 100,
                   facecolors='white', edgecolors=COLORS[condition], s=29, zorder=3)
        ax.errorbar(i, r['mean'] * 100, yerr=r['pretraining_sample_sd'] * 100,
                    fmt='_', color=COLORS[condition], ms=15, capsize=4, lw=1.3)
    baseline = (next(x for x in t['reference_cells'] if x['reference'] == 'surface'
                     and x['role'] == role)['metrics']['accuracy'] if role == 'source_test'
                else s['surface_reference']['metrics'][role]['accuracy'])
    ax.scatter(3, baseline * 100, marker='D', color='#555555', s=32)
    ax.axhline(50, color='#999999', ls='--', lw=.8)
    ax.set_xticks(range(4), ['EP', 'ES', 'EE', 'Surface'])
    ax.set_xlabel(label)
    ax.set_ylim(46, 95)
axes[0].set_ylabel('Accuracy (%)')
for i, ax in enumerate(axes):
    ax.text(-.12, 1.03, chr(97 + i), transform=ax.transAxes, fontweight='bold', fontsize=11)
save(fig, 'fig3_source')
