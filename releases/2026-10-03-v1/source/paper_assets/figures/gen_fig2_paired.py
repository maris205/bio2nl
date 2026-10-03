from plot_common import load, np, plt, save

d = load('transfer_summary.json')
fig, ax = plt.subplots(figsize=(5.8, 3.2), constrained_layout=True)
colors = ['#0072B2', '#D55E00', '#009E73']
for i, contrast in enumerate(['EP-ES', 'EP-EE', 'ES-EE']):
    r = next(x for x in d['paired_contrasts'] if x['contrast'] == contrast
             and x['role'] == 'target' and x['metric'] == 'auroc')
    for seed, offset in enumerate([-.10, 0, .10]):
        ax.scatter(i + offset, r[f'pt{seed}_difference'], color=colors[seed],
                   marker=['o', 's', '^'][seed], s=35, zorder=3,
                   label=f'PT seed {seed}' if i == 0 else None)
    ax.errorbar(i, r['mean_difference'], yerr=r['pretraining_sample_sd'],
                fmt='_', color='black', ms=17, lw=1.2, capsize=5, zorder=2)
ax.axhline(0, color='#777777', ls='--', lw=.8)
ax.set_xticks(range(3), ['EP−ES\n(primary)', 'EP−EE\n(contextual)', 'ES−EE\n(contextual)'])
ax.set_ylabel('Paired QQP AUC difference')
ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.17), fontsize=8)
save(fig, 'fig2_paired')
