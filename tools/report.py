import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.backbones import LABELS
from run_experiments import MAIN_BACKBONE, PROBE_BACKBONES
from util.utils import load_json, read_csv

COLOURS = {'dinov2_vitb14': '#4A7BA7', 'clip_vitb16': '#4E9B94', 'sam_vitb': '#D99A4E'}


def get_args_parser():
    parser = argparse.ArgumentParser('AquaProbe tables and figures', add_help=False)
    parser.add_argument('--output_dir', default='output', help='root written by tools/run_experiments.py')
    parser.add_argument('--task', default='all', choices=['all', 'tables', 'figures'])
    return parser


def fmt(value, digits=1):
    if value in (None, ''):
        return '--'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return '--' if not np.isfinite(value) else f'{value:.{digits}f}'


def csv_rows(root, name):
    path = root / 'results' / name
    return read_csv(path) if path.exists() else []


def tables(root):
    out = root / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    lines = ['\\begin{tabular}{llcccccc}', '\\toprule',
             '& Method & Signal & Params (M) & \\% backbone & mIoU & Bound. F & AUDRC \\\\', '\\midrule']
    for r in csv_rows(root, 'main_results.csv'):
        miou = fmt(r['miou_mean']) + (f'$\\pm${fmt(r["miou_std"])}' if r.get('miou_std') else '')
        lines.append(f'{r["id"]} & {r["method"]} & {r["allocation_signal"]} & {fmt(r["trainable_params_m"], 2)} & '
                     f'{fmt(r["trainable_pct_backbone"], 2)} & {miou} & {fmt(r["boundary_f"])} & {fmt(r["audrc"])} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (out / 'main_results.tex').write_text('\n'.join(lines) + '\n')
    lines = ['\\begin{tabular}{lccccc}', '\\toprule',
             'Backbone & $\\ell^*$ & std($\\ell^*$) & PDI & PDI/null & $\\rho(w,\\ell^*)$ \\\\', '\\midrule']
    for r in csv_rows(root, 'stationarity.csv'):
        lines.append(f'{LABELS.get(r["backbone"], r["backbone"])} & {r["peak_block_min"]}--{r["peak_block_max"]} & '
                     f'{fmt(r["peak_block_std"], 2)} & {fmt(r["pdi"], 3)} & {fmt(r["pdi_null_ratio"], 1)} & '
                     f'{fmt(r["spearman_peak_water_type"], 2)} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (out / 'stationarity.tex').write_text('\n'.join(lines) + '\n')
    lines = ['\\begin{tabular}{lcc}', '\\toprule', 'Run & $R_{\\text{tot}}$ & mIoU \\\\', '\\midrule']
    for r in csv_rows(root, 'ablations.csv'):
        lines.append(f'{r["tag"].replace("_", " ")} & {r["r_tot"]} & {fmt(r["miou"])} \\\\')
    for r in csv_rows(root, 'transfer_results.csv'):
        lines.append(f'{LABELS.get(r["backbone"], r["backbone"])} / {r["dataset"]} / {r["method"]} & 96 & '
                     f'{fmt(r["miou"])} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (out / 'ablations.tex').write_text('\n'.join(lines) + '\n')
    print(f'tables -> {out}')


def figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = root / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6})
    probe_dir = root / 'probe'
    fig, ax = plt.subplots(1, 3, figsize=(6.05, 1.6), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    for backbone in PROBE_BACKBONES:
        path = probe_dir / f'{backbone}__default.npz'
        if not path.exists():
            continue
        run = np.load(path, allow_pickle=True)
        g, m = run['G'].mean(0), run['field'].mean(0)
        colour, label = COLOURS[backbone], LABELS[backbone]
        ax[0].plot(range(len(g)), g, 'o-', ms=2, lw=1, color=colour, label=label)
        ax[1].plot(range(1, len(m) + 1), m, 'o-', ms=2, lw=1, color=colour)
        if 'boot_profile' in run.files and len(run['boot_profile']) > 1:
            sd = run['boot_profile'].std(0, ddof=1)
            ax[1].fill_between(range(1, len(m) + 1), m - sd, m + sd, color=colour, alpha=0.2, lw=0)
        if 'boot_cumulative' in run.files and len(run['boot_cumulative']) > 1:
            sd = run['boot_cumulative'].std(0, ddof=1)
            ax[0].fill_between(range(len(g)), g - sd, g + sd, color=colour, alpha=0.2, lw=0)
    ax[0].set(xlabel=r'block $\ell$', ylabel=r'cumulative $G^{(\ell)}$')
    ax[0].legend(frameon=False, loc='lower right')
    ax[1].set(xlabel=r'block $\ell$', ylabel=r'marginal $\Delta G^{(\ell)}$')
    path = probe_dir / f'{MAIN_BACKBONE}__default.npz'
    if path.exists():
        run = np.load(path, allow_pickle=True)
        field = run['field']
        conditions = [(str(w), float(s)) for w, s in run['conditions']]
        image = ax[2].imshow(field.T, origin='lower', aspect='auto', cmap='viridis',
                             extent=(-0.5, len(conditions) - 0.5, 0.5, field.shape[1] + 0.5))
        ax[2].step(range(len(conditions)), field.argmax(1) + 1, where='mid', color='w', ls='--', lw=1)
        types = list(dict.fromkeys(w for w, _ in conditions))
        per_type = len(conditions) // max(1, len(types))
        ax[2].set_xticks([i * per_type + (per_type - 1) / 2 for i in range(len(types))], types, fontsize=5)
        ax[2].set(xlabel=r'condition $\mathbf{c}$ (turbidity increasing)', ylabel=r'block $\ell$')
        fig.colorbar(image, ax=ax[2], fraction=0.05, pad=0.02).set_label(r'$\Delta G^{(\ell)}(\mathbf{c})$')
    fig.tight_layout(pad=0.3)
    fig.savefig(out / 'fig_gap_overview.pdf')
    fig.savefig(out / 'fig_gap_overview.png', dpi=200)
    plt.close(fig)

    utility = load_json(root / 'results' / 'utility_validity.json')
    fig, ax = plt.subplots(1, 2, figsize=(5.69, 1.6))
    if utility.get('utility') is not None:
        x, y = np.array(utility['marginal']), np.array(utility['utility'])
        ax[0].scatter(x, y, s=12, color='#D99A4E', edgecolor='k', lw=0.4, zorder=3)
        if np.ptp(x) > 0:
            fit = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 10)
            ax[0].plot(xs, np.polyval(fit, xs), '--', color='#8A8A8A', lw=0.8)
        for l, (a, b) in enumerate(zip(x, y), start=1):
            ax[0].annotate(str(l), (a, b), xytext=(2, 2), textcoords='offset points', fontsize=5)
        ax[0].text(0.03, 0.9, rf'$\rho_{{valid}}={fmt(utility.get("rho_valid"), 2)}$', transform=ax[0].transAxes)
    ax[0].set(xlabel=r'mean marginal gap $\Delta\bar G^{(\ell)}$', ylabel=r'utility $U^{(\ell)}$ (mIoU)')
    allocations = defaultdict(dict)
    for r in csv_rows(root, 'rank_allocations.csv'):
        allocations[r['strategy']][int(r['block'])] = float(r['rank'])
    strategies = [('aquaprobe', 'AquaProbe', '#D99A4E'), ('adalora', 'AdaLoRA', '#4E9B94'),
                  ('uniform', 'uniform', '#8A8A8A'), ('reversed', 'reversed', '#4A7BA7')]
    for j, (key, label, colour) in enumerate(strategies):
        if key in allocations:
            blocks = sorted(allocations[key])
            ax[1].bar(np.array(blocks) + (j - 1.5) * 0.21, [allocations[key][b] for b in blocks], 0.21,
                      color=colour, label=label)
    ax[1].set(xlabel=r'block $\ell$', ylabel=r'assigned rank $r^{(\ell)}$')
    ax[1].legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / 'fig_validity_allocation.pdf')
    fig.savefig(out / 'fig_validity_allocation.png', dpi=200)
    plt.close(fig)
    print(f'figures -> {out}')


def main(args):
    root = Path(args.output_dir).resolve()
    if args.task in ('all', 'tables'):
        tables(root)
    if args.task in ('all', 'figures'):
        figures(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AquaProbe tables and figures', parents=[get_args_parser()])
    main(parser.parse_args())
