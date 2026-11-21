import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiments import ABLATIONS, MAIN_ARMS, MAIN_BACKBONE, PROBE_BACKBONES, SENSITIVITY, SENSITIVITY_BACKBONE, TRANSFER
from util import allocation
from util.divergence import safe_spearman
from util.utils import load_json, save_json, write_csv

MAIN_IDS = {'frozen': 'B0', 'uniform': 'B2', 'random': 'B5a', 'reversed': 'B5b', 'adalora': 'B6a',
            'repsam': 'B6b', 'aquaprobe': 'B7', 'routed': 'B8', 'full': 'B1'}
SIGNALS = {'frozen': 'none', 'uniform': 'none', 'random': 'none', 'reversed': 'reversed marginal',
           'adalora': 'training importance', 'repsam': 'cross-domain cumulative', 'aquaprobe': 'marginal',
           'routed': 'per-cluster', 'full': 'none'}


def get_args_parser():
    parser = argparse.ArgumentParser('AquaProbe result aggregation', add_help=False)
    parser.add_argument('--task', default='all', choices=['all', 'select-gamma'])
    parser.add_argument('--output_dir', default='output', help='root written by tools/run_experiments.py')
    parser.add_argument('--gamma-grid', default=[0.0, 0.5, 1.0, 2.0, 4.0], nargs='+', type=float)
    parser.add_argument('--utility-split', default='test', choices=['test', 'heldout'])
    parser.add_argument('--utility-bootstrap', default=1000, type=int)
    parser.add_argument('--r-tot', default=96, type=int)
    parser.add_argument('--r-max', default=24, type=int)
    return parser


def load_runs(root):
    runs = defaultdict(list)
    for path in sorted((root / 'runs').glob('*/*/*/seed*/metrics.json')):
        m = json.loads(path.read_text())
        runs[(m['backbone'], m['dataset'], m['tag'])].append(m)
    return runs


def stat(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None, None
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def summarise(rs, key):
    return stat([r.get(key) for r in rs])


def probe_summary(root, backbone, tag='default'):
    return load_json(root / 'probe' / f'{backbone}__{tag}.json')


def select_gamma(args, root, runs):
    scores = {}
    for gamma in args.gamma_grid:
        mean, _ = summarise(runs.get((MAIN_BACKBONE, 'suim', f'gamma_select_g{gamma:g}'), []), 'val_miou')
        if mean is not None:
            scores[f'{gamma:g}'] = mean
    selected = float(max(scores, key=scores.get)) if scores else None
    out = {'val_miou': scores, 'selected_gamma': selected, 'n_settings': len(scores)}
    save_json(root / 'results' / 'gamma_selection.json', out)
    return out


def main_table(root, runs):
    rows = []
    for arm in MAIN_ARMS:
        rs = runs.get((MAIN_BACKBONE, 'suim', arm), [])
        if not rs:
            continue
        miou, miou_std = summarise(rs, 'miou')
        rows.append({'id': MAIN_IDS[arm], 'method': arm, 'allocation_signal': SIGNALS[arm], 'n_seeds': len(rs),
                     'trainable_params_m': rs[0]['trainable_params'] / 1e6,
                     'trainable_pct_backbone': rs[0]['trainable_pct_backbone'],
                     'miou_mean': miou, 'miou_std': miou_std, 'boundary_f': summarise(rs, 'boundary_f')[0],
                     'audrc': summarise(rs, 'audrc')[0], 'ranks_seed0': rs[0]['ranks']})
    write_csv(root / 'results' / 'main_results.csv', rows)
    return rows


def transfer_table(root, runs):
    rows = []
    for backbone, dataset, arms in TRANSFER:
        for arm in arms:
            rs = runs.get((backbone, dataset, arm), [])
            if rs:
                mean, std = summarise(rs, 'miou')
                rows.append({'backbone': backbone, 'dataset': dataset, 'method': arm, 'n_seeds': len(rs),
                             'miou': mean, 'miou_std': std, 'audrc': summarise(rs, 'audrc')[0]})
    write_csv(root / 'results' / 'transfer_results.csv', rows)
    return rows


def ablation_table(root, runs):
    rows = []
    for tag, (arm, extra) in ABLATIONS.items():
        rs = runs.get((MAIN_BACKBONE, 'suim', tag), [])
        if rs:
            mean, std = summarise(rs, 'miou')
            rows.append({'tag': tag, 'arm': arm, 'args': ' '.join(extra), 'r_tot': rs[0]['r_tot'],
                         'gamma': rs[0]['gamma'], 'n_seeds': len(rs), 'miou': mean, 'miou_std': std,
                         'ranks_seed0': rs[0]['ranks']})
    write_csv(root / 'results' / 'ablations.csv', rows)
    return rows


def utility_analysis(args, root, runs):
    key = 'miou' if args.utility_split == 'test' else 'val_miou'
    suffix = '' if args.utility_split == 'test' else '_holdout'
    frozen = {r['seed']: r[key] for r in runs.get((MAIN_BACKBONE, 'suim', f'frozen{suffix}'), [])}
    rows = []
    for block in range(1, 13):
        rs = runs.get((MAIN_BACKBONE, 'suim', f'single_block_b{block}{suffix}'), [])
        deltas = [r[key] - frozen[r['seed']] for r in rs if r['seed'] in frozen and r.get(key) is not None]
        if deltas:
            rows.append({'block': block, 'n_seeds': len(deltas), 'utility_miou': float(np.mean(deltas))})
    write_csv(root / 'results' / 'utility_sweep.csv', rows)
    out = {'eval_split': args.utility_split, 'blocks': [r['block'] for r in rows]}
    summary = probe_summary(root, MAIN_BACKBONE)
    if len(rows) == 12 and summary:
        U = np.array([r['utility_miou'] for r in rows])
        marginal = np.array(summary['profile'])
        cumulative = np.array(summary['cumulative'])[1:]
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(args.utility_bootstrap):
            index = rng.integers(0, 12, 12)
            if len(set(index.tolist())) > 2:
                boots.append(safe_spearman(marginal[index], U[index]))
        boots = [b for b in boots if np.isfinite(b)]
        out.update(rho_valid=safe_spearman(marginal, U), rho_cumulative=safe_spearman(cumulative, U),
                   ci95=list(np.percentile(boots, [2.5, 97.5])) if boots else None,
                   best_utility_block=int(np.argmax(U)) + 1, largest_increment_block=int(np.argmax(marginal)) + 1,
                   utility=U, marginal=marginal)
    save_json(root / 'results' / 'utility_validity.json', out)
    return out


def probe_tables(root):
    rows, agreement, sensitivity, profiles = [], {}, {}, []
    for backbone in PROBE_BACKBONES:
        summary = probe_summary(root, backbone)
        if not summary or 'stationarity' not in summary:
            continue
        s = summary['stationarity']
        rows.append({'backbone': backbone, **{k: v for k, v in s.items() if k != 'peak_blocks'},
                     'rectified_fraction': summary['rectified_fraction'], 'profile': summary['profile']})
        knn = probe_summary(root, backbone, 'knn')
        if knn:
            agreement[backbone] = safe_spearman(knn['profile'], summary['profile'])
        for l, v in enumerate(summary['cumulative']):
            profiles.append({'backbone': backbone, 'quantity': 'cumulative', 'block': l, 'value': v})
        for l, v in enumerate(summary['profile'], start=1):
            profiles.append({'backbone': backbone, 'quantity': 'marginal', 'block': l, 'value': v})
    write_csv(root / 'results' / 'stationarity.csv', rows)
    write_csv(root / 'results' / 'divergence_profiles.csv', profiles)
    base = probe_summary(root, SENSITIVITY_BACKBONE)
    for tag in SENSITIVITY:
        variant = probe_summary(root, SENSITIVITY_BACKBONE, tag)
        if base and variant:
            sensitivity[tag] = safe_spearman(variant['profile'], base['profile'])
    probe_dir = root / 'probe'
    return rows, {'statistic_agreement_knn_vs_cka': agreement, 'measurement_sensitivity': sensitivity,
                  'channel_attribution': load_json(probe_dir / 'attribution.json'),
                  'real_image_probe': load_json(probe_dir / 'realprobe.json'),
                  'rendering_fidelity': load_json(probe_dir / 'fid.json')}


def allocation_checks(args, root, runs):
    checks, allocations = {}, []
    summary = probe_summary(root, MAIN_BACKBONE)
    if summary:
        profile = np.array(summary['profile'])
        checks['gamma0_equals_uniform'] = bool(np.array_equal(
            allocation.allocate(profile, args.r_tot, 0.0, 0, args.r_max), allocation.uniform(len(profile), args.r_tot)))
    for arm in ('aquaprobe', 'adalora', 'uniform', 'reversed', 'random', 'repsam'):
        rs = runs.get((MAIN_BACKBONE, 'suim', arm), [])
        if rs:
            ranks = np.mean([r['ranks'] for r in rs], axis=0)
            for l, r in enumerate(ranks, start=1):
                allocations.append({'strategy': arm, 'block': l, 'rank': float(r), 'n_seeds': len(rs)})
    write_csv(root / 'results' / 'rank_allocations.csv', allocations)
    routed = runs.get((MAIN_BACKBONE, 'suim', 'routed'), [])
    static = runs.get((MAIN_BACKBONE, 'suim', 'aquaprobe'), [])
    efficiency = {}
    if static:
        efficiency['static'] = {k: static[0][k] for k in ('adapter_params', 'decoder_params', 'backbone_params',
                                                          'trainable_pct_backbone', 'eq7_params', 'ranks')}
    if routed:
        r = routed[0]
        efficiency['routed'] = {k: r.get(k) for k in ('adapter_params', 'router_params', 'trainable_params',
                                                      'stored_rank', 'active_rank', 'cluster_ranks', 'cluster_sizes',
                                                      'stored_budget')}
        efficiency['routed']['router_accuracy_mean'] = summarise(routed, 'router_accuracy')[0]
        if static:
            efficiency['routed_vs_static_param_gap_pct'] = \
                100.0 * (r['trainable_params'] - static[0]['trainable_params']) / static[0]['trainable_params']
    merges = [r['merge_max_abs_diff'] for rs in runs.values() for r in rs if 'merge_max_abs_diff' in r]
    checks['max_merge_abs_diff'] = max(merges) if merges else None
    save_json(root / 'results' / 'efficiency.json', efficiency)
    return checks


def main(args):
    root = Path(args.output_dir).resolve()
    runs = load_runs(root)
    gamma = select_gamma(args, root, runs)
    print(f'gamma selection: {gamma}')
    if args.task == 'select-gamma':
        return
    main_rows = main_table(root, runs)
    transfer_rows = transfer_table(root, runs)
    ablation_rows = ablation_table(root, runs)
    utility = utility_analysis(args, root, runs)
    stationarity_rows, analysis = probe_tables(root)
    analysis['checks'] = allocation_checks(args, root, runs)
    analysis['utility_validity'] = utility
    analysis['gamma_selection'] = gamma
    analysis['run_counts'] = {'/'.join(k): len(v) for k, v in sorted(runs.items())}
    save_json(root / 'results' / 'analysis.json', analysis)
    print(f'aggregated {sum(len(v) for v in runs.values())} runs: {len(main_rows)} main, {len(transfer_rows)} '
          f'transfer, {len(ablation_rows)} ablation, {len(stationarity_rows)} probe backbones -> {root / "results"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AquaProbe result aggregation', parents=[get_args_parser()])
    main(parser.parse_args())
