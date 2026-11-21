import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.datasets import trashcan_available, uieb_available
from util.options import get_data_args_parser
from util.utils import load_json, save_json

STAGES = ('depth', 'probe', 'sensitivity', 'attribution', 'realprobe', 'fid', 'crossdomain', 'utility', 'gamma',
          'main', 'ablations', 'transfer', 'aggregate', 'report')
PROBE_BACKBONES = ('dinov2_vitb14', 'clip_vitb16', 'sam_vitb')
MAIN_BACKBONE = 'dinov2_vitb14'
MAIN_ARMS = ('frozen', 'uniform', 'random', 'reversed', 'adalora', 'repsam', 'aquaprobe', 'routed', 'full')
GAMMA_ARMS = ('aquaprobe', 'reversed', 'routed')
SENSITIVITY = {
    'cls_token': ['--rows', 'cls'],
    'half_condition_grid': ['--half-grid'],
    'range_0.5_8m': ['--z-min', '0.5', '--z-max', '8.0'],
    'pool_256': ['--n-images', '256'],
}
ABLATIONS = {
    'aquaprobe_gamma0.5': ('aquaprobe', ['--gamma', '0.5']),
    'aquaprobe_gamma2': ('aquaprobe', ['--gamma', '2.0']),
    'aquaprobe_gamma4': ('aquaprobe', ['--gamma', '4.0']),
    'aquaprobe_cumulative': ('aquaprobe', ['--signal', 'cumulative']),
    'aquaprobe_knn': ('aquaprobe', ['--profile-tag', 'knn']),
    'aquaprobe_rmin1': ('aquaprobe', ['--r-min', '1']),
    'uniform_r192': ('uniform', ['--r-tot', '192']),
    'uniform_r288': ('uniform', ['--r-tot', '288']),
    'uniform_r384': ('uniform', ['--r-tot', '384']),
}
TRANSFER = (
    ('sam_vitb', 'suim', ('uniform', 'adalora', 'aquaprobe', 'routed', 'full')),
    ('dinov2_vitb14', 'trashcan', ('uniform', 'adalora', 'aquaprobe', 'routed', 'full')),
)
CROSSDOMAIN_BACKBONES = ('dinov2_vitb14',)
ATTRIBUTION_BACKBONE = 'dinov2_vitb14'
REALPROBE_BACKBONE = 'dinov2_vitb14'
SENSITIVITY_BACKBONE = 'dinov2_vitb14'
UTILITY_BLOCKS = 12


def get_args_parser():
    parser = argparse.ArgumentParser('AquaProbe experiment runner', add_help=False)
    parser.add_argument('--output_dir', default='output', help='root for probe/, runs/, results/, figures/, logs/')
    parser.add_argument('--stages', default=list(STAGES), nargs='+', choices=STAGES)
    parser.add_argument('--seeds', default=[0, 1, 2], nargs='+', type=int)
    parser.add_argument('--random-seeds', default=[0, 1, 2, 3, 4], nargs='+', type=int)
    parser.add_argument('--utility-seeds', default=[0, 1], nargs='+', type=int)
    parser.add_argument('--utility-split', default='test', choices=['test', 'heldout'])
    parser.add_argument('--gamma-seeds', default=[0], nargs='+', type=int)
    parser.add_argument('--gamma-grid', default=[0.0, 0.5, 1.0, 2.0, 4.0], nargs='+', type=float)
    parser.add_argument('--input-sizes', default=[], nargs='*', type=str,
                        help='per-backbone input size overrides, e.g. sam_vitb=512')
    parser.add_argument('--probe-args', default='', type=str, help='extra arguments for probe.py')
    parser.add_argument('--train-args', default='', type=str, help='extra arguments for main.py')
    parser.add_argument('--depth-model', default='DPT_Large', type=str)
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--keep-going', action='store_true')
    return parser


class Runner(object):
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output_dir).resolve()
        self.logs = self.root / 'logs'
        self.logs.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / 'results' / 'pipeline_status.json'
        self.status = {(s['stage'], s['name']): s for s in load_json(self.status_path, default=[])}

    def record(self, stage, name, state, detail='', seconds=0.0):
        self.status[(stage, name)] = {'stage': stage, 'name': name, 'state': state, 'detail': detail,
                                      'seconds': round(seconds, 1)}
        print(f'[{stage}] {name}: {state} {detail}', flush=True)

    def run(self, stage, name, script, script_args, done=None):
        if done is not None and Path(done).exists():
            self.record(stage, name, 'cached', str(done))
            return True
        command = [sys.executable, str(ROOT / script), *map(str, script_args)]
        if self.args.dry_run:
            self.record(stage, name, 'planned', ' '.join(command[1:]))
            return True
        start = time.time()
        log = self.logs / f'{stage}__{name}.log'
        with open(log, 'w') as handle:
            code = subprocess.call(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if code != 0:
            self.record(stage, name, 'failed', f'exit {code}, see {log}', time.time() - start)
            if not self.args.keep_going:
                self.save()
                raise SystemExit(code)
            return False
        self.record(stage, name, 'ok', str(log), time.time() - start)
        return True

    def skip(self, stage, name, reason):
        self.record(stage, name, 'skipped', reason)

    def save(self):
        save_json(self.status_path, list(self.status.values()))


def data_args(args):
    out = ['--nyu-format', args.nyu_format, '--nyu-split', args.nyu_split,
           '--heldout-fraction', args.heldout_fraction, '--heldout-seed', args.heldout_seed]
    for flag, value in (('--nyu-path', args.nyu_path), ('--suim-path', args.suim_path),
                        ('--trashcan-path', args.trashcan_path), ('--uieb-raw', args.uieb_raw),
                        ('--uieb-reference', args.uieb_reference), ('--depth-path', args.depth_path)):
        out += [flag, str(Path(value).resolve())]
    if args.limit is not None:
        out += ['--limit', args.limit]
    return out


def device_args(args):
    return ['--device', args.device] if args.device else []


def size_args(args, backbone):
    sizes = dict(item.split('=') for item in args.input_sizes)
    return ['--input-size', sizes[backbone]] if backbone in sizes else []


def probe_job(runner, args, stage, name, backbone, extra, done):
    command = ['--backbone', backbone, '--output_dir', runner.root / 'probe', *data_args(args), *device_args(args),
               *size_args(args, backbone), *shlex.split(args.probe_args), *extra]
    return runner.run(stage, name, 'probe.py', command, done)


def train_job(runner, args, stage, backbone, dataset, arm, tag, seed, extra=()):
    out = runner.root / 'runs' / backbone / dataset / tag / f'seed{seed}'
    command = ['--model', f'aquaprobe_{backbone}', '--arm', arm, '--data-set', dataset, '--seed', seed, '--tag', tag,
               '--output_dir', out, '--probe-path', runner.root / 'probe', *data_args(args), *device_args(args),
               *size_args(args, backbone), *shlex.split(args.train_args), *extra]
    return runner.run(stage, f'{backbone}-{dataset}-{tag}-s{seed}', 'main.py', command, out / 'metrics.json')


def seeds_for(args, arm):
    return args.random_seeds if arm == 'random' else args.seeds


def selected_gamma(runner):
    value = load_json(runner.root / 'results' / 'gamma_selection.json').get('selected_gamma')
    return 1.0 if value is None else value


def gamma_args(arm, gamma):
    return ['--gamma', gamma] if arm in GAMMA_ARMS else []


def main(args):
    runner = Runner(args)
    probe_dir = runner.root / 'probe'
    stages = set(args.stages)
    has_trashcan = trashcan_available(args)

    if 'depth' in stages:
        for dataset in ('suim', 'trashcan'):
            if dataset == 'trashcan' and not has_trashcan:
                runner.skip('depth', dataset, 'TrashCan not available')
                continue
            runner.run('depth', dataset, 'tools/cache_depth.py',
                       ['--data-set', dataset, '--depth-model', args.depth_model, *data_args(args), *device_args(args)])
    if 'probe' in stages:
        for backbone in PROBE_BACKBONES:
            probe_job(runner, args, 'probe', backbone, backbone, ['--task', 'profile', '--metrics', 'cka', 'knn'],
                      probe_dir / f'{backbone}__knn.json')
    if 'sensitivity' in stages:
        for tag, variant in SENSITIVITY.items():
            probe_job(runner, args, 'sensitivity', tag, SENSITIVITY_BACKBONE,
                      ['--task', 'profile', '--metrics', 'cka', '--point-only', '--tag', tag, *variant],
                      probe_dir / f'{SENSITIVITY_BACKBONE}__{tag}.json')
    if 'attribution' in stages:
        probe_job(runner, args, 'attribution', ATTRIBUTION_BACKBONE, ATTRIBUTION_BACKBONE, ['--task', 'attribution'],
                  probe_dir / 'attribution.json')
    if 'realprobe' in stages:
        if uieb_available(args):
            probe_job(runner, args, 'realprobe', 'uieb', REALPROBE_BACKBONE, ['--task', 'realprobe'],
                      probe_dir / 'realprobe.json')
        else:
            runner.skip('realprobe', 'uieb', f'UIEB reference images not found at {args.uieb_reference}')
    if 'fid' in stages:
        probe_job(runner, args, 'fid', 'suim+uieb', MAIN_BACKBONE, ['--task', 'fid'], probe_dir / 'fid.json')
    if 'crossdomain' in stages:
        for backbone in CROSSDOMAIN_BACKBONES:
            probe_job(runner, args, 'crossdomain', backbone, backbone, ['--task', 'crossdomain'],
                      probe_dir / f'{backbone}__crossdomain.npy')
    if 'utility' in stages:
        suffix = '' if args.utility_split == 'test' else '_holdout'
        extra = [] if args.utility_split == 'test' else ['--holdout']
        for seed in args.utility_seeds:
            train_job(runner, args, 'utility', MAIN_BACKBONE, 'suim', 'frozen', f'frozen{suffix}', seed, extra)
            for block in range(1, UTILITY_BLOCKS + 1):
                train_job(runner, args, 'utility', MAIN_BACKBONE, 'suim', 'single_block',
                          f'single_block_b{block}{suffix}', seed, ['--block', block, *extra])
    if 'gamma' in stages:
        for gamma in args.gamma_grid:
            for seed in args.gamma_seeds:
                train_job(runner, args, 'gamma', MAIN_BACKBONE, 'suim', 'aquaprobe', f'gamma_select_g{gamma:g}', seed,
                          ['--gamma', gamma, '--holdout'])
        runner.run('gamma', 'select', 'tools/aggregate.py',
                   ['--task', 'select-gamma', '--output_dir', runner.root, '--gamma-grid', *args.gamma_grid])
    gamma = selected_gamma(runner)
    if 'main' in stages:
        for arm in MAIN_ARMS:
            for seed in seeds_for(args, arm):
                train_job(runner, args, 'main', MAIN_BACKBONE, 'suim', arm, arm, seed, gamma_args(arm, gamma))
    if 'ablations' in stages:
        for tag, (arm, extra) in ABLATIONS.items():
            extra = list(extra) + (['--gamma', gamma] if arm == 'aquaprobe' and '--gamma' not in extra else [])
            for seed in args.seeds:
                train_job(runner, args, 'ablations', MAIN_BACKBONE, 'suim', arm, tag, seed, extra)
    if 'transfer' in stages:
        for backbone, dataset, arms in TRANSFER:
            if dataset == 'trashcan' and not has_trashcan:
                runner.skip('transfer', f'{backbone}-trashcan', 'TrashCan not available')
                continue
            for arm in arms:
                for seed in seeds_for(args, arm):
                    train_job(runner, args, 'transfer', backbone, dataset, arm, arm, seed, gamma_args(arm, gamma))
    if 'aggregate' in stages:
        runner.run('aggregate', 'results', 'tools/aggregate.py',
                   ['--task', 'all', '--output_dir', runner.root, '--utility-split', args.utility_split,
                    '--gamma-grid', *args.gamma_grid])
    if 'report' in stages:
        runner.run('report', 'outputs', 'tools/report.py', ['--output_dir', runner.root])
    runner.save()
    failed = [s for s in runner.status.values() if s['state'] == 'failed']
    print(f'finished: {len(runner.status)} steps recorded, {len(failed)} failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AquaProbe experiment runner', parents=[get_args_parser(), get_data_args_parser()])
    main(parser.parse_args())
