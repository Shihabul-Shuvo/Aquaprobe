import argparse
import time
from pathlib import Path

import numpy as np
import torch

from util import *
from models.backbones import backbone_meta, block_features, load_backbone
from util import divergence as D
from util.datasets import (UIEBPairs, center_crop, normalise, read_rgb, resize_shorter, suim_images,
                           uieb_available)
from util.render import restore_channel

TASKS = ('profile', 'attribution', 'realprobe', 'fid', 'crossdomain')


def get_args_parser():
    parser = argparse.ArgumentParser('AquaProbe measurement script', add_help=False)
    parser.add_argument('--task', default='profile', choices=TASKS, type=str)
    parser.add_argument('--backbone', default='dinov2_vitb14', choices=['dinov2_vitb14', 'clip_vitb16', 'sam_vitb'],
                        type=str)
    parser.add_argument('--input-size', default=None, type=int,
                        help='encoder input resolution, defaults to the pretrained resolution')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--rows', default='mean', choices=['mean', 'cls', 'tokens'], type=str,
                        help='rows of the similarity matrices: mean-pooled patch tokens, class token or tokens')
    parser.add_argument('--feature-dtype', default='float32', choices=['float32', 'float16'], type=str)
    parser.add_argument('--metrics', default=['cka', 'knn'], nargs='+', choices=['cka', 'knn'], type=str,
                        help='similarity statistics; the first one receives bootstrap and null')
    parser.add_argument('--knn-k', default=10, type=int)
    parser.add_argument('--n-bootstrap', default=20, type=int)
    parser.add_argument('--n-null', default=200, type=int)
    parser.add_argument('--null-draws', default=1000, type=int)
    parser.add_argument('--null-condition', default=['1C', '1.0'], nargs=2, type=str)
    parser.add_argument('--null-quantile', default=0.95, type=float)
    parser.add_argument('--dispersion-threshold', default=1.0, type=float)
    parser.add_argument('--n-images', default=None, type=int, help='random subset of the probe pool')
    parser.add_argument('--half-grid', action='store_true', help='use every other water type')
    parser.add_argument('--point-only', action='store_true', help='skip bootstrap and null')
    parser.add_argument('--tag', default='default', type=str)
    parser.add_argument('--attribution-water-types', default=['I', '9C'], nargs='+', type=str)
    parser.add_argument('--attribution-range', default=1.0, type=float)
    parser.add_argument('--fid-range', default=1.0, type=float)
    parser.add_argument('--fid-datasets', default=['suim', 'uieb'], nargs='+', choices=['suim', 'uieb'], type=str)
    parser.add_argument('--fid-batch-size', default=32, type=int)
    parser.add_argument('--realprobe-groups', default=3, type=int)
    return parser


def prepare(x, size):
    return center_crop(resize_shorter(x, size), size)


class Encoder(object):
    def __init__(self, args, rows=None):
        self.device = torch.device(args.device)
        self.model = load_backbone(args.backbone, args.input_size).to(self.device)
        self.meta = backbone_meta(self.model)
        self.rows = rows or args.rows
        self.dtype = torch.float16 if args.feature_dtype == 'float16' else torch.float32

    @torch.no_grad()
    def __call__(self, images):
        x = torch.stack([normalise(im.float(), self.meta['mean'], self.meta['std']) for im in images]).to(self.device)
        return block_features(self.model, x, self.rows).to('cpu', self.dtype)


def nyu_batches(args, size, device, n_images=None):
    dataset = build_nyu(args)
    indices = list(range(len(dataset)))
    if n_images and n_images < len(dataset):
        indices = sorted(np.random.default_rng(args.seed).permutation(len(dataset))[:n_images].tolist())
    for start in range(0, len(indices), args.batch_size):
        items = [dataset[i] for i in indices[start:start + args.batch_size]]
        clean = torch.stack([prepare(rgb, size) for rgb, _ in items])
        depth = torch.stack([prepare(d, size) for _, d in items])
        yield clean.to(device), depth.to(device)


@torch.no_grad()
def collect(encoder, batches, variants):
    out = {'clean': []}
    out.update({name: [] for name in variants})
    for clean, depth in batches:
        out['clean'].append(encoder(clean))
        for name, fn in variants.items():
            out[name].append(encoder(fn(clean, depth)))
    for name in list(out):
        parts = out.pop(name)
        out[name] = parts[0] if len(parts) == 1 else torch.cat(parts)
    return out


def gap_curve(clean, rendered, sim, index=None):
    if index is not None:
        index = index.to(clean.device)
        clean, rendered = clean[index], rendered[index]
    return D.cumulative_gap(clean, rendered, sim)


def measure_field(args, clean, rendered, conditions, metric, bootstrap, device):
    sim = D.similarity_fn(metric, args.knn_k)
    clean_d = clean.to(device)
    n_rows = clean.shape[0]
    boots = D.bootstrap_indices(n_rows, args.n_bootstrap, args.seed) if bootstrap else []
    G = np.zeros((len(conditions), clean.shape[1]))
    boot_G = np.zeros((len(boots), len(conditions), clean.shape[1]))
    for k, condition in enumerate(conditions):
        rend_d = rendered[condition].to(device)
        G[k] = gap_curve(clean_d, rend_d, sim)
        for b, index in enumerate(boots):
            boot_G[b, k] = gap_curve(clean_d, rend_d, sim, index)
        del rend_d
    out = {'G': G, 'raw': D.raw_increment(G), 'field': D.marginal_gap(G)}
    if bootstrap:
        out['boot_cumulative'] = boot_G.mean(1)
        out['boot_profile'] = D.marginal_gap(boot_G).mean(1)
        rend_d = rendered[null_condition(args)].to(device)
        null_boots = D.bootstrap_indices(n_rows, args.n_null, args.seed + 1)
        null_G = np.stack([gap_curve(clean_d, rend_d, sim, index) for index in null_boots])
        out['null_profiles'] = D.normalised_profile(D.marginal_gap(null_G))
        out['null_pdi'] = D.pdi_null(out['null_profiles'], len(conditions), args.null_draws, args.seed)
    return out


def null_condition(args):
    return str(args.null_condition[0]), float(args.null_condition[1])


def run_profile(args):
    device = torch.device(args.device)
    renderer = build_renderer(args)
    water = args.water_types[::2] if args.half_grid else args.water_types
    conditions = condition_grid(water, args.ranges)
    needed = list(dict.fromkeys(conditions + ([null_condition(args)] if not args.point_only else [])))
    encoder = Encoder(args)
    size = encoder.meta['resolution']
    variants = {c: (lambda J, Z, c=c: renderer(J, Z, c)) for c in needed}
    start = time.time()
    features = collect(encoder, nyu_batches(args, size, device, args.n_images), variants)
    clean = features.pop('clean')
    print(f'features: {clean.shape[0]} rows x {clean.shape[1]} depths x {clean.shape[2]} dims '
          f'in {time.time() - start:.1f}s', flush=True)
    output_dir = Path(args.output_dir)
    for position, metric in enumerate(args.metrics):
        primary = position == 0 and not args.point_only
        tag = args.tag if metric == 'cka' else (args.tag if args.tag != 'default' else metric)
        result = measure_field(args, clean, features, conditions, metric, primary, device)
        settings = {'backbone': args.backbone, 'tag': tag, 'metric': metric, 'rows': args.rows,
                    'z_min': args.z_min, 'z_max': args.z_max, 'n_rows': int(clean.shape[0]),
                    'n_images': args.n_images, 'water_types': list(water), 'ranges': args.ranges,
                    'resolution': size, 'veiling': [args.veiling_mode, args.veiling_level, args.veiling_exponent]}
        summary = {'settings': settings,
                   'rectified_fraction': float((result['raw'] < 0).mean()),
                   'rectified_blocks': sorted(set((np.where(result['raw'] < 0)[1] + 1).tolist())),
                   'profile': result['field'].mean(0), 'cumulative': result['G'].mean(0)}
        if primary:
            summary['stationarity'] = D.stationarity(result['field'], conditions, result['null_pdi'],
                                                     args.null_quantile, args.dispersion_threshold)
            boot = result['boot_profile']
            summary['profile_boot_std'] = boot.std(0, ddof=1) if len(boot) > 1 else None
        np.savez(output_dir / f'{args.backbone}__{tag}.npz', conditions=np.array(conditions, dtype=object), **result)
        utils.save_json(output_dir / f'{args.backbone}__{tag}.json', summary)
        print(f'[{args.backbone}/{tag}] profile {np.round(summary["profile"], 4).tolist()}', flush=True)
        if 'stationarity' in summary:
            s = summary['stationarity']
            print(f'[{args.backbone}/{tag}] peak {s["peak_block_min"]}-{s["peak_block_max"]} '
                  f'std {s["peak_block_std"]:.2f} PDI {s["pdi"]:.4f} ratio {s["pdi_null_ratio"]:.2f} '
                  f'rho_w {s["spearman_peak_water_type"]}', flush=True)


def run_attribution(args):
    device = torch.device(args.device)
    renderer = build_renderer(args)
    encoder = Encoder(args)
    size = encoder.meta['resolution']
    variants = {}
    for w in args.attribution_water_types:
        condition = (str(w), float(args.attribution_range))
        variants[(w, 'none')] = lambda J, Z, c=condition: renderer(J, Z, c)
        for index, channel in enumerate('RGB'):
            variants[(w, channel)] = lambda J, Z, c=condition, i=index: restore_channel(J, renderer(J, Z, c), i)
    features = collect(encoder, nyu_batches(args, size, device), variants)
    clean = features.pop('clean').to(device)
    sim = D.similarity_fn('cka')
    boots = D.bootstrap_indices(clean.shape[0], args.n_bootstrap, args.seed)
    entries = []
    for w in args.attribution_water_types:
        rendered = features[(w, 'none')].to(device)
        for channel in 'RGB':
            restored = features[(w, channel)].to(device)

            def attribution(index=None):
                a = D.marginal_gap(gap_curve(clean, rendered, sim, index)).sum()
                b = D.marginal_gap(gap_curve(clean, restored, sim, index)).sum()
                return float(a - b)

            point = attribution()
            samples = [attribution(index) for index in boots]
            lo, hi = np.percentile(samples, [2.5, 97.5]) if samples else (np.nan, np.nan)
            entries.append({'water_type': w, 'channel': channel, 'value': point, 'ci95': [lo, hi]})
            print(f'A_{channel}({w}) = {point:.4f} [{lo:.4f}, {hi:.4f}]', flush=True)
    utils.save_json(Path(args.output_dir) / 'attribution.json',
                    {'backbone': args.backbone, 'range': args.attribution_range, 'entries': entries})


def run_realprobe(args):
    if not uieb_available(args):
        raise SystemExit('UIEB raw/reference pairs are not available')
    device = torch.device(args.device)
    encoder = Encoder(args)
    size = encoder.meta['resolution']
    dataset = UIEBPairs(args.uieb_raw, args.uieb_reference, args.limit)
    clean, raw, quality, rows_per_image = [], [], [], []
    for start in range(0, len(dataset), args.batch_size):
        items = [dataset[i] for i in range(start, min(len(dataset), start + args.batch_size))]
        fc = encoder([prepare(ref, size) for ref, _ in items])
        fr = encoder([prepare(r, size) for _, r in items])
        clean.append(fc)
        raw.append(fr)
        rows_per_image.append(fc.shape[0] // len(items))
        quality += [D.uciqe(r) for _, r in items]
    clean, raw = torch.cat(clean).to(device), torch.cat(raw).to(device)
    per_image = rows_per_image[0]
    sim = D.similarity_fn('cka')
    increment = D.marginal_gap(gap_curve(clean, raw, sim))
    synthetic = np.load(Path(args.output_dir) / f'{args.backbone}__default.npz', allow_pickle=True)
    field = synthetic['field']
    conditions = [(str(w), float(s)) for w, s in synthetic['conditions']]
    distances = [D.jsd(D.normalised_profile(increment), p) for p in D.normalised_profile(field)]
    best = int(np.nanargmin(distances))
    quality = np.array(quality)
    order = np.argsort(-quality, kind='stable')
    groups = []
    for name, members in zip(('low', 'mid', 'high'), np.array_split(order, args.realprobe_groups)):
        if len(members) == 0:
            groups.append({'turbidity': name, 'n': 0, 'peak_block': None})
            continue
        rows = torch.cat([torch.arange(m * per_image, (m + 1) * per_image) for m in members.tolist()]).to(device)
        dg = D.marginal_gap(gap_curve(clean, raw, sim, rows))
        groups.append({'turbidity': name, 'n': int(len(members)), 'peak_block': int(np.argmax(dg)) + 1,
                       'uciqe_mean': float(quality[members].mean())})
    result = {'backbone': args.backbone, 'n_pairs': len(dataset), 'increment': increment,
              'spearman_vs_synthetic': D.safe_spearman(increment, field.mean(0)),
              'best_matching_condition': list(conditions[best]), 'jsd_to_conditions': distances,
              'uciqe_groups': groups}
    utils.save_json(Path(args.output_dir) / 'realprobe.json', result)
    print(f'UIEB: rho={result["spearman_vs_synthetic"]} best={conditions[best]} groups={groups}', flush=True)


def run_fid(args):
    from util import fid as FID
    device = torch.device(args.device)
    renderer = build_renderer(args)
    inception = FID.load_inception(device)
    nyu = build_nyu(args)
    rendered_stats = {}
    for w in args.water_types:
        images = []
        for i in range(len(nyu)):
            rgb, d = nyu[i]
            rendered = renderer(rgb[None], d[None], (w, float(args.fid_range)))[0]
            images.append(center_crop(resize_shorter(rendered, 299), 299))
        feats = FID.activations(inception, torch.stack(images), device, args.fid_batch_size)
        rendered_stats[w] = FID.statistics(feats)
    out = utils.load_json(Path(args.output_dir) / 'fid.json')
    for name in args.fid_datasets:
        if name == 'suim':
            paths, _ = suim_images(args.suim_path, 'train', args.limit)
        else:
            folder = Path(args.uieb_raw)
            paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg'))
            paths = paths if args.limit is None else paths[:args.limit]
        real = []
        for start in range(0, len(paths), args.fid_batch_size):
            batch = [center_crop(resize_shorter(read_rgb(p), 299), 299) for p in paths[start:start + args.fid_batch_size]]
            real.append(FID.activations(inception, torch.stack(batch), device, args.fid_batch_size))
        mu, sigma = FID.statistics(np.concatenate(real))
        scores = {w: FID.frechet_distance(mu, sigma, *rendered_stats[w]) for w in args.water_types}
        finite = {w: v for w, v in scores.items() if np.isfinite(v)}
        ranked = sorted(finite, key=finite.get)
        out[name] = {'range': args.fid_range, 'n_real': len(paths), 'n_rendered': len(nyu), 'fid': scores,
                     'best': ranked[0] if ranked else None, 'second': ranked[1] if len(ranked) > 1 else None}
        print(f'FID {name}: {scores}', flush=True)
    utils.save_json(Path(args.output_dir) / 'fid.json', out)


def run_crossdomain(args):
    encoder = Encoder(args)
    size = encoder.meta['resolution']
    nyu = build_nyu(args)
    paths, _ = suim_images(args.suim_path, 'train', args.limit)
    n = min(len(nyu), len(paths))
    terrestrial, underwater = [], []
    for start in range(0, n, args.batch_size):
        stop = min(n, start + args.batch_size)
        terrestrial.append(encoder([prepare(nyu[i][0], size) for i in range(start, stop)]))
        underwater.append(encoder([prepare(read_rgb(paths[i]), size) for i in range(start, stop)]))
    device = torch.device(args.device)
    G = gap_curve(torch.cat(terrestrial).to(device), torch.cat(underwater).to(device), D.similarity_fn('cka'))
    np.save(Path(args.output_dir) / f'{args.backbone}__crossdomain.npy', G)
    print(f'cross-domain cumulative gap {np.round(G, 4).tolist()}', flush=True)


def main(args):
    print(args)
    torch.set_grad_enabled(False)
    {'profile': run_profile, 'attribution': run_attribution, 'realprobe': run_realprobe,
     'fid': run_fid, 'crossdomain': run_crossdomain}[args.task](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AquaProbe measurement script',
                                     parents=[get_args_parser(), get_data_args_parser(), get_render_args_parser(),
                                              get_runtime_args_parser()])
    parser.set_defaults(output_dir='output/probe')
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
