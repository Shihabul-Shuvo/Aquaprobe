import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.datasets import SegDataset, dataset_root
from util.depth import DepthEstimator, cache_depth, pending
from util.options import get_data_args_parser
from util.utils import default_device


def get_args_parser():
    parser = argparse.ArgumentParser('AquaProbe monocular depth cache', add_help=False)
    parser.add_argument('--data-set', default='suim', choices=['suim', 'trashcan'], type=str)
    parser.add_argument('--depth-model', default='DPT_Large', type=str, help='MiDaS model type')
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(args):
    splits = ('train', 'test') if args.data_set == 'suim' else ('train', 'val')
    jobs = []
    for split in splits:
        dataset = SegDataset(args.data_set, dataset_root(args), split, 16, (0, 0, 0), (1, 1, 1), limit=args.limit)
        out = Path(args.depth_path) / args.data_set / split
        jobs.append((split, out, len(dataset.images), pending(dataset.images, out, args.overwrite)))
    estimator = DepthEstimator(args.depth_model, args.device) if any(todo for _, _, _, todo in jobs) else None
    for split, out, total, todo in jobs:
        n = cache_depth(todo, out, estimator) if todo else 0
        print(f'{args.data_set}/{split}: {n} new depth maps, {total} images, {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AquaProbe monocular depth cache',
                                     parents=[get_args_parser(), get_data_args_parser()])
    main(parser.parse_args())
