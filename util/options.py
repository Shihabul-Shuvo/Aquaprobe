import argparse

from .render import HELDOUT_RANGES, RANGES, WATER_TYPES
from .utils import default_device


def get_data_args_parser():
    parser = argparse.ArgumentParser('AquaProbe data options', add_help=False)
    parser.add_argument('--nyu-path', default='data/nyu_depth_v2_labeled.mat', type=str,
                        help='NYU Depth V2 labelled .mat file, or the DenseDepth export root')
    parser.add_argument('--nyu-format', default='labelled_mat', choices=['labelled_mat', 'densedepth'], type=str,
                        help='layout of --nyu-path')
    parser.add_argument('--nyu-split', default='test', type=str, help='DenseDepth split used as the probe pool')
    parser.add_argument('--suim-path', default='data/SUIM', type=str, help='SUIM root with train_val/ and TEST/')
    parser.add_argument('--trashcan-path', default='data/TrashCan', type=str,
                        help='TrashCan 1.0 root with material_version/')
    parser.add_argument('--uieb-raw', default='data/UIEB/raw-890', type=str, help='UIEB raw images')
    parser.add_argument('--uieb-reference', default='data/UIEB/reference-890', type=str,
                        help='UIEB reference images')
    parser.add_argument('--depth-path', default='data/depth', type=str,
                        help='monocular depth cache for SUIM and TrashCan')
    parser.add_argument('--limit', default=None, type=int, help='use only the first N images of every split')
    parser.add_argument('--heldout-fraction', default=0.1, type=float,
                        help='fraction of the training split held out for gamma selection')
    parser.add_argument('--heldout-seed', default=0, type=int)
    return parser


def get_render_args_parser():
    parser = argparse.ArgumentParser('AquaProbe rendering options', add_help=False)
    parser.add_argument('--water-types', default=list(WATER_TYPES), nargs='+', type=str,
                        help='Jerlov water types W')
    parser.add_argument('--ranges', default=list(RANGES), nargs='+', type=float, help='range multipliers S')
    parser.add_argument('--heldout-ranges', default=list(HELDOUT_RANGES), nargs='+', type=float,
                        help='held-out range multipliers used by AUDRC')
    parser.add_argument('--z-min', default=1.0, type=float, help='near end of the range interval in metres')
    parser.add_argument('--z-max', default=4.0, type=float, help='far end of the range interval in metres')
    parser.add_argument('--veiling-mode', default='power', choices=['power', 'constant'], type=str)
    parser.add_argument('--veiling-level', default=0.8, type=float)
    parser.add_argument('--veiling-exponent', default=1.15, type=float)
    return parser


def get_runtime_args_parser():
    parser = argparse.ArgumentParser('AquaProbe runtime options', add_help=False)
    parser.add_argument('--output_dir', default='output', help='path where to save results')
    parser.add_argument('--device', default=default_device(), help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--amp', action='store_true', help='bfloat16 autocast on CUDA')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)
    return parser
