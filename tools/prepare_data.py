import argparse
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.options import get_data_args_parser


def get_args_parser():
    parser = argparse.ArgumentParser('AquaProbe data preparation', add_help=False)
    parser.add_argument('--nyu-zip', default='', type=str, help='NYU Depth V2 DenseDepth archive (nyu_data.zip)')
    parser.add_argument('--suim-zip', default='', type=str, help='SUIM archive')
    return parser


def extract(archive, members, destination, strip):
    count = 0
    for member in members:
        relative = PurePosixPath(member).relative_to(strip)
        target = Path(destination).joinpath(*relative.parts)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(member))
        count += 1
    return count


def take(items, limit):
    return items if limit is None else items[:int(limit)]


def prepare_suim(args):
    with zipfile.ZipFile(args.suim_zip) as archive:
        names = set(archive.namelist())
        total = 0
        for sub in ('train_val', 'TEST'):
            images = sorted(n for n in names if n.startswith(f'{sub}/images/') and n.lower().endswith('.jpg'))
            images = take([n for n in images if f'{sub}/masks/{PurePosixPath(n).stem}.bmp' in names], args.limit)
            masks = [f'{sub}/masks/{PurePosixPath(n).stem}.bmp' for n in images]
            total += extract(archive, images + masks, args.suim_path, '')
    print(f'SUIM: extracted {total} files to {args.suim_path}')


def prepare_nyu(args):
    with zipfile.ZipFile(args.nyu_zip) as archive:
        listing = archive.read(f'nyu_data/data/nyu2_{args.nyu_split}.csv').decode().strip().splitlines()
        pairs = [line.split(',') for line in listing if line.strip()]
        if args.nyu_split == 'test':
            pairs = sorted(pairs)
        members = [f'nyu_data/{p.strip()}' for pair in take(pairs, args.limit) for p in pair]
        count = extract(archive, members, args.nyu_path, 'nyu_data/data')
    print(f'NYU ({args.nyu_split}): extracted {count} files to {args.nyu_path}')


def main(args):
    if args.nyu_zip:
        prepare_nyu(args)
    elif args.nyu_format == 'labelled_mat':
        print(f'NYU labelled .mat {"found" if Path(args.nyu_path).exists() else "MISSING"}: {args.nyu_path}')
    if args.suim_zip:
        prepare_suim(args)
    for name, path in (('UIEB references', args.uieb_reference), ('TrashCan', args.trashcan_path)):
        print(f'{name} {"found" if Path(path).exists() else "MISSING"}: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AquaProbe data preparation', parents=[get_args_parser(), get_data_args_parser()])
    main(parser.parse_args())
