import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

SUIM_CLASSES = ('BW', 'HD', 'PF', 'WR', 'RO', 'RI', 'FV', 'SR')
IGNORE_INDEX = 255
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg')


def take(items, limit):
    items = list(items)
    return items if limit is None else items[:int(limit)]


def to_tensor(image):
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def read_rgb(path):
    return to_tensor(Image.open(path).convert('RGB'))


def resize_shorter(x, size, mode='bilinear'):
    h, w = x.shape[-2:]
    scale = size / min(h, w)
    hw = (max(size, round(h * scale)), max(size, round(w * scale)))
    squeeze = x.dim() == 3
    x = x[None] if squeeze else x
    kwargs = {'align_corners': False} if mode == 'bilinear' else {}
    out = F.interpolate(x.float(), size=hw, mode=mode, **kwargs)
    return out[0] if squeeze else out


def center_crop(x, size):
    h, w = x.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


def normalise(x, mean, std):
    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    return (x - mean) / std


def prepare_input(image, meta):
    return normalise(resize_shorter(image, meta['resolution']), meta['mean'], meta['std'])


def depth_file(depth_root, dataset, split, image_path):
    return Path(depth_root) / dataset / split / (Path(image_path).stem + '.npy')


class NYULabelled(Dataset):
    def __init__(self, mat_path, limit=None):
        import h5py
        self.path = str(mat_path)
        with h5py.File(self.path, 'r') as handle:
            total = handle['images'].shape[0]
        self.indices = take(range(total), limit)
        self.handle = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import h5py
        if self.handle is None:
            self.handle = h5py.File(self.path, 'r')
        k = self.indices[i]
        rgb = np.asarray(self.handle['images'][k]).transpose(0, 2, 1).astype(np.float32) / 255.0
        depth = np.asarray(self.handle['depths'][k]).transpose(1, 0).astype(np.float32)
        return torch.from_numpy(rgb), torch.from_numpy(depth)[None]


class NYUDenseDepth(Dataset):
    def __init__(self, root, split='test', limit=None):
        folder = Path(root) / f'nyu2_{split}'
        if split == 'test':
            colours = sorted(folder.rglob('*_colors.png'))
            pairs = [(c, c.with_name(c.name.replace('_colors.png', '_depth.png'))) for c in colours]
        else:
            pairs = [(c, c.with_suffix('.png')) for c in sorted(folder.rglob('*.jpg'))]
        self.pairs = take([p for p in pairs if p[1].exists()], limit)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        colour, depth = self.pairs[i]
        d = np.asarray(Image.open(depth)).astype(np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        return read_rgb(colour), torch.from_numpy(d)[None]


def build_nyu(args, limit=None):
    limit = args.limit if limit is None else limit
    if args.nyu_format == 'labelled_mat':
        return NYULabelled(args.nyu_path, limit)
    if args.nyu_format == 'densedepth':
        return NYUDenseDepth(args.nyu_path, args.nyu_split, limit)
    raise ValueError(args.nyu_format)


def suim_mask(path):
    rgb = (np.asarray(Image.open(path).convert('RGB')) >= 128).astype(np.int64)
    return rgb[..., 0] * 4 + rgb[..., 1] * 2 + rgb[..., 2]


def suim_images(root, split, limit=None):
    sub = 'train_val' if split == 'train' else 'TEST'
    images = sorted((Path(root) / sub / 'images').glob('*.jpg'))
    masks = Path(root) / sub / 'masks'
    return take([p for p in images if (masks / (p.stem + '.bmp')).exists()], limit), masks


class SegDataset(Dataset):
    def __init__(self, name, root, split, input_size, mean, std, depth_root=None, renderer=None, train=False,
                 render_prob=0.0, conditions=None, cluster_of=None, indices=None, limit=None):
        self.name, self.split = name, split
        self.input_size, self.mean, self.std = input_size, mean, std
        self.depth_root, self.renderer = depth_root, renderer
        self.train, self.render_prob = train, render_prob
        self.conditions = list(conditions or [])
        self.cluster_of = cluster_of or {}
        if name == 'suim':
            self.images, self.mask_dir = suim_images(root, split, limit)
            self.num_classes = len(SUIM_CLASSES)
            self.class_names = list(SUIM_CLASSES)
        elif name == 'trashcan':
            from pycocotools.coco import COCO
            root = Path(root) / 'material_version'
            self.coco = COCO(str(root / f'instances_{split}_trashcan.json'))
            self.ids = take(sorted(self.coco.getImgIds()), limit)
            self.images = [root / split / self.coco.loadImgs(i)[0]['file_name'] for i in self.ids]
            categories = sorted(self.coco.getCatIds())
            self.category_index = {c: k + 1 for k, c in enumerate(categories)}
            self.num_classes = len(categories) + 1
            self.class_names = ['background'] + [self.coco.loadCats(c)[0]['name'] for c in categories]
        else:
            raise ValueError(name)
        if indices is not None:
            self.images = [self.images[i] for i in indices]
            if name == 'trashcan':
                self.ids = [self.ids[i] for i in indices]

    def __len__(self):
        return len(self.images)

    def mask(self, i):
        if self.name == 'suim':
            return suim_mask(self.mask_dir / (self.images[i].stem + '.bmp'))
        info = self.coco.loadImgs(self.ids[i])[0]
        out = np.zeros((info['height'], info['width']), dtype=np.int64)
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.ids[i], iscrowd=None))
        for ann in sorted(annotations, key=lambda a: -a['area']):
            out[self.coco.annToMask(ann) > 0] = self.category_index[ann['category_id']]
        return out

    def depth(self, i, shape):
        d = np.load(depth_file(self.depth_root, self.name, self.split, self.images[i])).astype(np.float32)
        d = torch.from_numpy(d)[None, None]
        return F.interpolate(d, size=tuple(shape), mode='bilinear', align_corners=False)[0]

    def load(self, i, condition=None):
        image = read_rgb(self.images[i])
        if condition is not None:
            image = self.renderer(image[None], self.depth(i, image.shape[-2:])[None], condition)[0]
        return image, torch.from_numpy(self.mask(i))

    def __getitem__(self, i):
        cluster, condition = -1, None
        if self.train and random.random() < self.render_prob:
            condition = random.choice(self.conditions)
            cluster = self.cluster_of.get(condition, -1)
        image, mask = self.load(i, condition)
        image = resize_shorter(image, self.input_size)
        mask = resize_shorter(mask[None].float(), self.input_size, mode='nearest')[0].long()
        if self.train:
            h, w = image.shape[-2:]
            top, left = random.randint(0, h - self.input_size), random.randint(0, w - self.input_size)
            image = image[:, top:top + self.input_size, left:left + self.input_size]
            mask = mask[top:top + self.input_size, left:left + self.input_size]
        return normalise(image, self.mean, self.std), mask, cluster


def test_split(name):
    return 'test' if name == 'suim' else 'val'


def dataset_root(args, name=None):
    name = name or args.data_set
    return args.suim_path if name == 'suim' else args.trashcan_path


def build_dataset(split, args, meta, renderer=None, train=False, conditions=None, cluster_of=None, indices=None,
                  name=None):
    name = name or args.data_set
    dataset = SegDataset(name, dataset_root(args, name), split, meta['resolution'], meta['mean'], meta['std'],
                         depth_root=args.depth_path, renderer=renderer, train=train,
                         render_prob=args.render_prob if train else 0.0, conditions=conditions,
                         cluster_of=cluster_of, indices=indices, limit=args.limit)
    return dataset, dataset.num_classes


class UIEBPairs(Dataset):
    def __init__(self, raw_dir, reference_dir, limit=None):
        raw_dir, reference_dir = Path(raw_dir), Path(reference_dir)
        raw = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.pairs = take([(p, reference_dir / p.name) for p in raw if (reference_dir / p.name).exists()], limit)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        raw, reference = self.pairs[i]
        return read_rgb(reference), read_rgb(raw)


def uieb_available(args):
    return Path(args.uieb_raw).exists() and Path(args.uieb_reference).exists()


def trashcan_available(args):
    return (Path(args.trashcan_path) / 'material_version').exists()


def heldout_split(n, fraction, seed):
    order = np.random.default_rng(seed).permutation(n)
    n_hold = min(n, max(1, int(round(fraction * n))))
    held = sorted(int(i) for i in order[:n_hold])
    train = sorted(int(i) for i in order[n_hold:]) or held
    return train, held
