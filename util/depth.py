from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_midas(model_type, device):
    model = torch.hub.load('intel-isl/MiDaS', model_type, trust_repo=True).to(device).eval()
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    transform = transforms.dpt_transform if model_type in ('DPT_Large', 'DPT_Hybrid') else transforms.small_transform
    return model, transform


@torch.no_grad()
def inverse_depth(model, transform, image, device):
    disparity = model(transform(image).to(device))
    disparity = F.interpolate(disparity[:, None], size=image.shape[:2], mode='bicubic', align_corners=False)[0, 0]
    return disparity.float().cpu().numpy()


def depth_from_inverse(disparity):
    lo, hi = float(disparity.min()), float(disparity.max())
    return 1.0 - (disparity - lo) / max(hi - lo, 1e-6)


class DepthEstimator(object):
    def __init__(self, model_type, device):
        self.device = device
        self.model, self.transform = load_midas(model_type, device)

    def __call__(self, image):
        return depth_from_inverse(inverse_depth(self.model, self.transform, image, self.device))


def pending(paths, out_dir, overwrite=False):
    return [p for p in paths if overwrite or not (Path(out_dir) / (Path(p).stem + '.npy')).exists()]


def cache_depth(paths, out_dir, estimator):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = np.asarray(Image.open(path).convert('RGB'))
        np.save(out_dir / (Path(path).stem + '.npy'), estimator(image).astype(np.float16))
    return len(paths)
