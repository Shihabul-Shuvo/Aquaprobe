import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


def load_inception(device):
    from pytorch_fid.inception import InceptionV3
    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()


@torch.no_grad()
def activations(model, images, device, batch_size=32):
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device).float().clamp(0, 1)
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        out.append(model(batch)[0].squeeze(-1).squeeze(-1).cpu())
    return torch.cat(out).double().numpy()


def statistics(features):
    if features.shape[0] < 2:
        return features.mean(0), None
    return features.mean(0), np.cov(features, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    if sigma1 is None or sigma2 is None:
        return float('nan')
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
