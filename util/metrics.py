import numpy as np
import torch
import torch.nn.functional as F


class SegMetrics(object):
    def __init__(self, num_classes, ignore_index=255, boundary_tol=0.008, boundary=True):
        self.num_classes, self.ignore_index, self.tol = num_classes, ignore_index, boundary_tol
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)
        self.boundary = boundary
        self.boundary_scores = []

    @torch.no_grad()
    def update(self, pred, target):
        pred, target = pred.cpu(), target.cpu()
        valid = target != self.ignore_index
        index = target[valid] * self.num_classes + pred[valid]
        counts = torch.bincount(index, minlength=self.num_classes ** 2)
        self.confusion += counts.view(self.num_classes, self.num_classes).double()
        if self.boundary:
            for p, t in zip(pred, target):
                self.boundary_scores.append(boundary_f(p, t, self.num_classes, self.tol, self.ignore_index))

    def per_class_iou(self):
        tp = self.confusion.diag()
        union = self.confusion.sum(0) + self.confusion.sum(1) - tp
        return torch.where(union > 0, tp / union.clamp_min(1), torch.full_like(tp, float('nan')))

    def miou(self):
        iou = self.per_class_iou()
        present = ~torch.isnan(iou)
        return 100.0 * iou[present].mean().item() if present.any() else float('nan')

    def boundary_f(self):
        return 100.0 * float(np.mean(self.boundary_scores)) if self.boundary_scores else float('nan')


def _edges(mask):
    m = mask.float()[None, None]
    eroded = -F.max_pool2d(-m, 3, stride=1, padding=1)
    return (m - eroded)[0, 0] > 0


def _dilate(mask, radius):
    return F.max_pool2d(mask.float()[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0] > 0


def boundary_f(pred, target, num_classes, tol=0.008, ignore_index=255):
    valid = target != ignore_index
    radius = max(1, int(round(tol * float(np.hypot(*target.shape)))))
    scores = []
    for c in range(num_classes):
        bp = _edges((pred == c) & valid)
        bt = _edges((target == c) & valid)
        n_pred, n_true = int(bp.sum()), int(bt.sum())
        if n_pred == 0 and n_true == 0:
            continue
        precision = int((bp & _dilate(bt, radius)).sum()) / max(1, n_pred)
        recall = int((bt & _dilate(bp, radius)).sum()) / max(1, n_true)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 1.0


def audrc(ranges, mious):
    s = np.asarray(ranges, dtype=np.float64)
    m = np.asarray(mious, dtype=np.float64)
    order = np.argsort(s)
    return float(np.trapezoid(m[order], s[order]) / (s.max() - s.min()))


def window_starts(length, crop, stride):
    starts = list(range(0, max(length - crop, 0) + 1, stride))
    if starts[-1] + crop < length:
        starts.append(length - crop)
    return starts


@torch.no_grad()
def slide_inference(model, image, crop, num_classes, stride_ratio):
    _, _, height, width = image.shape
    stride = max(1, int(crop * stride_ratio))
    logits = image.new_zeros(1, num_classes, height, width)
    count = image.new_zeros(1, 1, height, width)
    for top in window_starts(height, crop, stride):
        for left in window_starts(width, crop, stride):
            out, _ = model(image[:, :, top:top + crop, left:left + crop])
            logits[:, :, top:top + crop, left:left + crop] += out.float()
            count[:, :, top:top + crop, left:left + crop] += 1
    return logits / count
