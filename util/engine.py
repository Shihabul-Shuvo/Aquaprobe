import math
import sys

import torch
import torch.nn.functional as F

import util.utils as utils
from models.adapters import merge_adapters
from .datasets import IGNORE_INDEX, center_crop, prepare_input
from .metrics import SegMetrics, audrc, slide_inference


def train_one_epoch(model, criterion, data_loader, optimizer, lr_scheduler, device, epoch, step,
                    controller=None, amp=True, print_freq=20):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = f'Epoch: [{epoch}]'
    for inputs, targets, clusters in metric_logger.log_every(data_loader, print_freq, header):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        clusters = clusters.to(device, non_blocking=True)
        with utils.autocast(device, amp):
            logits, route_logits = model(inputs, clusters)
            loss = criterion(logits, route_logits, targets, clusters)
            if controller is not None:
                loss = loss + controller.orth_regulariser()
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training')
            sys.exit(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if controller is not None:
            controller.step(step)
        step += 1
        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]['lr'])
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, step


@torch.no_grad()
def evaluate(model, dataset, meta, device, stride_ratio=2 / 3, boundary_tol=0.008, amp=True, condition=None,
             metrics=None, boundary=True):
    model.eval()
    metrics = metrics or SegMetrics(dataset.num_classes, IGNORE_INDEX, boundary_tol, boundary)
    for i in range(len(dataset)):
        image, mask = dataset.load(i, condition)
        x = prepare_input(image, meta)[None].to(device)
        with utils.autocast(device, amp):
            logits = slide_inference(model, x, meta['resolution'], dataset.num_classes, stride_ratio)
        logits = F.interpolate(logits.float(), size=mask.shape, mode='bilinear', align_corners=False)
        metrics.update(logits.argmax(1).cpu(), mask[None])
    return metrics


@torch.no_grad()
def evaluate_robustness(model, dataset, meta, device, water_types, heldout_ranges, stride_ratio=2 / 3,
                        boundary_tol=0.008, amp=True):
    curve = {}
    for s in heldout_ranges:
        metrics = SegMetrics(dataset.num_classes, IGNORE_INDEX, boundary_tol, boundary=False)
        for w in water_types:
            evaluate(model, dataset, meta, device, stride_ratio, boundary_tol, amp, (str(w), float(s)), metrics)
        curve[str(s)] = metrics.miou()
    return audrc([float(s) for s in curve], list(curve.values())), curve


@torch.no_grad()
def router_accuracy(model, dataset, meta, conditions, cluster_of, device):
    model.eval()
    correct = total = 0
    for i in range(len(dataset)):
        for condition in conditions:
            image, _ = dataset.load(i, condition)
            x = center_crop(prepare_input(image, meta), meta['resolution'])[None].to(device)
            correct += int(model.route(x).argmax(1).item() == cluster_of[condition])
            total += 1
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def merge_check(model, dataset, meta, device):
    model.eval()
    image, _ = dataset.load(0)
    x = center_crop(prepare_input(image, meta), meta['resolution'])[None].to(device)
    before, _ = model(x)
    merge_adapters(model.backbone)
    after, _ = model(x)
    return float((before - after).abs().max().item())
