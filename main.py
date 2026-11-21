import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
from timm.models import create_model
from torch.utils.data import DataLoader

from util import *
from models import *
from models.adalora import AdaLoRAController
from models.adapters import adapter_parameters, set_active, stored_ranks
from models.aquaprobe import backbone_of
from models.backbones import backbone_meta, backbone_parameters, load_backbone

ARMS = ('frozen', 'uniform', 'random', 'reversed', 'adalora', 'repsam', 'aquaprobe', 'routed', 'full', 'single_block')
STATIC_ARMS = ('uniform', 'random', 'reversed', 'adalora', 'repsam', 'aquaprobe', 'single_block')


def get_args_parser():
    parser = argparse.ArgumentParser('AquaProbe adaptation script', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=40, type=int)

    parser.add_argument('--model', default='aquaprobe_dinov2_vitb14', type=str, metavar='MODEL',
                        choices=['aquaprobe_dinov2_vitb14', 'aquaprobe_clip_vitb16', 'aquaprobe_sam_vitb'],
                        help='Name of model to train')
    parser.add_argument('--input-size', default=None, type=int,
                        help='encoder input resolution, defaults to the pretrained resolution')
    parser.add_argument('--arm', default='aquaprobe', choices=ARMS, type=str, help='adaptation arm')
    parser.add_argument('--data-set', default='suim', choices=['suim', 'trashcan'], type=str)
    parser.add_argument('--probe-path', default='output/probe', type=str,
                        help='directory with the outputs of probe.py')

    parser.add_argument('--r-tot', default=96, type=int, help='total rank budget R_tot')
    parser.add_argument('--gamma', default=1.0, type=float, help='sharpening exponent')
    parser.add_argument('--r-min', default=0, type=int)
    parser.add_argument('--r-max', default=24, type=int)
    parser.add_argument('--signal', default='marginal', choices=['marginal', 'cumulative'], type=str,
                        help='profile that drives the allocation')
    parser.add_argument('--profile-tag', default='default', type=str, help='probe output used for allocation')
    parser.add_argument('--block', default=None, type=int, help='adapted block of the single_block arm')
    parser.add_argument('--probe-rank', default=4, type=int, help='rank of the single_block arm')
    parser.add_argument('--repsam-ratios', default=[4, 2, 1], nargs='+', type=int)

    parser.add_argument('--lora-alpha', default=16.0, type=float)
    parser.add_argument('--lora-init-std', default=0.02, type=float)
    parser.add_argument('--router-clusters', default=3, type=int)
    parser.add_argument('--router-layer', default=2, type=int)
    parser.add_argument('--router-hidden', default=64, type=int)

    parser.add_argument('--adalora-init-r', default=12, type=int)
    parser.add_argument('--adalora-t-init', default=0.1, type=float)
    parser.add_argument('--adalora-t-final', default=0.2, type=float)
    parser.add_argument('--adalora-delta-t', default=10, type=int)
    parser.add_argument('--adalora-beta1', default=0.85, type=float)
    parser.add_argument('--adalora-beta2', default=0.85, type=float)
    parser.add_argument('--adalora-orth-reg', default=0.1, type=float)
    parser.add_argument('--adalora-init-std', default=0.02, type=float)

    parser.add_argument('--lr-adapter', default=3e-4, type=float)
    parser.add_argument('--lr-decoder', default=1e-3, type=float)
    parser.add_argument('--lr-router', default=1e-3, type=float)
    parser.add_argument('--lr-full', default=3e-5, type=float)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--lambda-dice', default=1.0, type=float)
    parser.add_argument('--lambda-route', default=1.0, type=float)
    parser.add_argument('--render-prob', default=0.5, type=float, help='probability of render augmentation')

    parser.add_argument('--stride-ratio', default=0.6667, type=float, help='sliding-window stride / crop')
    parser.add_argument('--boundary-tol', default=0.008, type=float, help='boundary tolerance / image diagonal')
    parser.add_argument('--holdout', action='store_true',
                        help='train without the held-out slice and evaluate on it')
    parser.add_argument('--tag', default='', type=str, help='run name recorded in metrics.json, defaults to the arm')
    parser.add_argument('--print-freq', default=20, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    return parser


def probe_run(args, backbone, tag='default'):
    path = Path(args.probe_path) / f'{backbone}__{tag}.npz'
    if not path.exists():
        raise FileNotFoundError(f'probe output missing: {path}')
    return np.load(path, allow_pickle=True)


def allocation_profile(args, backbone, signal, tag):
    run = probe_run(args, backbone, tag)
    return run['field'].mean(0) if signal == 'marginal' else run['G'].mean(0)[1:]


def build_allocation(args, backbone, num_blocks, dim):
    kw = dict(gamma=args.gamma, r_min=args.r_min, r_max=args.r_max)
    ranks, cluster_ranks, cluster_of, info = np.zeros(num_blocks, dtype=int), None, {}, {}
    if args.arm == 'uniform':
        ranks = allocation.uniform(num_blocks, args.r_tot)
    elif args.arm == 'random':
        ranks = allocation.random_allocation(num_blocks, args.r_tot, args.seed, args.r_min, args.r_max)
    elif args.arm in ('reversed', 'aquaprobe'):
        profile = allocation_profile(args, backbone, args.signal, args.profile_tag)
        rule = allocation.reversed_profile if args.arm == 'reversed' else allocation.allocate
        ranks = rule(profile, args.r_tot, **kw)
        info['profile'] = profile
    elif args.arm == 'repsam':
        gap = np.load(Path(args.probe_path) / f'{backbone}__crossdomain.npy')
        ranks, bands, order = allocation.repsam_bands(gap[1:], args.r_tot, args.repsam_ratios, args.r_min,
                                                      args.r_max)
        info.update(cross_domain_gap=gap, bands=bands, band_order=order)
    elif args.arm == 'single_block':
        ranks = allocation.single_block(num_blocks, args.block, args.probe_rank)
    elif args.arm == 'routed':
        run = probe_run(args, backbone, args.profile_tag)
        conditions = [(str(w), float(s)) for w, s in run['conditions']]
        field = run['field']
        labels = allocation.cluster_conditions(field, args.router_clusters)
        profiles = [field[labels == m].mean(0) for m in range(args.router_clusters)]
        router_params = allocation.router_parameters(dim, args.router_hidden, args.router_clusters)
        stored_budget = args.r_tot - allocation.router_rank_charge(router_params, dim)
        cluster_ranks, ranks, active = allocation.nested_allocations(profiles, stored_budget, **kw)
        cluster_of = {c: int(m) for c, m in zip(conditions, labels)}
        info.update(cluster_ranks=cluster_ranks, active_rank=int(active), stored_rank=int(ranks.sum()),
                    stored_budget=int(stored_budget), cluster_profiles=profiles,
                    cluster_sizes=[int((labels == m).sum()) for m in range(args.router_clusters)],
                    cluster_of=[[w, s, m] for (w, s), m in cluster_of.items()])
    return np.asarray(ranks), cluster_ranks, cluster_of, info


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in names or k == 'cluster_ranks'}


def main(args):
    print(args)
    device = torch.device(args.device)
    utils.seed_everything(args.seed)
    backbone = backbone_of(args.model)
    output_dir = Path(args.output_dir)

    meta = backbone_meta(load_backbone(backbone, args.input_size, pretrained=False))
    renderer = build_renderer(args)
    conditions = condition_grid(args.water_types, args.ranges)
    full_train, nb_classes = build_dataset('train', args, meta, renderer)
    train_idx, held_idx = heldout_split(len(full_train), args.heldout_fraction, args.heldout_seed)
    train_idx = train_idx if args.holdout else None
    steps_per_epoch = max(1, (len(train_idx) if train_idx is not None else len(full_train)) // args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    ranks, cluster_ranks, cluster_of, info = build_allocation(args, backbone, meta['num_blocks'], meta['dim'])
    print(f'Creating model: {args.model} ({args.arm}), ranks {ranks.tolist()}')
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=nb_classes,
        input_size=args.input_size,
        ranks=ranks,
        cluster_ranks=cluster_ranks,
        adapter='svd' if args.arm == 'adalora' else 'lora',
        lora_alpha=args.lora_alpha,
        lora_init_std=args.lora_init_std,
        adalora_init_r=args.adalora_init_r,
        adalora_init_std=args.adalora_init_std,
        router_layer=args.router_layer,
        router_hidden=args.router_hidden,
        full_finetune=args.arm == 'full',
    )
    controller = None
    if args.arm == 'adalora':
        controller = AdaLoRAController(model.backbone, 4 * args.r_tot, total_steps, args.adalora_t_init,
                                       args.adalora_t_final, args.adalora_delta_t, args.adalora_beta1,
                                       args.adalora_beta2, args.adalora_orth_reg)
    model.to(device)

    dataset_train, _ = build_dataset('train', args, meta, renderer, train=True, conditions=conditions,
                                     cluster_of=cluster_of, indices=train_idx)
    generator = torch.Generator().manual_seed(args.seed)
    data_loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=args.pin_mem and device.type == 'cuda', generator=generator,
        persistent_workers=args.num_workers > 0,
    )

    param_groups = [{'params': list(model.decoder.parameters()), 'lr': args.lr_decoder}]
    adapters = [p for p in model.backbone.parameters() if p.requires_grad]
    if adapters:
        param_groups.append({'params': adapters, 'lr': args.lr_full if args.arm == 'full' else args.lr_adapter})
    if model.router is not None:
        param_groups.append({'params': list(model.router.parameters()), 'lr': args.lr_router})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    lr_scheduler = utils.cosine_with_warmup(optimizer, args.warmup_epochs * len(data_loader_train),
                                            args.epochs * len(data_loader_train))
    criterion = SegmentationLoss(nb_classes, args.lambda_dice, args.lambda_route)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if controller is not None and checkpoint.get('controller') is not None:
            controller.load_state_dict(checkpoint['controller'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            step = checkpoint.get('step', 0)

    history = []
    start_time = time.time()
    if not args.eval:
        print(f'Start training for {args.epochs} epochs')
        for epoch in range(args.start_epoch, args.epochs):
            train_stats, step = train_one_epoch(
                model, criterion, data_loader_train, optimizer, lr_scheduler, device, epoch, step,
                controller=controller, amp=args.amp, print_freq=args.print_freq,
            )
            history.append(train_stats.get('loss', float('nan')))
            if args.output_dir:
                torch.save({
                    'state_dict': trainable_state(model),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'controller': controller.state_dict() if controller is not None else None,
                    'epoch': epoch,
                    'step': step,
                    'args': args,
                }, output_dir / 'checkpoint.pth')
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch, 'step': step, 'n_parameters': n_parameters}
                with (output_dir / 'log.txt').open('a') as f:
                    f.write(json.dumps(log_stats) + '\n')
    train_hours = (time.time() - start_time) / 3600

    if controller is not None:
        controller.finalise()
        info['adalora_block_ranks'] = controller.block_ranks(meta['num_blocks'])
        info['adalora_init_triplets'] = controller.init_triplets
        info['adalora_target_triplets'] = controller.target
        ranks = np.array(info['adalora_block_ranks'])

    adapter_params = adapter_parameters(model.backbone)
    router_params = sum(p.numel() for p in model.router.parameters()) if model.router is not None else 0
    trainable = sum(p.numel() for p in model.backbone.parameters()) if args.arm == 'full' else adapter_params
    stored = stored_ranks(model.backbone)
    set_active(model.backbone, None)

    eval_start = time.time()
    if args.holdout:
        dataset_val, _ = build_dataset('train', args, meta, renderer, indices=held_idx)
    else:
        dataset_val, _ = build_dataset(test_split(args.data_set), args, meta, renderer)
    result = {}
    if args.arm in STATIC_ARMS and adapter_params > 0:
        result['merge_max_abs_diff'] = merge_check(model, dataset_val, meta, device)
    clean = evaluate(model, dataset_val, meta, device, args.stride_ratio, args.boundary_tol, args.amp)
    if args.holdout:
        result.update(val_miou=clean.miou(), val_boundary_f=clean.boundary_f(), n_eval=len(dataset_val))
    else:
        score, curve = evaluate_robustness(model, dataset_val, meta, device, args.water_types, args.heldout_ranges,
                                           args.stride_ratio, args.boundary_tol, args.amp)
        result.update(miou=clean.miou(), boundary_f=clean.boundary_f(), per_class_iou=clean.per_class_iou(),
                      audrc=score, heldout_range_miou=curve, n_eval=len(dataset_val))
        if model.router is not None:
            result['router_accuracy'] = router_accuracy(model, dataset_val, meta, conditions, cluster_of, device)
    eval_hours = (time.time() - eval_start) / 3600

    backbone_count = backbone_parameters(model.backbone)
    result.update(arm=args.arm, tag=args.tag or args.arm,
                  backbone=backbone, dataset=args.data_set, seed=args.seed, r_tot=int(args.r_tot), gamma=args.gamma,
                  r_min=args.r_min, r_max=args.r_max, signal=args.signal, profile_tag=args.profile_tag,
                  block=args.block, holdout=args.holdout, ranks=ranks.tolist(), stored_ranks=stored,
                  adapter_params=int(adapter_params), router_params=int(router_params),
                  trainable_params=int(trainable + router_params), backbone_params=int(backbone_count),
                  trainable_pct_backbone=100.0 * (trainable + router_params) / backbone_count,
                  decoder_params=int(sum(p.numel() for p in model.decoder.parameters())),
                  eq7_params=int(8 * meta['dim'] * int(round(float(np.sum(stored if args.arm != 'adalora' else ranks))))),
                  resolution=meta['resolution'], epochs=args.epochs, steps=step, n_train=len(dataset_train),
                  loss_history=history, train_hours=train_hours, eval_hours=eval_hours, **info)
    utils.save_json(output_dir / 'metrics.json', result)
    keys = ('miou', 'boundary_f', 'audrc', 'val_miou', 'router_accuracy', 'merge_max_abs_diff')
    print({k: result[k] for k in keys if k in result})
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training and evaluation time {total_time_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AquaProbe adaptation script',
                                     parents=[get_args_parser(), get_data_args_parser(), get_render_args_parser(),
                                              get_runtime_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
