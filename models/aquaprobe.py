import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model

from .adapters import inject_lora, inject_svd, set_active
from .backbones import BACKBONES, load_backbone, prefix_pooled, token_map

MODEL_PREFIX = 'aquaprobe_'


def _cfg(backbone, **kwargs):
    return {
        'backbone': backbone,
        'timm_name': BACKBONES[backbone],
        'decoder': 'conv1x1',
        'router_layer': 2,
        'router_hidden': 64,
        'router_clusters': 3,
        **kwargs
    }


default_cfgs = {
    'dinov2_vitb14': _cfg('dinov2_vitb14'),
    'clip_vitb16': _cfg('clip_vitb16'),
    'sam_vitb': _cfg('sam_vitb'),
}


def backbone_of(model_name):
    return model_name[len(MODEL_PREFIX):] if model_name.startswith(MODEL_PREFIX) else model_name


class Router(nn.Module):
    def __init__(self, dim, hidden, num_clusters):
        super(Router, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden, bias=False), nn.ReLU(),
                                 nn.Linear(hidden, num_clusters, bias=False))

    def forward(self, x):
        return self.net(x)


class Segmenter(nn.Module):
    def __init__(self, backbone, num_classes, cluster_ranks=None, router_layer=2, router_hidden=64):
        super(Segmenter, self).__init__()
        self.backbone = backbone
        self.dim = int(backbone.embed_dim)
        self.num_blocks = len(backbone.blocks)
        self.decoder = nn.Conv2d(self.dim, num_classes, kernel_size=1)
        self.router = None
        if cluster_ranks is not None:
            self.register_buffer('cluster_ranks', torch.as_tensor(np.asarray(cluster_ranks), dtype=torch.long))
            self.router = Router(self.dim, router_hidden, len(cluster_ranks))
            self.router_layer = int(router_layer)

    def route(self, x):
        zeros = torch.zeros(x.shape[0], self.num_blocks, dtype=torch.long, device=x.device)
        set_active(self.backbone, zeros)
        with torch.no_grad():
            pooled = prefix_pooled(self.backbone, x, self.router_layer)
        return self.router(pooled.float())

    def forward(self, x, cluster=None):
        route_logits = None
        if self.router is not None:
            route_logits = self.route(x)
            chosen = route_logits.argmax(1)
            if cluster is not None:
                chosen = torch.where(cluster >= 0, cluster, chosen)
            set_active(self.backbone, self.cluster_ranks[chosen])
        logits = self.decoder(token_map(self.backbone, x))
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits, route_logits


def build_segmenter(backbone, pretrained=True, num_classes=8, input_size=None, ranks=None, cluster_ranks=None,
                    adapter='lora', lora_alpha=16.0, lora_init_std=0.02, adalora_init_r=12, adalora_init_std=0.02,
                    router_layer=2, router_hidden=64, full_finetune=False, **kwargs):
    model = load_backbone(backbone, input_size, pretrained)
    if adapter == 'svd':
        inject_svd(model, adalora_init_r, lora_alpha, adalora_init_std)
    elif ranks is not None and int(np.sum(ranks)) > 0:
        inject_lora(model, ranks, lora_alpha, lora_init_std)
    if full_finetune:
        model.requires_grad_(True)
    segmenter = Segmenter(model, num_classes, cluster_ranks, router_layer, router_hidden)
    segmenter.default_cfg = default_cfgs[backbone]
    return segmenter


@register_model
def aquaprobe_dinov2_vitb14(pretrained=True, **kwargs):
    return build_segmenter('dinov2_vitb14', pretrained=pretrained, **kwargs)


@register_model
def aquaprobe_clip_vitb16(pretrained=True, **kwargs):
    return build_segmenter('clip_vitb16', pretrained=pretrained, **kwargs)


@register_model
def aquaprobe_sam_vitb(pretrained=True, **kwargs):
    return build_segmenter('sam_vitb', pretrained=pretrained, **kwargs)
