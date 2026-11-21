import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONES = {
    'dinov2_vitb14': 'vit_base_patch14_dinov2.lvd142m',
    'clip_vitb16': 'vit_base_patch16_clip_224.openai',
    'sam_vitb': 'samvit_base_patch16.sa1b',
}
LABELS = {'dinov2_vitb14': 'DINOv2 ViT-B/14', 'clip_vitb16': 'CLIP ViT-B/16', 'sam_vitb': 'SAM ViT-B'}
SAM_NATIVE_SIZE = 1024


def resample_sam_state(state, target):
    shapes = {k: v.shape for k, v in target.state_dict().items()}
    out = {}
    for key, value in state.items():
        if key in shapes and value.shape != shapes[key]:
            if key.endswith('pos_embed'):
                grid = value.permute(0, 3, 1, 2).float()
                grid = F.interpolate(grid, size=tuple(shapes[key][1:3]), mode='bicubic', align_corners=False)
                value = grid.permute(0, 2, 3, 1).to(value.dtype)
            elif 'rel_pos' in key:
                line = value.t()[None].float()
                value = F.interpolate(line, size=shapes[key][0], mode='linear')[0].t().to(value.dtype)
        out[key] = value
    return out


def load_backbone(name, input_size=None, pretrained=True):
    kwargs = {'num_classes': 0}
    if input_size:
        kwargs['img_size'] = int(input_size)
    if name == 'sam_vitb' and input_size and int(input_size) != SAM_NATIVE_SIZE and pretrained:
        source = timm.create_model(BACKBONES[name], pretrained=True, num_classes=0)
        model = timm.create_model(BACKBONES[name], pretrained=False, **kwargs)
        model.load_state_dict(resample_sam_state(source.state_dict(), model))
    else:
        model = timm.create_model(BACKBONES[name], pretrained=pretrained, **kwargs)
    if hasattr(model, 'neck'):
        model.neck = nn.Identity()
    model.requires_grad_(False)
    return model.eval()


def backbone_meta(model):
    cfg = model.pretrained_cfg
    size = model.patch_embed.img_size
    patch = model.patch_embed.patch_size
    return {
        'resolution': int(size[0] if isinstance(size, (tuple, list)) else size),
        'mean': tuple(cfg['mean']),
        'std': tuple(cfg['std']),
        'patch': int(patch[0] if isinstance(patch, (tuple, list)) else patch),
        'dim': int(model.embed_dim),
        'num_blocks': len(model.blocks),
        'num_prefix': int(getattr(model, 'num_prefix_tokens', 0) or 0),
    }


def backbone_parameters(model):
    return sum(p.numel() for n, p in model.named_parameters() if '.adapter' not in n)


def pool_tokens(x, num_prefix, rows):
    if x.dim() == 4:
        if rows == 'cls':
            raise ValueError('backbone has no class token')
        tokens = x.flatten(1, 2)
    else:
        if rows == 'cls':
            return x[:, 0]
        tokens = x[:, num_prefix:]
    if rows == 'mean':
        return tokens.mean(1)
    if rows == 'tokens':
        return tokens
    raise ValueError(rows)


@torch.no_grad()
def block_features(model, x, rows='mean'):
    num_prefix = int(getattr(model, 'num_prefix_tokens', 0) or 0)
    features, hooks = [], []
    hooks.append(model.blocks[0].register_forward_pre_hook(
        lambda module, args: features.append(pool_tokens(args[0], num_prefix, rows).float())))
    for block in model.blocks:
        hooks.append(block.register_forward_hook(
            lambda module, args, out: features.append(pool_tokens(out, num_prefix, rows).float())))
    try:
        model.forward_features(x)
    finally:
        for hook in hooks:
            hook.remove()
    stacked = torch.stack(features, 1)
    if rows == 'tokens':
        b, depth, n, d = stacked.shape
        stacked = stacked.permute(0, 2, 1, 3).reshape(b * n, depth, d)
    return stacked


def token_map(model, x):
    features = model.forward_features(x)
    if features.dim() == 3:
        num_prefix = int(getattr(model, 'num_prefix_tokens', 0) or 0)
        patch = model.patch_embed.patch_size
        ph, pw = (patch, patch) if isinstance(patch, int) else patch
        h, w = x.shape[-2] // ph, x.shape[-1] // pw
        features = features[:, num_prefix:].transpose(1, 2).reshape(x.shape[0], -1, h, w)
    return features


def prefix_pooled(model, x, num_blocks):
    out = model.forward_intermediates(x, indices=[num_blocks - 1], output_fmt='NCHW',
                                      intermediates_only=True, stop_early=True)
    return out[0].mean((2, 3))
