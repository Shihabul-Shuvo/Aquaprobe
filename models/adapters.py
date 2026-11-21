import torch
import torch.nn as nn


def expand_active(active, batch):
    if active.shape[0] == batch:
        return active
    if batch % active.shape[0] != 0:
        raise ValueError('active ranks do not match the batch')
    return active.repeat_interleave(batch // active.shape[0])


class NestedLoRA(nn.Module):
    def __init__(self, d_in, d_out, rank, alpha, init_std):
        super(NestedLoRA, self).__init__()
        self.rank, self.alpha = int(rank), float(alpha)
        self.A = nn.Parameter(torch.randn(self.rank, d_in) * init_std)
        self.B = nn.Parameter(torch.zeros(d_out, self.rank))
        self.active = None

    def forward(self, x):
        h = x @ self.A.t().to(x.dtype)
        if self.active is None:
            return (h @ self.B.t().to(x.dtype)) * (self.alpha / self.rank)
        r = expand_active(self.active.to(x.device), x.shape[0])
        shape = (r.shape[0],) + (1,) * (x.dim() - 2)
        mask = (torch.arange(self.rank, device=x.device)[None] < r[:, None]).to(h.dtype)
        h = h * mask.view(*shape, self.rank)
        scale = (self.alpha / r.clamp_min(1).to(h.dtype)).view(*shape, 1)
        return (h @ self.B.t().to(x.dtype)) * scale

    def delta_weight(self):
        return (self.alpha / self.rank) * self.B @ self.A

    def num_parameters(self):
        return self.A.numel() + self.B.numel()


class SVDLoRA(nn.Module):
    def __init__(self, d_in, d_out, rank, alpha, init_std):
        super(SVDLoRA, self).__init__()
        self.rank, self.alpha = int(rank), float(alpha)
        self.P = nn.Parameter(torch.randn(d_out, self.rank) * init_std)
        self.E = nn.Parameter(torch.zeros(self.rank, 1))
        self.Q = nn.Parameter(torch.randn(self.rank, d_in) * init_std)
        self.active = None
        self.kept_count = None

    def forward(self, x):
        return (x @ (self.Q * self.E).t().to(x.dtype)) @ self.P.t().to(x.dtype) * (self.alpha / self.rank)

    def delta_weight(self):
        return (self.alpha / self.rank) * self.P @ (self.Q * self.E)

    def kept(self):
        return self.rank if self.kept_count is None else self.kept_count

    def num_parameters(self):
        return self.kept() * (self.P.shape[0] + self.Q.shape[1])


class QKVAdapter(nn.Module):
    def __init__(self, base, adapters):
        super(QKVAdapter, self).__init__()
        self.base = base
        self.adapters = nn.ModuleList(adapters)

    def forward(self, x):
        return self.base(x) + torch.cat([a(x) for a in self.adapters], dim=-1)

    @torch.no_grad()
    def merged(self):
        d = self.base.weight.shape[0] // 3
        for i, adapter in enumerate(self.adapters):
            self.base.weight[i * d:(i + 1) * d] += adapter.delta_weight().to(self.base.weight.dtype)
        return self.base


class LinearAdapter(nn.Module):
    def __init__(self, base, adapter):
        super(LinearAdapter, self).__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, x):
        return self.base(x) + self.adapter(x)

    @torch.no_grad()
    def merged(self):
        self.base.weight += self.adapter.delta_weight().to(self.base.weight.dtype)
        return self.base


def inject(model, ranks, factory):
    for block, rank in zip(model.blocks, ranks):
        rank = int(rank)
        if rank <= 0:
            continue
        attn = block.attn
        d_in = attn.qkv.in_features
        d_out = attn.qkv.out_features // 3
        attn.qkv = QKVAdapter(attn.qkv, [factory(d_in, d_out, rank) for _ in range(3)])
        attn.proj = LinearAdapter(attn.proj, factory(attn.proj.in_features, attn.proj.out_features, rank))
    return model


def inject_lora(model, ranks, alpha, init_std):
    return inject(model, ranks, lambda i, o, r: NestedLoRA(i, o, r, alpha, init_std))


def inject_svd(model, rank, alpha, init_std):
    return inject(model, [rank] * len(model.blocks), lambda i, o, r: SVDLoRA(i, o, r, alpha, init_std))


def adapter_modules(model):
    for index, block in enumerate(model.blocks):
        for module in block.modules():
            if isinstance(module, (NestedLoRA, SVDLoRA)):
                yield index, module


def set_active(model, per_block):
    for index, module in adapter_modules(model):
        module.active = None if per_block is None else per_block[:, index]


def merge_adapters(model):
    for block in model.blocks:
        attn = block.attn
        if isinstance(attn.qkv, QKVAdapter):
            attn.qkv = attn.qkv.merged()
        if isinstance(attn.proj, LinearAdapter):
            attn.proj = attn.proj.merged()
    return model


def adapter_parameters(model):
    return sum(m.num_parameters() for _, m in adapter_modules(model))


def stored_ranks(model):
    ranks = [0] * len(model.blocks)
    for index, module in adapter_modules(model):
        if isinstance(module, NestedLoRA):
            ranks[index] = module.rank
    return ranks
