import torch

from .adapters import SVDLoRA, adapter_modules


class AdaLoRAController(object):
    def __init__(self, model, target_triplets, total_steps, t_init_frac=0.1, t_final_frac=0.2, delta_t=10,
                 beta1=0.85, beta2=0.85, orth_reg_weight=0.1):
        self.modules = [(index, m) for index, m in adapter_modules(model) if isinstance(m, SVDLoRA)]
        self.init_triplets = sum(m.rank for _, m in self.modules)
        self.target = int(min(target_triplets, self.init_triplets))
        self.total = int(total_steps)
        self.t_init = int(round(t_init_frac * self.total))
        self.t_final = int(round(t_final_frac * self.total))
        self.delta_t = max(1, int(delta_t))
        self.beta1, self.beta2 = float(beta1), float(beta2)
        self.orth_weight = float(orth_reg_weight)
        self.ipt, self.unc = {}, {}
        self.final_mask = None

    def parameters(self):
        return [p for _, m in self.modules for p in (m.P, m.E, m.Q)]

    def budget(self, step):
        span = self.total - self.t_final - self.t_init
        if step < self.t_init:
            return self.init_triplets
        if step >= self.total - self.t_final or span <= 0:
            return self.target
        progress = 1.0 - (step - self.t_init) / span
        return int(self.target + (self.init_triplets - self.target) * progress ** 3)

    def orth_regulariser(self):
        reg, count = 0.0, 0
        for _, m in self.modules:
            for matrix in (m.P.t() @ m.P, m.Q @ m.Q.t()):
                eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
                reg = reg + torch.norm(matrix - eye, p='fro')
                count += 1
        return self.orth_weight * reg / max(count, 1)

    @torch.no_grad()
    def update_importance(self):
        for p in self.parameters():
            if p.grad is None:
                continue
            key = id(p)
            ipt = (p * p.grad).abs()
            if key not in self.ipt:
                self.ipt[key] = torch.zeros_like(p)
                self.unc[key] = torch.zeros_like(p)
            self.ipt[key].mul_(self.beta1).add_(ipt, alpha=1 - self.beta1)
            self.unc[key].mul_(self.beta2).add_((ipt - self.ipt[key]).abs(), alpha=1 - self.beta2)

    def score(self, p):
        key = id(p)
        if key not in self.ipt:
            return torch.zeros_like(p)
        return self.ipt[key] * self.unc[key]

    @torch.no_grad()
    def triplet_scores(self):
        return [self.score(m.E).view(-1) + self.score(m.Q).mean(1) + self.score(m.P).mean(0) for _, m in self.modules]

    @torch.no_grad()
    def mask_to_budget(self, budget):
        scores = self.triplet_scores()
        flat = torch.cat(scores)
        keep = torch.zeros_like(flat, dtype=torch.bool)
        if budget > 0:
            keep[torch.topk(flat, int(budget)).indices] = True
        masks, offset = [], 0
        for (_, m), s in zip(self.modules, scores):
            mask = keep[offset:offset + s.numel()]
            offset += s.numel()
            m.E.masked_fill_(~mask.view(-1, 1), 0.0)
            masks.append(mask)
        return masks

    @torch.no_grad()
    def apply_final_mask(self):
        for (_, m), mask in zip(self.modules, self.final_mask):
            m.E.masked_fill_(~mask.view(-1, 1), 0.0)
            m.kept_count = int(mask.sum().item())

    @torch.no_grad()
    def step(self, step):
        if self.final_mask is not None:
            self.apply_final_mask()
            return
        self.update_importance()
        if step >= self.total - self.t_final - 1:
            self.final_mask = self.mask_to_budget(self.target)
        elif step >= self.t_init and (step - self.t_init) % self.delta_t == 0:
            self.mask_to_budget(self.budget(step))

    @torch.no_grad()
    def finalise(self):
        if self.final_mask is None:
            self.final_mask = self.mask_to_budget(self.target)
        self.apply_final_mask()

    def block_ranks(self, num_blocks):
        totals = [0] * num_blocks
        for index, m in self.modules:
            totals[index] += m.kept()
        return [t / 4.0 for t in totals]

    def state_dict(self):
        params = self.parameters()
        return {
            'ipt': [self.ipt.get(id(p)) for p in params],
            'unc': [self.unc.get(id(p)) for p in params],
            'final_mask': self.final_mask,
        }

    def load_state_dict(self, state):
        for p, ipt, unc in zip(self.parameters(), state['ipt'], state['unc']):
            if ipt is not None:
                self.ipt[id(p)] = ipt.to(p.device)
                self.unc[id(p)] = unc.to(p.device)
        self.final_mask = state['final_mask']
