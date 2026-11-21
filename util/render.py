import torch

JERLOV = {
    'I': (0.850, 0.961, 0.982),
    'IA': (0.840, 0.955, 0.975),
    'IB': (0.830, 0.950, 0.968),
    'II': (0.800, 0.925, 0.940),
    'III': (0.750, 0.885, 0.890),
    '1C': (0.750, 0.885, 0.875),
    '3C': (0.710, 0.820, 0.800),
    '5C': (0.670, 0.730, 0.670),
    '7C': (0.620, 0.610, 0.500),
    '9C': (0.550, 0.460, 0.290),
}
WATER_TYPES = tuple(JERLOV)
RANGES = (0.5, 1.0, 2.0, 4.0)
HELDOUT_RANGES = (0.25, 0.75, 1.5, 3.0, 5.0)


def condition_grid(water_types, ranges):
    return [(str(w), float(s)) for w in water_types for s in ranges]


def turbidity_index(water_type):
    return WATER_TYPES.index(str(water_type))


def survival(water_type, device=None, dtype=torch.float32):
    return torch.tensor(JERLOV[str(water_type)], device=device, dtype=dtype)


def attenuation(water_type, device=None, dtype=torch.float32):
    return -torch.log(survival(water_type, device, dtype))


def normalise_depth(depth):
    lo = depth.amin(dim=(-2, -1), keepdim=True)
    hi = depth.amax(dim=(-2, -1), keepdim=True)
    return (depth - lo) / (hi - lo).clamp_min(1e-6)


def scene_range(depth, s, z_min, z_max):
    return s * (z_min + (z_max - z_min) * normalise_depth(depth))


def restore_channel(clean, rendered, channel):
    out = rendered.clone()
    out[:, channel] = clean[:, channel]
    return out


class Renderer(object):
    def __init__(self, z_min=1.0, z_max=4.0, veiling_mode='power', veiling_level=0.8, veiling_exponent=1.15):
        self.z_min, self.z_max = float(z_min), float(z_max)
        self.veiling_mode = veiling_mode
        self.veiling_level = float(veiling_level)
        self.veiling_exponent = float(veiling_exponent)

    def veiling_light(self, water_type, device=None, dtype=torch.float32):
        n = survival(water_type, device, dtype)
        if self.veiling_mode == 'constant':
            return torch.full_like(n, self.veiling_level)
        if self.veiling_mode == 'power':
            return self.veiling_level * (n / n.max()) ** self.veiling_exponent
        raise ValueError(self.veiling_mode)

    def __call__(self, clean, depth, condition, z_min=None, z_max=None):
        water_type, s = condition
        z_min = self.z_min if z_min is None else z_min
        z_max = self.z_max if z_max is None else z_max
        z = scene_range(depth.to(clean), float(s), z_min, z_max)
        n = survival(water_type, clean.device, clean.dtype).view(1, 3, 1, 1)
        b = self.veiling_light(water_type, clean.device, clean.dtype).view(1, 3, 1, 1)
        t = n ** z
        return clean * t + b * (1.0 - t)


def build_renderer(args):
    return Renderer(args.z_min, args.z_max, args.veiling_mode, args.veiling_level, args.veiling_exponent)
