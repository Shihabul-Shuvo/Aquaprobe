import numpy as np
import torch
from scipy.stats import spearmanr

from .render import turbidity_index


def linear_cka(x, y):
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    numerator = torch.linalg.matrix_norm(y.t() @ x) ** 2
    denominator = torch.linalg.matrix_norm(x.t() @ x) * torch.linalg.matrix_norm(y.t() @ y)
    return (numerator / denominator).item()


def mutual_knn(x, y, k=10):
    def neighbours(z):
        z = torch.nn.functional.normalize(z.double(), dim=1)
        sim = z @ z.t()
        sim.fill_diagonal_(-float('inf'))
        return sim.topk(min(k, z.shape[0] - 1), dim=1).indices

    a, b = neighbours(x), neighbours(y)
    shared = (a.unsqueeze(2) == b.unsqueeze(1)).any(2).double().sum(1)
    return (shared / a.shape[1]).mean().item()


def similarity_fn(metric, k=10):
    if metric == 'cka':
        return linear_cka
    if metric == 'knn':
        return lambda x, y: mutual_knn(x, y, k)
    raise ValueError(metric)


def cumulative_gap(clean, rendered, sim):
    return np.array([1.0 - sim(clean[:, l], rendered[:, l]) for l in range(clean.shape[1])])


def raw_increment(gap):
    return np.diff(gap, axis=-1)


def marginal_gap(gap):
    return np.maximum(0.0, raw_increment(gap))


def normalised_profile(increment):
    increment = np.asarray(increment, dtype=np.float64)
    total = increment.sum(-1, keepdims=True)
    uniform = np.full_like(increment, 1.0 / increment.shape[-1])
    return np.where(total > 0, increment / np.where(total > 0, total, 1.0), uniform)


def jsd(p, q, eps=1e-12):
    p, q = np.asarray(p) + eps, np.asarray(q) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def jsd_matrix(profiles):
    n = len(profiles)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = jsd(profiles[i], profiles[j])
    return out


def pdi(profiles):
    matrix = jsd_matrix(profiles)
    return float(matrix[np.triu_indices(len(profiles), 1)].mean())


def pdi_null(null_profiles, n_conditions, n_draws, seed):
    rng = np.random.default_rng(seed)
    matrix = jsd_matrix(null_profiles)
    size = min(n_conditions, len(null_profiles))
    upper = np.triu_indices(size, 1)
    values = []
    for _ in range(n_draws):
        subset = rng.choice(len(null_profiles), size, replace=False)
        values.append(matrix[np.ix_(subset, subset)][upper].mean())
    return np.array(values)


def bootstrap_indices(n_rows, n_boot, seed):
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(rng.integers(0, n_rows, n_rows)) for _ in range(n_boot)]


def safe_spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]) or np.isnan(a).any() or np.isnan(b).any():
        return float('nan')
    return float(spearmanr(a, b)[0])


def stationarity(field, conditions, null_values, quantile, dispersion_threshold):
    field = np.asarray(field)
    peak = field.argmax(1) + 1
    turbidity = np.array([turbidity_index(w) for w, _ in conditions])
    ranges = np.array([s for _, s in conditions])
    value = pdi(normalised_profile(field))
    null_mean = float(np.mean(null_values))
    null_upper = float(np.quantile(null_values, quantile))
    dispersion = float(peak.std(ddof=1)) if len(peak) > 1 else float('nan')
    return {
        'n_conditions': len(conditions),
        'peak_block_min': int(peak.min()),
        'peak_block_max': int(peak.max()),
        'peak_block_std': dispersion,
        'pdi': value,
        'pdi_null_mean': null_mean,
        'pdi_null_upper': null_upper,
        'pdi_null_ratio': value / null_mean if null_mean > 0 else float('nan'),
        'pdi_exceeds_null': bool(value > null_upper),
        'dispersion_exceeds_threshold': bool(dispersion > dispersion_threshold),
        'non_stationary': bool(value > null_upper and dispersion > dispersion_threshold),
        'spearman_peak_water_type': safe_spearman(turbidity, peak),
        'spearman_peak_range': safe_spearman(ranges, peak),
        'peak_blocks': peak.tolist(),
    }


def uciqe(image):
    from skimage.color import rgb2lab
    lab = rgb2lab(image.permute(1, 2, 0).clamp(0, 1).numpy())
    lightness, a, b = lab[..., 0] / 100.0, lab[..., 1] / 100.0, lab[..., 2] / 100.0
    chroma = np.sqrt(a ** 2 + b ** 2)
    lo, hi = np.percentile(lightness, [1, 99])
    saturation = chroma / np.maximum(lightness, 1e-6)
    return float(0.4680 * chroma.std() + 0.2745 * (hi - lo) + 0.2576 * saturation.mean())
