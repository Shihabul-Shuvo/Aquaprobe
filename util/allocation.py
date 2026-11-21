import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .divergence import jsd_matrix, normalised_profile


def allocate(profile, r_tot, gamma=1.0, r_min=0, r_max=24, iterations=200):
    profile = np.asarray(profile, dtype=np.float64)
    weights = np.power(profile, gamma) if gamma != 0 else np.ones_like(profile)
    num_blocks = len(weights)
    capacity = np.where(weights > 0, r_max, r_min).sum()
    if not num_blocks * r_min <= r_tot <= capacity:
        raise ValueError(f'budget {r_tot} infeasible for bounds [{r_min}, {r_max}] and profile {profile}')

    def total(tau):
        return np.clip(tau * weights, r_min, r_max).sum()

    lo, hi = 0.0, 1.0
    while total(hi) < r_tot:
        hi *= 2.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if total(mid) < r_tot else (lo, mid)
    target = np.clip(hi * weights, r_min, r_max)
    ranks = np.floor(target + 1e-9)
    remainder = int(round(r_tot - ranks.sum()))
    fractional = target - ranks
    if remainder > 0:
        order = [i for i in np.argsort(-fractional, kind='stable') if ranks[i] < r_max][:remainder]
        ranks[order] += 1
    elif remainder < 0:
        order = [i for i in np.argsort(fractional, kind='stable') if ranks[i] > r_min][:-remainder]
        ranks[order] -= 1
    ranks = ranks.astype(int)
    if ranks.sum() != r_tot:
        raise RuntimeError('allocation does not meet the budget')
    return ranks


def uniform(num_blocks, r_tot):
    return allocate(np.ones(num_blocks), r_tot, gamma=0.0, r_min=0, r_max=r_tot)


def reversed_profile(profile, r_tot, gamma=1.0, r_min=0, r_max=24):
    return allocate(np.asarray(profile)[::-1], r_tot, gamma, r_min, r_max)


def random_allocation(num_blocks, r_tot, seed, r_min=0, r_max=24):
    weights = np.random.default_rng(seed).uniform(size=num_blocks)
    return allocate(weights, r_tot, 1.0, r_min, r_max)


def repsam_bands(cross_domain_gap, r_tot, ratios=(4, 2, 1), r_min=0, r_max=24):
    gap = np.asarray(cross_domain_gap, dtype=np.float64)
    bands = np.array_split(np.arange(len(gap)), len(ratios))
    order = np.argsort([-gap[band].mean() for band in bands], kind='stable')
    weights = np.zeros(len(gap))
    for position, band_index in enumerate(order):
        weights[bands[band_index]] = ratios[position]
    return allocate(weights, r_tot, 1.0, r_min, r_max), [b.tolist() for b in bands], order.tolist()


def single_block(num_blocks, block, rank):
    ranks = np.zeros(num_blocks, dtype=int)
    ranks[block - 1] = rank
    return ranks


def cluster_conditions(field, num_clusters):
    profiles = normalised_profile(field)
    distance = np.sqrt(np.clip(jsd_matrix(profiles), 0.0, None))
    tree = linkage(squareform(distance, checks=False), method='average')
    return fcluster(tree, num_clusters, criterion='maxclust') - 1


def router_parameters(dim, hidden, num_clusters):
    return dim * hidden + hidden * num_clusters


def router_rank_charge(router_params, dim):
    return int(round(router_params / (8 * dim)))


def nested_allocations(cluster_profiles, stored_budget, gamma=1.0, r_min=0, r_max=24):
    for active in range(stored_budget, 0, -1):
        try:
            ranks = np.stack([allocate(p, active, gamma, r_min, r_max) for p in cluster_profiles])
        except ValueError:
            continue
        stored = ranks.max(0)
        if stored.sum() <= stored_budget:
            return ranks, stored, active
    raise ValueError('stored budget too small for nested allocations')
