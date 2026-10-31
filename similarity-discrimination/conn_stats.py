"""conn_stats.py - V6 shared statistics: decoder-level MI (Miller-Madow, per-entropy),
hierarchical (seed -> trial) bootstrap, and seed-stratified permutation test."""
import numpy as np

def mi_mm(t, o, K=4, clip=True):
    t = np.asarray(t, int); o = np.asarray(o, int); n = len(t)
    if n == 0: return 0.0
    J = np.zeros((K, K))
    for a, b in zip(t, o): J[a, b] += 1
    def H(c):
        p = c[c > 0] / c.sum(); return -(p * np.log2(p)).sum(), int((c > 0).sum())
    hx, mx = H(J.sum(1)); hy, my = H(J.sum(0)); hxy, mxy = H(J.ravel())
    v = hx + hy - hxy + ((mx - 1) + (my - 1) - (mxy - 1)) / (2 * n * np.log(2))
    return float(np.clip(v, 0.0, np.log2(K))) if clip else float(v)

def hier_boot(seed, t, o, B=2000, rng=None, K=4):
    """Resample seeds with replacement, then trials within each drawn seed."""
    rng = rng or np.random.default_rng(7)
    seed = np.asarray(seed); t = np.asarray(t); o = np.asarray(o)
    u = np.unique(seed); by = {s: np.where(seed == s)[0] for s in u}
    d = []; m = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(by[s], len(by[s])) for s in rng.choice(u, len(u))])
        d.append(np.mean(t[idx] == o[idx])); m.append(mi_mm(t[idx], o[idx], K))
    return np.percentile(d, [2.5, 97.5]), np.percentile(m, [2.5, 97.5])

def perm_strat(seed, t, o, P=2000, rng=None, K=4):
    """Permute identified labels within each seed; returns (P+1)-corrected p."""
    rng = rng or np.random.default_rng(9)
    seed = np.asarray(seed); t = np.asarray(t); o = np.asarray(o)
    obs = mi_mm(t, o, K); groups = [np.where(seed == s)[0] for s in np.unique(seed)]; c = 0
    for _ in range(P):
        o2 = o.copy()
        for g in groups: o2[g] = rng.permutation(o[g])
        c += mi_mm(t, o2, K) >= obs
    return (c + 1) / (P + 1)
