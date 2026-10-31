"""
conn_hopfield_depth.py — V5 Experiment 13: Hopfield baselines at controlled integration depth.

Question: is the failure of Hopfield retrieval caused by the sign() nonlinearity itself,
or by running the dynamics to convergence?
Conditions (N=80, 3 families x 4 members, 85% similarity, 5 unrelated, 25% noise, 20 seeds):
  - Continuous Hopfield dx/dt = -x + tanh(beta W x), dt=0.1, beta=2.0, k Euler steps
  - Asynchronous binary Hopfield: k single-unit updates in random order
  - Synchronous binary Hopfield: 1 step, and run to convergence (published baseline)
Readout: nearest family member by overlap of sign(state). MI: pooled over all trials,
Miller-Madow corrected (bits), clipped to [0, log2(4)].
"""
import numpy as np, csv
from conn_stats import hier_boot, perm_strat
RNG_BASE = 42

def generate_family(N, n_members, similarity, rng):
    proto = rng.choice([-1, 1], size=N); n_diff = int((1 - similarity) * N)
    fam = [proto.copy()]
    for _ in range(1, n_members):
        m = proto.copy(); m[rng.choice(N, size=n_diff, replace=False)] *= -1; fam.append(m)
    return fam

def make_patterns(N, n_fam, members, similarity, n_unrel, rng):
    allp = []; fam_map = {}
    for f in range(n_fam):
        fam_map[f] = []
        for m in generate_family(N, members, similarity, rng):
            fam_map[f].append(len(allp)); allp.append(m)
    for _ in range(n_unrel): allp.append(rng.choice([-1, 1], size=N))
    return np.array(allp), fam_map

def classify(out, patterns, fam_map, f):
    ov = [np.dot(out, patterns[j]) for j in fam_map[f]]
    return int(np.argmax(ov))

def mi_mm(t, o, K=4):
    t = np.asarray(t); o = np.asarray(o); n = len(t)
    J = np.zeros((K, K))
    for a, b in zip(t, o): J[a, b] += 1
    def H(c):
        p = c[c > 0] / c.sum(); return -(p * np.log2(p)).sum(), int((c > 0).sum())
    hx, mx = H(J.sum(1)); hy, my = H(J.sum(0)); hxy, mxy = H(J.ravel())
    v = (hx + hy - hxy) + ((mx - 1) + (my - 1) - (mxy - 1)) / (2 * n * np.log(2))
    return float(np.clip(v, 0, np.log2(K)))

def sgn(x):
    s = np.sign(x); s[s == 0] = 1; return s

def cont_hopfield(W, x0, k, dt=0.1, beta=2.0):
    x = x0.copy()
    for _ in range(k): x = x + dt * (-x + np.tanh(beta * (W @ x)))
    return sgn(x)

def async_hopfield(W, s0, k, rng):
    s = s0.copy(); N = len(s)
    for i in rng.integers(0, N, size=k): s[i] = 1 if W[i] @ s >= 0 else -1
    return s

def sync_hopfield(W, s0, steps):
    s = s0.copy()
    for _ in range(steps):
        s2 = sgn(W @ s)
        if np.array_equal(s2, s): break
        s = s2
    return s

if __name__ == "__main__":
    N = 80; noise = 0.25; n_seeds = 20
    conds = ([("cue", "Raw cue (no dynamics)")] +
             [(f"cont{k}", f"Continuous Hopfield, {k} steps (t={k*0.1:g})") for k in [1, 2, 5, 10, 25, 150]] +
             [(f"async{k}", f"Async binary Hopfield, {k} unit updates") for k in [8, 20, 40, 80, 800]] +
             [("sync1", "Sync binary Hopfield, 1 step"), ("syncC", "Sync binary Hopfield, converged (published)")])
    R = {c: {"d": [], "t": [], "o": [], "mv": [], "s": []} for c, _ in conds}
    for seed in range(n_seeds):
        rng = np.random.default_rng(RNG_BASE + seed)
        P, fam = make_patterns(N, 3, 4, 0.85, 5, rng)
        W = sum(np.outer(p, p) for p in P) / N; np.fill_diagonal(W, 0)
        for f in range(3):
            for mi, midx in enumerate(fam[f]):
                cue = P[midx].copy(); cue[rng.random(N) < noise] *= -1
                xc = cue + rng.normal(0, 0.15, N)
                outs = {"cue": cue}
                for k in [1, 2, 5, 10, 25, 150]: outs[f"cont{k}"] = cont_hopfield(W, xc, k)
                for k in [8, 20, 40, 80, 800]: outs[f"async{k}"] = async_hopfield(W, cue, k, rng)
                outs["sync1"] = sync_hopfield(W, cue, 1); outs["syncC"] = sync_hopfield(W, cue, 100)
                for c, out in outs.items():
                    pr = classify(out, P, fam, f)
                    R[c]["d"].append(pr == mi); R[c]["t"].append(mi); R[c]["o"].append(pr)
                    R[c]["mv"].append(np.mean(out != cue) * 100); R[c]["s"].append(seed)
    rows = []
    print(f"{'Condition':<50}{'Disc':>8}{'CI':>16}{'MI':>8}{'CI':>16}{'p_strat':>9}{'Moved%':>8}")
    for c, lab in conds:
        r = R[c]; (dl, dh), (ml, mh) = hier_boot(r["s"], r["t"], r["o"]); p = perm_strat(r["s"], r["t"], r["o"])
        row = [lab, 100 * np.mean(r["d"]), 100 * dl, 100 * dh, mi_mm(r["t"], r["o"]), ml, mh, p, np.mean(r["mv"])]
        rows.append(row)
        print(f"{lab:<50}{row[1]:7.1f}% [{row[2]:5.1f},{row[3]:5.1f}]{row[4]:8.3f} [{ml:.3f},{mh:.3f}]{p:9.4f}{row[8]:7.1f}%")
    with open("hopfield_depth_results.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["condition", "discrimination", "disc_ci_low", "disc_ci_high", "mi_mm_bits", "mi_ci_low", "mi_ci_high", "p_seed_stratified", "bits_moved_pct"])
        for row in rows: w.writerow([row[0]] + [f"{v:.4f}" for v in row[1:]])
