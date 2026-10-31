"""conn_decoder_cv.py - V7/V8 Experiment 15c: is there member information in converged outputs
that the nearest-neighbour decoder misses?  A trained, seed-held-out decoder is compared with NN.

Member labels are arbitrary within each seed, so the decoder must be family-relative. For each
candidate member k it computes features of the output z relative to the stored family S:
  f1 = agreement of z with member k on all bits
  f2 = agreement on bits where member k differs from the family majority (its 'own' bits)
  f3 = agreement on bits where member k agrees with the majority but some other member does not
  f4 = number of own bits of k where z equals k AND differs from the prototype/majority
A conditional-logit model, score_k = w . f_k (+ quadratic terms), is fitted on training seeds and
tested on held-out seeds (2-fold by seed, both directions). An upper bound is also reported:
trials whose output is identical for all possible cued members carry no information for ANY decoder.
Z is the BINARISED output. For continuous Hopfield the raw state x is also compared across the four cues
(max |x_a - x_b| below 1e-6 and 1e-3). The identical-output fraction q is resampled at the seed level
(B=5000) to give 95% CIs for the bounds 2(1-q) bits and 0.25q+(1-q).
Decoder details: features z-scored with training-fold mean/sd; L2 = 0.01 fixed in advance (not tuned);
3000 gradient steps, rate 0.5; held-out predictions of both folds are pooled before computing accuracy,
MI and the seed-stratified permutation test."""
import numpy as np, csv
from conn_stats import mi_mm, hier_boot, perm_strat
src = open("conn_family_geometry.py").read().split("OUT=[]")[0]
fg = {}; exec(compile(src, "fg", "exec"), fg)
N = fg["N"]; SEEDS = 40      # more seeds than Exp 15 to give the decoder training data

def features(z, F):
    maj = np.sign(F.sum(0)); maj[maj == 0] = 1
    out = []
    for k in range(4):
        own = F[k] != maj
        oth = (F[k] == maj) & np.any(F != maj, axis=0)
        f1 = np.mean(z == F[k]); f2 = np.mean(z[own] == F[k][own]) if own.any() else 0.5
        f3 = np.mean(z[oth] == F[k][oth]) if oth.any() else 0.5; f4 = np.sum((z == F[k]) & own)
        b = np.array([f1, f2, f3, f4 / 12.0])
        out.append(np.concatenate([b, b**2]))
    return np.array(out)                                   # 4 x d

def fit_clogit(X, y, l2=1e-2, iters=3000, lr=0.5):
    d = X.shape[2]; w = np.zeros(d)
    mu = X.reshape(-1, d).mean(0); sd = X.reshape(-1, d).std(0) + 1e-9; Xs = (X - mu) / sd
    for _ in range(iters):
        s = Xs @ w; s -= s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
        g = -(Xs[np.arange(len(y)), y] - (p[:, :, None] * Xs).sum(1)).mean(0) + l2 * w
        w -= lr * g
    return lambda Xn: np.argmax(((Xn - mu) / sd) @ w, axis=1)

rows = []; FAM = []; RAW = []
for mode in ["A_star_stored", "B_star_hidden", "C_equidistant85"]:
    D = {c: {"X": [], "y": [], "nn": [], "seed": [], "ident": []} for c in ["raw cue", "cont. converged", "sync converged"]}
    for sd in range(SEEDS):
        rng = np.random.default_rng(1000 + sd); P, fams, protos = fg["make"](rng, mode)
        W = sum(np.outer(p, p) for p in P) / N; np.fill_diagonal(W, 0); st = {"fixed": 0, "cycle2": 0, "none": 0}
        for f in range(3):
            F = P[fams[f]]
            # outputs for every possible cued member with the SAME noise draw -> identity check
            noise_mask = rng.random(N) < fg["NOISE"]; jit = rng.normal(0, 0.15, N)
            outs_all = {c: [] for c in D}; raw = []
            for mi in range(4):
                cue = F[mi].copy(); cue[noise_mask] *= -1
                outs_all["raw cue"].append(cue)
                x = cue + jit
                for _ in range(150): x = x + 0.1 * (-x + np.tanh(2.0 * (W @ x)))
                raw.append(x); outs_all["cont. converged"].append(fg["sgn"](x.copy()))
                outs_all["sync converged"].append(fg["sync"](W, cue, st))
            dev = max(np.max(np.abs(raw[0] - r)) for r in raw[1:])
            RAW.append((mode, sd, dev))
            for c in D:
                same = all(np.array_equal(outs_all[c][0], o) for o in outs_all[c][1:])
                for mi in range(4):
                    z = outs_all[c][mi]
                    D[c]["X"].append(features(z, F)); D[c]["y"].append(mi); D[c]["seed"].append(sd)
                    D[c]["nn"].append(int(np.argmax(F @ z))); D[c]["ident"].append(same)
                FAM.append((mode, c, sd, same))
    for c, d in D.items():
        X = np.array(d["X"]); y = np.array(d["y"]); s = np.array(d["seed"]); pred = np.zeros_like(y)
        half = s < SEEDS // 2
        for tr, te in [(half, ~half), (~half, half)]:
            pred[te] = fit_clogit(X[tr], y[tr])(X[te])
        nn = np.array(d["nn"])
        fam = [(sd, same) for (m, cc, sd, same) in FAM if m == mode and cc == c]
        fs = np.array([a for a, _ in fam]); fq = np.array([b for _, b in fam], float); q = fq.mean()
        rng = np.random.default_rng(11); u = np.unique(fs); qb = []
        for _ in range(5000):
            pick = rng.choice(u, len(u)); qb.append(np.concatenate([fq[fs == k] for k in pick]).mean())
        qlo, qhi = np.percentile(qb, [2.5, 97.5])
        r = [mode, c, 100*np.mean(nn == y), mi_mm(y, nn), mi_mm(y, nn, clip=False), 100*np.mean(pred == y), mi_mm(y, pred),
             mi_mm(y, pred, clip=False), perm_strat(s, y, pred), 100*q, 100*qlo, 100*qhi,
             2*(1-q), 2*(1-qhi), 2*(1-qlo), 100*(0.25*q+(1-q)), 100*(0.25*qhi+(1-qhi)), 100*(0.25*qlo+(1-qlo))]
        rows.append(r)
        print(f"{mode:<17}{c:<16} NN {r[2]:5.1f}% {r[3]:.3f}b (raw {r[4]:+.3f}) | trained {r[5]:5.1f}% {r[6]:.3f}b (raw {r[7]:+.3f}) p={r[8]:.3f} | "
              f"q={r[9]:.1f}% [{r[10]:.1f},{r[11]:.1f}] -> MI<= {r[12]:.2f} [{r[13]:.2f},{r[14]:.2f}], acc<= {r[15]:.1f}% [{r[16]:.1f},{r[17]:.1f}]")
    devs = np.array([v for (m, sd, v) in RAW if m == mode])
    print(f"   raw continuous states identical across the 4 cues: tol 1e-6 {100*np.mean(devs<1e-6):.1f}%, tol 1e-3 {100*np.mean(devs<1e-3):.1f}%, median max-dev {np.median(devs):.2e}")
    rows.append([mode, "cont. raw state identical (tol 1e-6 / 1e-3)", "", "", "", "", "", "", "", 100*np.mean(devs<1e-6), 100*np.mean(devs<1e-3)] + [""]*7)
with open("decoder_cv_results.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["design", "condition", "nn_disc", "nn_mi_clipped", "nn_mi_unclipped", "trained_heldout_disc",
        "trained_mi_clipped", "trained_mi_unclipped", "p_seed_stratified", "q_identical_binarised_pct", "q_ci_low", "q_ci_high",
        "mi_bound_bits", "mi_bound_ci_low", "mi_bound_ci_high", "acc_bound_pct", "acc_bound_ci_low", "acc_bound_ci_high"])
    [w.writerow(r) for r in rows]
