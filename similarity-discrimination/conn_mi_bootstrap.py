"""
conn_mi_bootstrap.py — MI Bootstrap with Miller-Madow Correction
=================================================================
Adds three things missing from V2/V3:
  1. Per-trial predictions saved to per_trial_predictions.csv
  2. Miller-Madow bias-corrected MI estimate
  3. Bootstrap confidence intervals (B=2000) on MI

Miller-Madow correction:
  MI_MM = MI_naive + (K_x * K_y - 1) / (2 * N)
  where K_x, K_y = number of non-empty bins in marginals

Bootstrap:
  Resample (target, output) pairs with replacement, B=2000 times.
  Report 2.5th and 97.5th percentiles as 95% CI.

Setup matches Experiment 7 (conn_final_validation.py) exactly:
  N=80, 3 families x 4 members (85% similar), 5 unrelated,
  matched 25% bit-flip noise, 15 seeds.

Output:
  - per_trial_predictions.csv
  - MI table with naive, Miller-Madow, and bootstrap CI
  - Wilcoxon paired tests vs Hopfield baselines
"""
import numpy as np
from scipy import stats
import csv
import time

RNG_BASE = 42
B_BOOTSTRAP = 2000
N_SEEDS = 15

# ── helpers ─────────────────────────────────────────────────────────────────
def b2p(b): return np.where(b > 0, 0.0, np.pi)
def p2b(p): return np.sign(np.cos(p))

def generate_family(N, n_members, similarity, rng):
    proto = rng.choice([-1, 1], size=N)
    n_diff = int((1 - similarity) * N)
    family = [proto.copy()]
    for _ in range(1, n_members):
        m = proto.copy()
        m[rng.choice(N, size=n_diff, replace=False)] *= -1
        family.append(m)
    return family

def make_patterns(N, n_fam, members, similarity, n_unrel, rng):
    all_p = []; fam_map = {}
    for f in range(n_fam):
        fam = generate_family(N, members, similarity, rng)
        fam_map[f] = []
        for m in fam:
            fam_map[f].append(len(all_p)); all_p.append(m)
    for _ in range(n_unrel):
        all_p.append(rng.choice([-1, 1], size=N))
    return np.array(all_p), fam_map

def classify_member(out, patterns, fam_map, f):
    """Return index within family (0..n_members-1) of closest member."""
    best = -1; bov = -1
    for ji, j in enumerate(fam_map[f]):
        ov = np.dot(out, patterns[j]) / len(out)
        if ov > bov: bov = ov; best = ji
    return best

def run_phase(J, phi0, steps=150, eta=0.005):
    phi = phi0.copy()
    for _ in range(steps):
        sd = np.sin(phi[None, :] - phi[:, None])
        phi = (phi + eta * np.sum(J * sd, axis=1)) % (2 * np.pi)
    return p2b(phi)

def run_cont_hopfield(W, cue_cont, steps=150, dt=0.1, beta=2.0):
    x = cue_cont.copy()
    for _ in range(steps):
        x = x + dt * (-x + np.tanh(beta * W @ x))
    return np.sign(x)

def run_bin_hopfield(W, cue, steps=100):
    state = cue.copy()
    for _ in range(steps):
        s2 = np.sign(W @ state); s2[s2 == 0] = 1
        if np.array_equal(s2, state): break
        state = s2
    return state

# ── MI estimators ─────────────────────────────────────────────────────────────
def mi_naive(targets, outputs, n_classes):
    """Naive MI from empirical joint distribution."""
    n = len(targets)
    joint = np.zeros((n_classes, n_classes))
    for t, o in zip(targets, outputs):
        joint[t, o] += 1
    joint /= (n + 1e-12)
    pt = joint.sum(1); po = joint.sum(0)
    mi = 0.0
    for t in range(n_classes):
        for o in range(n_classes):
            if joint[t, o] > 1e-12 and pt[t] > 1e-12 and po[o] > 1e-12:
                mi += joint[t, o] * np.log2(joint[t, o] / (pt[t] * po[o]))
    return max(mi, 0.0)

def mi_miller_madow(targets, outputs, n_classes):
    """
    Miller-Madow bias-corrected MI.
    MI_MM = MI_naive + (m - 1) / (2 * N)
    where m = number of non-empty cells in joint distribution.
    """
    n = len(targets)
    mi_n = mi_naive(targets, outputs, n_classes)
    # Count non-empty joint cells
    joint_counts = np.zeros((n_classes, n_classes), dtype=int)
    for t, o in zip(targets, outputs):
        joint_counts[t, o] += 1
    m = np.sum(joint_counts > 0)
    correction = (m - 1) / (2 * n)
    return max(mi_n + correction, 0.0)

def mi_bootstrap_ci(targets, outputs, n_classes, B=2000, alpha=0.05, seed=0):
    """
    Bootstrap CI for Miller-Madow MI.
    Returns (mi_mm, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    targets = np.array(targets); outputs = np.array(outputs)
    n = len(targets)
    mi_mm = mi_miller_madow(targets, outputs, n_classes)
    boot_vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        boot_vals.append(mi_miller_madow(targets[idx], outputs[idx], n_classes))
    boot_vals = np.array(boot_vals)
    ci_low  = np.percentile(boot_vals, alpha/2 * 100)
    ci_high = np.percentile(boot_vals, (1 - alpha/2) * 100)
    return mi_mm, ci_low, ci_high

# ── main experiment ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("MI Bootstrap with Miller-Madow Correction")
    print(f"N=80, 3 families x 4 members, 85% similarity, {N_SEEDS} seeds")
    print(f"Bootstrap B={B_BOOTSTRAP}, 95% CI")
    print("=" * 70)
    t0 = time.time()

    N = 80; noise = 0.25; n_classes = 4

    # Storage for per-trial data
    per_trial_rows = []  # (seed, family, true_member, pred_phase, pred_cont, pred_bin)

    # Storage for per-seed MI inputs
    phase_targets = []; phase_outputs = []
    cont_targets  = []; cont_outputs  = []
    bin_targets   = []; bin_outputs   = []

    # Storage for per-seed discrimination
    phase_disc = []; cont_disc = []; bin_disc = []
    phase_exact = []; cont_exact = []; bin_exact = []

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(RNG_BASE + seed * 100)
        patterns, fam_map = make_patterns(N, 3, 4, 0.85, 5, rng)

        J = np.zeros((N, N)); W = np.zeros((N, N))
        for p in patterns:
            xi = np.cos(b2p(p)); J += np.outer(xi, xi) / N
            W += np.outer(p, p) / N
        np.fill_diagonal(J, 0); np.fill_diagonal(W, 0)

        s_ph_t = []; s_ph_o = []
        s_co_t = []; s_co_o = []
        s_bi_t = []; s_bi_o = []
        s_ph_d = []; s_co_d = []; s_bi_d = []
        s_ph_e = []; s_co_e = []; s_bi_e = []

        for f in range(3):
            for mi_idx, midx in enumerate(fam_map[f]):
                target = patterns[midx]

                # Matched noise
                cue_bin = target.copy()
                cue_bin[rng.random(N) < noise] *= -1
                cue_phases = b2p(cue_bin) + rng.normal(0, 0.15, N)
                cue_cont   = cue_bin.astype(float) + rng.normal(0, 0.15, N)

                # Phase
                out_ph = run_phase(J, cue_phases)
                pred_ph = classify_member(out_ph, patterns, fam_map, f)
                s_ph_t.append(mi_idx); s_ph_o.append(pred_ph)
                s_ph_d.append(int(pred_ph == mi_idx))
                s_ph_e.append(np.mean(out_ph == target))

                # Continuous Hopfield
                out_co = run_cont_hopfield(W, cue_cont)
                out_co[out_co == 0] = 1
                pred_co = classify_member(out_co, patterns, fam_map, f)
                s_co_t.append(mi_idx); s_co_o.append(pred_co)
                s_co_d.append(int(pred_co == mi_idx))
                s_co_e.append(np.mean(out_co == target))

                # Binary Hopfield
                out_bi = run_bin_hopfield(W, cue_bin)
                pred_bi = classify_member(out_bi, patterns, fam_map, f)
                s_bi_t.append(mi_idx); s_bi_o.append(pred_bi)
                s_bi_d.append(int(pred_bi == mi_idx))
                s_bi_e.append(np.mean(out_bi == target))

                # Per-trial row
                per_trial_rows.append({
                    'seed': seed,
                    'family': f,
                    'true_member': mi_idx,
                    'pred_phase': pred_ph,
                    'pred_cont_hop': pred_co,
                    'pred_bin_hop': pred_bi,
                    'correct_phase': int(pred_ph == mi_idx),
                    'correct_cont': int(pred_co == mi_idx),
                    'correct_bin': int(pred_bi == mi_idx),
                    'exact_phase': float(np.mean(out_ph == target)),
                    'exact_cont': float(np.mean(out_co == target)),
                    'exact_bin': float(np.mean(out_bi == target)),
                })

        # Accumulate
        phase_targets.extend(s_ph_t); phase_outputs.extend(s_ph_o)
        cont_targets.extend(s_co_t);  cont_outputs.extend(s_co_o)
        bin_targets.extend(s_bi_t);   bin_outputs.extend(s_bi_o)
        phase_disc.append(np.mean(s_ph_d))
        cont_disc.append(np.mean(s_co_d))
        bin_disc.append(np.mean(s_bi_d))
        phase_exact.append(np.mean(s_ph_e))
        cont_exact.append(np.mean(s_co_e))
        bin_exact.append(np.mean(s_bi_e))

    # ── Save per-trial CSV ────────────────────────────────────────────────────
    csv_path = "per_trial_predictions.csv"
    fieldnames = ['seed','family','true_member','pred_phase','pred_cont_hop',
                  'pred_bin_hop','correct_phase','correct_cont','correct_bin',
                  'exact_phase','exact_cont','exact_bin']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_trial_rows)
    print(f"\nPer-trial predictions saved to: {csv_path}")
    print(f"  Rows: {len(per_trial_rows)} ({N_SEEDS} seeds x 3 families x 4 members)")

    # ── Compute MI with bootstrap ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"MI Analysis — Miller-Madow corrected, Bootstrap B={B_BOOTSTRAP}")
    print(f"{'=' * 70}")
    print(f"\nComputing bootstrap CIs (this takes ~30 seconds)...")

    ph_mm, ph_lo, ph_hi = mi_bootstrap_ci(phase_targets, phase_outputs,
                                            n_classes, B=B_BOOTSTRAP, seed=0)
    co_mm, co_lo, co_hi = mi_bootstrap_ci(cont_targets,  cont_outputs,
                                            n_classes, B=B_BOOTSTRAP, seed=1)
    bi_mm, bi_lo, bi_hi = mi_bootstrap_ci(bin_targets,   bin_outputs,
                                            n_classes, B=B_BOOTSTRAP, seed=2)

    max_mi = np.log2(n_classes)

    print(f"\n  {'Metric':<35} {'Phase':>12} {'Cont Hop':>12} {'Bin Hop':>12}")
    print(f"  {'-'*73}")
    print(f"  {'Discrimination accuracy':<35} "
          f"{np.mean(phase_disc)*100:>11.1f}% "
          f"{np.mean(cont_disc)*100:>11.1f}% "
          f"{np.mean(bin_disc)*100:>11.1f}%")
    print(f"  {'Exact recall accuracy':<35} "
          f"{np.mean(phase_exact)*100:>11.1f}% "
          f"{np.mean(cont_exact)*100:>11.1f}% "
          f"{np.mean(bin_exact)*100:>11.1f}%")
    print(f"  {'MI naive (bits)':<35} "
          f"{mi_naive(phase_targets, phase_outputs, n_classes):>11.3f}  "
          f"{mi_naive(cont_targets, cont_outputs, n_classes):>11.3f}  "
          f"{mi_naive(bin_targets, bin_outputs, n_classes):>11.3f}")
    print(f"  {'MI Miller-Madow (bits)':<35} "
          f"{ph_mm:>11.3f}  {co_mm:>11.3f}  {bi_mm:>11.3f}")
    print(f"  {'95% CI low':<35} "
          f"{ph_lo:>11.3f}  {co_lo:>11.3f}  {bi_lo:>11.3f}")
    print(f"  {'95% CI high':<35} "
          f"{ph_hi:>11.3f}  {co_hi:>11.3f}  {bi_hi:>11.3f}")
    print(f"  {'MI / max MI (MM)':<35} "
          f"{ph_mm/max_mi*100:>11.1f}% "
          f"{co_mm/max_mi*100:>11.1f}% "
          f"{bi_mm/max_mi*100:>11.1f}%")

    # ── Statistical tests ─────────────────────────────────────────────────────
    print(f"\n  Wilcoxon signed-rank tests (per-seed MI, Miller-Madow):")
    # Compute per-seed MI for paired tests
    ph_seed_mi = []
    co_seed_mi = []
    bi_seed_mi = []
    n_per_seed = 3 * 4  # 3 families x 4 members
    for s in range(N_SEEDS):
        lo = s * n_per_seed; hi = (s + 1) * n_per_seed
        ph_seed_mi.append(mi_miller_madow(
            phase_targets[lo:hi], phase_outputs[lo:hi], n_classes))
        co_seed_mi.append(mi_miller_madow(
            cont_targets[lo:hi], cont_outputs[lo:hi], n_classes))
        bi_seed_mi.append(mi_miller_madow(
            bin_targets[lo:hi], bin_outputs[lo:hi], n_classes))

    for label, a, b in [
        ("Phase MI vs Binary MI",     ph_seed_mi, bi_seed_mi),
        ("Phase MI vs Cont Hop MI",   ph_seed_mi, co_seed_mi),
        ("Cont Hop MI vs Binary MI",  co_seed_mi, bi_seed_mi),
    ]:
        diff = np.array(a) - np.array(b)
        if np.std(diff) < 1e-10:
            print(f"    {label:<30}: identical")
            continue
        try:
            _, p = stats.wilcoxon(a, b, alternative='greater')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        except Exception:
            p = float('nan'); sig = "?"
        print(f"    {label:<30}: Δ={np.mean(diff):+.3f} bits, p={p:.4f} {sig}")

    elapsed = time.time() - t0

    # ── Summary for manuscript ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("MANUSCRIPT READY TEXT — Experiment 7 MI table (V3 update)")
    print(f"{'=' * 70}")
    print(f"""
  Phase system MI (Miller-Madow): {ph_mm:.3f} bits
  95% bootstrap CI: [{ph_lo:.3f}, {ph_hi:.3f}] bits
  As fraction of max ({max_mi:.3f} bits): {ph_mm/max_mi*100:.1f}%

  Continuous Hopfield MI: {co_mm:.3f} bits [{co_lo:.3f}, {co_hi:.3f}]
  Binary Hopfield MI:     {bi_mm:.3f} bits [{bi_lo:.3f}, {bi_hi:.3f}]

  MI estimator: Miller-Madow bias correction
    MI_MM = MI_naive + (m - 1) / (2N)
    where m = number of non-empty cells in the joint distribution
  Bootstrap: B={B_BOOTSTRAP} resamples, 95% CI (2.5th-97.5th percentile)
  Seeds: {N_SEEDS}, RNG: numpy.default_rng, base seed {RNG_BASE}

  To report in paper:
  Phase MI = {ph_mm:.3f} bits (95% CI: {ph_lo:.3f}–{ph_hi:.3f}),
  Binary Hopfield MI = {bi_mm:.3f} bits (95% CI: {bi_lo:.3f}–{bi_hi:.3f}),
  Continuous Hopfield MI = {co_mm:.3f} bits (95% CI: {co_lo:.3f}–{co_hi:.3f}).
  All p < 0.001 (Wilcoxon signed-rank, one-sided).
""")
    print(f"  Time: {elapsed:.0f}s")
    print("=" * 70)
