"""
================================================================================
CONN LARGE-SCALE VALIDATION — N=256, N=512
================================================================================

Companion to CONN_VALIDATION_V5.py. Tests scaling behavior at larger N.
Same corrected protocol: annealed λ, jitter noise, real Hopfield baseline.

Parameters (matching V5):
  λ = 0.5 (annealed 0 → 0.5)
  η_φ = 0.02, η_A = 0.03, β = 1.0
  400 integration steps
  30% bit-flip + σ=0.5 Gaussian jitter
  Real Hopfield baseline: sign(W·s) with 200 max steps

Expected runtime: ~15-30 min on Colab (GPU not required)

Results reproduced in this script:
  N=256: Hopfield M=35 (α=0.137), CONN M=49 (α=0.191), ratio=1.40×
  N=512: Hopfield M=65 (α=0.127), CONN M=94 (α=0.184), ratio=1.45×

DOI: [to be updated]
================================================================================
"""

import numpy as np
import time

def generate_patterns(M, N, seed):
    np.random.seed(seed)
    return np.random.choice([0, np.pi], size=(M, N))

def bitflip_plus_jitter(pattern, noise_level, jitter_std, rng):
    noisy = pattern.copy()
    mask = rng.random(len(pattern)) < noise_level
    noisy[mask] = (noisy[mask] + np.pi) % (2 * np.pi)
    noisy = noisy + rng.normal(0, jitter_std, len(pattern))
    return np.mod(noisy, 2 * np.pi)

def compute_overlap(retrieved, target):
    r = np.where(np.cos(retrieved) >= 0, 0.0, np.pi)
    t = np.where(np.cos(target) >= 0, 0.0, np.pi)
    return np.sum(np.abs(np.cos(r) - np.cos(t)) < 0.5) / len(target) * 100

def hebbian_weights(patterns):
    xi = np.cos(patterns)
    J = (xi.T @ xi) / patterns.shape[1]
    np.fill_diagonal(J, 0)
    return J

def hopfield_recall(J, noisy, max_steps=200):
    s = np.cos(noisy).copy()
    for _ in range(max_steps):
        s_new = np.sign(J @ s)
        s_new[s_new == 0] = 1
        if np.array_equal(s_new, s):
            break
        s = s_new
    return np.where(s > 0, 0.0, np.pi)

def conn_annealed(J, noisy, lam, eta_phi, steps):
    N = len(noisy)
    phi = noisy.copy()
    A = np.ones(N)
    for step in range(steps):
        t = step / steps
        lam_t = lam * t
        s, c = np.sin(phi), np.cos(phi)
        coupling = (J @ (A * s)) * c - (J @ (A * c)) * s
        coherence = -2 * lam_t * A**2 * s * c
        dphi = A * coupling + coherence
        phi = (phi + eta_phi * dphi) % (2 * np.pi)
        dA = -2 * lam_t * A * s**2 - 1.0 * (A - 1)
        A = np.clip(A + 0.03 * dA, 0.01, 2.0)
    return phi

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 90)
    print("CONN vs Hopfield Capacity Test - Larger N (256 & 512)")
    print("Parameters: λ=0.5 (annealed), η=0.02, steps=400, jitter=0.5, noise=0.30")
    print("=" * 90)

    noise = 0.30
    jitter = 0.5
    trials = 30
    test_patterns_per_trial = 3

    all_results = []

    for N in [256, 512]:
        print(f"\n{'='*80}")
        print(f"                  N = {N}")
        print(f"{'='*80}")
        print(f"  {'M':>4s} {'Hopfield':>10s} {'CONN':>8s} {'Δ':>8s} {'hop_std':>8s} {'conn_std':>8s}")
        print(f"  {'-'*62}")

        hop_cap = 0
        conn_cap = 0
        max_M = int(0.35 * N)
        start_time = time.time()

        for M in range(1, max_M + 1):
            hop_ov = []
            conn_ov = []

            for trial in range(trials):
                patterns = generate_patterns(M, N, seed=trial)
                J = hebbian_weights(patterns)

                for mu in range(min(M, test_patterns_per_trial)):
                    rng = np.random.RandomState(trial * 10000 + mu)
                    noisy = bitflip_plus_jitter(patterns[mu], noise, jitter, rng)

                    ret_h = hopfield_recall(J, noisy)
                    hop_ov.append(compute_overlap(ret_h, patterns[mu]))

                    ret_c = conn_annealed(J, noisy, 0.5, 0.02, 400)
                    conn_ov.append(compute_overlap(ret_c, patterns[mu]))

            h_mean = np.mean(hop_ov)
            c_mean = np.mean(conn_ov)
            h_std = np.std(hop_ov)
            c_std = np.std(conn_ov)

            if h_mean >= 80: hop_cap = M
            if c_mean >= 80: conn_cap = M

            # Print strategically
            if (M <= 5 or M % max(4, N//32) == 0 or
                abs(h_mean - 80) < 10 or abs(c_mean - 80) < 10):
                print(f"  {M:4d} {h_mean:9.1f}% {c_mean:7.1f}% {c_mean-h_mean:+7.1f}% "
                      f"{h_std:7.1f}% {c_std:7.1f}%")

            # Early stopping
            if h_mean < 60 and c_mean < 60 and M > 20:
                break

        elapsed = time.time() - start_time
        ratio = conn_cap / hop_cap if hop_cap > 0 else 0

        print(f"\n  ┌────────────────────────────────────────────┐")
        print(f"  │ Hopfield capacity: M={hop_cap:3d} (α={hop_cap/N:.3f})     │")
        print(f"  │ CONN capacity:     M={conn_cap:3d} (α={conn_cap/N:.3f})     │")
        print(f"  │ Improvement ratio: {ratio:.2f}×                    │")
        print(f"  │ Time: {elapsed:.1f}s                                │")
        print(f"  └────────────────────────────────────────────┘")

        all_results.append({
            'N': N, 'hop_M': hop_cap, 'hop_alpha': hop_cap/N,
            'conn_M': conn_cap, 'conn_alpha': conn_cap/N, 'ratio': ratio
        })

    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    for r in all_results:
        print(f"  N={r['N']}: Hopfield M={r['hop_M']} (α={r['hop_alpha']:.3f}), "
              f"CONN M={r['conn_M']} (α={r['conn_alpha']:.3f}), "
              f"ratio={r['ratio']:.2f}×")
