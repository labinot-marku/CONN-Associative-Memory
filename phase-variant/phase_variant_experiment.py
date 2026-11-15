"""
Phase-variant storage experiment (Geometry/Topology preprint, Version 2, Section 5.1)
====================================================================================

Question: can a CONN network store K globally phase-shifted variants
xi^mu * exp(i*theta_k), theta_k = 2*pi*k/K, of each spatial pattern and
recover both the spatial pattern and the variant from a noisy cue?

Dynamics: identical to validation/CONN_VALIDATION_V5.py (CONN v2):
  eta_phi = 0.02, eta_A = 0.03, beta = 1.0, 400 steps, lambda annealed 0 -> lambda_final.
Noise: identical to CONN v2: 30% pi-flips + Gaussian jitter (sigma = 0.5 rad).
Weights: Hebbian from the spatial patterns (J is the same for every global phase variant).

Conditions:
  lambda_final = 0.5 (standard CONN; coherence term pins phases to {0, pi};
                      symmetry of the dynamics: Z2, phi -> phi + pi)
  lambda_final = 0.0 (no coherence term; symmetry: full U(1), phi -> phi + theta)

Decoder: for every (mu, k) compute score = mean_j cos(phi_j - xi^mu_j - theta_k)
on the retrieved phases; pick the argmax. Spatial correct if mu matches,
variant correct if k matches, success if both match.

Predictions written before running (from Proposition 3 of the preprint):
  P1  K=2, both lambda: variants are xi and -xi (exact Z2 symmetry); the cue's
      sign is carried to the output, so variant accuracy ~ spatial accuracy (not chance).
  P2  K=4, lambda=0: U(1) symmetry; the global phase is carried, variant accuracy high.
  P3  K=4, lambda=0.5: variants at pi/2 and 3*pi/2 are not equilibria and get pushed
      to 0 or pi; those variants fail, the others succeed -> overall variant
      accuracy near 50%, not 25%.

Usage: python phase_variant_experiment.py  (writes phase_variant_results.csv,
phase_variant_per_k.csv). Seeded; fully reproducible. Runtime ~1 min.
"""
import csv
import numpy as np
from scipy.stats import binomtest

ETA_PHI, ETA_A, BETA, STEPS = 0.02, 0.03, 1.0, 400
NOISE, JITTER = 0.30, 0.5
TRIALS_PER_PATTERN = 30


def conn_recall(J, phi0, lam_final):
    """CONN v2 recall (copied from validation/CONN_VALIDATION_V5.py)."""
    phi = phi0.copy()
    A = np.ones(len(phi))
    for step in range(STEPS):
        lam = lam_final * (step / STEPS)
        s, c = np.sin(phi), np.cos(phi)
        coupling = (J @ (A * s)) * c - (J @ (A * c)) * s
        dphi = A * coupling - 2 * lam * A**2 * s * c
        phi = (phi + ETA_PHI * dphi) % (2 * np.pi)
        dA = -2 * lam * A * s**2 - 2 * BETA * (A - 1)
        A = np.clip(A + ETA_A * dA, 0.01, 2.0)
    return phi


def add_noise(target, rng):
    noisy = target.copy()
    mask = rng.random(len(noisy)) < NOISE
    noisy[mask] += np.pi
    noisy += rng.normal(0, JITTER, len(noisy))
    return np.mod(noisy, 2 * np.pi)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return 100 * (centre - half), 100 * (centre + half)


def run(N, M, K, lam, seed):
    rng = np.random.default_rng(seed)
    xi = rng.choice([0.0, np.pi], size=(M, N))
    s = np.cos(xi)
    J = s.T @ s / N
    np.fill_diagonal(J, 0)
    thetas = 2 * np.pi * np.arange(K) / K
    n = sp = ph = both = 0
    per_k = np.zeros((K, 2), dtype=int)  # [correct, total]
    for mu in range(M):
        for k in range(K):
            for _ in range(TRIALS_PER_PATTERN):
                cue = add_noise(np.mod(xi[mu] + thetas[k], 2 * np.pi), rng)
                out = conn_recall(J, cue, lam)
                scores = np.cos(out[None, None, :] - xi[:, None, :] - thetas[None, :, None]).mean(-1)
                m_hat, k_hat = np.unravel_index(np.argmax(scores), scores.shape)
                n += 1
                sp += m_hat == mu
                ph += k_hat == k
                both += (m_hat == mu) and (k_hat == k)
                per_k[k] += [k_hat == k, 1]
    return n, sp, ph, both, per_k


def main():
    rows, krows = [], []
    configs = [(32, 10), (128, 10)]
    for N, M in configs:
        for lam in (0.5, 0.0):
            for K in (2, 4):
                n, sp, ph, both, per_k = run(N, M, K, lam, seed=1000 * N + 10 * K + int(10 * lam))
                p = binomtest(int(ph), n, 1 / K).pvalue
                lo, hi = wilson(ph, n)
                row = dict(N=N, M_spatial=M, K=K, lambda_final=lam, trials=n,
                           success=round(100 * both / n, 1), spatial=round(100 * sp / n, 1),
                           variant=round(100 * ph / n, 1), variant_CI95=f"{lo:.1f}-{hi:.1f}",
                           chance=round(100 / K, 1), p_vs_chance=f"{p:.2g}")
                rows.append(row)
                print(row)
                for k in range(K):
                    krows.append(dict(N=N, K=K, lambda_final=lam, k=k,
                                      theta=f"{2*k/K:g}pi",
                                      variant_acc=round(100 * per_k[k, 0] / per_k[k, 1], 1)))
    with open("phase_variant_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with open("phase_variant_per_k.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=krows[0].keys()); w.writeheader(); w.writerows(krows)
    for r in krows:
        print(r)


if __name__ == "__main__":
    main()
