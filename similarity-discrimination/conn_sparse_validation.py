"""
Experiment 11: Sparse Connectivity Test
=========================================
Does pairwise phase-difference coupling maintain discrimination
under sparse (25%) connectivity at N=1024?

Uses exact per-edge Hebbian weights on a directed random graph.
All models share the same sparse topology.

Based on corrected implementation after Gemini identified a bug
in Copilot's original sparse code (mean-field weight approximation
instead of exact per-edge Hebbian computation).
"""
import numpy as np
from scipy import stats
import time

def b2p(b): return np.where(b > 0, 0.0, np.pi)
def p2b(p): return np.sign(np.cos(p))

def generate_family(N, n_members, similarity, rng):
    proto = rng.choice([-1, 1], size=N)
    n_diff = int(N * (1 - similarity))
    family = [proto.copy()]
    for m in range(1, n_members):
        member = proto.copy()
        pos = rng.choice(N, size=n_diff, replace=False)
        member[pos] *= -1
        family.append(member)
    return family

def build_sparse_topology(N, k, rng):
    """Directed k-neighbor graph with no self-links."""
    neighbors = np.zeros((N, k), dtype=int)
    all_idx = np.arange(N)
    for i in range(N):
        choices = np.delete(all_idx, i)
        neighbors[i] = rng.choice(choices, size=k, replace=False)
    return neighbors

def compute_true_sparse_weights(P, neighbors):
    """
    Compute exact Hebbian weights on active edges.
    P: shape (P_count, N)
    neighbors: shape (N, k)
    Returns: J_sparse shape (N, k)
    J_sparse[i, ell] = (1/N) sum_mu P[mu,i] * P[mu, neighbors[i,ell]]
    """
    P = np.asarray(P)
    P_neighbors = P[:, neighbors]            # (P_count, N, k)
    P_targets = P[:, :, np.newaxis]          # (P_count, N, 1)
    J_sparse = np.sum(P_targets * P_neighbors, axis=0) / float(P.shape[1])
    return J_sparse

def retrieve_phase_sparse(J_sparse, neighbors, cue_phases, f_type,
                          steps=150, eta=0.005, tol=1e-8):
    phi = cue_phases.copy()
    N, k = neighbors.shape
    for _ in range(steps):
        phi_j = phi[neighbors]
        phi_i = phi[:, None]
        diff = phi_j - phi_i
        if f_type == 'sin':
            F = np.sin(diff)
        elif f_type == 'cos':
            F = np.cos(diff)
        elif f_type == 'linear':
            F = np.arctan2(np.sin(diff), np.cos(diff))
        coupling = np.sum(J_sparse * F, axis=1)
        delta = eta * coupling
        phi = (phi + delta) % (2 * np.pi)
        if np.max(np.abs(delta)) < tol:
            return np.where(np.cos(phi) > 0, 0.0, np.pi)
    return np.where(np.cos(phi) > 0, 0.0, np.pi)

def retrieve_tanh_sparse(J_sparse, neighbors, cue_phases,
                         steps=150, eta=0.02, beta=2.0, tol=1e-8):
    phi = cue_phases.copy()
    for _ in range(steps):
        phi_centered = (phi + np.pi) % (2*np.pi) - np.pi
        phi_j = phi_centered[neighbors]
        field = np.sum(J_sparse * phi_j, axis=1) / np.pi
        delta = eta * (-phi_centered + np.tanh(beta * field))
        phi = (phi + delta) % (2 * np.pi)
        if np.max(np.abs(delta)) < tol:
            return np.where(np.cos(phi) > 0, 0.0, np.pi)
    return np.where(np.cos(phi) > 0, 0.0, np.pi)

def retrieve_binary_hopfield_sparse(W_sparse, neighbors, cue_binary, steps=100):
    state = cue_binary.copy()
    for _ in range(steps):
        state_j = state[neighbors]
        field = np.sum(W_sparse * state_j, axis=1)
        s2 = np.sign(field)
        s2[s2 == 0] = 1
        if np.array_equal(s2, state): break
        state = s2
    return state


if __name__ == "__main__":
    print("=" * 80)
    print("EXPERIMENT 11: Sparse Connectivity (N=1024, k=256, 25% connectivity)")
    print("Exact per-edge Hebbian weights, same topology for all models")
    print("=" * 80)

    N = 1024
    sparse_k = 256
    n_fam = 3; members = 4; similarity = 0.85
    noise = 0.25; n_seeds = 20
    n_unrelated = int(5 * (N / 80))
    jitter = 0.15

    print(f"\nN={N}, k={sparse_k} ({sparse_k/N:.0%} connectivity)")
    print(f"{n_fam} families x {members} members, {n_unrelated} unrelated")
    print(f"Total patterns: {n_fam * members + n_unrelated}, "
          f"alpha = {(n_fam * members + n_unrelated)/N:.3f}")
    print(f"{n_seeds} seeds\n")

    t0 = time.time()
    results = {k: [] for k in ['sin', 'linear', 'cos', 'tanh', 'binary']}

    for seed_idx in range(n_seeds):
        rng = np.random.RandomState(7 + seed_idx * 101)

        # Build patterns
        all_patterns = []
        family_members = {}
        for f in range(n_fam):
            fam = generate_family(N, members, similarity, rng)
            family_members[f] = []
            for member in fam:
                family_members[f].append(len(all_patterns))
                all_patterns.append(member)
        for _ in range(n_unrelated):
            all_patterns.append(rng.choice([-1, 1], size=N))
        patterns = np.array(all_patterns)
        patterns_cos = np.cos(b2p(patterns))

        # Build sparse topology and exact weights
        neighbors = build_sparse_topology(N, sparse_k, rng)
        J_sparse = compute_true_sparse_weights(patterns_cos, neighbors)
        W_sparse = compute_true_sparse_weights(patterns, neighbors)

        # Test each family member
        for f in range(n_fam):
            for midx in family_members[f]:
                target = patterns[midx]
                cue_binary = target.copy()
                cue_binary[rng.random(N) < noise] *= -1
                cue_phases = b2p(cue_binary) + rng.randn(N) * jitter

                def classify(out_bin):
                    best = -1; best_ov = -1
                    for j in family_members[f]:
                        ov = np.mean(out_bin == patterns[j])
                        if ov > best_ov: best_ov = ov; best = j
                    return 1 if best == midx else 0

                out = retrieve_phase_sparse(J_sparse, neighbors, cue_phases, 'sin')
                results['sin'].append(classify(p2b(out)))
                out = retrieve_phase_sparse(J_sparse, neighbors, cue_phases, 'linear')
                results['linear'].append(classify(p2b(out)))
                out = retrieve_phase_sparse(J_sparse, neighbors, cue_phases, 'cos')
                results['cos'].append(classify(p2b(out)))
                out = retrieve_tanh_sparse(J_sparse, neighbors, cue_phases)
                results['tanh'].append(classify(p2b(out)))
                out_bin = retrieve_binary_hopfield_sparse(W_sparse, neighbors, cue_binary)
                results['binary'].append(classify(out_bin))

        if (seed_idx + 1) % 5 == 0:
            print(f"  Seeds completed: {seed_idx + 1}/{n_seeds}")

    elapsed = time.time() - t0

    # Results
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")

    models = [
        ('A. sin(phi_j - phi_i)', 'sin'),
        ('B. linear circular', 'linear'),
        ('C. cos(phi_j - phi_i)', 'cos'),
        ('D. tanh on circle', 'tanh'),
        ('E. Binary Hopfield', 'binary'),
    ]

    print(f"\n  {'Model':<25s}  {'Discrimination':>13s}  {'±Std':>7s}")
    print(f"  {'-'*48}")
    for label, key in models:
        arr = np.array(results[key])
        print(f"  {label:<25s}  {arr.mean():12.1%}  ±{arr.std():.1%}")

    # Statistical tests
    print(f"\n  Pairwise vs Binary (Wilcoxon):")
    for label, key in models[:-1]:
        a = results[key]; b = results['binary']
        diff = np.array(a) - np.array(b)
        if np.all(diff == 0) or np.std(diff) == 0:
            print(f"    {label:<25s}: constant (no variance)")
            continue
        try:
            stat, p = stats.wilcoxon(a, b, alternative='greater')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        except:
            p = float('nan'); sig = "?"
        print(f"    {label:<25s}: p={p:.4f} {sig}")

    # Bernoulli check
    print(f"\n  Bernoulli SD check:")
    theoretical_sd = np.sqrt(0.25 * 0.75)
    for label, key in [('D. tanh', 'tanh'), ('E. Binary', 'binary')]:
        empirical_sd = np.std(results[key])
        print(f"    {label}: empirical SD = {empirical_sd:.4f}, "
              f"theoretical = {theoretical_sd:.4f}, "
              f"match = {'YES' if abs(empirical_sd - theoretical_sd) < 0.01 else 'NO'}")

    print(f"\n  Time: {elapsed:.0f}s")
    print("=" * 80)
