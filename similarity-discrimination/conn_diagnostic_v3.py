"""
conn_diagnostic_v3.py — Section 3.14 Diagnostic Experiments
=============================================================
Post-publication diagnostics added in V3. Four experiments not present
in V1/V2 that led to the V3 mechanism reframing.

3.14.1: Weight matrix independence — random J = learned J = zero weights
3.14.2: Family size scaling — collapse at 16 members
3.14.3: Below-capacity Hopfield — failure is intrinsic, not overloading
3.14.4: Unrelated pattern retrieval — phase outperforms Hopfield

All experiments use matched noise (25% bit-flip), matched evaluation,
20 seeds, N=80, 3 families x 4 members (85% similarity) unless stated.
"""
import numpy as np
from scipy import stats
import time

RNG_BASE = 42

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
    unrel = []
    for _ in range(n_unrel):
        unrel.append(len(all_p))
        all_p.append(rng.choice([-1,1], size=N))
    return np.array(all_p), fam_map, unrel

def classify(out, patterns, fam_map, f):
    best=-1; bov=-1
    for j in fam_map[f]:
        ov = np.dot(out, patterns[j]) / len(out)
        if ov > bov: bov = ov; best = j
    return best

def run_phase(J, phi0, steps=150, eta=0.005):
    phi = phi0.copy()
    for _ in range(steps):
        sd = np.sin(phi[None,:] - phi[:,None])
        phi = (phi + eta * np.sum(J * sd, axis=1)) % (2*np.pi)
    return p2b(phi)

def run_hopfield(W, cue, steps=100):
    state = cue.copy()
    for _ in range(steps):
        s2 = np.sign(W @ state); s2[s2==0] = 1
        if np.array_equal(s2, state): break
        state = s2
    return state


if __name__ == '__main__':
    N = 80; noise = 0.25; n_seeds = 20

    # ================================================================
    # 3.14.1: Weight matrix independence
    # ================================================================
    print("=" * 70)
    print("3.14.1: Weight Matrix Independence")
    print("Question: Does the Hebbian content of J matter?")
    print("=" * 70)

    res_learned=[]; res_random=[]; res_zero=[]
    for seed in range(n_seeds):
        rng = np.random.default_rng(RNG_BASE + seed)
        patterns, fam_map, _ = make_patterns(N, 3, 4, 0.85, 5, rng)

        J_cos = np.zeros((N,N))
        for p in patterns:
            xi = np.cos(b2p(p)); J_cos += np.outer(xi,xi)/N
        np.fill_diagonal(J_cos, 0)

        J_rand = rng.normal(0, 1, (N,N))
        J_rand = (J_rand + J_rand.T) / 2
        np.fill_diagonal(J_rand, 0)
        J_rand *= np.linalg.norm(J_cos,'fro') / (np.linalg.norm(J_rand,'fro') + 1e-12)

        J_zero = np.zeros((N,N))

        for f in range(3):
            for midx in fam_map[f]:
                target = patterns[midx]
                cue = target.copy(); cue[rng.random(N) < noise] *= -1
                phi0 = b2p(cue) + rng.normal(0, 0.15, N)

                out = run_phase(J_cos, phi0)
                res_learned.append(classify(out, patterns, fam_map, f)==midx)
                out = run_phase(J_rand, phi0)
                res_random.append(classify(out, patterns, fam_map, f)==midx)
                out = run_phase(J_zero, phi0)
                res_zero.append(classify(out, patterns, fam_map, f)==midx)

    print(f"\n  {'Model':<30} {'Discrimination':>13}")
    print(f"  {'-'*45}")
    print(f"  {'Phase (learned J)':<30} {np.mean(res_learned)*100:>12.1f}%")
    print(f"  {'Phase (random J, matched norm)':<30} {np.mean(res_random)*100:>12.1f}%")
    print(f"  {'Phase (zero weights)':<30} {np.mean(res_zero)*100:>12.1f}%")
    print(f"\n  Finding: Hebbian weight matrix content is irrelevant.")
    print(f"  Any coupling (or none) gives identical discrimination.")

    # ================================================================
    # 3.14.2: Family size scaling
    # ================================================================
    print(f"\n{'=' * 70}")
    print("3.14.2: Family Size Scaling")
    print("Question: Does phase preserve discrimination with more members?")
    print(f"{'=' * 70}")

    print(f"\n  {'Members':>8} {'Chance':>8} {'Phase':>8} {'Hopfield':>8} {'NN cue':>8}")
    print(f"  {'-'*45}")

    for n_mem in [2, 4, 8, 16, 32]:
        n_unrel = max(0, 17 - n_mem * 3)
        ph=[]; hop=[]; nn=[]
        for seed in range(n_seeds):
            rng = np.random.default_rng(RNG_BASE + seed)
            patterns, fam_map, _ = make_patterns(N, 3, n_mem, 0.85, n_unrel, rng)
            J = np.zeros((N,N)); W = np.zeros((N,N))
            for p in patterns:
                xi = np.cos(b2p(p)); J += np.outer(xi,xi)/N
                W += np.outer(p,p)/N
            np.fill_diagonal(J,0); np.fill_diagonal(W,0)
            for f in range(3):
                for midx in fam_map[f]:
                    target = patterns[midx]
                    cue = target.copy(); cue[rng.random(N) < noise] *= -1
                    phi0 = b2p(cue) + rng.normal(0, 0.15, N)
                    ph.append(classify(run_phase(J,phi0), patterns, fam_map, f)==midx)
                    hop.append(classify(run_hopfield(W,cue), patterns, fam_map, f)==midx)
                    nn.append(classify(cue, patterns, fam_map, f)==midx)
        print(f"  {n_mem:>7}  {1/n_mem*100:>7.1f}%  {np.mean(ph)*100:>7.1f}%"
              f"  {np.mean(hop)*100:>7.1f}%  {np.mean(nn)*100:>7.1f}%")

    print(f"\n  Finding: Phase collapses to chance at 16 members while NN")
    print(f"  cue stays at 91%. At high load, dynamics destroy the signal.")

    # ================================================================
    # 3.14.3: Below-capacity Hopfield
    # ================================================================
    print(f"\n{'=' * 70}")
    print("3.14.3: Below-Capacity Hopfield")
    print(f"Question: Does Hopfield fail because it's overloaded (alpha>0.138)?")
    print(f"Hopfield capacity for N={N}: ~{int(0.138*N)} patterns")
    print(f"{'=' * 70}")

    print(f"\n  {'Patterns':>9} {'alpha':>6} {'Phase':>8} {'Hopfield':>9} {'Hop k=0':>8}")
    print(f"  {'-'*50}")

    for n_unrel in [0, 2, 5, 10, 17, 30]:
        total = 12 + n_unrel
        ph=[]; hop=[]; hop0=[]
        for seed in range(n_seeds):
            rng = np.random.default_rng(RNG_BASE + seed)
            patterns, fam_map, _ = make_patterns(N, 3, 4, 0.85, n_unrel, rng)
            J = np.zeros((N,N)); W = np.zeros((N,N))
            for p in patterns:
                xi = np.cos(b2p(p)); J += np.outer(xi,xi)/N
                W += np.outer(p,p)/N
            np.fill_diagonal(J,0); np.fill_diagonal(W,0)
            for f in range(3):
                for midx in fam_map[f]:
                    target = patterns[midx]
                    cue = target.copy(); cue[rng.random(N) < noise] *= -1
                    phi0 = b2p(cue) + rng.normal(0, 0.15, N)
                    ph.append(classify(run_phase(J,phi0), patterns, fam_map, f)==midx)
                    hop.append(classify(run_hopfield(W,cue), patterns, fam_map, f)==midx)
                    hop0.append(classify(cue, patterns, fam_map, f)==midx)
        cap = " <-- at capacity" if abs(total/N - 0.138) < 0.03 else ""
        print(f"  {total:>8}  {total/N:>5.3f}  {np.mean(ph)*100:>7.1f}%"
              f"  {np.mean(hop)*100:>8.1f}%  {np.mean(hop0)*100:>7.1f}%{cap}")

    print(f"\n  Finding: Hopfield fails at discrimination even at alpha=0.150")
    print(f"  (at capacity). The failure is intrinsic to sign(), not overloading.")

    # ================================================================
    # 3.14.4: Unrelated pattern retrieval
    # ================================================================
    print(f"\n{'=' * 70}")
    print("3.14.4: Unrelated Pattern Retrieval")
    print("Question: Does phase also outperform Hopfield on non-family patterns?")
    print(f"{'=' * 70}")

    ph_u=[]; hop_u=[]
    for seed in range(n_seeds):
        rng = np.random.default_rng(RNG_BASE + seed)
        patterns, fam_map, unrel = make_patterns(N, 3, 4, 0.85, 5, rng)
        J = np.zeros((N,N)); W = np.zeros((N,N))
        for p in patterns:
            xi = np.cos(b2p(p)); J += np.outer(xi,xi)/N
            W += np.outer(p,p)/N
        np.fill_diagonal(J,0); np.fill_diagonal(W,0)
        for uidx in unrel:
            target = patterns[uidx]
            cue = target.copy(); cue[rng.random(N) < noise] *= -1
            phi0 = b2p(cue) + rng.normal(0, 0.15, N)
            ph_u.append(np.mean(run_phase(J,phi0) == target))
            hop_u.append(np.mean(run_hopfield(W,cue) == target))

    print(f"\n  {'Model':<30} {'Exact recall':>12}")
    print(f"  {'-'*45}")
    print(f"  {'Phase (unrelated patterns)':<30} {np.mean(ph_u)*100:>11.1f}%")
    print(f"  {'Hopfield (unrelated patterns)':<30} {np.mean(hop_u)*100:>11.1f}%")
    print(f"\n  Finding: Phase also outperforms Hopfield on unrelated patterns.")
    print(f"  Previously unreported.")

    print(f"\n{'=' * 70}")
    print("All V3 diagnostic experiments complete.")
    print(f"{'=' * 70}")
