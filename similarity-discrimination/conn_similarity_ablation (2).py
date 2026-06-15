"""
CONN Similarity Structure — Ablation Study
=============================================
ChatGPT's requested controls:
  1. Remove amplitude dynamics — is it the phases or the amplitudes?
  2. Remove coherence prior — is it the sin²(φ) term?
  3. Similarity sweep (70%, 80%, 85%, 90%, 95%)
  4. Computational budget control — give Hopfield more attempts
"""
import numpy as np
from scipy import stats
import time

class CONNAblation:
    def __init__(self, N, seed=42):
        self.N = N; self.rng = np.random.RandomState(seed)
        self.patterns = []; self.J = np.zeros((N, N))
    
    def store(self, phases):
        self.patterns.append(phases.copy())
        self.J = np.zeros((self.N, self.N))
        for mu in range(len(self.patterns)):
            xi = np.cos(self.patterns[mu])
            self.J += np.outer(xi, xi) / self.N
        np.fill_diagonal(self.J, 0)
    
    def retrieve(self, cue, lam=4.0, eta_phi=0.005, eta_A=0.03, steps=150,
                 use_amplitude=True, use_coherence=True):
        N = self.N; phi = cue.copy().astype(float)
        A = 0.5 + self.rng.random(N) * 0.3
        
        for step in range(steps):
            sd = np.sin(phi[np.newaxis,:] - phi[:,np.newaxis])
            
            if use_amplitude:
                coupling = np.sum(self.J * A[np.newaxis,:] * sd, axis=1)
            else:
                coupling = np.sum(self.J * sd, axis=1)  # no amplitude weighting
            
            if use_coherence:
                coh_term = -lam * (A*A if use_amplitude else 1.0) * np.sin(2 * phi)
            else:
                coh_term = 0  # no coherence prior
            
            dphi = (A if use_amplitude else 1.0) * coupling + coh_term
            phi = (phi + eta_phi * dphi) % (2*np.pi)
            
            if use_amplitude:
                dA = -lam * 2 * A * np.sin(phi)**2
                A = np.clip(A + eta_A * dA, 0.01, 2.0)
        
        out = np.where(np.cos(phi) > 0, 0.0, np.pi)
        return out
    
    def add_noise(self, phases, level):
        noisy = phases.copy()
        mask = self.rng.random(self.N) < level
        noisy[mask] = self.rng.random(np.sum(mask)) * 2 * np.pi
        return noisy
    
    def overlap(self, out, target):
        err = np.abs(np.arctan2(np.sin(out-target), np.cos(out-target)))
        return np.mean(err < np.pi/4)
    
    @property
    def n_stored(self): return len(self.patterns)


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


def run_discrimination_test(N, n_families, members, similarity, n_unrelated,
                            noise, seed, use_amplitude=True, use_coherence=True,
                            hopfield_attempts=1):
    rng = np.random.RandomState(seed)
    
    all_patterns = []
    family_members = {}
    
    for f in range(n_families):
        fam = generate_family(N, members, similarity, rng)
        family_members[f] = []
        for member in fam:
            family_members[f].append(len(all_patterns))
            all_patterns.append(member)
    
    for _ in range(n_unrelated):
        all_patterns.append(rng.choice([-1, 1], size=N))
    
    # Build CONN
    conn = CONNAblation(N, seed=seed+5000)
    for p in all_patterns:
        conn.store(b2p(p))
    
    # Build Hopfield
    W = np.zeros((N, N))
    for p in all_patterns:
        W += np.outer(p, p) / N
    np.fill_diagonal(W, 0)
    
    conn_correct = []; hop_correct = []
    
    for f in range(n_families):
        for midx in family_members[f]:
            target = all_patterns[midx]
            tp = b2p(target)
            
            # CONN
            cue = conn.add_noise(tp, noise)
            out = conn.retrieve(cue, lam=4.0, steps=150,
                              use_amplitude=use_amplitude, use_coherence=use_coherence)
            out_b = p2b(out)
            
            best = -1; best_ov = -1
            for j in family_members[f]:
                ov = np.mean(out_b == all_patterns[j])
                if ov > best_ov: best_ov = ov; best = j
            conn_correct.append(1 if best == midx else 0)
            
            # Hopfield (with multiple attempts if specified)
            best_hop = -1; best_hop_ov = -1
            for attempt in range(hopfield_attempts):
                cue_b = target.copy()
                cue_b[rng.random(N) < noise] *= -1
                state = cue_b.copy()
                for _ in range(100):
                    s2 = np.sign(W @ state); s2[s2==0] = 1
                    if np.array_equal(s2, state): break
                    state = s2
                
                for j in family_members[f]:
                    ov = np.mean(state == all_patterns[j])
                    if ov > best_hop_ov:
                        best_hop_ov = ov; best_hop = j
            
            hop_correct.append(1 if best_hop == midx else 0)
    
    return np.mean(conn_correct), np.mean(hop_correct)


if __name__ == "__main__":
    print("=" * 80)
    print("CONN SIMILARITY — ABLATION STUDY")
    print("=" * 80)
    
    N = 80; n_fam = 3; members = 4; n_unrel = 5; noise = 0.25; n_seeds = 15
    
    t0 = time.time()
    
    # ================================================================
    # ABLATION 1: Component removal
    # ================================================================
    print(f"\nABLATION 1: Which component drives discrimination?")
    print(f"N={N}, 3 families × 4 members, 85% similar, {n_seeds} seeds")
    print("-" * 80)
    
    configs = [
        ("Full CONN",           True,  True),
        ("No amplitude",        False, True),
        ("No coherence prior",  True,  False),
        ("No amplitude + no coherence", False, False),
    ]
    
    print(f"\n  {'Configuration':<30s}  {'CONN disc':>9s}  {'Hop disc':>8s}  {'Δ':>7s}")
    print(f"  {'-'*58}")
    
    for label, use_amp, use_coh in configs:
        cc = []; hh = []
        for seed in range(n_seeds):
            c, h = run_discrimination_test(
                N, n_fam, members, 0.85, n_unrel, noise, seed*100,
                use_amplitude=use_amp, use_coherence=use_coh)
            cc.append(c); hh.append(h)
        cm = np.mean(cc); hm = np.mean(hh)
        print(f"  {label:<30s}  {cm:8.1%}  {hm:7.1%}  {cm-hm:+6.1%}")
    
    # ================================================================
    # ABLATION 2: Similarity sweep
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"ABLATION 2: Similarity sweep (full CONN)")
    print("-" * 80)
    
    print(f"\n  {'Similarity':>10s}  {'CONN disc':>9s}  {'Hop disc':>8s}  {'Δ':>7s}")
    print(f"  {'-'*38}")
    
    for sim in [0.70, 0.80, 0.85, 0.90, 0.95]:
        cc = []; hh = []
        for seed in range(n_seeds):
            c, h = run_discrimination_test(
                N, n_fam, members, sim, n_unrel, noise, seed*100)
            cc.append(c); hh.append(h)
        print(f"  {sim:9.0%}  {np.mean(cc):8.1%}  {np.mean(hh):7.1%}  {np.mean(cc)-np.mean(hh):+6.1%}")
    
    # ================================================================
    # ABLATION 3: Computational budget control
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"ABLATION 3: Give Hopfield multiple attempts (85% similarity)")
    print("-" * 80)
    
    print(f"\n  {'Hop attempts':>12s}  {'CONN disc':>9s}  {'Hop disc':>8s}  {'Δ':>7s}")
    print(f"  {'-'*42}")
    
    for n_attempts in [1, 3, 5, 10]:
        cc = []; hh = []
        for seed in range(n_seeds):
            c, h = run_discrimination_test(
                N, n_fam, members, 0.85, n_unrel, noise, seed*100,
                hopfield_attempts=n_attempts)
            cc.append(c); hh.append(h)
        print(f"  {n_attempts:11d}  {np.mean(cc):8.1%}  {np.mean(hh):7.1%}  {np.mean(cc)-np.mean(hh):+6.1%}")
    
    elapsed = time.time() - t0
    
    # ================================================================
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print(f"{'=' * 80}")
    print(f"""
  ABLATION 1 answers: Which CONN component drives family discrimination?
    - If removing amplitude kills it: amplitude carries the signal
    - If removing coherence kills it: the sin²(φ) prior is essential
    - If removing both kills it: the full dynamics are needed
    - If nothing kills it: bare phase coupling alone is sufficient
    
  ABLATION 2 answers: At what similarity level does discrimination fail?
    - If CONN discriminates at 95%: very fine-grained phase structure
    - If CONN fails at 90%: the mechanism has a resolution limit
    - If CONN fails at 80%: the effect is narrow
    
  ABLATION 3 answers: Is Hopfield's failure computational or representational?
    - If 10 Hopfield attempts close the gap: the difference is 
      computational budget, not representation
    - If 10 attempts don't help: Hopfield genuinely lacks the 
      discriminative information — it's representational
    
  Time: {elapsed:.0f}s
""")
    print("=" * 80)
