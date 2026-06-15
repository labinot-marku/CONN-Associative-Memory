"""
Fair Comparison: Phase-coupled vs Continuous Hopfield vs Binary Hopfield
=========================================================================
Addresses ChatGPT's methodological concerns:
  1. Matched noise model across all systems
  2. Continuous Hopfield baseline (tanh, not sign)
  3. Same evaluation metric for all
  4. Tests whether the effect is phase-specific or continuity-general
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


def retrieve_phase_coupled(J, cue_phases, rng, steps=150, eta=0.005):
    """Bare phase coupling: dphi = eta * sum_j J_ij * sin(phi_j - phi_i)
    No amplitude, no coherence prior. The minimal oscillatory system."""
    N = len(cue_phases)
    phi = cue_phases.copy()
    for step in range(steps):
        sd = np.sin(phi[np.newaxis,:] - phi[:,np.newaxis])
        coupling = np.sum(J * sd, axis=1)
        phi = (phi + eta * coupling) % (2 * np.pi)
    return np.where(np.cos(phi) > 0, 0.0, np.pi)


def retrieve_continuous_hopfield(W, cue_continuous, steps=150, dt=0.1, beta=2.0):
    """Continuous Hopfield: dx/dt = -x + tanh(beta * W @ x)
    Continuous-valued neurons, same weight matrix as binary Hopfield."""
    x = cue_continuous.copy()
    for step in range(steps):
        x = x + dt * (-x + np.tanh(beta * W @ x))
    return x


def retrieve_binary_hopfield(W, cue_binary, steps=100):
    """Standard binary Hopfield: s = sign(W @ s)"""
    state = cue_binary.copy()
    for _ in range(steps):
        s_new = np.sign(W @ state); s_new[s_new == 0] = 1
        if np.array_equal(s_new, state): break
        state = s_new
    return state


def run_fair_test(N, n_families, members, similarity, n_unrelated, noise, seed):
    rng = np.random.RandomState(seed)
    
    # Generate patterns
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
    
    # Build weight matrices
    # Phase-coupled: J from cos encoding
    J_phase = np.zeros((N, N))
    for p in all_patterns:
        xi = np.cos(b2p(p))
        J_phase += np.outer(xi, xi) / N
    np.fill_diagonal(J_phase, 0)
    
    # Hopfield: standard W
    W_hop = np.zeros((N, N))
    for p in all_patterns:
        W_hop += np.outer(p, p) / N
    np.fill_diagonal(W_hop, 0)
    
    phase_correct = []
    cont_hop_correct = []
    bin_hop_correct = []
    
    for f in range(n_families):
        for midx in family_members[f]:
            target = all_patterns[midx]
            
            # === MATCHED NOISE: binary bit-flip for ALL systems ===
            cue_binary = target.copy()
            flip_mask = rng.random(N) < noise
            cue_binary[flip_mask] *= -1
            
            # Convert the SAME noisy binary cue to phases for the phase system
            cue_phases = b2p(cue_binary)
            # Add small jitter to activate dynamics (minimal, matched)
            cue_phases += rng.randn(N) * 0.15
            
            # Convert the SAME noisy binary cue to continuous for cont. Hopfield
            cue_continuous = cue_binary.astype(float) + rng.randn(N) * 0.15
            
            # --- Phase-coupled retrieval ---
            out_phases = retrieve_phase_coupled(J_phase, cue_phases, rng)
            out_phase_binary = p2b(out_phases)
            
            # --- Continuous Hopfield retrieval ---
            out_cont = retrieve_continuous_hopfield(W_hop, cue_continuous)
            out_cont_binary = np.sign(out_cont)
            out_cont_binary[out_cont_binary == 0] = 1
            
            # --- Binary Hopfield retrieval ---
            out_bin = retrieve_binary_hopfield(W_hop, cue_binary)
            
            # === SAME evaluation for all: which family member is closest? ===
            def classify(output, fam_indices):
                best = -1; best_ov = -1
                for j in fam_indices:
                    ov = np.mean(output == all_patterns[j])
                    if ov > best_ov: best_ov = ov; best = j
                return best
            
            phase_correct.append(1 if classify(out_phase_binary, family_members[f]) == midx else 0)
            cont_hop_correct.append(1 if classify(out_cont_binary, family_members[f]) == midx else 0)
            bin_hop_correct.append(1 if classify(out_bin, family_members[f]) == midx else 0)
    
    return np.mean(phase_correct), np.mean(cont_hop_correct), np.mean(bin_hop_correct)


if __name__ == "__main__":
    print("=" * 80)
    print("FAIR THREE-WAY COMPARISON")
    print("Matched noise, matched evaluation, matched computation")
    print("=" * 80)
    
    N = 80; n_fam = 3; members = 4; n_unrel = 5; noise = 0.25; n_seeds = 20
    
    print(f"\nN={N}, {n_fam} families × {members} members, {n_seeds} seeds")
    print(f"All systems receive the SAME binary-corrupted cue")
    print(f"All systems evaluated with the SAME binarized comparison\n")
    
    t0 = time.time()
    
    # ================================================================
    # TEST 1: Main comparison at 85% similarity
    # ================================================================
    print("TEST 1: Family discrimination at 85% similarity")
    print("-" * 80)
    
    phase_all = []; cont_all = []; bin_all = []
    for seed in range(n_seeds):
        p, c, b = run_fair_test(N, n_fam, members, 0.85, n_unrel, noise, seed*100)
        phase_all.append(p); cont_all.append(c); bin_all.append(b)
    
    print(f"\n  {'System':<25s}  {'Discrimination':>13s}  {'±Std':>7s}")
    print(f"  {'-'*48}")
    print(f"  {'Phase-coupled (sin)':25s}  {np.mean(phase_all):12.1%}  ±{np.std(phase_all):.1%}")
    print(f"  {'Continuous Hopfield (tanh)':25s}  {np.mean(cont_all):12.1%}  ±{np.std(cont_all):.1%}")
    print(f"  {'Binary Hopfield (sign)':25s}  {np.mean(bin_all):12.1%}  ±{np.std(bin_all):.1%}")
    
    # Statistical tests
    print(f"\n  Pairwise tests (Wilcoxon):")
    for label, a, b_data in [
        ("Phase vs Binary Hop", phase_all, bin_all),
        ("Phase vs Cont Hop", phase_all, cont_all),
        ("Cont Hop vs Binary Hop", cont_all, bin_all),
    ]:
        diff = np.array(a) - np.array(b_data)
        if np.all(diff == 0):
            print(f"    {label:<25s}: identical")
            continue
        try:
            stat, p = stats.wilcoxon(a, b_data, alternative='greater')
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        except:
            p = 1; sig = "?"
        print(f"    {label:<25s}: Δ={np.mean(diff):+.1%} p={p:.4f} {sig}")
    
    # ================================================================
    # TEST 2: Similarity sweep (all three systems)
    # ================================================================
    print(f"\n{'=' * 80}")
    print("TEST 2: Similarity sweep (all three systems)")
    print("-" * 80)
    
    print(f"\n  {'Sim':>5s}  {'Phase':>8s}  {'Cont Hop':>8s}  {'Bin Hop':>8s}  {'Ph-Bin':>7s}  {'Ph-Cont':>7s}")
    print(f"  {'-'*50}")
    
    for sim in [0.70, 0.80, 0.85, 0.90, 0.95]:
        pp = []; cc = []; bb = []
        for seed in range(n_seeds):
            p, c, b = run_fair_test(N, n_fam, members, sim, n_unrel, noise, seed*100)
            pp.append(p); cc.append(c); bb.append(b)
        pm = np.mean(pp); cm = np.mean(cc); bm = np.mean(bb)
        print(f"  {sim:4.0%}  {pm:7.1%}  {cm:7.1%}  {bm:7.1%}  {pm-bm:+6.1%}  {pm-cm:+6.1%}")
    
    elapsed = time.time() - t0
    
    # ================================================================
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print(f"{'=' * 80}")
    print(f"""
  This test addresses ChatGPT's methodological concerns:
  
  1. MATCHED NOISE: All systems receive the same binary-corrupted cue.
     Phase system gets cue converted to phases + minimal jitter.
     Continuous Hopfield gets cue as float + minimal jitter.
     Binary Hopfield gets cue directly.
     
  2. MATCHED EVALUATION: All outputs binarized, then compared
     to binary family members via overlap.
     
  3. THREE-WAY separates the questions:
     Phase vs Binary:     Does continuous oscillatory dynamics help?
     Cont Hop vs Binary:  Does continuity alone help?
     Phase vs Cont Hop:   Does PHASE COUPLING specifically help
                          beyond generic continuity?
     
  POSSIBLE OUTCOMES:
  
  A. Phase >> Cont Hop >> Binary:
     Phase coupling is specifically advantageous.
     Continuity helps some, phase geometry helps more.
     
  B. Phase ≈ Cont Hop >> Binary:
     Continuity is what matters, not phase coupling specifically.
     Any continuous system would work.
     
  C. Phase >> Cont Hop ≈ Binary:
     Phase coupling is essential. Continuity without phase
     coupling doesn't help.
     
  D. Phase ≈ Cont Hop ≈ Binary:
     The original result was a methodological artifact.
     
  Time: {elapsed:.0f}s
""")
    print("=" * 80)
