"""
Phase System Decomposition
============================
ChatGPT's most important remaining question:
Is it the sinusoid, the circle, or both?

Five coupling models on the same task:
  A. sin(φ_j - φ_i)     — sinusoidal on circle (current system)
  B. (φ_j - φ_i) mod 2π — linear difference on circle
  C. cos(φ_j - φ_i)     — cosine on circle (different function)
  D. tanh(W @ φ)         — continuous on circle but no pairwise phase diff
  E. Binary Hopfield     — baseline

If only A works: the sinusoidal coupling function is essential
If A and B work: circular topology matters, coupling function doesn't
If A and C work: any periodic function on the circle works
If A, B, C all work: it's the circle, not the coupling
If only A fails and others work: something else is going on
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


def retrieve_sin_coupling(J, cue_phases, steps=150, eta=0.005):
    """Model A: dφ_i = η Σ_j J_ij sin(φ_j - φ_i)"""
    phi = cue_phases.copy()
    for _ in range(steps):
        sd = np.sin(phi[np.newaxis,:] - phi[:,np.newaxis])
        coupling = np.sum(J * sd, axis=1)
        phi = (phi + eta * coupling) % (2 * np.pi)
    return np.where(np.cos(phi) > 0, 0.0, np.pi)


def retrieve_linear_coupling(J, cue_phases, steps=150, eta=0.002):
    """Model B: dφ_i = η Σ_j J_ij (φ_j - φ_i), wrapped to circle
    Linear phase difference, still on circular manifold."""
    phi = cue_phases.copy()
    for _ in range(steps):
        # Circular difference: wrap to [-π, π]
        diff = phi[np.newaxis,:] - phi[:,np.newaxis]
        diff = np.arctan2(np.sin(diff), np.cos(diff))  # proper circular diff
        coupling = np.sum(J * diff, axis=1)
        phi = (phi + eta * coupling) % (2 * np.pi)
    return np.where(np.cos(phi) > 0, 0.0, np.pi)


def retrieve_cos_coupling(J, cue_phases, steps=150, eta=0.005):
    """Model C: dφ_i = η Σ_j J_ij cos(φ_j - φ_i)
    Cosine coupling on circle — gradient of -sin interaction energy."""
    phi = cue_phases.copy()
    for _ in range(steps):
        cd = np.cos(phi[np.newaxis,:] - phi[:,np.newaxis])
        coupling = np.sum(J * cd, axis=1)
        phi = (phi + eta * coupling) % (2 * np.pi)
    return np.where(np.cos(phi) > 0, 0.0, np.pi)


def retrieve_tanh_on_circle(J, cue_phases, steps=150, eta=0.05, beta=2.0):
    """Model D: dφ_i = -φ_i + tanh(β Σ_j J_ij φ_j), wrapped to circle
    Continuous Hopfield-like dynamics but living on the circle."""
    phi = cue_phases.copy()
    for _ in range(steps):
        field = J @ phi
        phi = (phi + eta * (-phi + np.tanh(beta * field))) % (2 * np.pi)
    return np.where(np.cos(phi) > 0, 0.0, np.pi)


def retrieve_binary_hopfield(W, cue_binary, steps=100):
    """Model E: Standard binary Hopfield."""
    state = cue_binary.copy()
    for _ in range(steps):
        s2 = np.sign(W @ state); s2[s2==0] = 1
        if np.array_equal(s2, state): break
        state = s2
    return state


def run_decomposition(N, n_families, members, similarity, n_unrelated, noise, seed):
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
    
    # Build weight matrices
    J = np.zeros((N, N))
    for p in all_patterns:
        xi = np.cos(b2p(p))
        J += np.outer(xi, xi) / N
    np.fill_diagonal(J, 0)
    
    W = np.zeros((N, N))
    for p in all_patterns:
        W += np.outer(p, p) / N
    np.fill_diagonal(W, 0)
    
    results = {name: [] for name in ['sin', 'linear', 'cos', 'tanh_circle', 'binary']}
    
    for f in range(n_families):
        for midx in family_members[f]:
            target = all_patterns[midx]
            
            # Matched binary noise
            cue_binary = target.copy()
            cue_binary[rng.random(N) < noise] *= -1
            cue_phases = b2p(cue_binary) + rng.randn(N) * 0.15
            
            def classify(output_binary):
                best = -1; best_ov = -1
                for j in family_members[f]:
                    ov = np.mean(output_binary == all_patterns[j])
                    if ov > best_ov: best_ov = ov; best = j
                return 1 if best == midx else 0
            
            # Model A: sin coupling
            out = retrieve_sin_coupling(J, cue_phases)
            results['sin'].append(classify(p2b(out)))
            
            # Model B: linear coupling on circle
            out = retrieve_linear_coupling(J, cue_phases)
            results['linear'].append(classify(p2b(out)))
            
            # Model C: cos coupling
            out = retrieve_cos_coupling(J, cue_phases)
            results['cos'].append(classify(p2b(out)))
            
            # Model D: tanh on circle
            out = retrieve_tanh_on_circle(J, cue_phases)
            results['tanh_circle'].append(classify(p2b(out)))
            
            # Model E: binary Hopfield
            out_bin = retrieve_binary_hopfield(W, cue_binary)
            results['binary'].append(classify(out_bin))
    
    return {k: np.mean(v) for k, v in results.items()}


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE SYSTEM DECOMPOSITION")
    print("Is it the sinusoid, the circle, or both?")
    print("=" * 80)
    
    N = 80; n_fam = 3; members = 4; n_unrel = 5; noise = 0.25; n_seeds = 20
    
    print(f"\nN={N}, {n_fam} families × {members} members, {n_seeds} seeds")
    print(f"All systems: matched noise, matched evaluation\n")
    
    t0 = time.time()
    
    # ================================================================
    # TEST 1: Main comparison at 85% similarity
    # ================================================================
    print("TEST 1: Five coupling models at 85% similarity")
    print("-" * 80)
    
    all_results = {k: [] for k in ['sin', 'linear', 'cos', 'tanh_circle', 'binary']}
    
    for seed in range(n_seeds):
        r = run_decomposition(N, n_fam, members, 0.85, n_unrel, noise, seed * 100)
        for k in all_results:
            all_results[k].append(r[k])
    
    models = [
        ('A. sin(φ_j − φ_i)', 'sin', 'Sinusoidal on circle'),
        ('B. (φ_j − φ_i) circ', 'linear', 'Linear diff on circle'),
        ('C. cos(φ_j − φ_i)', 'cos', 'Cosine on circle'),
        ('D. tanh(J@φ) on S¹', 'tanh_circle', 'tanh dynamics on circle'),
        ('E. Binary Hopfield', 'binary', 'sign(Wx) baseline'),
    ]
    
    print(f"\n  {'Model':<25s}  {'Discrimination':>13s}  {'±Std':>7s}  {'Description'}")
    print(f"  {'-'*70}")
    for label, key, desc in models:
        m = np.mean(all_results[key]); s = np.std(all_results[key])
        print(f"  {label:<25s}  {m:12.1%}  ±{s:.1%}  {desc}")
    
    # Statistical tests vs binary
    print(f"\n  Pairwise vs Binary Hopfield (Wilcoxon):")
    for label, key, _ in models[:-1]:
        a = all_results[key]; b = all_results['binary']
        diff = np.array(a) - np.array(b)
        if np.all(diff == 0):
            print(f"    {label:<25s}: identical")
            continue
        try:
            stat, p = stats.wilcoxon(a, b, alternative='greater')
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        except:
            p = 1; sig = "?"
        print(f"    {label:<25s}: Δ={np.mean(diff):+.1%} p={p:.4f} {sig}")
    
    # Pairwise: sin vs each other model
    print(f"\n  Pairwise vs sin-coupling (Wilcoxon):")
    for label, key, _ in models[1:]:
        a = all_results['sin']; b = all_results[key]
        diff = np.array(a) - np.array(b)
        if np.all(diff == 0):
            print(f"    sin vs {label:<20s}: identical")
            continue
        try:
            stat, p = stats.wilcoxon(a, b, alternative='greater')
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        except:
            p = 1; sig = "?"
        print(f"    sin vs {label:<20s}: Δ={np.mean(diff):+.1%} p={p:.4f} {sig}")
    
    # ================================================================
    # TEST 2: Similarity sweep for all models
    # ================================================================
    print(f"\n{'=' * 80}")
    print("TEST 2: Similarity sweep (all five models)")
    print("-" * 80)
    
    print(f"\n  {'Sim':>5s}  {'sin':>7s}  {'linear':>7s}  {'cos':>7s}  {'tanh':>7s}  {'binary':>7s}")
    print(f"  {'-'*45}")
    
    for sim in [0.70, 0.80, 0.85, 0.90, 0.95]:
        sim_results = {k: [] for k in all_results}
        for seed in range(n_seeds):
            r = run_decomposition(N, n_fam, members, sim, n_unrel, noise, seed * 100)
            for k in sim_results:
                sim_results[k].append(r[k])
        
        print(f"  {sim:4.0%}  {np.mean(sim_results['sin']):6.1%}  "
              f"{np.mean(sim_results['linear']):6.1%}  "
              f"{np.mean(sim_results['cos']):6.1%}  "
              f"{np.mean(sim_results['tanh_circle']):6.1%}  "
              f"{np.mean(sim_results['binary']):6.1%}")
    
    elapsed = time.time() - t0
    
    # ================================================================
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print(f"{'=' * 80}")
    print(f"""
  POSSIBLE OUTCOMES:
  
  1. ONLY sin works (A >> B,C,D,E):
     The sinusoidal coupling function is specifically essential.
     Neither the circle nor other periodic functions replicate it.
     
  2. sin AND linear work, cos fails (A ≈ B >> C,D,E):
     Phase-DIFFERENCE on a circle matters.
     The specific function (sin vs linear) doesn't.
     Cosine (which measures alignment, not difference) doesn't help.
     
  3. All circle models work (A ≈ B ≈ C >> D,E):
     The circular topology is what matters.
     Any coupling function on S¹ preserves discrimination.
     
  4. All circle models + tanh-on-circle work (A ≈ B ≈ C ≈ D >> E):
     Living on the circle is sufficient regardless of coupling.
     The topology alone does the work.
     
  5. Everything except binary works (A ≈ B ≈ C ≈ D >> E):
     Continuity on ANY topology works after all.
     The earlier tanh comparison was misleading.
     
  Time: {elapsed:.0f}s
""")
    print("=" * 80)
