"""
Final Validation Experiments
==============================
ChatGPT's remaining requests:
  1. Mutual information: I(target; output) for all three systems
  2. Scaling: N = 80, 128, 192, 256
  3. Capacity load: discrimination at increasing α
  4. Discrimination vs retrieval: measure BOTH exact recall AND family discrimination
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

def retrieve_phase(J, cue_phases, rng, steps=150, eta=0.005):
    phi = cue_phases.copy()
    for _ in range(steps):
        sd = np.sin(phi[np.newaxis,:] - phi[:,np.newaxis])
        coupling = np.sum(J * sd, axis=1)
        phi = (phi + eta * coupling) % (2 * np.pi)
    return np.where(np.cos(phi) > 0, 0.0, np.pi)

def retrieve_cont_hopfield(W, cue, steps=150, dt=0.1, beta=2.0):
    x = cue.copy()
    for _ in range(steps):
        x = x + dt * (-x + np.tanh(beta * W @ x))
    return x

def retrieve_bin_hopfield(W, cue, steps=100):
    state = cue.copy()
    for _ in range(steps):
        s2 = np.sign(W @ state); s2[s2==0] = 1
        if np.array_equal(s2, state): break
        state = s2
    return state

def mutual_info_bits(target_indices, output_indices, n_classes):
    """Compute mutual information I(target; output) in bits."""
    n = len(target_indices)
    # Joint distribution
    joint = np.zeros((n_classes, n_classes))
    for t, o in zip(target_indices, output_indices):
        joint[t, o] += 1
    joint /= n + 1e-12
    
    # Marginals
    p_t = joint.sum(axis=1)
    p_o = joint.sum(axis=0)
    
    # MI = sum p(t,o) log2(p(t,o) / (p(t)*p(o)))
    mi = 0
    for t in range(n_classes):
        for o in range(n_classes):
            if joint[t, o] > 1e-12 and p_t[t] > 1e-12 and p_o[o] > 1e-12:
                mi += joint[t, o] * np.log2(joint[t, o] / (p_t[t] * p_o[o]))
    return mi

def run_full_test(N, n_families, members, similarity, n_unrelated, noise, seed):
    """Run all three systems, return discrimination, exact recall, and MI data."""
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
    
    results = {sys: {'discrim': [], 'exact': [], 'target_idx': [], 'output_idx': []}
               for sys in ['phase', 'cont', 'binary']}
    
    for f in range(n_families):
        for midx in family_members[f]:
            target = all_patterns[midx]
            member_within_family = family_members[f].index(midx)
            
            # Matched binary noise
            cue_binary = target.copy()
            cue_binary[rng.random(N) < noise] *= -1
            cue_phases = b2p(cue_binary) + rng.randn(N) * 0.15
            cue_cont = cue_binary.astype(float) + rng.randn(N) * 0.15
            
            def classify_and_measure(output_binary, sys_name):
                # Exact recall accuracy
                exact = np.mean(output_binary == target)
                results[sys_name]['exact'].append(exact)
                
                # Family discrimination
                best = -1; best_ov = -1
                for ji, j in enumerate(family_members[f]):
                    ov = np.mean(output_binary == all_patterns[j])
                    if ov > best_ov: best_ov = ov; best = ji
                correct = 1 if family_members[f][best] == midx else 0
                results[sys_name]['discrim'].append(correct)
                
                # For MI: target member index and output member index
                results[sys_name]['target_idx'].append(member_within_family)
                results[sys_name]['output_idx'].append(best)
            
            # Phase
            out_p = retrieve_phase(J, cue_phases, rng)
            classify_and_measure(p2b(out_p), 'phase')
            
            # Continuous Hopfield
            out_c = retrieve_cont_hopfield(W, cue_cont)
            out_c_bin = np.sign(out_c); out_c_bin[out_c_bin == 0] = 1
            classify_and_measure(out_c_bin, 'cont')
            
            # Binary Hopfield
            out_b = retrieve_bin_hopfield(W, cue_binary)
            classify_and_measure(out_b, 'binary')
    
    # Compute MI for each system
    mi = {}
    for sys in ['phase', 'cont', 'binary']:
        mi[sys] = mutual_info_bits(
            results[sys]['target_idx'], results[sys]['output_idx'], members)
    
    return {
        'phase_disc': np.mean(results['phase']['discrim']),
        'cont_disc': np.mean(results['cont']['discrim']),
        'bin_disc': np.mean(results['binary']['discrim']),
        'phase_exact': np.mean(results['phase']['exact']),
        'cont_exact': np.mean(results['cont']['exact']),
        'bin_exact': np.mean(results['binary']['exact']),
        'phase_mi': mi['phase'],
        'cont_mi': mi['cont'],
        'bin_mi': mi['binary'],
    }


if __name__ == "__main__":
    print("=" * 80)
    print("FINAL VALIDATION: MI, Scaling, Capacity, Discrimination vs Recall")
    print("=" * 80)
    
    t0 = time.time()
    n_seeds = 15
    noise = 0.25
    
    # ================================================================
    # TEST 1: Mutual Information Analysis
    # ================================================================
    print(f"\nTEST 1: Mutual Information — I(target member; identified member)")
    print(f"N=80, 3 families × 4 members, 85% similarity, {n_seeds} seeds")
    print("-" * 80)
    
    mi_phase = []; mi_cont = []; mi_bin = []
    disc_phase = []; disc_cont = []; disc_bin = []
    exact_phase = []; exact_cont = []; exact_bin = []
    
    for seed in range(n_seeds):
        r = run_full_test(80, 3, 4, 0.85, 5, noise, seed*100)
        mi_phase.append(r['phase_mi']); mi_cont.append(r['cont_mi']); mi_bin.append(r['bin_mi'])
        disc_phase.append(r['phase_disc']); disc_cont.append(r['cont_disc']); disc_bin.append(r['bin_disc'])
        exact_phase.append(r['phase_exact']); exact_cont.append(r['cont_exact']); exact_bin.append(r['bin_exact'])
    
    max_mi = np.log2(4)  # 2 bits for 4 family members
    
    print(f"\n  {'Metric':<30s}  {'Phase':>8s}  {'Cont Hop':>8s}  {'Bin Hop':>8s}")
    print(f"  {'-'*58}")
    print(f"  {'Discrimination accuracy':<30s}  {np.mean(disc_phase):7.1%}  {np.mean(disc_cont):7.1%}  {np.mean(disc_bin):7.1%}")
    print(f"  {'Exact recall accuracy':<30s}  {np.mean(exact_phase):7.1%}  {np.mean(exact_cont):7.1%}  {np.mean(exact_bin):7.1%}")
    mi_label = f"MI (bits, max={max_mi:.2f})"
    print(f"  {mi_label:<30s}  {np.mean(mi_phase):7.3f}  {np.mean(mi_cont):7.3f}  {np.mean(mi_bin):7.3f}")
    print(f"  {'MI / max MI':<30s}  {np.mean(mi_phase)/max_mi:7.1%}  {np.mean(mi_cont)/max_mi:7.1%}  {np.mean(mi_bin)/max_mi:7.1%}")
    
    # Statistical tests on MI
    print(f"\n  MI statistical tests:")
    for label, a, b in [
        ("Phase MI vs Binary MI", mi_phase, mi_bin),
        ("Phase MI vs Cont MI", mi_phase, mi_cont),
        ("Cont MI vs Binary MI", mi_cont, mi_bin),
    ]:
        diff = np.array(a) - np.array(b)
        try:
            stat, p = stats.wilcoxon(a, b, alternative='greater')
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        except:
            p = 1; sig = "?"
        print(f"    {label:<25s}: \u0394={np.mean(diff):+.3f} bits, p={p:.4f} {sig}")
    
    # ================================================================
    # TEST 2: Scaling
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"TEST 2: Scaling — does the effect survive larger networks?")
    print(f"3 families \u00d7 4 members, 85% similarity, {n_seeds} seeds")
    print("-" * 80)
    
    print(f"\n  {'N':>5s}  {'M':>3s}  {'\u03b1':>5s}  {'Ph disc':>7s}  {'Bn disc':>7s}  "
          f"{'Ph MI':>6s}  {'Bn MI':>6s}  {'Ph exact':>8s}  {'Bn exact':>8s}")
    print(f"  {'-'*68}")
    
    for N in [80, 128, 192, 256]:
        n_fam = 3; members = 4; n_unrel = max(3, N//20)
        total = n_fam * members + n_unrel
        
        pd = []; bd = []; pm = []; bm = []; pe = []; be = []
        for seed in range(n_seeds):
            r = run_full_test(N, n_fam, members, 0.85, n_unrel, noise, seed*100 + N)
            pd.append(r['phase_disc']); bd.append(r['bin_disc'])
            pm.append(r['phase_mi']); bm.append(r['bin_mi'])
            pe.append(r['phase_exact']); be.append(r['bin_exact'])
        
        print(f"  {N:5d}  {total:3d}  {total/N:.3f}  {np.mean(pd):6.1%}  {np.mean(bd):6.1%}  "
              f"{np.mean(pm):5.3f}  {np.mean(bm):5.3f}  {np.mean(pe):7.1%}  {np.mean(be):7.1%}")
    
    # ================================================================
    # TEST 3: Capacity load — discrimination under increasing α
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"TEST 3: Discrimination under increasing pattern load")
    print(f"N=80, 3 families \u00d7 4 members, 85% similarity, varying unrelated")
    print("-" * 80)
    
    print(f"\n  {'Total':>6s}  {'\u03b1':>5s}  {'Ph disc':>7s}  {'Bn disc':>7s}  "
          f"{'Ph MI':>6s}  {'Ph exact':>8s}  {'Bn exact':>8s}")
    print(f"  {'-'*55}")
    
    N = 80; n_fam = 3; members = 4
    for n_unrel in [0, 5, 10, 20, 30, 40]:
        total = n_fam * members + n_unrel
        pd = []; bd = []; pm = []; pe = []; be = []
        for seed in range(n_seeds):
            r = run_full_test(N, n_fam, members, 0.85, n_unrel, noise, seed*100 + n_unrel)
            pd.append(r['phase_disc']); bd.append(r['bin_disc'])
            pm.append(r['phase_mi'])
            pe.append(r['phase_exact']); be.append(r['bin_exact'])
        print(f"  {total:6d}  {total/N:.3f}  {np.mean(pd):6.1%}  {np.mean(bd):6.1%}  "
              f"{np.mean(pm):5.3f}  {np.mean(pe):7.1%}  {np.mean(be):7.1%}")
    
    # ================================================================
    # TEST 4: Discrimination vs exact recall — are they different?
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"TEST 4: Discrimination vs exact recall across similarity")
    print(f"N=80, {n_seeds} seeds")
    print("-" * 80)
    
    print(f"\n  {'Sim':>5s}  {'Ph disc':>7s}  {'Ph exact':>8s}  {'Bn disc':>7s}  {'Bn exact':>8s}  {'Ph MI':>6s}")
    print(f"  {'-'*48}")
    
    for sim in [0.70, 0.80, 0.85, 0.90, 0.95]:
        pd = []; pe = []; bd = []; be = []; pm = []
        for seed in range(n_seeds):
            r = run_full_test(80, 3, 4, sim, 5, noise, seed*100)
            pd.append(r['phase_disc']); pe.append(r['phase_exact'])
            bd.append(r['bin_disc']); be.append(r['bin_exact'])
            pm.append(r['phase_mi'])
        print(f"  {sim:4.0%}  {np.mean(pd):6.1%}  {np.mean(pe):7.1%}  {np.mean(bd):6.1%}  "
              f"{np.mean(be):7.1%}  {np.mean(pm):5.3f}")
    
    elapsed = time.time() - t0
    
    # ================================================================
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"""
  TEST 1 (Mutual Information):
    Phase system preserves significantly more member-specific information
    than both Hopfield baselines? Check MI values and p-values above.
    Maximum possible MI = {max_mi:.2f} bits (4 family members).
    
  TEST 2 (Scaling):
    Does the discrimination advantage hold at N=128, 192, 256?
    Does MI remain higher for phase coupling at larger N?
    
  TEST 3 (Capacity):
    Does discrimination survive when the network is loaded with
    many additional unrelated patterns (increasing \u03b1)?
    
  TEST 4 (Discrimination vs Recall):
    Are discrimination and exact recall measuring different things?
    If phase exact recall \u2248 Hopfield exact recall but phase discrimination
    >> Hopfield discrimination, then the phase system encodes structural
    information beyond what bit-level accuracy captures.
    
  Time: {elapsed:.0f}s
""")
    print("=" * 80)
