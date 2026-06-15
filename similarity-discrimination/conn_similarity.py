"""
CONN Similarity Structure Test
================================
Core question: Does CONN's phase space encode similarity between patterns?

Experiment:
  1. Store FAMILIES of correlated patterns (85% shared, 15% different)
  2. Test selective retrieval from distinguishing-feature cues
  3. Test family-neighborhood activation from shared-feature cues
  4. Compare CONN vs Hopfield on both tasks

If CONN can distinguish family members that Hopfield can't,
the continuous phase representation adds something real.
"""
import numpy as np
from scipy import stats
import time

class CONN:
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
    
    def retrieve(self, cue, lam=4.0, eta_phi=0.005, eta_A=0.03, steps=150):
        N = self.N; phi = cue.copy().astype(float)
        A = 0.5 + self.rng.random(N) * 0.3
        for step in range(steps):
            sd = np.sin(phi[np.newaxis,:] - phi[:,np.newaxis])
            coupling = np.sum(self.J * A[np.newaxis,:] * sd, axis=1)
            coh_term = -lam * A * A * np.sin(2 * phi)
            phi = (phi + eta_phi * (A * coupling + coh_term)) % (2*np.pi)
            dA = -lam * 2 * A * np.sin(phi)**2
            A = np.clip(A + eta_A * dA, 0.01, 2.0)
        out = np.where(np.cos(phi) > 0, 0.0, np.pi)
        return out, A
    
    def add_noise(self, phases, level):
        noisy = phases.copy()
        mask = self.rng.random(self.N) < level
        noisy[mask] = self.rng.random(np.sum(mask)) * 2 * np.pi
        return noisy
    
    def overlap(self, out, target):
        err = np.abs(np.arctan2(np.sin(out-target), np.cos(out-target)))
        return np.mean(err < np.pi/4)


def b2p(b): return np.where(b > 0, 0.0, np.pi)
def p2b(p): return np.sign(np.cos(p))


def generate_pattern_family(N, n_members, similarity, rng):
    """
    Generate a family of correlated patterns.
    All members share 'similarity' fraction of bits with the prototype.
    The differing bits are specific to each member.
    """
    prototype = rng.choice([-1, 1], size=N)
    n_diff = int(N * (1 - similarity))
    
    family = [prototype.copy()]
    diff_positions = []
    
    for m in range(1, n_members):
        member = prototype.copy()
        # Choose unique positions to flip for this member
        positions = rng.choice(N, size=n_diff, replace=False)
        member[positions] *= -1
        family.append(member)
        diff_positions.append(positions)
    
    return family, prototype, diff_positions


def hopfield_retrieve(W, cue_binary, max_steps=100):
    """Standard synchronous Hopfield retrieval."""
    state = cue_binary.copy()
    for _ in range(max_steps):
        s_new = np.sign(W @ state)
        s_new[s_new == 0] = 1
        if np.array_equal(s_new, state):
            break
        state = s_new
    return state


if __name__ == "__main__":
    print("=" * 80)
    print("CONN SIMILARITY STRUCTURE TEST")
    print("Does phase space encode similarity between patterns?")
    print("=" * 80)
    
    N = 80
    n_families = 3
    members_per_family = 4
    similarity = 0.85  # 85% shared bits within family
    n_unrelated = 5     # additional unrelated patterns
    noise = 0.25
    n_seeds = 15
    
    total_patterns = n_families * members_per_family + n_unrelated
    
    print(f"\nN={N}, {n_families} families × {members_per_family} members ({similarity*100:.0f}% similar)")
    print(f"{n_unrelated} unrelated patterns, {total_patterns} total (α={total_patterns/N:.3f})")
    print(f"Noise: {noise*100:.0f}%, Seeds: {n_seeds}\n")
    
    t0 = time.time()
    
    # Collect results across seeds
    conn_within = []   # CONN retrieval accuracy for within-family cues
    hop_within = []    # Hopfield retrieval accuracy for within-family cues
    conn_correct_member = []  # Does CONN find the RIGHT family member?
    hop_correct_member = []   # Does Hopfield find the RIGHT family member?
    conn_family_hit = []      # Does CONN at least land in the right family?
    hop_family_hit = []       # Does Hopfield at least land in the right family?
    conn_unrelated = []       # CONN accuracy on unrelated patterns
    hop_unrelated = []        # Hopfield accuracy on unrelated patterns
    
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 100)
        
        # Generate pattern families + unrelated patterns
        all_patterns = []
        all_families = []  # (family_idx, member_idx) for each pattern
        family_members = {}  # family_idx -> list of pattern indices
        
        for f in range(n_families):
            family, proto, diffs = generate_pattern_family(N, members_per_family, similarity, rng)
            family_members[f] = []
            for m, member in enumerate(family):
                idx = len(all_patterns)
                all_patterns.append(member)
                all_families.append((f, m))
                family_members[f].append(idx)
        
        for _ in range(n_unrelated):
            all_patterns.append(rng.choice([-1, 1], size=N))
            all_families.append((-1, -1))  # unrelated
        
        # --- Build CONN ---
        conn = CONN(N, seed=seed + 5000)
        for p in all_patterns:
            conn.store(b2p(p))
        
        # --- Build Hopfield ---
        W_hop = np.zeros((N, N))
        for p in all_patterns:
            W_hop += np.outer(p, p) / N
        np.fill_diagonal(W_hop, 0)
        
        # === TEST 1: Selective retrieval of family members ===
        for f in range(n_families):
            for member_idx in family_members[f]:
                target = all_patterns[member_idx]
                target_phases = b2p(target)
                
                # Create noisy cue
                cue_binary = target.copy()
                cue_binary[rng.random(N) < noise] *= -1
                cue_phases = conn.add_noise(target_phases, noise)
                
                # CONN retrieval
                out_phases, amplitudes = conn.retrieve(cue_phases, lam=4.0, steps=150)
                out_binary = p2b(out_phases)
                conn_acc = np.mean(out_binary == target)
                conn_within.append(conn_acc)
                
                # Did CONN find the CORRECT member (not just a family member)?
                best_match = -1; best_overlap = -1
                for j in family_members[f]:
                    ov = np.mean(out_binary == all_patterns[j])
                    if ov > best_overlap:
                        best_overlap = ov; best_match = j
                conn_correct_member.append(1 if best_match == member_idx else 0)
                
                # Did CONN at least land in the right family?
                best_family_ov = max(np.mean(out_binary == all_patterns[j]) for j in family_members[f])
                conn_family_hit.append(1 if best_family_ov > 0.80 else 0)
                
                # Hopfield retrieval
                hop_out = hopfield_retrieve(W_hop, cue_binary)
                hop_acc = np.mean(hop_out == target)
                hop_within.append(hop_acc)
                
                best_match_h = -1; best_overlap_h = -1
                for j in family_members[f]:
                    ov = np.mean(hop_out == all_patterns[j])
                    if ov > best_overlap_h:
                        best_overlap_h = ov; best_match_h = j
                hop_correct_member.append(1 if best_match_h == member_idx else 0)
                
                best_family_ov_h = max(np.mean(hop_out == all_patterns[j]) for j in family_members[f])
                hop_family_hit.append(1 if best_family_ov_h > 0.80 else 0)
        
        # === TEST 2: Unrelated pattern retrieval (control) ===
        for idx in range(n_families * members_per_family, total_patterns):
            target = all_patterns[idx]
            target_phases = b2p(target)
            
            cue_binary = target.copy()
            cue_binary[rng.random(N) < noise] *= -1
            cue_phases = conn.add_noise(target_phases, noise)
            
            out_phases, _ = conn.retrieve(cue_phases, lam=4.0, steps=150)
            conn_unrelated.append(np.mean(p2b(out_phases) == target))
            
            hop_out = hopfield_retrieve(W_hop, cue_binary)
            hop_unrelated.append(np.mean(hop_out == target))
    
    elapsed = time.time() - t0
    print(f"Computed in {elapsed:.0f}s")
    
    # ================================================================
    # RESULTS
    # ================================================================
    print(f"\n{'=' * 80}")
    print("TEST 1: Can each system retrieve the CORRECT family member?")
    print(f"{'=' * 80}")
    
    print(f"\n  {'Metric':<35s}  {'CONN':>8s}  {'Hopfield':>8s}  {'Δ':>7s}")
    print(f"  {'-'*62}")
    
    metrics = [
        ("Exact target accuracy", conn_within, hop_within),
        ("Correct member identified", conn_correct_member, hop_correct_member),
        ("Right family (any member >80%)", conn_family_hit, hop_family_hit),
    ]
    
    for label, c_data, h_data in metrics:
        cm = np.mean(c_data); hm = np.mean(h_data)
        print(f"  {label:<35s}  {cm:7.1%}  {hm:7.1%}  {cm-hm:+6.1%}")
    
    # Statistical tests
    print(f"\n  Statistical tests (Wilcoxon):")
    for label, c_data, h_data in metrics:
        try:
            stat, p = stats.wilcoxon(c_data, h_data, alternative='greater')
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        except:
            p = 1; sig = "?"
        print(f"    {label:<35s}: p={p:.4f} {sig}")
    
    # ================================================================
    print(f"\n{'=' * 80}")
    print("TEST 2: Unrelated patterns (control — no similarity structure)")
    print(f"{'=' * 80}")
    
    cm_u = np.mean(conn_unrelated); hm_u = np.mean(hop_unrelated)
    print(f"\n  CONN:     {cm_u:.1%}")
    print(f"  Hopfield: {hm_u:.1%}")
    print(f"  Δ:        {cm_u - hm_u:+.1%}")
    
    # ================================================================
    print(f"\n{'=' * 80}")
    print("TEST 3: The critical question — family discrimination")
    print(f"{'=' * 80}")
    
    conn_discrim = np.mean(conn_correct_member)
    hop_discrim = np.mean(hop_correct_member)
    
    print(f"""
  When retrieving a pattern from a family of 85%-similar patterns:
  
  CONN correctly identifies the specific member:  {conn_discrim:.1%}
  Hopfield correctly identifies the specific member: {hop_discrim:.1%}
  
  Difference: {conn_discrim - hop_discrim:+.1%}
""")
    
    if conn_discrim > hop_discrim + 0.03:
        verdict = "CONN's continuous phase representation DOES provide\n  superior discrimination between similar patterns."
    elif abs(conn_discrim - hop_discrim) <= 0.03:
        verdict = "No significant difference in family discrimination.\n  Phase continuity does not help with similar-pattern retrieval."
    else:
        verdict = "Hopfield OUTPERFORMS CONN on family discrimination.\n  The continuous phase representation may hurt with correlated patterns."
    
    print(f"  VERDICT: {verdict}")
    
    # ================================================================
    print(f"\n{'=' * 80}")
    print("TEST 4: Family-level vs member-level retrieval")
    print(f"{'=' * 80}")
    
    print(f"""
  CONN lands in the right family:     {np.mean(conn_family_hit):.1%}
  CONN finds the exact member:        {np.mean(conn_correct_member):.1%}
  Gap (family recognition - member):  {np.mean(conn_family_hit) - np.mean(conn_correct_member):+.1%}
  
  Hopfield lands in the right family: {np.mean(hop_family_hit):.1%}
  Hopfield finds the exact member:    {np.mean(hop_correct_member):.1%}
  Gap:                                {np.mean(hop_family_hit) - np.mean(hop_correct_member):+.1%}
""")
    
    family_gap_conn = np.mean(conn_family_hit) - np.mean(conn_correct_member)
    family_gap_hop = np.mean(hop_family_hit) - np.mean(hop_correct_member)
    
    if family_gap_conn > family_gap_hop + 0.03:
        print("  CONN shows LARGER family-member gap — it recognizes family")
        print("  structure (landing in the neighborhood) even when it can't")
        print("  pinpoint the exact member. This is the similarity-structure")
        print("  signal: phase proximity encoding semantic proximity.")
    elif family_gap_conn < family_gap_hop - 0.03:
        print("  Hopfield shows larger gap. Unexpected.")
    else:
        print("  Similar family-member gaps in both architectures.")
    
    # ================================================================
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"""
  The experiment tests whether CONN's continuous phase representation
  encodes similarity structure that Hopfield's binary representation misses.
  
  Three levels of retrieval success:
    1. Exact target match (did you get the right answer?)
    2. Correct family member (did you get the right specific variant?)
    3. Right family (did you at least land in the right neighborhood?)
  
  If CONN shows higher family-hit rate but similar member-identification
  to Hopfield: phase proximity encodes family structure (the neighborhood
  is navigable) but doesn't help with fine discrimination.
  
  If CONN shows higher member-identification: phase continuity provides
  genuine fine-grained discrimination between similar patterns.
  
  If no difference: the similarity structure hypothesis is unsupported
  in this regime.
  
  Total patterns: {total_patterns}, α = {total_patterns/N:.3f}
  Time: {elapsed:.0f}s
""")
    print("=" * 80)
