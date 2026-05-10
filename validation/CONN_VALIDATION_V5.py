"""
================================================================================
CONN VALIDATION SUITE v5.0 — CORRECTED IMPLEMENTATION
================================================================================

Coherence-Oscillator Neural Network (CONN) Validation
Author: Labinot Marku
Corrected: May 2026

WHAT THIS VERSION FIXES:
  1. Noise model: bit-flip + Gaussian jitter (continuous off-manifold init)
  2. Hopfield baseline: real discrete sign(W·s) updates, not CONN with λ=0
  3. Parameters: η_φ=0.02, λ=0.5 (annealed), β=1.0, 400 steps
  4. Protocol: 50 trials per M, test multiple patterns, linear sweep
  5. Deterministic seeding throughout

WHY JITTER IS NEEDED:
  Pure bit-flip noise produces phases exactly at {0, π}. Since sin(0)=sin(π)=0,
  all CONN dynamics evaluate to zero — the network does nothing. Adding small
  Gaussian jitter moves phases off the fixed points, activating the continuous
  dynamics. Hopfield's sign() operation is immune to small jitter, so this is
  fair to both models.

USAGE:
  python CONN_VALIDATION_V5.py              # Full validation (~5 min)
  python CONN_VALIDATION_V5.py --quick      # Quick check (~1 min)
  python CONN_VALIDATION_V5.py --experiment capacity

DOI: [to be updated]
================================================================================
"""

import numpy as np
import csv
import time
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Corrected experimental parameters"""
    
    # CONN dynamics (CORRECTED — validated optimal)
    LAMBDA = 0.5             # Coherence strength (annealed from 0)
    ETA_PHI = 0.02           # Phase learning rate
    ETA_A = 0.03             # Amplitude learning rate
    BETA = 1.0               # Amplitude regularization
    MAX_STEPS = 400          # Integration steps
    ANNEAL = True            # Anneal λ from 0 → LAMBDA
    
    # Noise protocol
    NOISE_LEVEL = 0.30       # 30% bit-flip rate
    JITTER_STD = 0.5         # Gaussian jitter std (radians)
    
    # Experiment 1: Capacity
    N_VALUES = [32, 64, 128]
    CAPACITY_TRIALS = 50     # Trials per M value
    PATTERNS_PER_TRIAL = 3   # Patterns tested per trial
    TARGET_RECALL = 80.0     # Percent threshold
    
    # Experiment 2: Noise robustness
    NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    NOISE_M = 6
    NOISE_TRIALS = 40
    
    # Experiment 3: Ablation
    ABLATION_N = 64
    ABLATION_M = 10
    ABLATION_TRIALS = 40
    
    # Hopfield baseline
    HOPFIELD_STEPS = 200     # Sufficient for convergence
    
    # Output
    OUTPUT_DIR = Path("conn_validation_v5")


# ============================================================================
# PATTERN GENERATION & NOISE
# ============================================================================

def generate_patterns(M, N, seed):
    """Generate M random binary phase patterns {0, π}"""
    rng = np.random.RandomState(seed)
    return rng.choice([0.0, np.pi], size=(M, N))


def apply_noise(pattern, noise_level, jitter_std, rng):
    """
    Bit-flip + Gaussian jitter noise.
    
    Step 1: Flip each bit with probability noise_level (same as Hopfield protocol)
    Step 2: Add Gaussian jitter to ALL phases (activates CONN dynamics)
    
    Both CONN and Hopfield receive identical corrupted input.
    Hopfield's sign() is immune to small jitter, so this is fair.
    """
    noisy = pattern.copy()
    # Bit flips
    mask = rng.random(len(pattern)) < noise_level
    noisy[mask] = (noisy[mask] + np.pi) % (2 * np.pi)
    # Continuous jitter
    noisy = noisy + rng.normal(0, jitter_std, len(pattern))
    return np.mod(noisy, 2 * np.pi)


# ============================================================================
# HEBBIAN WEIGHTS (shared by CONN and Hopfield)
# ============================================================================

def hebbian_weights(patterns):
    """Compute Hebbian weight matrix J from {0,π} patterns"""
    M, N = patterns.shape
    xi = np.cos(patterns)  # Convert {0,π} → {+1,−1}
    J = (xi.T @ xi) / N
    np.fill_diagonal(J, 0)
    return J


# ============================================================================
# CONN RECALL
# ============================================================================

def conn_recall(J, noisy_phases, config=None):
    """
    CONN recall with annealed coherence.
    
    Dynamics:
      dφ_j/dt = A_j * Σ_i J_ji sin(φ_i - φ_j) - 2λ(t) A_j² sin(φ_j)cos(φ_j)
      dA_j/dt = -2λ(t) A_j sin²(φ_j) - 2β(A_j - 1)
    
    where λ(t) = λ_final × (t / T) ramps linearly from 0.
    """
    if config is None:
        config = Config()
    
    N = len(noisy_phases)
    phi = noisy_phases.copy()
    A = np.ones(N)
    
    lam = config.LAMBDA
    eta_phi = config.ETA_PHI
    eta_A = config.ETA_A
    beta = config.BETA
    steps = config.MAX_STEPS
    anneal = config.ANNEAL
    
    for step in range(steps):
        # Annealed coherence: ramp λ from 0
        lam_t = lam * (step / steps) if anneal else lam
        
        s = np.sin(phi)
        c = np.cos(phi)
        
        # Coupling: Σ_i J_ji A_i sin(φ_i - φ_j) via sin addition formula
        coupling = (J @ (A * s)) * c - (J @ (A * c)) * s
        
        # Coherence: -2λ A² sin(φ)cos(φ) = -λ A² sin(2φ)
        coherence = -2 * lam_t * A**2 * s * c
        
        # Phase update
        dphi = A * coupling + coherence
        phi = (phi + eta_phi * dphi) % (2 * np.pi)
        
        # Amplitude update
        dA = -2 * lam_t * A * s**2 - 2 * beta * (A - 1)
        A = np.clip(A + eta_A * dA, 0.01, 2.0)
    
    return phi


# ============================================================================
# HOPFIELD RECALL (true discrete baseline)
# ============================================================================

def hopfield_recall(J, noisy_phases, max_steps=200):
    """
    Standard Hopfield recall with synchronous sign updates.
    
    Converts phase input to ±1, runs sign(W·s), converts back.
    The sign() operation is immune to small phase jitter.
    """
    s = np.cos(noisy_phases).copy()  # {0,π} + jitter → approximately ±1
    
    for step in range(max_steps):
        s_new = np.sign(J @ s)
        s_new[s_new == 0] = 1  # Break ties
        if np.array_equal(s_new, s):
            break
        s = s_new
    
    return np.where(s > 0, 0.0, np.pi)


# ============================================================================
# OVERLAP METRIC
# ============================================================================

def compute_overlap(retrieved_phases, target_phases):
    """
    Quantize retrieved phases to nearest {0, π} and compute fraction correct.
    Both CONN and Hopfield are judged by the same binary criterion.
    """
    retrieved_binary = np.where(np.cos(retrieved_phases) >= 0, 0.0, np.pi)
    target_binary = np.where(np.cos(target_phases) >= 0, 0.0, np.pi)
    correct = np.sum(np.abs(np.cos(retrieved_binary) - np.cos(target_binary)) < 0.5)
    return (correct / len(target_phases)) * 100


# ============================================================================
# EXPERIMENT 1: CAPACITY (LINEAR SWEEP)
# ============================================================================

def experiment_capacity(N, config=None, verbose=True):
    """
    Measure capacity via linear sweep.
    Protocol: for each M, run trials with deterministic seeds,
    test multiple patterns per trial. Capacity = largest M with ≥80% mean recall.
    """
    if config is None:
        config = Config()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  CAPACITY: N={N}")
        print(f"{'='*70}")
    
    max_M = int(0.45 * N)
    hop_capacity = 0
    conn_capacity = 0
    
    results_table = []
    
    for M in range(1, max_M + 1):
        hop_overlaps = []
        conn_overlaps = []
        
        for trial in range(config.CAPACITY_TRIALS):
            patterns = generate_patterns(M, N, seed=trial)
            J = hebbian_weights(patterns)
            
            n_test = min(M, config.PATTERNS_PER_TRIAL)
            for mu in range(n_test):
                rng = np.random.RandomState(trial * 10000 + mu)
                noisy = apply_noise(patterns[mu], config.NOISE_LEVEL, 
                                    config.JITTER_STD, rng)
                
                # CONN
                ret_c = conn_recall(J, noisy, config)
                conn_overlaps.append(compute_overlap(ret_c, patterns[mu]))
                
                # Hopfield
                ret_h = hopfield_recall(J, noisy, config.HOPFIELD_STEPS)
                hop_overlaps.append(compute_overlap(ret_h, patterns[mu]))
        
        h_mean = np.mean(hop_overlaps)
        c_mean = np.mean(conn_overlaps)
        h_std = np.std(hop_overlaps)
        c_std = np.std(conn_overlaps)
        delta = c_mean - h_mean
        
        if h_mean >= config.TARGET_RECALL:
            hop_capacity = M
        if c_mean >= config.TARGET_RECALL:
            conn_capacity = M
        
        results_table.append({
            'N': N, 'M': M, 'alpha': M/N,
            'hopfield_mean': h_mean, 'hopfield_std': h_std,
            'conn_mean': c_mean, 'conn_std': c_std,
            'delta': delta
        })
        
        if verbose:
            # Print key rows
            show = (M <= 3 or abs(h_mean - 80) < 8 or abs(c_mean - 80) < 5
                    or M % max(1, N//16) == 0)
            if show:
                print(f"  M={M:3d}: Hop={h_mean:5.1f}±{h_std:4.1f}%  "
                      f"CONN={c_mean:5.1f}±{c_std:4.1f}%  Δ={delta:+5.1f}%")
        
        # Early stop
        if h_mean < 55 and c_mean < 55:
            break
    
    ratio = conn_capacity / hop_capacity if hop_capacity > 0 else 0
    
    if verbose:
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │ Hopfield:  M={hop_capacity:3d}  α={hop_capacity/N:.3f}             │")
        print(f"  │ CONN:      M={conn_capacity:3d}  α={conn_capacity/N:.3f}             │")
        print(f"  │ Ratio:     {ratio:.2f}×                        │")
        print(f"  └─────────────────────────────────────────┘")
    
    return {
        'N': N,
        'hop_capacity': hop_capacity, 'hop_alpha': hop_capacity/N,
        'conn_capacity': conn_capacity, 'conn_alpha': conn_capacity/N,
        'ratio': ratio,
        'detail': results_table
    }


# ============================================================================
# EXPERIMENT 2: NOISE ROBUSTNESS
# ============================================================================

def experiment_noise_robustness(config=None, verbose=True):
    """Test recall accuracy across noise levels"""
    if config is None:
        config = Config()
    
    N = config.ABLATION_N
    M = config.NOISE_M
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  NOISE ROBUSTNESS: N={N}, M={M}")
        print(f"{'='*70}")
        print(f"\n  {'Noise':>6s}  {'CONN':>8s}  {'±':>5s}  {'Hopfield':>8s}  {'±':>5s}  {'Δ':>7s}")
        print(f"  {'-'*50}")
    
    results = []
    
    for noise_level in config.NOISE_LEVELS:
        conn_ov = []
        hop_ov = []
        
        for trial in range(config.NOISE_TRIALS):
            patterns = generate_patterns(M, N, seed=trial)
            J = hebbian_weights(patterns)
            
            for mu in range(min(M, 3)):
                rng = np.random.RandomState(trial * 10000 + mu)
                noisy = apply_noise(patterns[mu], noise_level,
                                    config.JITTER_STD, rng)
                
                ret_c = conn_recall(J, noisy, config)
                conn_ov.append(compute_overlap(ret_c, patterns[mu]))
                
                ret_h = hopfield_recall(J, noisy, config.HOPFIELD_STEPS)
                hop_ov.append(compute_overlap(ret_h, patterns[mu]))
        
        c_mean, c_std = np.mean(conn_ov), np.std(conn_ov)
        h_mean, h_std = np.mean(hop_ov), np.std(hop_ov)
        delta = c_mean - h_mean
        
        results.append({
            'noise': noise_level,
            'conn_mean': c_mean, 'conn_std': c_std,
            'hop_mean': h_mean, 'hop_std': h_std,
            'delta': delta
        })
        
        if verbose:
            print(f"  {noise_level:5.0%}  {c_mean:7.1f}% {c_std:4.1f}%  "
                  f"{h_mean:7.1f}% {h_std:4.1f}%  {delta:+6.1f}%")
    
    return results


# ============================================================================
# EXPERIMENT 3: ABLATION STUDY
# ============================================================================

def experiment_ablation(config=None, verbose=True):
    """Test contribution of coherence and amplitude"""
    if config is None:
        config = Config()
    
    N = config.ABLATION_N
    M = config.ABLATION_M
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  ABLATION: N={N}, M={M}")
        print(f"{'='*70}")
    
    # Define configurations
    class ConnConfig:
        def __init__(self, lam, anneal, eta_A, eta_phi, beta, steps):
            self.LAMBDA = lam
            self.ANNEAL = anneal
            self.ETA_PHI = eta_phi
            self.ETA_A = eta_A
            self.BETA = beta
            self.MAX_STEPS = steps
    
    conditions = [
        ("Full CONN (annealed λ=0.5)", 
         ConnConfig(0.5, True, 0.03, 0.02, 1.0, 400)),
        ("No coherence (λ=0)", 
         ConnConfig(0.0, False, 0.03, 0.02, 1.0, 400)),
        ("No amplitude dynamics", 
         ConnConfig(0.5, True, 0.0, 0.02, 1.0, 400)),
        ("Bare coupling (λ=0, no amp)", 
         ConnConfig(0.0, False, 0.0, 0.02, 1.0, 400)),
    ]
    
    results = []
    
    for name, cfg in conditions:
        overlaps = []
        for trial in range(config.ABLATION_TRIALS):
            patterns = generate_patterns(M, N, seed=trial)
            J = hebbian_weights(patterns)
            
            for mu in range(min(M, 3)):
                rng = np.random.RandomState(trial * 10000 + mu)
                noisy = apply_noise(patterns[mu], config.NOISE_LEVEL,
                                    config.JITTER_STD, rng)
                
                ret = conn_recall(J, noisy, cfg)
                overlaps.append(compute_overlap(ret, patterns[mu]))
        
        mean_ov = np.mean(overlaps)
        std_ov = np.std(overlaps)
        results.append({'name': name, 'mean': mean_ov, 'std': std_ov})
        
        if verbose:
            print(f"  {name:35s}: {mean_ov:5.1f}% ± {std_ov:4.1f}%")
    
    # Add Hopfield baseline
    hop_overlaps = []
    for trial in range(config.ABLATION_TRIALS):
        patterns = generate_patterns(M, N, seed=trial)
        J = hebbian_weights(patterns)
        for mu in range(min(M, 3)):
            rng = np.random.RandomState(trial * 10000 + mu)
            noisy = apply_noise(patterns[mu], config.NOISE_LEVEL,
                                config.JITTER_STD, rng)
            ret = hopfield_recall(J, noisy, config.HOPFIELD_STEPS)
            hop_overlaps.append(compute_overlap(ret, patterns[mu]))
    
    hop_mean = np.mean(hop_overlaps)
    hop_std = np.std(hop_overlaps)
    results.append({'name': 'Hopfield (discrete)', 'mean': hop_mean, 'std': hop_std})
    
    if verbose:
        print(f"  {'Hopfield (discrete)':35s}: {hop_mean:5.1f}% ± {hop_std:4.1f}%")
        
        # Component contributions
        full = results[0]['mean']
        no_coh = results[1]['mean']
        no_amp = results[2]['mean']
        print(f"\n  Component contributions:")
        print(f"    Coherence (λ):    {full - no_coh:+5.1f}%")
        print(f"    Amplitude:        {full - no_amp:+5.1f}%")
        print(f"    vs Hopfield:      {full - hop_mean:+5.1f}%")
    
    return results


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(capacity_results, noise_results, ablation_results, output_dir):
    """Save all results to CSV files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Capacity summary
    with open(output_dir / "capacity_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'Hopfield_M', 'Hopfield_alpha', 
                         'CONN_M', 'CONN_alpha', 'Ratio'])
        for r in capacity_results:
            writer.writerow([r['N'], r['hop_capacity'], f"{r['hop_alpha']:.3f}",
                            r['conn_capacity'], f"{r['conn_alpha']:.3f}",
                            f"{r['ratio']:.2f}"])
    
    # Capacity detail
    with open(output_dir / "capacity_detail.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'M', 'alpha', 'hopfield_mean', 'hopfield_std',
                         'conn_mean', 'conn_std', 'delta'])
        for r in capacity_results:
            for d in r['detail']:
                writer.writerow([d['N'], d['M'], f"{d['alpha']:.3f}",
                                f"{d['hopfield_mean']:.1f}", f"{d['hopfield_std']:.1f}",
                                f"{d['conn_mean']:.1f}", f"{d['conn_std']:.1f}",
                                f"{d['delta']:.1f}"])
    
    # Noise robustness
    with open(output_dir / "noise_robustness.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['noise_level', 'conn_mean', 'conn_std',
                         'hop_mean', 'hop_std', 'delta'])
        for r in noise_results:
            writer.writerow([f"{r['noise']:.2f}", f"{r['conn_mean']:.1f}",
                            f"{r['conn_std']:.1f}", f"{r['hop_mean']:.1f}",
                            f"{r['hop_std']:.1f}", f"{r['delta']:.1f}"])
    
    # Ablation
    with open(output_dir / "ablation.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'mean_recall', 'std_recall'])
        for r in ablation_results:
            writer.writerow([r['name'], f"{r['mean']:.1f}", f"{r['std']:.1f}"])
    
    # Parameters used
    with open(output_dir / "parameters.txt", 'w') as f:
        f.write("CONN Validation v5.0 — Corrected Parameters\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"λ (coherence):     {Config.LAMBDA} (annealed from 0)\n")
        f.write(f"η_φ (phase LR):    {Config.ETA_PHI}\n")
        f.write(f"η_A (amp LR):      {Config.ETA_A}\n")
        f.write(f"β (amp reg):       {Config.BETA}\n")
        f.write(f"Integration steps: {Config.MAX_STEPS}\n")
        f.write(f"Noise level:       {Config.NOISE_LEVEL}\n")
        f.write(f"Jitter std:        {Config.JITTER_STD}\n")
        f.write(f"Capacity trials:   {Config.CAPACITY_TRIALS}\n")
        f.write(f"Hopfield steps:    {Config.HOPFIELD_STEPS}\n")


# ============================================================================
# MAIN
# ============================================================================

def run_full_validation(quick=False):
    """Run complete validation suite"""
    config = Config()
    
    if quick:
        config.N_VALUES = [32, 64]
        config.CAPACITY_TRIALS = 20
        config.PATTERNS_PER_TRIAL = 2
        config.NOISE_TRIALS = 20
        config.ABLATION_TRIALS = 20
    
    print("\n" + "=" * 70)
    print("  CONN VALIDATION SUITE v5.0 — CORRECTED")
    print("=" * 70)
    print(f"  Parameters: λ={config.LAMBDA} (annealed), η_φ={config.ETA_PHI}, "
          f"steps={config.MAX_STEPS}")
    print(f"  Noise: {config.NOISE_LEVEL:.0%} bit-flip + σ={config.JITTER_STD} jitter")
    print(f"  Mode: {'QUICK' if quick else 'FULL'}")
    print("=" * 70)
    
    start = time.time()
    
    # Experiment 1: Capacity
    capacity_results = []
    for N in config.N_VALUES:
        result = experiment_capacity(N, config)
        capacity_results.append(result)
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"  CAPACITY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'N':>4s}  {'Hop M':>6s}  {'Hop α':>7s}  {'CONN M':>7s}  {'CONN α':>7s}  {'Ratio':>6s}")
    print(f"  {'-'*45}")
    for r in capacity_results:
        print(f"  {r['N']:4d}  {r['hop_capacity']:6d}  {r['hop_alpha']:7.3f}  "
              f"{r['conn_capacity']:7d}  {r['conn_alpha']:7.3f}  {r['ratio']:5.2f}×")
    
    avg_ratio = np.mean([r['ratio'] for r in capacity_results if r['ratio'] > 0])
    print(f"\n  Average improvement: {avg_ratio:.2f}×")
    
    # Experiment 2: Noise robustness
    noise_results = experiment_noise_robustness(config)
    
    # Experiment 3: Ablation
    ablation_results = experiment_ablation(config)
    
    # Save
    save_results(capacity_results, noise_results, ablation_results, 
                 config.OUTPUT_DIR)
    
    elapsed = time.time() - start
    
    print(f"\n{'='*70}")
    print(f"  VALIDATION COMPLETE ({elapsed:.1f}s)")
    print(f"  Results saved to: {config.OUTPUT_DIR}/")
    print(f"{'='*70}\n")
    
    return capacity_results, noise_results, ablation_results


if __name__ == "__main__":
    quick = '--quick' in sys.argv
    
    if '--experiment' in sys.argv:
        idx = sys.argv.index('--experiment')
        exp = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'all'
        config = Config()
        if exp == 'capacity':
            N = 64
            experiment_capacity(N, config)
        elif exp == 'noise':
            experiment_noise_robustness(config)
        elif exp == 'ablation':
            experiment_ablation(config)
        else:
            run_full_validation(quick)
    else:
        run_full_validation(quick)
