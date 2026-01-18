"""
================================================================================
CONN (Coherence Oscillatory Neural Network) - CORRECTED IMPLEMENTATION
================================================================================

⚠️ IMPORTANT: EXPLORATORY VALIDATION CODE ⚠️

This code provides exploratory validation of the CONN mechanism.
Results at small N (e.g., N=32) in quick mode are UPPER BOUNDS and should
NOT be interpreted as rigorous capacity measurements.

Published paper reports CONSERVATIVE estimates (~1.38× improvement) based
on rigorous validation. Quick-mode results may show higher numbers (2-4×)
due to:
  - Small network size effects
  - Optimistic binary search thresholds  
  - Limited trial counts
  - Non-asymptotic regime

Use this code for:
  ✓ Mechanism exploration
  ✓ Implementation verification
  ✓ Educational purposes

Do NOT use for:
  ✗ Benchmark claims
  ✗ Capacity performance claims
  ✗ Comparisons without matched baselines

For validated results, see published manuscript (DOI: 10.13140/RG.2.2.25288.99841)

VERSION HISTORY:
v1.0 (Jan 11, 2026): Initial implementation  
v2.0 (Jan 18, 2026): CORRECTED - Fixed coherence gradient sign
                     Added exploratory vs validated distinction

CRITICAL FIX: Changed coherence term from +2λ to -2λ (gradient descent)

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Experimental parameters matching the manuscript"""
    
    # Network sizes
    N_VALUES = [32, 64, 128]
    
    # Experiment 1: Capacity
    CAPACITY_TRIALS = 10
    CAPACITY_NOISE = 0.1
    
    # Experiment 2: Noise Robustness  
    NOISE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    NOISE_M = 8
    NOISE_TRIALS = 5
    
    # Experiment 3: Ablation
    ABLATION_N = 32
    ABLATION_M = 8
    ABLATION_TRIALS = 20
    
    # CONN dynamics parameters (CORRECTED)
    LAMBDA = 4.0           # Coherence strength
    ETA_PHI = 0.005        # Phase learning rate
    ETA_A = 0.03           # Amplitude learning rate
    MAX_STEPS = 150        # Integration steps
    
    # Output
    OUTPUT_DIR = Path("conn_validation_corrected")
    VERBOSE = True

# ============================================================================
# CORE CONN IMPLEMENTATION (CORRECTED)
# ============================================================================

class CONN:
    """
    Coherence Oscillatory Neural Network
    
    Implements phase-amplitude dynamics with topological coherence constraint.
    This is the CORRECTED version with proper gradient descent.
    """
    
    def __init__(self, N, lambda_coh=4.0, eta_phi=0.005, eta_A=0.03, 
                 enable_amplitude=True):
        """
        Initialize CONN network
        
        Args:
            N: Number of neurons
            lambda_coh: Coherence strength (default: 4.0)
            eta_phi: Phase learning rate (default: 0.005)
            eta_A: Amplitude learning rate (default: 0.03)
            enable_amplitude: Use amplitude dynamics (default: True)
        """
        self.N = N
        self.lambda_coh = lambda_coh
        self.eta_phi = eta_phi
        self.eta_A = eta_A
        self.enable_amplitude = enable_amplitude
        
        # Hebbian weights (computed from patterns)
        self.J = np.zeros((N, N))
        self.patterns = None
        
    def store_patterns(self, patterns):
        """
        Compute Hebbian weights from binary phase patterns {0, π}
        
        Args:
            patterns: (M, N) array where each row is a pattern with phases in {0, π}
        """
        M, N = patterns.shape
        assert N == self.N, f"Pattern size {N} doesn't match network size {self.N}"
        
        self.patterns = patterns.copy()
        
        # Hebbian learning: J_ij = (1/N) Σ_μ cos(φ_i^μ) cos(φ_j^μ)
        # For binary {0,π}: cos(0)=1, cos(π)=-1, so this is ±1 outer products
        self.J = np.zeros((N, N))
        
        for mu in range(M):
            xi = np.cos(patterns[mu])  # Convert {0,π} to {1,-1}
            self.J += np.outer(xi, xi)
        
        self.J /= N
        np.fill_diagonal(self.J, 0)  # No self-connections
        
        return self.J
    
    def recall(self, initial_phases, initial_amplitudes=None, max_steps=None):
        """
        Recall pattern using CORRECTED CONN dynamics
        
        Args:
            initial_phases: (N,) array of initial phases (with noise)
            initial_amplitudes: (N,) array of initial amplitudes (optional)
            max_steps: Maximum integration steps (default: Config.MAX_STEPS)
            
        Returns:
            final_phases: (N,) array of recalled phases
            final_amplitudes: (N,) array of final amplitudes
            trajectory: List of (phases, amplitudes) at each step
        """
        if max_steps is None:
            max_steps = Config.MAX_STEPS
            
        # Initialize state
        phi = initial_phases.copy()
        
        if initial_amplitudes is not None:
            A = initial_amplitudes.copy()
        else:
            # Default: random amplitudes in [0.5, 0.8]
            A = 0.5 + np.random.rand(self.N) * 0.3
        
        # Track trajectory
        trajectory = []
        
        # CORRECTED DYNAMICS
        for step in range(max_steps):
            
            # ================================================================
            # PHASE UPDATE (CORRECTED)
            # ================================================================
            
            # Coupling term: Σ_i J_ji A_i sin(φ_i - φ_j)
            # Compute pairwise phase differences: φ_i - φ_j
            phi_diff = phi[:, np.newaxis] - phi[np.newaxis, :]  # (N, N) matrix
            
            # Coupling: J_ji * A_i * sin(φ_i - φ_j)
            coupling_matrix = self.J * A[np.newaxis, :] * np.sin(phi_diff)
            coupling = np.sum(coupling_matrix, axis=1)  # Sum over i
            
            # *** CRITICAL FIX ***
            # Coherence term: -2λ A² sin(φ)cos(φ) [NEGATIVE for gradient descent]
            coherence = -2 * self.lambda_coh * A**2 * np.sin(phi) * np.cos(phi)
            
            # Phase dynamics: dφ/dt = A * coupling + coherence
            dphi = A * coupling + coherence
            
            # Update phase
            phi = phi + self.eta_phi * dphi
            phi = np.mod(phi, 2 * np.pi)  # Keep in [0, 2π]
            
            # ================================================================
            # AMPLITUDE UPDATE
            # ================================================================
            
            if self.enable_amplitude and self.lambda_coh > 0:
                # dA/dt = -2λ A sin²(φ)
                dA = -2 * self.lambda_coh * A * np.sin(phi)**2
                
                # Update amplitude
                A = A + self.eta_A * dA
                A = np.clip(A, 0.01, 2.0)  # Keep in reasonable range
            elif not self.enable_amplitude:
                # Keep amplitudes fixed at initial values
                pass
            
            # Store trajectory (every 10 steps to save memory)
            if step % 10 == 0:
                trajectory.append((phi.copy(), A.copy()))
        
        return phi, A, trajectory

# ============================================================================
# PATTERN GENERATION & UTILITIES
# ============================================================================

def generate_binary_patterns(M, N, seed=None):
    """Generate M random binary phase patterns {0, π}"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice([0, np.pi], size=(M, N))

def add_phase_noise(pattern, noise_level):
    """
    Add noise to binary phase pattern
    
    For {0, π} patterns, noise flips bits with probability noise_level
    """
    noisy = pattern.copy()
    flip_mask = np.random.rand(len(pattern)) < noise_level
    noisy[flip_mask] = (noisy[flip_mask] + np.pi) % (2 * np.pi)
    return noisy

def compute_overlap(retrieved_phases, target_phases, threshold=np.pi/4):
    """
    Compute pattern overlap
    
    Fraction of neurons within threshold of target phase
    """
    # Compute phase errors (accounting for 2π periodicity)
    errors = retrieved_phases - target_phases
    errors = np.arctan2(np.sin(errors), np.cos(errors))  # Wrap to [-π, π]
    
    # Count how many are within threshold
    correct = np.sum(np.abs(errors) < threshold)
    return correct / len(target_phases)

# ============================================================================
# EXPERIMENT 1: CAPACITY MEASUREMENT
# ============================================================================

def measure_capacity(N, trials=10, target_recall=0.80, verbose=True):
    """
    Measure maximum capacity M where recall ≥ target_recall
    
    Uses binary search to find M_max
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 1: Capacity Measurement (N={N})")
        print(f"{'='*80}")
        print(f"Finding M where recall ≥ {target_recall*100:.0f}%...")
    
    # Binary search bounds
    M_low = int(0.1 * N)
    M_high = int(0.5 * N)
    M_best = M_low
    
    for iteration in range(12):  # Max 12 iterations
        M = (M_low + M_high) // 2
        
        # Test M with multiple trials
        overlaps = []
        for trial in range(trials):
            # Generate patterns
            patterns = generate_binary_patterns(M, N, seed=trial)
            
            # Create and train CONN
            conn = CONN(N, lambda_coh=Config.LAMBDA, eta_phi=Config.ETA_PHI,
                       eta_A=Config.ETA_A, enable_amplitude=True)
            conn.store_patterns(patterns)
            
            # Test recall for each pattern
            for mu in range(M):
                # Add noise
                noisy = add_phase_noise(patterns[mu], Config.CAPACITY_NOISE)
                
                # Recall
                retrieved, _, _ = conn.recall(noisy)
                
                # Measure overlap
                overlap = compute_overlap(retrieved, patterns[mu])
                overlaps.append(overlap)
        
        mean_overlap = np.mean(overlaps)
        
        # Update binary search
        if mean_overlap >= target_recall:
            M_best = M
            M_low = M + 1
            status = "✓ PASS"
        else:
            M_high = M - 1
            status = "✗ FAIL"
        
        if verbose:
            print(f"  Iter {iteration+1:2d}: M={M:3d}, "
                  f"recall={mean_overlap*100:5.1f}%  {status}")
        
        # Check convergence
        if M_low > M_high:
            break
    
    # Compute metrics
    alpha = M_best / N
    hopfield_M = int(0.138 * N)  # Classical Hopfield capacity
    improvement = M_best / hopfield_M if hopfield_M > 0 else 0
    
    if verbose:
        print(f"\n  RESULT:")
        print(f"    M_max = {M_best}")
        print(f"    α = {alpha:.3f}")
        print(f"    Hopfield M = {hopfield_M}")
        print(f"    Improvement = {improvement:.2f}×")
        print(f"{'='*80}\n")
    
    return {
        'N': N,
        'M_max': M_best,
        'alpha': alpha,
        'hopfield_M': hopfield_M,
        'improvement': improvement
    }

# ============================================================================
# EXPERIMENT 2: NOISE ROBUSTNESS
# ============================================================================

def test_noise_robustness(N=32, M=8, noise_levels=None, trials=5, verbose=True):
    """Test recall accuracy vs noise level"""
    if noise_levels is None:
        noise_levels = Config.NOISE_LEVELS
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 2: Noise Robustness (N={N}, M={M})")
        print(f"{'='*80}")
    
    results = []
    
    for noise in noise_levels:
        overlaps = []
        
        for trial in range(trials):
            # Generate patterns
            patterns = generate_binary_patterns(M, N, seed=trial)
            
            # Create and train CONN
            conn = CONN(N, lambda_coh=Config.LAMBDA, eta_phi=Config.ETA_PHI,
                       eta_A=Config.ETA_A, enable_amplitude=True)
            conn.store_patterns(patterns)
            
            # Test each pattern
            for mu in range(M):
                # Add noise
                noisy = add_phase_noise(patterns[mu], noise)
                
                # Recall
                retrieved, _, _ = conn.recall(noisy)
                
                # Measure overlap
                overlap = compute_overlap(retrieved, patterns[mu])
                overlaps.append(overlap)
        
        mean_recall = np.mean(overlaps)
        std_recall = np.std(overlaps)
        
        if verbose:
            print(f"  Noise={noise:.1f}: recall={mean_recall*100:5.1f}% "
                  f"± {std_recall*100:4.1f}%")
        
        results.append({
            'noise_level': noise,
            'mean_recall': mean_recall,
            'std_recall': std_recall
        })
    
    if verbose:
        print(f"{'='*80}\n")
    
    return results

# ============================================================================
# EXPERIMENT 3: ABLATION STUDY
# ============================================================================

def ablation_study(N=32, M=8, trials=20, verbose=True):
    """Test contribution of each component"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 3: Ablation Study (N={N}, M={M})")
        print(f"{'='*80}")
    
    conditions = [
        ("Full CONN", Config.LAMBDA, True),
        ("No λ (λ=0)", 0.0, True),
        ("No Amplitude", Config.LAMBDA, False),
        ("Baseline (Hopfield)", 0.0, False),
    ]
    
    results = []
    
    for name, lambda_val, use_amplitude in conditions:
        overlaps = []
        
        for trial in range(trials):
            # Generate patterns
            patterns = generate_binary_patterns(M, N, seed=trial)
            
            # Create CONN with specific configuration
            conn = CONN(N, lambda_coh=lambda_val, eta_phi=Config.ETA_PHI,
                       eta_A=Config.ETA_A, enable_amplitude=use_amplitude)
            conn.store_patterns(patterns)
            
            # Test each pattern
            for mu in range(M):
                # Add noise
                noisy = add_phase_noise(patterns[mu], Config.CAPACITY_NOISE)
                
                # Recall
                retrieved, _, _ = conn.recall(noisy)
                
                # Measure overlap
                overlap = compute_overlap(retrieved, patterns[mu])
                overlaps.append(overlap)
        
        mean_recall = np.mean(overlaps)
        std_recall = np.std(overlaps)
        
        if verbose:
            print(f"  {name:25s}: recall={mean_recall*100:5.1f}% "
                  f"± {std_recall*100:4.1f}%")
        
        results.append({
            'condition': name,
            'lambda': lambda_val,
            'amplitude': use_amplitude,
            'mean_recall': mean_recall,
            'std_recall': std_recall
        })
    
    # Compute component contributions
    full = next(r['mean_recall'] for r in results if r['condition'] == 'Full CONN')
    no_lambda = next(r['mean_recall'] for r in results if 'No λ' in r['condition'])
    no_amp = next(r['mean_recall'] for r in results if 'No Amplitude' in r['condition'])
    baseline = next(r['mean_recall'] for r in results if 'Baseline' in r['condition'])
    
    if verbose:
        print(f"\n  Component Contributions:")
        print(f"    λ (coherence):  {(full - no_lambda)*100:+.1f}%")
        print(f"    Amplitude:      {(full - no_amp)*100:+.1f}%")
        print(f"    Total gain:     {(full - baseline)*100:+.1f}%")
        print(f"{'='*80}\n")
    
    return results

# ============================================================================
# MAIN VALIDATION SUITE
# ============================================================================

def run_full_validation(output_dir=None, quick_mode=False):
    """Run complete validation suite"""
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("  CONN VALIDATION SUITE - CORRECTED IMPLEMENTATION")
    print("="*80)
    print(f"\n  Output directory: {output_dir}")
    print(f"  Mode: {'QUICK' if quick_mode else 'FULL'}")
    print(f"\n  CRITICAL: Using corrected gradient (coherence term negative)")
    print("="*80)
    
    start_time = time.time()
    
    # Experiment 1: Capacity
    if quick_mode:
        n_values = [32]
    else:
        n_values = Config.N_VALUES
    
    capacity_results = []
    for N in n_values:
        result = measure_capacity(N, trials=Config.CAPACITY_TRIALS)
        capacity_results.append(result)
    
    # Experiment 2: Noise robustness
    noise_results = test_noise_robustness(
        N=32, M=Config.NOISE_M, 
        noise_levels=Config.NOISE_LEVELS,
        trials=Config.NOISE_TRIALS
    )
    
    # Experiment 3: Ablation
    ablation_results = ablation_study(
        N=Config.ABLATION_N,
        M=Config.ABLATION_M,
        trials=Config.ABLATION_TRIALS
    )
    
    # Save results
    print(f"\nSaving results to {output_dir}/...")
    
    with open(output_dir / "capacity_results.csv", 'w', newline='') as f:
        if capacity_results:
            writer = csv.DictWriter(f, fieldnames=capacity_results[0].keys())
            writer.writeheader()
            writer.writerows(capacity_results)
    
    with open(output_dir / "noise_robustness.csv", 'w', newline='') as f:
        if noise_results:
            writer = csv.DictWriter(f, fieldnames=noise_results[0].keys())
            writer.writeheader()
            writer.writerows(noise_results)
    
    with open(output_dir / "ablation_results.csv", 'w', newline='') as f:
        if ablation_results:
            writer = csv.DictWriter(f, fieldnames=ablation_results[0].keys())
            writer.writeheader()
            writer.writerows(ablation_results)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"  VALIDATION COMPLETE!")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Results saved to: {output_dir}/")
    print(f"{'='*80}\n")
    
    # Summary
    avg_improvement = np.mean([r['improvement'] for r in capacity_results])
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Average capacity improvement: {avg_improvement:.2f}×")
    print(f"  (vs. classical Hopfield α=0.138)")
    print(f"{'='*80}\n")
    
    return capacity_results, noise_results, ablation_results

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Detect if running in Colab/Jupyter (no command-line args)
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    # Check if real command-line args (not Jupyter kernel args)
    has_real_args = len(sys.argv) > 1 and not any(
        arg.startswith('-f') or 'kernel' in arg.lower() 
        for arg in sys.argv[1:]
    )
    
    if IN_COLAB or not has_real_args:
        # Interactive mode (Colab/Jupyter) - just run full validation
        print("🔬 Running in interactive mode (Colab/Jupyter)")
        print("   For command-line options, run as: python script.py --help")
        print()
        
        # Run quick validation by default in Colab
        run_full_validation(quick_mode=True)
        
    else:
        # Command-line mode with argparse
        import argparse
        
        parser = argparse.ArgumentParser(
            description="CONN Validation Suite (CORRECTED)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run full validation
  python CONN_CORRECT_FROM_TSX.py
  
  # Quick test
  python CONN_CORRECT_FROM_TSX.py --quick
  
  # Single experiment
  python CONN_CORRECT_FROM_TSX.py --experiment capacity
            """
        )
        
        parser.add_argument('--quick', action='store_true',
                           help='Quick mode (fewer trials, smaller N)')
        parser.add_argument('--experiment', choices=['capacity', 'noise', 'ablation', 'all'],
                           default='all', help='Which experiment to run')
        parser.add_argument('--output', type=str, default=None,
                           help='Output directory (default: conn_validation_corrected)')
        
        args = parser.parse_args()
        
        if args.experiment == 'all':
            run_full_validation(output_dir=args.output, quick_mode=args.quick)
        elif args.experiment == 'capacity':
            N = 32 if args.quick else 64
            measure_capacity(N, trials=Config.CAPACITY_TRIALS)
        elif args.experiment == 'noise':
            test_noise_robustness(N=32, M=Config.NOISE_M, 
                                 noise_levels=Config.NOISE_LEVELS,
                                 trials=Config.NOISE_TRIALS)
        elif args.experiment == 'ablation':
            ablation_study(N=Config.ABLATION_N, M=Config.ABLATION_M,
                          trials=Config.ABLATION_TRIALS)
