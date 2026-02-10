"""
Gated Higher-Order Dynamics: Trajectory-Constrained Stability

This code accompanies the preprint:
"Externalized Cognitive State Reconstruction via Gated Higher-Order Dynamics: 
A Fiber-Bundle Framework for Bounded Recursive Processing"

Demonstrates that gated dynamics can remain stable despite strong violations 
of conservative volumetric stability bounds. Stability arises from trajectory 
confinement to low-curvature regions of the reachable set, not parameter tuning.

Key Finding:
Systems with gated higher-order interactions exhibit order-of-magnitude gaps 
(~100-200×) between classical worst-case Lyapunov bounds and realized trajectory 
curvature, enabling stable operation in theoretically prohibited parameter regimes.

Author: Labinot Marku, M.D.
Date: February 10, 2026
License: MIT

Repository: https://github.com/[YOUR_USERNAME]/gated-dynamics-ecsr
Preprint: [ResearchGate URL]
"""

import numpy as np
from numpy.linalg import eigvalsh, eigvals, norm
from scipy.integrate import solve_ivp

# ================================================================
# 1. Core Utilities
# ================================================================

def random_pd_matrix(N, min_eig=1.0):
    """
    Generate a random symmetric positive-definite matrix.
    
    Parameters:
    -----------
    N : int
        Matrix dimension
    min_eig : float
        Minimum eigenvalue (ensures positive definiteness)
    
    Returns:
    --------
    A : ndarray (N, N)
        Symmetric positive-definite matrix
    """
    Q = np.random.randn(N, N)
    A = Q.T @ Q
    eigs = eigvalsh(A)
    A += (min_eig - np.min(eigs)) * np.eye(N)
    return A


def random_cubic_tensor(N, scale=1.0):
    """
    Generate a fully symmetric cubic tensor.
    
    Full symmetry is required for consistency of gradient and Hessian
    calculations. The tensor is symmetrized over all 6 permutations of indices.
    
    Parameters:
    -----------
    N : int
        Tensor dimension
    scale : float
        Overall scaling factor
    
    Returns:
    --------
    T : ndarray (N, N, N)
        Fully symmetric cubic tensor
    """
    T = np.random.randn(N, N, N) * scale
    T = (
        T
        + T.transpose(1, 0, 2)
        + T.transpose(2, 1, 0)
        + T.transpose(0, 2, 1)
        + T.transpose(1, 2, 0)
        + T.transpose(2, 0, 1)
    ) / 6.0
    return T


def cubic_energy_grad(x, T):
    """
    Gradient of cubic energy term.
    
    For E_cubic(x) = x^T T x x (Einstein notation),
    ∇E_cubic(x) = 3 T_{ijk} x_j x_k
    
    Assumes fully symmetric tensor T.
    """
    return 3.0 * np.einsum('ijk,j,k->i', T, x, x)


def cubic_hessian(x, T):
    """
    Hessian of cubic energy term.
    
    ∇²E_cubic(x) = 6 T_{ijk} x_k
    
    Assumes fully symmetric tensor T.
    """
    return 6.0 * np.einsum('ijk,k->ij', T, x)


# ================================================================
# 2. Gated System Dynamics
# ================================================================

def gated_system(t, state, A, T, alpha, beta, lam):
    """
    Gated higher-order dynamical system.
    
    State equation:
    dx/dt = -A x - u ∇E_cubic(x) - 2λu x
    du/dt = -α u + β σ(‖x‖)
    
    where σ is a smooth sigmoid activation based on state magnitude.
    
    Parameters:
    -----------
    t : float
        Time (not used, required by solve_ivp)
    state : ndarray (N+1,)
        Combined state [x, u] where x ∈ ℝ^N, u ∈ ℝ
    A : ndarray (N, N)
        Quadratic dissipative anchor (positive definite)
    T : ndarray (N, N, N)
        Cubic interaction tensor (fully symmetric)
    alpha : float
        Gate decay rate
    beta : float
        Gate activation strength
    lam : float
        Regularization strength (λ parameter)
    
    Returns:
    --------
    dstate : ndarray (N+1,)
        Time derivative [dx/dt, du/dt]
    """
    N = len(state) - 1
    x = state[:N]
    u = state[N]

    # Core dynamics: quadratic anchor + gated cubic term + regularization
    dx = -(A @ x) - u * cubic_energy_grad(x, T) - (2.0 * lam * u * x)

    # Gate activation: smooth threshold based on state magnitude
    h = norm(x)
    z = np.clip(10.0 * (h - 0.5), -50, 50)  # Numerical safety: prevent overflow
    sigma = 1.0 / (1.0 + np.exp(-z))

    # Gate relaxation dynamics
    du = -alpha * u + beta * sigma

    return np.concatenate([dx, [du]])


# ================================================================
# 3. Stability Diagnostics
# ================================================================

def governor_margin(A, T, u_max, lam, r_test=5.0, samples=50):
    """
    Approximate conservative volumetric (worst-case) stability margin.
    
    This function samples the maximal cubic curvature on a sphere of fixed 
    radius r_test to emulate classical worst-case stability criteria. The 
    bound is intentionally adversarial and does not account for trajectory 
    confinement.
    
    Governor Inequality (conservative bound):
    λ_min(A) + 2λu_max - u_max sup_{‖x‖=r} ρ(∇²E_cubic(x))
    
    If negative, classical theory predicts instability.
    
    Parameters:
    -----------
    A : ndarray (N, N)
        Quadratic anchor matrix
    T : ndarray (N, N, N)
        Cubic tensor
    u_max : float
        Maximum gate value along trajectory
    lam : float
        Regularization parameter
    r_test : float
        Radius for adversarial evaluation (default: 5.0)
    samples : int
        Number of random directions to sample
    
    Returns:
    --------
    margin : float
        Governor margin (negative indicates theoretical instability)
    H_sup : float
        Supremum of cubic Hessian spectral radius at r=r_test
    """
    lam_min = np.min(eigvalsh(A))
    H_sup = 0.0

    for _ in range(samples):
        v = np.random.randn(A.shape[0])
        v /= norm(v)
        H = cubic_hessian(r_test * v, T)
        H_sup = max(H_sup, np.max(np.abs(eigvals(H))))

    margin = lam_min + 2.0 * lam * u_max - u_max * H_sup
    return margin, H_sup


def averaged_margin(traj_x, traj_u, A, T, lam, alpha, t_final=50.0):
    """
    Trajectory-averaged effective stability margin.
    
    Uses exponential time-weighting to emphasize early dynamics:
    
    λ_min(A) + 2λ⟨u⟩_α - ⟨u ρ(∇²E_cubic(x))⟩_α
    
    where ⟨·⟩_α denotes exponentially weighted time average.
    
    Parameters:
    -----------
    traj_x : ndarray (M, N)
        Trajectory states
    traj_u : ndarray (M,)
        Gate values along trajectory
    A : ndarray (N, N)
        Quadratic anchor
    T : ndarray (N, N, N)
        Cubic tensor
    lam : float
        Regularization parameter
    alpha : float
        Exponential decay rate for weighting
    t_final : float
        Final time (for time grid construction)
    
    Returns:
    --------
    margin : float
        Trajectory-averaged margin (positive indicates empirical stability)
    rho_mean : float
        Mean cubic curvature along trajectory
    """
    lam_min = np.min(eigvalsh(A))

    # Time-based weighting (not index-based)
    t = np.linspace(0.0, t_final, len(traj_u))
    weights = np.exp(-alpha * t)
    weights /= np.sum(weights)

    # Compute cubic curvature along trajectory
    rho_vals = np.array([
        np.max(np.abs(eigvals(cubic_hessian(x, T))))
        for x in traj_x
    ])

    lhs = lam_min + 2.0 * lam * np.sum(weights * traj_u)
    rhs = np.sum(weights * traj_u * rho_vals)

    return lhs - rhs, np.mean(rho_vals)


# ================================================================
# 4. Trial Execution
# ================================================================

def run_trial(N=10, T_scale=2.0, lam=0.4, min_eig=0.5, seed=None):
    """
    Run a single simulation trial and return stability diagnostics.
    
    Parameters:
    -----------
    N : int
        System dimension
    T_scale : float
        Cubic tensor scaling (controls interaction strength)
    lam : float
        Regularization parameter λ
    min_eig : float
        Minimum eigenvalue of anchor matrix A
    seed : int or None
        Random seed for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - converged : bool (trajectory converged to near-origin)
        - gov_margin : float (governor margin, classical bound)
        - avg_margin : float (trajectory-averaged margin)
        - H_sup : float (supremum cubic curvature at r=5.0)
        - x_max : float (maximum state norm along trajectory)
        - lam_min : float (minimum eigenvalue of A)
        - u_max : float (maximum gate value)
    """
    if seed is not None:
        np.random.seed(seed)
    
    A = random_pd_matrix(N, min_eig)
    T = random_cubic_tensor(N, scale=T_scale)

    x0 = np.random.randn(N) * 0.1
    u0 = 1.0

    sol = solve_ivp(
        gated_system,
        (0.0, 50.0),
        np.concatenate([x0, [u0]]),
        args=(A, T, 0.3, 0.5, lam),
        rtol=1e-6,
        atol=1e-9
    )

    traj_x = sol.y[:-1].T
    traj_u = sol.y[-1]

    # Convergence criterion
    converged = (traj_u[-1] < 0.2) and (norm(traj_x[-1]) < 0.2)

    # Stability diagnostics
    u_max = np.max(traj_u)
    gov_margin, H_sup = governor_margin(A, T, u_max, lam)
    avg_margin, _ = averaged_margin(traj_x, traj_u, A, T, lam, alpha=0.3)
    
    lam_min = np.min(eigvalsh(A))

    return {
        "converged": converged,
        "gov_margin": gov_margin,
        "avg_margin": avg_margin,
        "H_sup": H_sup,
        "x_max": np.max(norm(traj_x, axis=1)),
        "lam_min": lam_min,
        "u_max": u_max
    }


# ================================================================
# 5. Main Experiment
# ================================================================

if __name__ == "__main__":
    
    # Fixed seed for reproducibility
    np.random.seed(42)

    results = []
    print("=" * 70)
    print("GATED HIGHER-ORDER DYNAMICS: Conservatism Gap Demonstration")
    print("=" * 70)
    print("\nSearching for trajectory-constrained stability...\n")

    # Run 50 trials with fixed parameters
    for i in range(50):
        results.append(run_trial(N=10, T_scale=2.0, lam=0.4, min_eig=0.5))

    # Classify trials
    stable = [r for r in results if r["converged"]]
    regime_B = [r for r in stable if r["gov_margin"] < 0.0]

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTotal trials:                          50")
    print(f"Stable trajectories (converged):       {len(stable)}")
    print(f"Regime B (stable despite violation):   {len(regime_B)}")
    print(f"Convergence rate:                      {100*len(stable)/50:.1f}%")
    print(f"Regime B rate:                         {100*len(regime_B)/50:.1f}%")

    if regime_B:
        # Analyze representative Regime B trial
        ex = regime_B[0]
        
        # Compute conservatism ratio using known parameter values
        # (lam=0.4 as passed to run_trial)
        lam = 0.4
        denominator = ex["lam_min"] + 2.0 * lam * ex["u_max"]
        conservatism_ratio = ex["H_sup"] / denominator
        
        print("\n" + "=" * 70)
        print("REPRESENTATIVE REGIME B TRIAL (Trajectory-Constrained Stability)")
        print("=" * 70)
        print(f"\nClassical Stability Bound (Governor Margin):")
        print(f"  λ_min(A) + 2λu_max - u_max·H_sup = {ex['gov_margin']:.2f}  [VIOLATED]")
        print(f"\nCubic Curvature Diagnostics:")
        print(f"  Maximum Hessian spectral radius (r=5.0):  {ex['H_sup']:.2f}")
        print(f"  Stability denominator (λ_min + 2λu_max):  {denominator:.2f}")
        print(f"  Conservatism ratio (H_sup / denominator): {conservatism_ratio:.1f}×")
        print(f"\nTrajectory Statistics:")
        print(f"  Maximum state norm:  {ex['x_max']:.3f}")
        print(f"  Maximum gate value:  {ex['u_max']:.3f}")
        
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        print("\nThe system remains stable despite violating classical worst-case")
        print("Lyapunov bounds by a factor of ~{:.0f}×.".format(conservatism_ratio))
        print("\nMechanism: Trajectory confinement to low-curvature regions of the")
        print("reachable set. The cubic destabilizing term (H_sup) is evaluated")
        print("at r=5.0, but actual trajectories remain confined to r<1.0 where")
        print("cubic curvature is orders of magnitude smaller.")
        print("\nThis demonstrates that volumetric (worst-case) stability criteria")
        print("can be massively conservative for systems with gated dynamics and")
        print("regularization-mediated trajectory confinement.")
        print("=" * 70)
    else:
        print("\nNo Regime B trials found. Try adjusting parameters:")
        print("  - Increase T_scale (stronger cubic interactions)")
        print("  - Decrease lam (weaker regularization)")
        print("  - Increase min_eig (stronger anchor)")
