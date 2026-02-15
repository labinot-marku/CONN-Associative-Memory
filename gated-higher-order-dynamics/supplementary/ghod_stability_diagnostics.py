"""
Discrete GHOD Stability Diagnostics (CORRECTED VERSION)

Supplementary code for: "Gated Higher-Order Dynamics with Guaranteed Return"
Author: Labinot Marku, M.D.
DOI: 10.13140/RG.2.2.23260.24962

This module provides reproducible stability diagnostics for discrete GHOD systems.

CORRECTION: Fixed gate dynamics (line 330) to match Section 2.5 of main manuscript.
"""

import numpy as np
from numpy.linalg import eigvalsh, eigvals, norm

# ==============================
# Core Functions
# ==============================

def cubic_gradient(x, T):
    """
    Compute cubic gradient: g(x) = 3 T(x,x)
    
    Args:
        x: State vector (N,)
        T: Cubic tensor (N,N,N)
    
    Returns:
        Gradient vector (N,)
    """
    return 3 * np.einsum('ijk,j,k->i', T, x, x)


def cubic_hessian(x, T):
    """
    Compute cubic Hessian: H(x) = 6 T(x,.)
    
    Args:
        x: State vector (N,)
        T: Cubic tensor (N,N,N)
    
    Returns:
        Hessian matrix (N,N)
    """
    return 6 * np.einsum('ijk,k->ij', T, x)


def vector_field(x, u, A, T, lam):
    """
    GHOD vector field: f(x,u) = -Ax - ug(x) - 2λux
    
    Args:
        x: State vector
        u: Gate value
        A: Anchor matrix
        T: Cubic tensor
        lam: Regularization parameter
    
    Returns:
        Vector field f(x,u)
    """
    g = cubic_gradient(x, T)
    return -A @ x - u * g - 2 * lam * u * x


def sigmoid(h, k=10.0, theta=0.5):
    """
    Sigmoid activation function for gate dynamics.
    
    Args:
        h: Input (typically norm of state)
        k: Steepness parameter
        theta: Threshold
    
    Returns:
        sigmoid(h) in [0,1]
    """
    return 1.0 / (1.0 + np.exp(-k * (h - theta)))


# ==============================
# Spectral Estimation
# ==============================

def estimate_rho_max(T, R, n_samples=200, seed=None):
    """
    Estimate worst-case Hessian spectral radius on ball of radius R.
    
    Args:
        T: Cubic tensor (N,N,N)
        R: Ball radius
        n_samples: Number of random samples
        seed: Random seed for reproducibility
    
    Returns:
        rho_max: Estimated supremum of ρ(H(x)) for ‖x‖ ≤ R
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = T.shape[0]
    rho_max = 0.0
    
    for _ in range(n_samples):
        # Sample random direction
        x = np.random.randn(N)
        x = R * x / norm(x)
        
        # Compute Hessian spectral radius
        H = cubic_hessian(x, T)
        rho = max(abs(eigvals(H)))
        rho_max = max(rho_max, rho)
    
    return rho_max


# ==============================
# Main Diagnostics
# ==============================

def stability_diagnostics(A, T, lam, u_min, u_max, dt, R, 
                         n_samples=200, seed=None, verbose=True):
    """
    Compute discrete Lyapunov stability diagnostics.
    
    This function computes all quantities required to verify
    the discrete stability theorem from Appendix A.x.
    
    Args:
        A: Anchor matrix (N,N)
        T: Cubic tensor (N,N,N)
        lam: Regularization parameter λ
        u_min: Minimum gate value
        u_max: Maximum gate value
        dt: Discrete time step Δt
        R: Ball radius
        n_samples: Number of samples for ρ_max estimation
        seed: Random seed for reproducibility
        verbose: Print detailed diagnostics
    
    Returns:
        dict with keys:
            - a_min: λ_min(A)
            - a: ‖A‖
            - rho_max: Estimated sup ρ(H(x))
            - M_min: Stability margin
            - C_max: Vector field bound
            - dt_bound: Theoretical step-size upper bound
            - condition_satisfied: Boolean
            - safety_ratio: dt / dt_bound
    """
    
    # 1. Linear anchor statistics
    eigvals_A = eigvalsh(A)
    a_min = eigvals_A.min()
    a = norm(A, 2)
    
    # 2. Estimate rho_max
    rho_max = estimate_rho_max(T, R, n_samples, seed)
    
    # 3. Compute stability margin and bound (CORRECTED FORMULA)
    M_min = a_min + 2*lam*u_min - 0.5*u_max*rho_max
    C_max = a + 2*lam*u_max + 0.5*u_max*rho_max
    
    # 4. Compute step-size bound
    if M_min <= 0:
        dt_bound = 0.0
        condition_satisfied = False
        safety_ratio = np.inf
    else:
        dt_bound = 2*M_min / (C_max**2)
        condition_satisfied = (dt < dt_bound)
        safety_ratio = dt / dt_bound if dt_bound > 0 else np.inf
    
    # 5. Print diagnostics if verbose
    if verbose:
        print("=" * 60)
        print("DISCRETE GHOD STABILITY DIAGNOSTICS")
        print("=" * 60)
        print("\nSystem Parameters:")
        print(f"  Dimension N        = {A.shape[0]}")
        print(f"  Regularization λ   = {lam:.6f}")
        print(f"  Ball radius R      = {R:.6f}")
        print(f"  Time step Δt       = {dt:.6f}")
        print(f"\nLinear Anchor Statistics:")
        print(f"  a_min = λ_min(A)   = {a_min:.6f}")
        print(f"  a = ‖A‖            = {a:.6f}")
        print(f"\nCubic Hessian Envelope:")
        print(f"  ρ_max (estimated)  = {rho_max:.6f}")
        print(f"  (from {n_samples} samples)")
        print(f"\nGate Statistics:")
        print(f"  u_min              = {u_min:.6f}")
        print(f"  u_max              = {u_max:.6f}")
        print(f"\nDerived Bounds:")
        print(f"  M_min              = {M_min:.6f}")
        print(f"  C_max              = {C_max:.6f}")
        print(f"  Δt_bound           = {dt_bound:.6f}")
        print(f"\nStability Condition:")
        print(f"  M_min > 0?         : {'YES' if M_min > 0 else 'NO ✗'}")
        print(f"  Δt < Δt_bound?     : {'YES' if condition_satisfied else 'NO ✗'}")
        print(f"  Safety ratio       : {safety_ratio:.4f}")
        print("=" * 60)
        
        if condition_satisfied:
            print("✓ Theoretical stability condition SATISFIED")
        else:
            print("✗ Theoretical stability condition NOT satisfied")
            if M_min <= 0:
                print("  → Increase λ or decrease u_max")
            else:
                print("  → Decrease Δt or increase λ")
        print("=" * 60)
    
    return {
        'a_min': a_min,
        'a': a,
        'rho_max': rho_max,
        'M_min': M_min,
        'C_max': C_max,
        'dt_bound': dt_bound,
        'condition_satisfied': condition_satisfied,
        'safety_ratio': safety_ratio
    }


# ==============================
# Empirical Validation
# ==============================

def validate_lyapunov_descent(trajectory_x, verbose=True):
    """
    Validate empirical Lyapunov descent on a trajectory.
    
    Args:
        trajectory_x: List or array of state vectors (K, N)
        verbose: Print validation results
    
    Returns:
        dict with validation statistics
    """
    trajectory_x = np.array(trajectory_x)
    
    # Compute V(x) = 0.5‖x‖²
    V_values = 0.5 * np.sum(trajectory_x**2, axis=1)
    
    # Check monotonicity
    diffs = np.diff(V_values)
    is_decreasing = np.all(diffs <= 0)
    
    # Empirical descent rate
    if len(V_values) > 1:
        empirical_gamma = -np.mean(diffs) / np.mean(V_values[:-1])
    else:
        empirical_gamma = 0.0
    
    # Trajectory norms
    norms = np.linalg.norm(trajectory_x, axis=1)
    max_norm = np.max(norms)
    final_norm = norms[-1]
    
    if verbose:
        print("\n" + "=" * 60)
        print("EMPIRICAL LYAPUNOV VALIDATION")
        print("=" * 60)
        print(f"\nTrajectory Statistics:")
        print(f"  Number of steps    : {len(V_values)}")
        print(f"  Initial V(x_0)     : {V_values[0]:.6f}")
        print(f"  Final V(x_K)       : {V_values[-1]:.6f}")
        print(f"  Total decrease     : {V_values[0] - V_values[-1]:.6f}")
        print(f"\nMonotonicity:")
        print(f"  Monotone decrease? : {'YES ✓' if is_decreasing else 'NO ✗'}")
        if not is_decreasing:
            n_increases = np.sum(diffs > 0)
            print(f"  Increases detected : {n_increases}/{len(diffs)}")
        print(f"\nDescent Rate:")
        print(f"  Empirical γ        : {empirical_gamma:.6f}")
        print(f"\nState Norms:")
        print(f"  Maximum ‖x_k‖      : {max_norm:.6f}")
        print(f"  Final ‖x_k‖        : {final_norm:.6f}")
        print("=" * 60)
    
    return {
        'V_values': V_values,
        'is_decreasing': is_decreasing,
        'empirical_gamma': empirical_gamma,
        'max_norm': max_norm,
        'final_norm': final_norm
    }


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    """
    Example: Verify stability for a synthetic GHOD system
    
    CORRECTED: Gate dynamics now use sigmoid activation (matches Section 2.5)
    """
    
    # System parameters
    N = 20
    lam = 0.4
    dt = 0.01
    R = 5.0
    u_min = 0.0
    u_max = 1.0
    
    # Generate system
    np.random.seed(42)
    
    # Stable anchor
    M = np.random.randn(N, N)
    A = (M.T @ M) / N  # Positive definite
    
    # Symmetric cubic tensor
    T = np.random.randn(N, N, N) * 0.5
    T = (T + T.transpose(1,0,2) + 
         T.transpose(2,1,0) + 
         T.transpose(0,2,1) +
         T.transpose(1,2,0) + 
         T.transpose(2,0,1)) / 6.0
    
    # Run diagnostics
    print("\nRunning stability diagnostics...")
    results = stability_diagnostics(
        A, T, lam, u_min, u_max, dt, R,
        n_samples=200,
        seed=42,
        verbose=True
    )
    
    # Simulate trajectory with CORRECTED gate dynamics
    print("\nSimulating trajectory...")
    x = np.random.randn(N) * 2.0
    u = 0.0
    alpha, beta = 1.0, 2.0
    
    trajectory = []
    for _ in range(500):
        trajectory.append(x.copy())
        
        # Update state
        f = vector_field(x, u, A, T, lam)
        x_next = x + dt * f
        
        # ✅ CORRECTED gate dynamics (matches Section 2.5)
        h = norm(x)
        u_next = u + dt * (-alpha * u + beta * sigmoid(h))
        u_next = np.clip(u_next, 0, u_max)
        
        x, u = x_next, u_next
    
    # Validate
    validation = validate_lyapunov_descent(trajectory, verbose=True)
    
    print("\n✓ Diagnostics complete")
    print("\nNOTE: Gate dynamics corrected to match Section 2.5 of main manuscript.")
    print("Previous version had incorrect cubic-dependent gate update.")
