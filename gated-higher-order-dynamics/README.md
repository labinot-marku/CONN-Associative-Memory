Gated Higher-Order Dynamics: Trajectory-Constrained Stability
Part of: CONN Associative Memory Research
Preprint: ResearchGate Publication
DOI: 10.13140/RG.2.2.23260.24962
This folder contains simulation code accompanying the preprint:
"Gated Higher-Order Dynamics with Guaranteed Return: An Exploratory Framework for the Capacity-Stability Trade-Off"
Author: Labinot Marku, M.D.
The code demonstrates that nonlinear systems with gated higher-order interactions can remain dynamically stable even when classical worst-case (volumetric) stability criteria are violated by orders of magnitude.

Quick Start
bash# Install dependencies
pip install -r requirements.txt

# Run demonstration
python gated_dynamics.py
Expected output: ~37 stable trajectories (out of 50) with conservatism ratio of ~165.7×.

Key Finding
Order-of-magnitude conservatism gaps (~150-170×) exist between classical Lyapunov worst-case bounds and realized trajectory curvature in gated higher-order dynamics. This enables stable operation in parameter regimes that volumetric stability analysis deems unstable.
Mechanism: The conservatism gap arises from two complementary factors:

Geometric scaling: For homogeneous cubic energy E₁(x) = T(x,x,x), the Hessian magnitude scales linearly with state norm: ||∇²E₁(x)|| ∝ ||x||. Volumetric bounds evaluate cubic curvature at fixed radius r=5.0, while actual trajectories remain confined to ||x|| < 0.5. This geometric effect alone contributes a factor of approximately ~10× from radius difference.
Spectral concentration: Random tensor realizations exhibit spectral properties where maximum eigenvalues at the boundary of large evaluation spheres systematically exceed trajectory-averaged curvature, amplifying the geometric effect.

The phenomenon is reproducible across random initializations and represents trajectory confinement to low-curvature regions of the reachable set through gating and regularization.

Relation to CONN Paper
This work extends the CONN framework by:

Formalizing gated dynamics with rigorous stability guarantees (Theorem 1)
Identifying three-regime hierarchy (A/B/C) for stability-expressivity trade-offs
Quantifying conservatism gaps between worst-case theory and realized behavior
Providing fiber-bundle interpretation of higher-order gated systems

While CONN demonstrated practical benefits of complex-valued gating, this work provides the theoretical foundation explaining why such architectures remain stable.

Model Overview
Dynamics
ẋ = -A x - u ∇E_cubic(x) - 2λu x
u̇ = -α u + β σ(‖x‖)
where:

A: Quadratic dissipative anchor (positive definite)
E_cubic(x): Fully symmetric cubic energy (higher-order interactions)
u: Temporal gate variable (controls higher-order activation)
λ: Regularization parameter (trajectory confinement strength)

Stability Diagnostics
Governor (Worst-Case) Margin:
λ_min(A) + 2λu_max - u_max sup_{‖x‖=r} ρ(∇²E_cubic(x))
Frequently negative (classical theory predicts instability).
Trajectory-Averaged Margin:
λ_min(A) + 2λ⟨u⟩ - ⟨u ρ(x)⟩
Typically positive (empirical stability observed).
Gap between these: ~150-170×, explained by trajectory confinement.

Key Results
From 50 randomized trials (N=10, λ=0.4, T_scale=2.0):

Convergence rate: 74% (37 stable trajectories)
Regime B rate: 100% of stable trials violate classical bounds
Conservatism ratio: 165.7× (representative trial)
Trajectory confinement: ‖x‖ = 0.465 despite evaluation at ‖x‖ = 5.0

Representative Trial (seed=42)
Classical Stability Bound (Governor Margin):
  λ_min(A) + 2λu_max - u_max·H_sup = -214.09  [VIOLATED]

Cubic Curvature Diagnostics:
  Maximum Hessian spectral radius (r=5.0):  215.39
  Stability denominator (λ_min + 2λu_max):  1.30
  Conservatism ratio (H_sup / denominator): 165.7×

Trajectory Statistics:
  Maximum state norm:  0.465
  Maximum gate value:  1.000

Reproducibility and Verification
Code Execution
The conservatism gap phenomenon has been independently reproduced via multiple code executions with controlled random initialization (fixed seed=42). Results match reported values within numerical precision.
Verification Steps

Clone repository:

bash   git clone https://github.com/labinot-marku/CONN-Associative-Memory.git
   cd CONN-Associative-Memory/gated-higher-order-dynamics

Install dependencies:

bash   pip install numpy scipy matplotlib

Run 50-trial sweep:

bash   python gated_dynamics.py

Expected output:

   Total trials:                          50
   Stable trajectories (converged):       37
   Regime B (stable despite violation):   37
   Convergence rate:                      74.0%
   
   Representative Trial:
   Conservatism ratio: 165.7×
   Governor margin: -214.09 [VIOLATED]
   Max trajectory norm: 0.465
Mathematical Validation
Structure verified:

✅ Cubic tensor full symmetry (T_ijk = T_σ(i,j,k) for all permutations σ)
✅ Gradient formula: g(x) = 3T(x,x) (correct for homogeneous cubic)
✅ Hessian formula: H(x) = 6T(x,·,·) (consistent with second derivative)
✅ Euler's homogeneous function theorem: x^T g(x) = (1/2) x^T H(x) x
✅ ODE well-posed: Smooth right-hand side with numerical clipping

Numerical consistency:

✅ Results reproducible with fixed random seed
✅ Scaling relationships consistent (cubic curvature ∝ ||x||)
✅ Spectral radius magnitudes plausible for N=10, T_scale=2.0
✅ Convergence criterion stable across trials

Reproducibility Features

Fixed seed: np.random.seed(42) ensures exact reproducibility
No tuning: Parameters fixed across all trials
Full symmetry: Cubic tensors symmetrized over all 6 permutations
Numerical safety: Sigmoid overflow prevention via clipping


Files

gated_dynamics.py: Complete implementation with diagnostics
README.md: This file
requirements.txt: Python dependencies (NumPy, SciPy)

Key Functions

random_cubic_tensor(): Generates fully symmetric T with enforced 6-fold symmetry
cubic_energy_grad(): Computes ∇E₁(x) = 3T(x,x)
cubic_hessian(): Computes ∇²E₁(x) = 6T(x,·,·)
governor_margin(): Adversarial worst-case bound at r=5.0
averaged_margin(): Trajectory-localized empirical bound
run_trial(): Single simulation with full diagnostics


Theoretical Foundation
This implementation validates:

Theorem 1 (Guaranteed Return): Asymptotic convergence under dominance condition
Theorem 3 (Governor Inequality): Classical worst-case spectral bound
Appendix A.3 (Discrete-Time Stability): Explicit Lyapunov decrease conditions
Appendix D (Maximum Entropy Connections): Relationship to neural population models

See preprint for complete mathematical proofs and supplementary materials for detailed appendices.

Current Status
Published Materials:

Main manuscript: ResearchGate (DOI: 10.13140/RG.2.2.23260.24962)
Appendix A.3: Discrete-time stability theorem
Appendix D: Connections to maximum entropy neural models
Source code: MIT License (this repository)

Preprint Level: ✅ Ready
Journal Submission: Technical exposition being enhanced (explicit derivations of key identities for reviewer transparency)

Citation
If you use this code or reference this work, please cite:
bibtex@article{marku2026gated,
  title={Gated Higher-Order Dynamics with Guaranteed Return: 
         An Exploratory Framework for the Capacity-Stability Trade-Off},
  author={Marku, Labinot},
  journal={ResearchGate Preprint},
  year={2026},
  doi={10.13140/RG.2.2.23260.24962},
  url={https://www.researchgate.net/publication/387585009}
}
For the CONN paper:
bibtex@article{marku2024conn,
  title={Complex-Valued Oscillatory Neural Networks (CONN) for 
         Distributed Associative Memory},
  author={Marku, Labinot},
  journal={ResearchGate Preprint},
  year={2024},
  url={https://github.com/labinot-marku/CONN-Associative-Memory/tree/main/validation}
}

Contact

ResearchGate: Labinot Marku
GitHub: @labinot-marku
Email: labinot.marku@krh.de
Issues: Report here


Acknowledgments
Mathematical analysis developed with assistance from AI systems (Claude 3.5 Sonnet, ChatGPT-4, Gemini 2.0) for symbolic computation, proof verification, numerical stability improvements, and professional documentation. All theoretical insights, scientific judgments, interpretations, and final content decisions are the author's responsibility. AI contributions documented in preprint Appendix C.

License
See repository LICENSE

Published: February 10, 2026
DOI: 10.13140/RG.2.2.23260.24962
Status: Public preprint with verified reproducible code
Last Updated: February 21, 2026
