# Gated Higher-Order Dynamics: Trajectory-Constrained Stability

**Part of**: [CONN Associative Memory Research](https://github.com/labinot-marku/CONN-Associative-Memory)

---

This folder contains simulation code accompanying the preprint:

**"Externalized Cognitive State Reconstruction via Gated Higher-Order Dynamics: A Fiber-Bundle Framework for Bounded Recursive Processing"**  
Author: Labinot Marku, M.D.

The code demonstrates that nonlinear systems with gated higher-order interactions can remain dynamically stable even when classical worst-case (volumetric) stability criteria are violated by orders of magnitude.

---

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run demonstration
python gated_dynamics.py
```

Expected output: ~37 stable trajectories (out of 50) with conservatism ratio of ~165.7×.

---

## Key Finding

**Order-of-magnitude conservatism gaps** (~150-170×) exist between classical Lyapunov worst-case bounds and realized trajectory curvature in gated higher-order dynamics. This enables stable operation in parameter regimes that volumetric stability analysis deems unstable.

**Mechanism**: Trajectory confinement to low-curvature regions of the reachable set through gating and regularization.

---

## Relation to CONN Paper

This work extends the [CONN framework](../validation/) by:

1. **Formalizing gated dynamics** with rigorous stability guarantees (Theorem 1)
2. **Identifying three-regime hierarchy** (A/B/C) for stability-expressivity trade-offs
3. **Quantifying conservatism gaps** between worst-case theory and realized behavior
4. **Providing fiber-bundle interpretation** of higher-order gated systems

While CONN demonstrated practical benefits of complex-valued gating, this work provides the **theoretical foundation** explaining why such architectures remain stable.

---

## Model Overview

### Dynamics
```
ẋ = -A x - u ∇E_cubic(x) - 2λu x
u̇ = -α u + β σ(‖x‖)
```

where:
- `A`: Quadratic dissipative anchor (positive definite)
- `E_cubic(x)`: Fully symmetric cubic energy (higher-order interactions)
- `u`: Temporal gate variable (controls higher-order activation)
- `λ`: Regularization parameter (trajectory confinement strength)

---

## Stability Diagnostics

### Governor (Worst-Case) Margin
```
λ_min(A) + 2λu_max - u_max sup_{‖x‖=r} ρ(∇²E_cubic(x))
```
Frequently **negative** (classical theory predicts instability).

### Trajectory-Averaged Margin
```
λ_min(A) + 2λ⟨u⟩ - ⟨u ρ(x)⟩
```
Typically **positive** (empirical stability observed).

**Gap between these**: ~150-170×, explained by trajectory confinement.

---

## Key Results

From 50 randomized trials (N=10, λ=0.4, T_scale=2.0):

- **Convergence rate**: 74% (37 stable trajectories)
- **Regime B rate**: 100% of stable trials violate classical bounds
- **Conservatism ratio**: 165.7× (representative trial)
- **Trajectory confinement**: ‖x‖ = 0.465 despite evaluation at ‖x‖ = 5.0

---

## Reproducibility

- **Fixed seed**: `np.random.seed(42)` ensures exact reproducibility
- **No tuning**: Parameters fixed across all trials
- **Full symmetry**: Cubic tensors symmetrized over all 6 permutations
- **Numerical safety**: Sigmoid overflow prevention via clipping

---

## Files

- `gated_dynamics.py`: Complete implementation with diagnostics
- `README.md`: This file
- `requirements.txt`: Python dependencies (NumPy, SciPy)

---

## Theoretical Foundation

This implementation validates:

- **Theorem 1** (Guaranteed Return): Asymptotic convergence under dominance condition
- **Theorem 3** (Governor Inequality): Classical worst-case spectral bound
- **Appendix A.5** (Boundary-Limited Capacity): C_eff ≤ k · Area(∂R)

See preprint for complete mathematical proofs.

---

## Citation
```bibtex
@article{marku2026gated,
  title={Externalized Cognitive State Reconstruction via Gated Higher-Order Dynamics: 
         A Fiber-Bundle Framework for Bounded Recursive Processing},
  author={Marku, Labinot},
  journal={ResearchGate Preprint},
  year={2026},
  url={https://github.com/labinot-marku/CONN-Associative-Memory/tree/main/Gated%20Higher-Order%20Dynamics}
}
```

For the CONN paper:
```bibtex
@article{marku2024conn,
  title={Complex-Valued Oscillatory Neural Networks (CONN) for 
         Distributed Associative Memory},
  author={Marku, Labinot},
  journal={ResearchGate Preprint},
  year={2024},
  url={https://github.com/labinot-marku/CONN-Associative-Memory/tree/main/validation}
}
```

---

## Contact

- **ResearchGate**: [Labinot Marku](https://www.researchgate.net/profile/Labinot-Marku)
- **GitHub**: [@labinot-marku](https://github.com/labinot-marku)
- **Issues**: [Report here](https://github.com/labinot-marku/CONN-Associative-Memory/issues)

---

## Acknowledgments

Code development benefited from multi-AI collaboration (Claude, ChatGPT, Gemini) for:
- Numerical stability improvements
- Mathematical verification
- Professional documentation

All theoretical insights attributed to the human author (documented in preprint Appendix C).

---

## License

See [repository LICENSE](../../LICENSE)

---

**Last Updated**: February 10, 2026
```

---

## **AFTER PASTING**:

1. **Commit message** box will appear at bottom
2. **Paste this commit message**:
```
Add gated-dynamics folder with README

New preprint on trajectory-constrained stability in gated higher-order 
systems. Demonstrates order-of-magnitude conservatism gaps (~165.7×) 
between classical worst-case bounds and realized trajectory curvature.

Key results: 74% convergence in Regime B, all stable trials violate 
governor inequality, trajectory confinement mechanism validated.
```

3. **Click**: "Commit new file"

---

## **⚠️ IMPORTANT NOTE**

I see you named the folder `Gated Higher-Order Dynamics` (with spaces).

**This will work**, but the URL will have `%20` for spaces:
```
.../tree/main/Gated%20Higher-Order%20Dynamics
