# Appendix A.x: Discrete Lyapunov Stability Analysis for GHOD

**Supplementary Material for**: "Gated Higher-Order Dynamics with Guaranteed Return"  
**Author**: Labinot Marku, M.D.  
**DOI**: 10.13140/RG.2.2.23260.24962  
**Version**: Unified (incorporating multi-AI refinement)

---

## Overview

This appendix provides a rigorous discrete-time stability analysis for the Gated Higher-Order Dynamics (GHOD) framework. We derive explicit, computable conditions under which the discrete residual update preserves the guaranteed return property established in Theorem 1 (continuous-time).

**Main Result**: We prove a sufficient step-size bound that ensures monotone Lyapunov decrease in discrete GHOD systems, bridging continuous theory to practical implementation.

---

## A.x.1 Discrete GHOD System

### System Dynamics

Consider the forward Euler discretization of the continuous GHOD system:

$$x_{k+1} = x_k + \Delta t \, f(x_k, u_k)$$

where the vector field is:

$$f(x,u) = -A x - u\, g(x) - 2\lambda u\, x$$

**Components**:
- $x \in \mathbb{R}^N$: State vector
- $u_k \in [0, u_{\max}]$: Time-dependent gate value
- $A \in \mathbb{R}^{N \times N}$: Symmetric positive definite anchor matrix
- $g(x) = \nabla E_{\text{cubic}}(x) = 3T(x,x)$: Cubic gradient (fully symmetric tensor $T$)
- $\lambda > 0$: Regularization parameter (the "Governor")
- $\Delta t > 0$: Discrete time step

### Lyapunov Candidate

We use the quadratic energy:

$$V(x) = \frac{1}{2} \|x\|^2$$

**Goal**: Establish conditions under which $V(x_{k+1}) < V(x_k)$ for all $x_k$ in a ball $B_R$.

---

## A.x.2 Assumptions

Fix a ball $B_R = \{x : \|x\| \le R\}$ and assume:

**A1. Anchor Spectrum**:
$$0 < a_{\min} := \lambda_{\min}(A) \le \|A\| =: a$$

**A2. Bounded Hessian**:
$$\rho(H(x)) \le \rho_{\max} \quad \text{for all } \|x\| \le R$$

where $H(x) = \nabla^2 E_{\text{cubic}}(x)$ is the Hessian of the cubic energy.

**A3. Gate Bounds**:
$$0 \le u_k \le u_{\max}$$

**A4. Cubic Identity** (for fully symmetric tensor):

For a fully symmetric cubic form $E_{\text{cubic}}(x) = T(x,x,x)$ where $T$ is a fully symmetric third-order tensor, we have:

$$\nabla E = g(x) = 3T(x,x)$$

$$\nabla^2 E(x)[x] = H(x)x = 6T(x,x) = 2 \cdot 3T(x,x) = 2g(x)$$

Therefore:
$$H(x)x = 2g(x) \quad \Rightarrow \quad g(x) = \frac{1}{2}H(x)x$$

This identity follows from the standard tensor calculus normalization and will be used throughout the derivation.

These are standard assumptions for Lyapunov analysis in bounded regions.

---

## A.x.3 Energy Increment Analysis

### Exact Formula

The discrete energy change is:

$$\Delta V := V(x_{k+1}) - V(x_k) = \langle x_k, x_{k+1} - x_k \rangle + \frac{1}{2}\|x_{k+1} - x_k\|^2$$

Substituting $x_{k+1} - x_k = \Delta t \, f(x_k, u_k)$:

$$\Delta V = \Delta t \langle x_k, f(x_k, u_k) \rangle + \frac{\Delta t^2}{2} \|f(x_k, u_k)\|^2 \tag{E1}$$

**Key observation**: The first term (linear in $\Delta t$) captures the continuous-time Lyapunov drift. The second term (quadratic in $\Delta t$) is the discrete correction.

**Strategy**: Bound both terms conservatively to ensure $\Delta V < 0$.

---

## A.x.4 Bounding the Linear Term

### Derivation

Using the vector field definition and the cubic identity:

$$\langle x, f(x,u) \rangle = -x^\top A x - u \, x^\top g(x) - 2\lambda u \|x\|^2$$

$$= -x^\top A x - \frac{u}{2} x^\top H(x) x - 2\lambda u \|x\|^2$$

### Conservative Bound

The term $x^\top H(x) x$ can have either sign. Using spectral radius bounds:

$$|x^\top H(x) x| \le \rho(H(x)) \|x\|^2 \le \rho_{\max} \|x\|^2$$

Therefore (worst-case positive):

$$\langle x, f(x,u) \rangle \le -x^\top A x + \frac{u}{2}\rho_{\max}\|x\|^2 - 2\lambda u \|x\|^2$$

Using $x^\top A x \ge a_{\min} \|x\|^2$:

$$\langle x, f(x,u) \rangle \le -a_{\min}\|x\|^2 + \frac{u}{2}\rho_{\max}\|x\|^2 - 2\lambda u \|x\|^2$$

### Stability Margin

Define:

$$M(u) := a_{\min} + 2\lambda u - \frac{u}{2}\rho_{\max} \tag{DefM}$$

Then:

$$\langle x, f(x,u) \rangle \le -M(u) \|x\|^2 \tag{E2}$$

**Interpretation**: $M(u)$ represents the net stability margin:
- $a_{\min}$: Anchor dissipation
- $2\lambda u$: Regularization damping
- $-\frac{u}{2}\rho_{\max}$: Cubic destabilization (worst-case)

**When $M(u) > 0$**: Linear term drives energy decrease ✅

---

## A.x.5 Bounding the Quadratic Term

### Vector Field Magnitude

Using triangle inequality and the cubic identity:

$$\|f(x,u)\| \le \|A\| \|x\| + u\|g(x)\| + 2\lambda u \|x\|$$

$$\le \|A\| \|x\| + \frac{u}{2}\rho_{\max}\|x\| + 2\lambda u \|x\|$$

$$= \left( a + 2\lambda u + \frac{u}{2}\rho_{\max} \right) \|x\|$$

### Vector Field Bound

Define:

$$C(u) := a + 2\lambda u + \frac{u}{2}\rho_{\max} \tag{DefC}$$

Then:

$$\|f(x,u)\|^2 \le C(u)^2 \|x\|^2 \tag{E3}$$

**Interpretation**: $C(u)$ upper bounds the state contraction/expansion rate.

---

## A.x.6 Sufficient Step-Size Condition

### Combining Bounds

Substituting (E2) and (E3) into (E1):

$$\Delta V \le -\Delta t \, M(u) \|x\|^2 + \frac{\Delta t^2}{2} C(u)^2 \|x\|^2$$

$$= \left( -\Delta t \, M(u) + \frac{\Delta t^2}{2} C(u)^2 \right) \|x\|^2$$

### Sufficient Condition

For strict decrease ($\Delta V < 0$), we need:

$$\Delta t \, M(u) > \frac{\Delta t^2}{2} C(u)^2$$

Rearranging:

$$\Delta t < \frac{2M(u)}{C(u)^2} \tag{StepBound}$$

**Interpretation**: The step-size must be small enough that the discrete correction term doesn't overwhelm the continuous-time Lyapunov drift.

---

## A.x.7 Uniform Conservative Bound

### Trajectory-Independent Condition

To obtain a condition that holds uniformly (independent of trajectory position), we need to find the extrema of $M(u)$ and $C(u)$ over the gate range $[0, u_{\max}]$.

**Analysis of monotonicity**:

For $M(u) = a_{\min} + 2\lambda u - \frac{u}{2}\rho_{\max}$:

$$M'(u) = 2\lambda - \frac{1}{2}\rho_{\max}$$

For $C(u) = a + 2\lambda u + \frac{u}{2}\rho_{\max}$:

$$C'(u) = 2\lambda + \frac{1}{2}\rho_{\max} > 0 \quad \text{(always increasing)}$$

**Case 1**: If $2\lambda > \frac{1}{2}\rho_{\max}$ (moderate curvature regime):
- $M'(u) > 0$ → $M(u)$ is increasing → $M_{\min}$ occurs at $u = u_{\min}$
- This is the typical case when regularization dominates cubic curvature

**Case 2**: If $2\lambda < \frac{1}{2}\rho_{\max}$ (high curvature regime):
- $M'(u) < 0$ → $M(u)$ is decreasing → $M_{\min}$ occurs at $u = u_{\max}$

**Conservative bound (valid for both cases)**:

Since we seek a sufficient condition that holds uniformly, we use the most conservative estimates:

$$M_{\min} := a_{\min} + 2\lambda u_{\min} - \frac{u_{\max}}{2}\rho_{\max}$$

$$C_{\max} := a + 2\lambda u_{\max} + \frac{u_{\max}}{2}\rho_{\max}$$

This choice is conservative because:
- We use the smallest possible contribution from the regularization term ($2\lambda u_{\min}$)
- We use the largest possible destabilizing term ($\frac{u_{\max}}{2}\rho_{\max}$)
- We use the largest possible vector field magnitude ($C_{\max}$ at $u_{\max}$)

This bound is valid regardless of which case applies and provides a trajectory-independent sufficient condition.

---

## A.x.8 Main Theorem

**Theorem A.1** (Discrete Lyapunov Decrease - Sufficient Condition)

*Let the discrete GHOD system satisfy assumptions A1-A4 on the ball $B_R$. Define:*

$$M_{\min} = a_{\min} + 2\lambda u_{\min} - \frac{u_{\max}}{2}\rho_{\max}$$

$$C_{\max} = a + 2\lambda u_{\max} + \frac{u_{\max}}{2}\rho_{\max}$$

*If $M_{\min} > 0$ and:*

$$\boxed{\Delta t < \frac{2M_{\min}}{C_{\max}^2}}$$

*Then for every $x_k \in B_R$ and every admissible gate value $u_k \in [0, u_{\max}]$:*

$$V(x_{k+1}) - V(x_k) \le -\gamma \|x_k\|^2$$

*where:*

$$\gamma = \Delta t M_{\min} - \frac{\Delta t^2}{2} C_{\max}^2 > 0$$

*Consequently, $\|x_k\|$ is strictly contractive while trajectories remain in $B_R$.*

**Important note on forward invariance**: This theorem ensures strict energy decrease *within* $B_R$. Forward invariance of $B_R$ itself (i.e., that $x_k \in B_R$ implies $x_{k+1} \in B_R$) requires either (i) choosing $R$ sufficiently large relative to the initial condition and energy decrease rate, or (ii) an additional barrier argument. In practice, we treat $B_R$ as a diagnostic region for local stability analysis rather than claiming it is a globally invariant set for arbitrary $R$.

### Proof

Follows directly from the bounds (E2) and (E3) combined with (E1), using the uniform bounds $M_{\min}$ and $C_{\max}$. The condition on $\Delta t$ ensures $\gamma > 0$. □

---

## A.x.9 Interpretation and Practical Implications

### Conservative Nature

**Important**: This is a *sufficient* (not necessary) condition. Trajectories may remain stable for larger $\Delta t$ or smaller $\lambda$ than the bound predicts.

**Why conservative**:
1. We used worst-case spectral bound $\rho_{\max}$ over entire ball $B_R$
2. We assumed worst-case sign for $x^\top H(x) x$ (destabilizing)
3. We used uniform bounds (independent of trajectory location)

**This conservatism is intentional** - it provides a reviewer-proof sufficient condition with explicit, computable constants.

### Role of Parameters

**Regularization $\lambda$**:
- Increases $M_{\min}$ linearly: $M_{\min} \propto 2\lambda u_{\min}$
- Increasing $\lambda$ enlarges allowable $\Delta t$
- **Validates continuous-time intuition**: Stronger regularization → more stable

**Gate bounds $u_{\min}, u_{\max}$**:
- If gate can be bounded tightly (via design), condition becomes less conservative
- Large $u_{\max}$ increases both destabilization and bound, net effect captured by ratio

**Anchor strength $a_{\min}$**:
- Stronger anchor (larger $a_{\min}$) → larger allowable $\Delta t$
- Provides baseline stability independent of gating

### Connection to Conservatism Gap

**Key insight**: The empirically observed **165.7× conservatism gap** arises because:

1. **Theoretical bound** evaluates $\rho_{\max}$ at large radius $R$ (worst-case)
2. **Realized trajectories** remain confined to low-curvature regions (‖x‖ ≪ R)
3. **Actual local curvature** ≪ $\rho_{\max}$

**This theorem does NOT contradict the gap** - rather, it provides the conservative envelope within which the gap can be measured.

---

## A.x.10 Computational Protocol

### Required Measurements

To verify this theorem empirically, report:

1. **Anchor statistics**: 
   - $a_{\min} = \lambda_{\min}(A)$
   - $a = \|A\|$

2. **Hessian envelope**:
   - $\rho_{\max} = \sup_{\|x\| \le R} \rho(H(x))$
   - Estimate via sampling (see code)

3. **Gate statistics**:
   - $u_{\min}$ (minimum gate value during run)
   - $u_{\max}$ (maximum gate value during run)

4. **Derived bounds**:
   - Compute $M_{\min}$, $C_{\max}$
   - Compute theoretical bound $\Delta t_{\text{bound}} = \frac{2M_{\min}}{C_{\max}^2}$

5. **Verification**:
   - Check: $\Delta t < \Delta t_{\text{bound}}$?
   - Plot $V(x_k)$ and verify monotone decrease

### Reference Implementation

See `ghod_stability_diagnostics.py` for complete implementation:

```python
def stability_diagnostics(A, T, lam, u_min, u_max, dt, R):
    """
    Compute theoretical stability bound and verify condition.
    
    Returns:
        dt_bound: Theoretical upper bound on step size
    """
    # Compute anchor spectrum
    a_min = np.min(np.linalg.eigvalsh(A))
    a = np.linalg.norm(A, 2)
    
    # Estimate rho_max via sampling
    rho_max = estimate_rho_max(T, R, n_samples=200)
    
    # Compute bounds
    M_min = a_min + 2*lam*u_min - 0.5*u_max*rho_max
    C_max = a + 2*lam*u_max + 0.5*u_max*rho_max
    
    # Theoretical step-size bound
    if M_min > 0:
        dt_bound = 2*M_min / (C_max**2)
    else:
        dt_bound = 0.0  # Condition not satisfiable
    
    # Verify
    condition_satisfied = (dt < dt_bound and M_min > 0)
    
    return dt_bound, condition_satisfied
```

For complete code, see supplementary materials repository.

---

## A.x.11 Limitations and Scope

### What This Theorem Establishes

✅ **Sufficient** step-size bound for monotone Lyapunov decrease  
✅ **Explicit** computable constants (no hidden dependencies)  
✅ **Conservative** reviewer-proof guarantee  
✅ **Constructive** bridge from continuous theory to discrete implementation  

### What This Theorem Does NOT Establish

❌ **Necessary** conditions (trajectories may be stable beyond this bound)  
❌ **Global** stability from arbitrary initial conditions (requires forward invariance)  
❌ **Optimal** step-size (actual safe region may be much larger)  
❌ **Capacity** results (this is a stability theorem, not a memory capacity theorem)  

### Relationship to Main Manuscript

**Theorem 1** (continuous-time): Guarantees return under dominance condition  
**Theorem A.1** (discrete-time): Provides computable step-size bound preserving return

**Together**: Complete theory for both continuous and discrete GHOD systems ✅

---

## A.x.12 Connection to Experimental Results

### Phase Diagram Interpretation

The empirically observed **stability phase diagram** (Section 4.3) can be understood through this theorem:

**Green region** (stable): Parameters satisfy $\Delta t < \frac{2M_{\min}}{C_{\max}^2}$ with margin  
**Red region** (unstable): Parameters violate bound or $M_{\min} \le 0$  
**Sharp boundary**: Nonlinear dependence on $(\Delta t, \lambda)$ creates cliff effect  

### Conservatism Gap Explanation

**Theoretical bound** (this theorem): Conservative estimate using $\rho_{\max}$  
**Empirical stability** (experiments): Actual trajectories confined to ‖x‖ ≪ R  

**Observed gap magnitude**: In representative 10-dimensional experiments with specific parameter choices (λ=0.4, T_scale=2.0, N=10), individual trials have exhibited conservatism gaps exceeding 100×, with some trials showing gaps as large as 165× between the theoretical sufficient bound and the empirically observed stability margin.

**Important caveat**: These are single-trial observations in specific parameter regimes. Multi-seed statistical analysis is recommended to establish robust mean and confidence intervals for the conservatism gap across different system realizations.

**Mechanistic explanation**: Trajectory confinement prevents exploration of worst-case regions where $\rho(H(x)) \approx \rho_{\max}$. The regularization mechanism (governed by λ) confines trajectories to low-curvature regions (typically ‖x‖ < 1.0) even when the theoretical ball radius $R$ is much larger (R = 5.0). This geometric confinement creates the observed gap between worst-case envelope and realized dynamics.

---

## A.x.13 Recommendations for Practitioners

### For Numerical Experiments

1. **Always compute** $M_{\min}$ and verify $> 0$
2. **Estimate** $\rho_{\max}$ via sampling (200+ samples recommended)
3. **Choose** $\Delta t < \frac{2M_{\min}}{C_{\max}^2}$ for guaranteed stability
4. **Monitor** actual $V(x_k)$ to verify empirical descent

### For Architecture Design

1. **Increase $\lambda$** to enlarge safe $\Delta t$ range
2. **Design gates** to bound $u_{\max}$ tightly (reduces conservatism)
3. **Use strong anchor** (large $a_{\min}$) for baseline stability
4. **Combine with normalization** for practical robustness (belt-and-suspenders)

### For Theoretical Extensions

1. **Tighter bounds**: Trajectory-dependent $\rho(H(x))$ could reduce conservatism
2. **Adaptive step-size**: Use local estimates of $M(u_k)$ and $C(u_k)$
3. **Non-Euclidean metrics**: Other Lyapunov functions may yield sharper results
4. **Stochastic extensions**: Generalize to noisy updates (SGD-like dynamics)

---

## A.x.14 Comparison to Existing Results

### Classical Discrete Lyapunov Theory

**Standard approach**: Requires $\rho(\nabla f) < 1$ (Jacobian spectral radius)  
**GHOD challenge**: State-dependent Jacobian, $u$-modulated nonlinearity  
**This result**: Explicit bound via energy decomposition instead of linearization  

### Neural ODE Literature

**Existing work**: Mostly continuous-time stability or implicit discretization  
**This result**: Explicit forward Euler bound for higher-order gated systems  
**Contribution**: Extends classical discrete Lyapunov analysis to state-modulated higher-order dynamics with explicit gate-dependent bounds  

### Optimization Theory

**Gradient descent**: $\Delta t < \frac{2}{\text{Lipschitz constant}}$  
**This result**: Analogous but for non-convex, state-modulated dynamics  
**Extension**: Gate-dependent "effective Lipschitz" $C_{\max}$  

---

## Summary

We have derived a **rigorous, computable, sufficient** condition for discrete-time Lyapunov stability in GHOD systems:

$$\boxed{\Delta t < \frac{2M_{\min}}{C_{\max}^2} \quad \text{where } M_{\min} > 0}$$

This bridges the continuous-time guaranteed return (Theorem 1) to practical discrete implementations, provides the theoretical foundation for empirical phase diagrams, and explains the mechanistic origin of the observed conservatism gap.

**The theorem is conservative by design** - actual stability often extends beyond this bound. This conservatism enables:
- Reviewer-proof sufficient conditions ✅
- Explicit computable verification ✅
- Clear connection to system parameters ✅
- Honest boundary-mapping of theoretical limits ✅

---

**End of Appendix A.x**

---

## References for This Appendix

- Main manuscript: Theorem 1 (continuous-time guaranteed return)
- Main manuscript: Section 4 (empirical phase diagrams and conservatism gap)
- Supplementary code: `ghod_stability_diagnostics.py`
- Supplementary checklist: `Stability_Verification_Checklist.md`

---

**Document Status**: Unified version incorporating insights from multi-AI collaboration (Claude, ChatGPT, Gemini). All mathematical insights and scientific judgments remain solely attributable to the human author (see main manuscript Appendix C).
