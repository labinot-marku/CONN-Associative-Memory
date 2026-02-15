# Discrete Stability Verification Checklist

**Supplementary Material for**: "Gated Higher-Order Dynamics with Guaranteed Return"  
**Author**: Labinot Marku, M.D.  
**DOI**: 10.13140/RG.2.2.23260.24962

---

## Purpose

This checklist enables reproducible verification of the discrete Lyapunov stability bound from Appendix A.x.

Reviewers and practitioners can use this to:
1. Verify theoretical predictions
2. Reproduce empirical results
3. Design new experiments with guaranteed stability

---

## Required Measurements

### 1. Linear Anchor Statistics

**Compute**:
- $a_{\min} = \lambda_{\min}(A)$ (smallest eigenvalue)
- $a = \|A\|$ (operator norm / largest singular value)

**Method**:
```python
import numpy as np
eigvals = np.linalg.eigvalsh(A)
a_min = eigvals.min()
a = np.linalg.norm(A, 2)
```

---

### 2. Cubic Hessian Envelope

**Estimate**: $\rho_{\max} = \sup_{\|x\| \le R} \rho(H(x))$

**Method**: Sample random directions on sphere of radius $R$

```python
def estimate_rho_max(T, R, n_samples=200):
    N = T.shape[0]
    rho_max = 0.0
    for _ in range(n_samples):
        x = np.random.randn(N)
        x = R * x / np.linalg.norm(x)
        H = cubic_hessian(x, T)
        rho = max(abs(np.linalg.eigvals(H)))
        rho_max = max(rho_max, rho)
    return rho_max
```

**Report**:
- Sampling procedure (number of samples, radius $R$)
- Estimated $\rho_{\max}$
- Confidence interval (if applicable)

---

### 3. Gate Statistics

**Measure during simulation**:
- $u_{\min}$ (minimum gate value observed)
- $u_{\max}$ (maximum gate value observed)

**Method**:
```python
u_values = []
# During simulation loop:
u_values.append(u_k)

# After simulation:
u_min = min(u_values)
u_max = max(u_values)
```

---

### 4. Derived Quantities

**Compute**:

$$M_{\min} = a_{\min} + 2\lambda u_{\min} - \frac{u_{\max}}{2}\rho_{\max}$$

$$C_{\max} = a + 2\lambda u_{\max} + \frac{u_{\max}}{2}\rho_{\max}$$

$$\Delta t_{\text{bound}} = \frac{2M_{\min}}{C_{\max}^2}$$

**Code**:
```python
M_min = a_min + 2*lam*u_min - 0.5*u_max*rho_max
C_max = a + 2*lam*u_max + 0.5*u_max*rho_max

if M_min > 0:
    dt_bound = 2*M_min / (C_max**2)
else:
    dt_bound = 0.0  # Condition not satisfied
```

---

### 5. Stability Condition

**Verify**:

$$\Delta t < \Delta t_{\text{bound}}$$

**Report**:
- Chosen $\Delta t$
- Theoretical bound $\Delta t_{\text{bound}}$
- Ratio $\Delta t / \Delta t_{\text{bound}}$
- Whether condition is satisfied (Yes/No)

---

### 6. Empirical Validation

**Measure**:
- Plot $V(x_k) = \frac{1}{2}\|x_k\|^2$ over time
- Verify monotone decrease
- Compute actual contraction rate

**Code**:
```python
V_values = []
for x_k in trajectory:
    V_values.append(0.5 * np.dot(x_k, x_k))

# Check monotonicity
is_decreasing = all(V_values[i+1] <= V_values[i] 
                    for i in range(len(V_values)-1))

# Compute empirical contraction rate
empirical_gamma = np.mean([V_values[i] - V_values[i+1] 
                           for i in range(len(V_values)-1)])
```

**Report**:
- Whether $V$ decreases monotonically
- Empirical vs theoretical contraction rate
- Plot of $V(x_k)$ vs iteration

---

## Complete Reporting Template

### Experimental Parameters
- Dimension: $N = $ ___
- Regularization: $\lambda = $ ___
- Step size: $\Delta t = $ ___
- Ball radius: $R = $ ___

### Measured Statistics
- $a_{\min} = $ ___
- $a = \|A\| = $ ___
- $\rho_{\max} = $ ___ (from ___ samples)
- $u_{\min} = $ ___
- $u_{\max} = $ ___

### Derived Bounds
- $M_{\min} = $ ___
- $C_{\max} = $ ___
- $\Delta t_{\text{bound}} = $ ___
- Ratio $\Delta t / \Delta t_{\text{bound}} = $ ___

### Stability Verification
- [ ] $M_{\min} > 0$? (Yes/No)
- [ ] $\Delta t < \Delta t_{\text{bound}}$? (Yes/No)
- [ ] $V$ decreases monotonically? (Yes/No)

### Empirical Observations
- Theoretical $\gamma = $ ___
- Empirical $\gamma_{\text{obs}} = $ ___
- Maximum $\|x_k\|$ observed: ___
- Final $\|x_k\|$: ___

---

## Example Report

**Experimental Parameters**
- $N = 32$
- $\lambda = 0.4$
- $\Delta t = 0.01$
- $R = 5.0$

**Measured Statistics**
- $a_{\min} = 0.85$
- $a = 1.12$
- $\rho_{\max} = 45.3$ (from 200 samples)
- $u_{\min} = 0.0$
- $u_{\max} = 1.0$

**Derived Bounds**
- $M_{\min} = 0.85 + 0 - 22.65 = -21.8$ ❌
- $C_{\max} = 1.12 + 0.8 + 22.65 = 24.57$
- Condition NOT satisfied (requires larger $\lambda$ or smaller $u_{\max}$)

**Interpretation**: Conservative bound predicts instability, but empirical trajectory may remain stable due to trajectory confinement to low-curvature regions (conservatism gap phenomenon).

---

## Notes for Reproducibility

1. **Random seed**: Always report random seed for reproducibility
2. **Sampling**: Use sufficient samples ($n \geq 200$) for $\rho_{\max}$ estimation
3. **Precision**: Report values to at least 2 significant figures
4. **Plots**: Include $V(x_k)$ plot in supplementary materials
5. **Code**: Link to implementation (GitHub repository)

---

**End of Checklist**
