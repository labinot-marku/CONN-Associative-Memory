# Supplementary Materials
## Gated Higher-Order Dynamics with Guaranteed Return

**Author**: Labinot Marku, M.D.  
**Preprint DOI**: [10.13140/RG.2.2.23260.24962](https://doi.org/10.13140/RG.2.2.23260.24962)  
**Preprint**: [ResearchGate](https://www.researchgate.net/profile/Labinot-Marku)  
**Main Repository**: [GitHub](https://github.com/labinot-marku/CONN-Associative-Memory/tree/main/gated-higher-order-dynamics)

---

## Overview

This folder contains mathematical analysis and reproducible code supporting the preprint:

**"Gated Higher-Order Dynamics with Guaranteed Return"**

The supplementary materials provide:

- Explicit discrete-time Lyapunov analysis  
- Conservative sufficient stability bounds  
- Computational diagnostics  
- Reproducibility protocols  
- Empirical illustration of conservatism gaps  

The analysis is intentionally conservative and local in scope.  
It establishes **sufficient** (not necessary) conditions for discrete Lyapunov decrease within a bounded region.

No global or necessary stability claims are made.

**Note**: Structural confinement mechanisms are under investigation. Current bounds are conservative sufficient conditions; empirical observations suggest actual stability regions may be substantially larger.

---

## Contents

### 1. Appendix A.x — Unified Discrete Stability Analysis

**File**: [Appendix_Ax_Unified_Discrete_Stability.md](Appendix_Ax_Unified_Discrete_Stability.md)

This document provides:

- Full derivation of discrete step-size bounds  
- Explicit computable sufficient conditions  
- Spectral envelope formulation  
- Monotonicity analysis of stability margin  
- Forward-invariance scope clarification  
- Discussion of conservatism mechanisms

**Main Sufficient Condition (Theorem A.1)**:

$$\Delta t < \frac{2 M_{\min}}{C_{\max}^2}, \quad \text{with } M_{\min} > 0$$

Where:

$$M_{\min} = a_{\min} + 2\lambda u_{\min} - \frac{u_{\max}}{2}\rho_{\max}$$

$$C_{\max} = a + 2\lambda u_{\max} + \frac{u_{\max}}{2}\rho_{\max}$$

These quantities are computed from:

- the anchor spectrum (`a_min`, `a`)  
- gate bounds (`u_min`, `u_max`)  
- a sampled Hessian spectral envelope (`rho_max`) over a bounded region

The result guarantees discrete Lyapunov decrease inside the chosen bounded region under the stated assumptions.

---

### 2. Stability Diagnostics Code

**File**: [ghod_stability_diagnostics.py](ghod_stability_diagnostics.py)

This module provides:

- Computation of $M_{\min}$, $C_{\max}$, and $\Delta t$ bounds  
- Monte Carlo estimation of $\rho_{\max}$  
- Empirical Lyapunov descent verification  
- Diagnostic summaries

**Example usage**:
```python
from ghod_stability_diagnostics import stability_diagnostics

results = stability_diagnostics(
    A=A,
    T=T,
    lam=0.4,
    u_min=0.0,
    u_max=1.0,
    dt=0.01,
    R=5.0,
    n_samples=200
)

print(results["dt_bound"])
```

The diagnostics illustrate the difference between conservative worst-case envelopes and realized trajectory curvature.

---

### 3. Stability Verification Checklist

**File**: [Stability_Verification_Checklist.md](Stability_Verification_Checklist.md)

Step-by-step protocol for:

- Estimating spectral quantities  
- Computing sufficient discrete bounds  
- Verifying empirical descent  
- Reporting results transparently

Intended for replication and independent validation.

---

### 4. Reproducible Experiment Framework (Optional)

**File**: [gated_dynamics_reproducible.py](gated_dynamics_reproducible.py) *(if available)*

Provides:

- Multi-seed statistical evaluation  
- Confidence interval reporting  
- Dimension sweeps  
- CSV export

Recommended for characterizing empirical behavior beyond single-trial demonstrations.

---

## Mathematical Context

### Conservative Nature of the Discrete Bounds

The discrete theorem provides **sufficient conditions only**.

Conservatism arises from:

1. Use of worst-case Hessian spectral envelope over a bounded ball  
2. Uniform (trajectory-independent) bounds  
3. Non-adaptive step-size constraints

Actual trajectories frequently remain confined to lower-curvature regions of state space.

This creates a gap between worst-case theoretical bounds and observed stable operation.

### Conservatism Gap (Empirical Observation)

In representative experiments (e.g., $N = 10$, $\lambda = 0.4$, $T_{\text{scale}} = 2.0$), sampled trials may exhibit order-of-magnitude gaps between:

- conservative sufficient bounds, and  
- empirically observed stable operation

These observations are:

- parameter-dependent  
- based on sampled trials  
- not claimed to be universal

Multi-seed statistical analysis is recommended for quantitative characterization.

### Ongoing Theoretical Work

The mechanisms underlying trajectory confinement to low-curvature regions are under active investigation. Potential explanations include:

- Adaptive damping through gate modulation
- Favorable alignment properties in random tensor realizations  
- Regularization-mediated trajectory geometry

Formal structural theorems characterizing these mechanisms are in development.

---

## Installation
```bash
# Clone repository
git clone https://github.com/labinot-marku/CONN-Associative-Memory.git
cd CONN-Associative-Memory/gated-higher-order-dynamics/supplementary

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install numpy scipy matplotlib
```

**Run diagnostics**:
```bash
python ghod_stability_diagnostics.py
```

---

## Relationship to Main Preprint

**The main preprint presents**:
- Conceptual framework  
- Continuous-time stability results  
- Empirical demonstrations

**The supplementary materials provide**:
- Explicit discrete sufficient bounds  
- Conservative envelope derivations  
- Reproducibility protocols

The appendix adopts a deliberately cautious and technical tone, clearly separating:
- proven sufficient conditions  
- empirical observations  
- exploratory architectural interpretation

---

## AI Assistance Disclosure

Portions of the mathematical exposition and code refinement were developed with the assistance of large language models (including Claude, ChatGPT, and Gemini) used as interactive technical tools.

All derivations, parameter choices, interpretations, and final formulations were reviewed and approved by the author. Responsibility for the content rests solely with the author.

---

## Citation

If you use these materials, please cite:
```bibtex
@article{marku2026gated,
  title={Gated Higher-Order Dynamics with Guaranteed Return},
  author={Marku, Labinot},
  journal={ResearchGate Preprint},
  year={2026},
  doi={10.13140/RG.2.2.23260.24962}
}
```

---

## License

MIT License — see main repository for details.

---

## Contact

For questions, discussion, or collaboration inquiries:
- Open a [GitHub issue](https://github.com/labinot-marku/CONN-Associative-Memory/issues)
- Contact via [ResearchGate](https://www.researchgate.net/profile/Labinot-Marku)

---

**Last Updated**: February 2026  
**Version**: 1.0 (Initial supplementary release)
```

---

## 🔧 **KEY FIXES APPLIED**

### **What I Changed**:

1. ✅ **File links** - Changed from `./filename` to just `filename` (relative links)
   - `[Appendix_Ax_Unified_Discrete_Stability.md](Appendix_Ax_Unified_Discrete_Stability.md)`
   - `[ghod_stability_diagnostics.py](ghod_stability_diagnostics.py)`
   - `[Stability_Verification_Checklist.md](Stability_Verification_Checklist.md)`

2. ✅ **Math formatting** - Restored `$$` for display equations

3. ✅ **Inline math** - Restored `$` for inline math (like `$M_{\min}$`)

4. ✅ **Section formatting** - Cleaned up bullets and structure

---

## 📋 **HOW TO UPDATE**

1. **Go to your README on GitHub**:
   - Navigate to: https://github.com/labinot-marku/CONN-Associative-Memory/tree/main/gated-higher-order-dynamics/supplementary
   - Click on `README.md`

2. **Click pencil icon** (Edit this file)

3. **Select all and delete** current content

4. **Paste the entire corrected version** above

5. **Commit changes**:
```
   Fix file links to use relative paths
```

6. **Extended description**:
```
   Updated all file references to use proper relative markdown links.
   Restored LaTeX math formatting for proper equation rendering.
