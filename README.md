# CONN-Associative-Memory – Research Repository

**Coherence Oscillatory Neural Networks (CONN) + Gated Higher-Order Dynamics (GHOD)**

[![DOI](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.21347.00801-blue)](https://doi.org/10.13140/RG.2.2.21347.00801)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

**Author:** Labinot Marku, M.D. — Department of Neurosurgery, KRH Klinikum Nordstadt Hannover, Germany
**AI Collaboration:** Claude (Anthropic), ChatGPT (OpenAI), Gemini (Google)

---

## 🔬 Research Contents

This repository contains two connected research programmes:

| Framework | Status | Preprint | Code |
|-----------|--------|----------|------|
| **GHOD** — Gated Higher-Order Dynamics | Active (2026) | [See below](#-ghod-gated-higher-order-dynamics) | [`/gated-higher-order-dynamics`](gated-higher-order-dynamics/) |
| **CONN** — Coherence Oscillatory Neural Networks | Complete (2025) | [See below](#-conn-coherence-oscillatory-neural-networks) | Root directory |

---

## 📐 GHOD: Gated Higher-Order Dynamics

The GHOD framework provides a rigorous mathematical theory of stability in systems where higher-order interactions are gated — transiently activated to enhance expressivity, then decayed to ensure return to a stable anchor.

### Preprints

**Complete merged preprint (March 2026) — current canonical version:**
> Marku, L. (2026). *Metric-Induced Stability Beyond Spectral Criteria in Gated Higher-Order Dynamical Systems.*
> ResearchGate. [DOI: 10.13140/RG.2.2.15154.57288](https://doi.org/10.13140/RG.2.2.15154.57288)

Contains: Theorem 1 (Metric-Nonlinear Stabilisation), Theorem HG (Metric-Hypocoercivity), Theorem H-LSI (Log-Sobolev upgrade), Theorem S (Spectral Resolution), Lemma CR (explicit confinement radius R*), Proposition SC (structural conditions for Assumption A5), four corrected hypocoercivity errors, and engagement with Goto et al. 2025 (arXiv:2512.13859).

**Original GHOD framework (February 2026):**
> Marku, L. (2026). *Gated Higher-Order Dynamics with Guaranteed Return: An Exploratory Framework for the Capacity-Stability Trade-Off.*
> ResearchGate / Zenodo. [DOI: 10.13140/RG.2.2.23260.24962](https://doi.org/10.13140/RG.2.2.23260.24962)

### Core Mathematical Result

The fundamental observation is an exact decomposition:

```
ẋ = -G(x,t)·x - g(t)·r(x)
```

where `G(x,t) = A + g(t)·H(x)` is the effective quadratic operator and `r(x) = ∇Φ(x) - H(x)·x` is the nonlinear remainder. Global stability emerges from the competition between local quadratic behaviour (governed by G) and global nonlinear dissipation (governed by r(x)) — even when G has negative eigenvalues.

### Key Results

- **Theorem 1**: Global boundedness holds under superquadratic restoring condition on r(x), even when local eigenvalues of J = -G are positive
- **Theorem HG**: Exponential entropy decay persists for time-varying g(t) under the unified condition `||g - ḡ||∞ · L < 2ε√μ_LS`
- **Theorem S**: The transport-corrected operator `J_eff = J + A_G` satisfies `λ_GHOD < 0`, explaining why locally positive eigenvalues do not produce global instability
- **Lemma CR**: Explicit confinement radius `R* = (δ/C₂)^{1/δ}` — computable without simulation

### GHOD Code

```
gated-higher-order-dynamics/
├── gated_dynamics.py                  # Core GHOD simulation (corrected gate dynamics)
├── supplementary/
│   └── ghod_stability_diagnostics.py  # Discrete stability diagnostics, Lyapunov validation
└── README.md
```

**Three stability regimes** emerge from the balance between dissipation margin δ = λ_min(A) - g_max·L and restoring amplitude C₂ = g_max·β₀:

| Regime | Condition | Behaviour |
|--------|-----------|-----------|
| A | δ >> 0, g_max small | Classical stability, no gap |
| B | δ > 0 small | Confined exploration, conservatism gap (100×–10,000×) |
| C | δ ≤ 0 | Governor Inequality violated, divergence |

**Reproducible result (Regime B):** 37/50 trials converge despite Governor Inequality violations averaging -214.09; trajectory curvature 10,770× smaller than worst-case predictions.

### Related Work

Independent convergent evidence: Goto, Lopez Rios, Scholz, Vaikuntanathan (2025). *Neuromodulation-inspired gated associative memory networks.* arXiv:2512.13859. Using DMFT and many-body simulations at N=1000, they show multiplicative neuromodulatory gating reorganises attractor structure and bypasses the classical spin-glass transition — convergent with the core GHOD insight that gating decouples capacity from stability.

---

## 🔵 CONN: Coherence Oscillatory Neural Networks

### Preprint

> Marku, L. (2025). *Topological Phase Constraints and Amplitude Dynamics Improve Associative Memory in Oscillatory Neural Networks.*
> ResearchGate. [DOI: 10.13140/RG.2.2.21347.00801](https://doi.org/10.13140/RG.2.2.21347.00801)

### ⚠️ Hyperparameter Configuration

The repository code uses **λ=4.0** (exploratory). The paper validates **λ=1.0** as optimal (50% higher capacity at N=32).

| λ value | N=32 Capacity | Status |
|---------|--------------|--------|
| 1.0 | α=0.375 (M=12) | ✅ Optimal — use for citations |
| 2.0 | α=0.344 (M=11) | ⚠️ Acceptable |
| 4.0 | α=0.250 (M=8) | ❌ Suboptimal (-33%) — current repo default |

To reproduce paper results, change line 61 in `CONN_VALIDATION_V4_FINAL.py`:
```python
LAMBDA = 1.0   # Change from 4.0 to 1.0 (optimal)
```

### Quick Start

```bash
git clone https://github.com/labinot-marku/CONN-Associative-Memory.git
cd CONN-Associative-Memory
pip install -r requirements.txt
python CONN_VALIDATION_V4_FINAL.py
```

### CONN Dynamics

Energy function:
```
E(φ, A) = -½ Σᵢⱼ Jᵢⱼ Aᵢ Aⱼ cos(φᵢ - φⱼ) + λ Σⱼ Aⱼ² sin²(φⱼ)
```

### Expected Results (λ=1.0, paper-validated)

- **Capacity:** ~1.38× average over Hopfield baseline (N=32: 1.09×, N=64: 1.35×, N=128: 1.69×)
- **Noise robustness:** ≥80% recall at 30% phase-flip noise
- **Ablation (N=32):** Full CONN 85.3% vs Baseline 68.5% (+16.8 pp)

---

## 📋 Citation

**GHOD complete preprint (cite this for the full framework):**
```bibtex
@article{marku2026ghod_merged,
  title={Metric-Induced Stability Beyond Spectral Criteria in 
         Gated Higher-Order Dynamical Systems},
  author={Marku, Labinot},
  journal={ResearchGate Preprint},
  year={2026},
  doi={10.13140/RG.2.2.15154.57288},
  note={Code: https://github.com/labinot-marku/CONN-Associative-Memory/
        tree/main/gated-higher-order-dynamics}
}
```

**Original GHOD preprint:**
```bibtex
@article{marku2026ghod,
  title={Gated Higher-Order Dynamics with Guaranteed Return},
  author={Marku, Labinot},
  journal={ResearchGate / Zenodo Preprint},
  year={2026},
  doi={10.13140/RG.2.2.23260.24962}
}
```

**CONN preprint:**
```bibtex
@article{marku2025conn,
  title={Topological Phase Constraints and Amplitude Dynamics Improve 
         Associative Memory in Oscillatory Neural Networks},
  author={Marku, Labinot},
  journal={ResearchGate Preprint},
  year={2025},
  doi={10.13140/RG.2.2.21347.00801},
  note={Code: https://github.com/labinot-marku/CONN-Associative-Memory}
}
```

---

## 👤 Author & Contact

**Labinot Marku, M.D.**
Department of Neurosurgery, KRH Klinikum Nordstadt Hannover, Germany

📧 labinot.marku@krh.de
🔬 [ResearchGate Profile](https://www.researchgate.net/profile/Labinot-Marku)

**AI Collaboration:** Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google) provided formalization assistance, mathematical validation, implementation support, and systematic parameter search. All theoretical insights, experimental designs, and scientific judgments are the author's. AI contributions are fully disclosed in manuscript acknowledgments.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Last updated: April 2026*
