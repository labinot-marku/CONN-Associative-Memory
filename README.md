# CONN-Associative-Memory – Research Repository
### Coherence Oscillatory Neural Networks (CONN) + Gated Higher-Order Dynamics (GHOD)

![DOI](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.21347.00801-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)

**Author:** Labinot Marku, M.D. — Department of Neurosurgery, KRH Klinikum Nordstadt Hannover, Germany  
**AI Collaboration:** Claude (Anthropic), ChatGPT (OpenAI), Gemini (Google), Grok (xAI)

---

## 🔬 Research Contents

This repository contains two connected research programmes:

| Framework | Status | Preprint | Code |
|-----------|--------|----------|------|
| **GHOD** — Gated Higher-Order Dynamics | Active (2026) | [See below](#-ghod-gated-higher-order-dynamics) | `/gated-higher-order-dynamics` |
| **CONN** — Coherence Oscillatory Neural Networks | **Revised (v2, May 2026)** | [See below](#-conn-coherence-oscillatory-neural-networks) | `/validation` |

---

## 📐 GHOD: Gated Higher-Order Dynamics

The GHOD framework provides a rigorous mathematical theory of stability in systems where higher-order interactions are gated — transiently activated to enhance expressivity, then decayed to ensure return to a stable anchor.

### Preprints

**Complete merged preprint (April 2026)** — current canonical version:

> Marku, L. (2026). *Metric-Induced Stability Beyond Spectral Criteria in Gated Higher-Order Dynamical Systems.* ResearchGate. DOI: [10.13140/RG.2.2.15913.15209](https://doi.org/10.13140/RG.2.2.15913.15209)

Contains: Theorem 1 (Metric-Nonlinear Stabilisation), Theorem HG (Metric-Hypocoercivity), Theorem H-LSI (Log-Sobolev upgrade), Theorem S (Spectral Resolution), Lemma CR (explicit confinement radius R\*), Proposition SC (structural conditions for Assumption A5), four corrected hypocoercivity errors, and engagement with Goto et al. 2025 (arXiv:2512.13859).

**Original GHOD framework (February 2026)** with appendices:

> Marku, L. (2026). *Gated Higher-Order Dynamics with Guaranteed Return: An Exploratory Framework for the Capacity-Stability Trade-Off.* ResearchGate / Zenodo. DOI: [10.13140/RG.2.2.23260.24962](https://doi.org/10.13140/RG.2.2.23260.24962)

Supplementary materials published alongside the original preprint:
- **Appendix A3** — Discrete-Time GHOD Stability Diagnostics (corrected version): discrete Lyapunov descent theorem, corrected gate dynamics, explicit step-size bounds, reproducible 50-trial sweep results (conservatism ratio ~164×, 36/50 convergence). [ResearchGate](https://www.researchgate.net/)
- **Appendix D** — Relationship to Maximum Entropy Neural Population Models: connects the GHOD framework to maximum entropy principles in neural population coding. [ResearchGate](https://www.researchgate.net/)
- **Appendix E** — Neurobiological and Clinical Implications of Gated Higher-Order Dynamics: connects the framework to clinical neuroscience and seizure prediction. [ResearchGate](https://www.researchgate.net/)

### Core Mathematical Result

The fundamental observation is an exact decomposition:

```
ẋ = -G(x,t)·x - g(t)·r(x)
```

where `G(x,t) = A + g(t)·H(x)` is the effective quadratic operator and `r(x) = ∇Φ(x) - H(x)·x` is the nonlinear remainder. Global stability emerges from the competition between local quadratic behaviour (governed by G) and global nonlinear dissipation (governed by r(x)) — even when G has negative eigenvalues.

### Key Results

- **Theorem 1:** Global boundedness holds under superquadratic restoring condition on r(x), even when local eigenvalues of J = -G are positive
- **Theorem HG:** Exponential entropy decay persists for time-varying g(t) under the unified condition `||g - ḡ||∞ · L < 2ε√μ_LS`
- **Theorem S:** The transport-corrected operator `J_eff = J + A_G` satisfies `λ_GHOD < 0`, explaining why locally positive eigenvalues do not produce global instability
- **Lemma CR:** Explicit confinement radius `R* = (δ/C₂)^{1/δ}` — computable without simulation

### GHOD Code

```
gated-higher-order-dynamics/
├── gated_dynamics.py                  # Core GHOD simulation (corrected gate dynamics)
├── supplementary/
│   └── ghod_stability_diagnostics.py  # Discrete stability diagnostics, Lyapunov validation
└── README.md
```

Three stability regimes emerge from the balance between dissipation margin `δ = λ_min(A) - g_max·L` and restoring amplitude `C₂ = g_max·β₀`:

| Regime | Condition | Behaviour |
|--------|-----------|-----------|
| **A** | δ >> 0, g_max small | Classical stability, no gap |
| **B** | δ > 0 small | Confined exploration, conservatism gap (100×–10,000×) |
| **C** | δ ≤ 0 | Governor Inequality violated, divergence |

**Reproducible result (Regime B):** 37/50 trials converge despite Governor Inequality violations averaging -214.09; trajectory curvature 10,770× smaller than worst-case predictions.

### Related Work

Independent convergent evidence: Goto, Lopez Rios, Scholz, Vaikuntanathan (2025). *Neuromodulation-inspired gated associative memory networks.* arXiv:2512.13859. Using DMFT and many-body simulations at N=1000, they show multiplicative neuromodulatory gating reorganises attractor structure and bypasses the classical spin-glass transition — convergent with the core GHOD insight that gating decouples capacity from stability.

---

## 🔵 CONN: Coherence Oscillatory Neural Networks (v2 — Revised May 2026)

> ⚠️ **v2 Revision Notice:** This section reflects the corrected v2 preprint (May 2026), which fixes implementation errors in the original v1 (December 2025). See Section 9 (Errata) of the v2 preprint for full details. The v1 validation code is archived in `validation/archive/`.

### Preprint

> Marku, L. (2025/2026). *Phase Coherence Constraints and Amplitude Dynamics Improve Associative Memory in Oscillatory Neural Networks.* ResearchGate. DOI: [10.13140/RG.2.2.21347.00801](https://doi.org/10.13140/RG.2.2.21347.00801)
>
> v2 PDF available in `preprint/CONN_preprint_v2_final.pdf`

### Key Results (v2)

| N    | Hopfield M | Hopfield α | CONN M | CONN α | Ratio  | Trials |
|------|-----------|-----------|--------|--------|--------|--------|
| 32   | 6         | 0.188     | 6      | 0.188  | 1.00×  | 50     |
| 64   | 10        | 0.156     | 13     | 0.203  | 1.30×  | 60     |
| 128  | 19        | 0.148     | 25     | 0.195  | 1.32×  | 80     |
| 256  | 35        | 0.137     | 49     | 0.191  | 1.40×  | 30     |
| 512  | 65        | 0.127     | 94     | 0.184  | 1.45×  | 30     |
| 1024 | 124       | 0.121     | 195    | 0.190  | 1.57×  | 30     |
| 2048 | 238       | 0.116     | 390    | 0.190  | 1.64×  | 30     |

**CONN α ≈ 0.19** remains stable across all N ≥ 64 under the tested protocol. The data suggest an asymptotic improvement near **1.38×** (CONN α ≈ 0.19 vs theoretical Hopfield α ≈ 0.138). At large N (≥1024), part of the ratio increase reflects the synchronous Hopfield baseline underperforming below its theoretical capacity.

CONN also exhibits **up to 8× lower recall variance** near capacity at large N.

### Corrected Parameters (v2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| λ         | 0.5 (annealed 0 → 0.5) | Coherence strength |
| η_φ       | 0.02  | Phase learning rate |
| η_A       | 0.03  | Amplitude learning rate |
| β         | 1.0   | Amplitude regularization |
| Steps     | 400   | Integration steps |
| Noise     | 30% bit-flip + σ=0.5 jitter | Corruption protocol |

### Energy Function (v2)

```
E(φ, A) = −½ Σᵢⱼ Jᵢⱼ Aᵢ Aⱼ cos(φᵢ − φⱼ) + λ Σⱼ Aⱼ² sin²(φⱼ) + (β/2) Σⱼ (Aⱼ − 1)²
```

### Validation Scripts

All scripts require only **NumPy** and run on standard hardware.

| Script | Covers | Runtime |
|--------|--------|---------|
| `validation/CONN_VALIDATION_V5.py` | N=32, 64, 128 + noise + ablation | ~5 min |
| `validation/CONN_LARGE_SCALE_VALIDATION.py` | N=256, 512 | ~15-30 min |
| `validation/CONN_N1024_N2048_VALIDATION.py` | N=1024, 2048 | ~2-4 hours |

```bash
# Quick check
cd validation
python CONN_VALIDATION_V5.py --quick

# Full validation (reproduces Tables 1-3)
python CONN_VALIDATION_V5.py

# Large-scale
python CONN_LARGE_SCALE_VALIDATION.py
python CONN_N1024_N2048_VALIDATION.py
```

### Important Notes

**Noise model:** The Gaussian jitter (σ=0.5) is required because CONN's continuous dynamics produce zero gradients when phases are exactly at {0, π}. The jitter is small relative to the π-radian separation between states and does not provide information about which state is correct — it merely breaks the equilibrium degeneracy. Hopfield's discrete sign() operation is robust to this jitter.

**Capacity definition:** Capacity is defined operationally as the largest M where mean recall ≥ 80% under the tested noise protocol. This is an operational capacity and should not be conflated with the thermodynamic storage capacity of classical Hopfield networks.

**v1 code (archived):** The original `CONN_VALIDATION_V4_FINAL.py` is preserved in `validation/archive/` for transparency. It contained errors documented in the v2 errata: incorrect noise model (bit-flip only, causing zero dynamics), no real Hopfield baseline, λ=4.0 default, η_φ=0.005, 150 steps.

### v1 → v2 Corrections Summary

| Issue | v1 | v2 |
|-------|----|----|
| Noise model | Bit-flip only (dynamics = zero) | Bit-flip + σ=0.5 jitter |
| Hopfield baseline | CONN with λ=0 (not Hopfield) | True sign(J·s) updates |
| λ | 4.0 constant | 0.5 annealed |
| η_φ | 0.005 | 0.02 |
| Steps | 150 | 400 |
| Terminology | "Topological prior" | "Phase coherence prior" |
| Amplitude dynamics | "Gradient descent on E" | "Confidence-collapse rule" |
| Network sizes | N=32, 64, 128 | N=32 to N=2048 |

---

## 📁 Repository Structure

```
CONN-Associative-Memory/
├── README.md
├── validation/
│   ├── CONN_VALIDATION_V5.py              # v2: N=32-128 + noise + ablation
│   ├── CONN_LARGE_SCALE_VALIDATION.py     # v2: N=256, 512
│   ├── CONN_N1024_N2048_VALIDATION.py     # v2: N=1024, 2048
│   └── archive/
│       └── CONN_VALIDATION_V4_FINAL.py    # v1 (superseded)
├── results/
│   ├── capacity_summary.csv
│   ├── capacity_detail.csv
│   ├── noise_robustness.csv
│   └── ablation.csv
├── figures/
│   ├── CONN_Figure1_scaling.png
│   ├── CONN_Figure2_variance.png
│   └── CONN_Figure3_N2048_detail.png
├── preprint/
│   └── CONN_preprint_v2_final.pdf
├── gated-higher-order-dynamics/           # GHOD (separate project)
│   ├── gated_dynamics.py
│   ├── supplementary/
│   └── README.md
└── [legacy PDFs from v1]
```

---

## 📋 Citation

**CONN preprint (cite this for CONN):**

```bibtex
@article{marku2025conn,
  title={Phase Coherence Constraints and Amplitude Dynamics Improve 
         Associative Memory in Oscillatory Neural Networks},
  author={Marku, Labinot},
  year={2025},
  note={Revised May 2026 (v2)},
  doi={10.13140/RG.2.2.21347.00801}
}
```

**GHOD complete preprint (cite this for the full framework):**

```bibtex
@article{marku2026ghod_merged,
  title={Metric-Induced Stability Beyond Spectral Criteria in 
         Gated Higher-Order Dynamical Systems},
  author={Marku, Labinot},
  journal={ResearchGate Preprint},
  year={2026},
  doi={10.13140/RG.2.2.28366.63048}
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

---

## 👤 Author & Contact

**Labinot Marku, M.D.**  
Department of Neurosurgery, KRH Klinikum Nordstadt Hannover, Germany

📧 labinot.marku@krh.de  
🔬 [ResearchGate Profile](https://www.researchgate.net/profile/Labinot-Marku)

**AI Collaboration:** Claude (Anthropic), ChatGPT (OpenAI), Gemini (Google), and Grok (xAI) provided formalization assistance, mathematical validation, implementation support, systematic parameter search, and identification of implementation errors in v1. All theoretical insights, experimental designs, and scientific judgments are the author's. AI contributions are fully disclosed in manuscript acknowledgments.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

*Last updated: May 2026*
