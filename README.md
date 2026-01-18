# CONN Validation Suite – Research Implementation

**Coherence Oscillatory Neural Networks (CONN)**

[![DOI](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.21347.00801-blue)](https://doi.org/10.13140/RG.2.2.21347.00801)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

---

## 📄 Manuscript Reference

**Title:** Topological Phase Constraints and Amplitude Dynamics Improve Associative Memory in Oscillatory Neural Networks

**Author:** Labinot Marku, M.D.  
**Institution:** KRH Klinikum Nordstadt Hannover, Germany  
**Date:** December 21, 2025  
**DOI:** [10.13140/RG.2.2.21347.00801](https://doi.org/10.13140/RG.2.2.21347.00801)  
**AI Collaboration:** Claude (Anthropic), ChatGPT (OpenAI)

---

## ⚠️ Important: Hyperparameter Configuration

This repository contains a **valid exploratory implementation** of CONN dynamics at **λ=4.0** (coherence strength).

The **published paper** subsequently found **λ=1.0 to be optimal** through systematic grid search validation (36 configurations tested).

### Current Repository Configuration (λ=4.0)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **λ** (lambda_coh) | **4.0** | Coherence strength (suboptimal) |
| β (beta) | 0.5 | Amplitude regularization |
| η_φ (eta_phi) | 0.005 | Phase learning rate |
| η_A (eta_A) | 0.03 | Amplitude learning rate |
| Steps | 150 | Integration timesteps |

**Expected results with λ=4.0:**
- Capacity: α ≈ 0.25 at N=32 (8 patterns for 32 neurons, suboptimal)
- Ablation: Minimal separation between conditions

### Paper-Validated Optimal Configuration (λ=1.0)

**From paper Table S1 (page 3), Hyperparameter Optimization (page 5-6):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **λ** (lambda_coh) | **1.0** | Coherence strength (OPTIMAL) |
| β (beta) | 0.5 | Amplitude regularization |
| η_φ (eta_phi) | 0.005 | Phase learning rate |
| η_A (eta_A) | 0.03 | Amplitude learning rate |
| Steps | 150 | Integration timesteps |

**Expected results with λ=1.0:**
- Capacity: α ≈ 0.375 at N=32 (**+50% vs λ=4.0**, 12 patterns for 32 neurons)
- Ablation shows clear separation:
  - Full CONN: 85.3%
  - No λ: 72.1% (-13.2 pp)
  - No amplitude: 80.9% (-4.4 pp)
  - Baseline: 68.5%

### 🔧 How to Use Optimal Parameters

**To reproduce published paper results, modify line 61 in the code:**

```python
# In Config class:
LAMBDA = 1.0   # Change from 4.0 to 1.0 (OPTIMAL from paper)
BETA = 0.5     # Keep as-is
```

**Performance comparison:**

| λ value | N=32 Capacity | Status | Reference |
|---------|---------------|--------|-----------|
| 1.0 | α=0.375 (M=12) | ✅ **Optimal** | Paper Table S1 |
| 2.0 | α=0.344 (M=11) | ⚠️ Acceptable | Paper Table S1 |
| 4.0 | α=0.250 (M=8) | ❌ **Suboptimal (-33%)** | Current repo |

**From paper (page 6):** *"λ=1.0 emerged as optimal, yielding 50% higher capacity at N=32 compared to λ=4.0"*

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/labinot-marku/CONN-Associative-Memory.git
cd CONN-Associative-Memory
pip install -r requirements.txt
```

### Run Validation

**Main implementation:** [`CONN_VALIDATION_V4_FINAL.py`](CONN_VALIDATION_V4_FINAL.py)

```bash
# Full validation suite (uses current λ=4.0)
python CONN_VALIDATION_V4_FINAL.py

# Quick test (N=32 only)
python CONN_VALIDATION_V4_FINAL.py --quick

# Single experiments
python CONN_VALIDATION_V4_FINAL.py --experiment capacity
python CONN_VALIDATION_V4_FINAL.py --experiment noise
python CONN_VALIDATION_V4_FINAL.py --experiment ablation
```

**Note:** To get paper-validated results, first change λ to 1.0 (see above).

---

## 📊 Expected Results

### With Published Paper Parameters (λ=1.0) ✅

**Validated claims from manuscript:**

1. **Capacity Improvement:** ~1.38× average over Hopfield baseline
   - **Note:** Capacity α = M/N (patterns per neuron)
   - N=32: α=0.375 (M=12) vs Hopfield α=0.344 (M=11) → 1.09×
   - N=64: α=0.359 (M=23) vs Hopfield α=0.266 (M=17) → 1.35×
   - N=128: α=0.383 (M=49) vs Hopfield α=0.227 (M=29) → 1.69×

2. **Noise Robustness:** ≥80% recall up to 30% phase-flip noise

3. **Ablation Study (N=32, M=8):**
   - Full CONN: **85.3%**
   - No λ (λ=0): **72.1%** (-13.2 pp)
   - No amplitude: **80.9%** (-4.4 pp)
   - Baseline (Hopfield): **68.5%**

**Status:** ✅ Validated across 30 trials per configuration – use these numbers for citations

### With Current Repository Parameters (λ=4.0) ⚠️

**Expected results:**
- Capacity: α ≈ 0.25 at N=32 (33% lower than optimal)
- Ablation: Minimal separation (coherence over-constrains dynamics)

**Status:** ⚠️ Valid exploratory implementation, but suboptimal for performance claims

**Key Point:** Always use the published validated estimates (1.38×) for scientific claims. To reproduce them, switch to λ=1.0.

---

## 🔬 Implementation Details

### CONN Dynamics

The implementation uses **gradient descent** on the energy function:

```
E(φ, A) = -½ Σᵢⱼ Jᵢⱼ Aᵢ Aⱼ cos(φᵢ - φⱼ) + λ Σⱼ Aⱼ² sin²(φⱼ)
          └──────────┬──────────┘   └────────┬──────────┘
           Hebbian coupling      Topological prior
```

### Phase Update

```python
# Coupling term: Σᵢ Jⱼᵢ Aᵢ sin(φᵢ - φⱼ)
phi_diff = phi[:, np.newaxis] - phi[np.newaxis, :]
coupling_matrix = J * A[np.newaxis, :] * np.sin(phi_diff)
coupling = np.sum(coupling_matrix, axis=1)

# Coherence term: -2λ A² sin(φ)cos(φ)
# Negative sign implements gradient descent (essential for convergence)
coherence = -2 * lambda_coh * A**2 * np.sin(phi) * np.cos(phi)

# Phase dynamics: dφ/dt = Aⱼ * coupling + coherence
dphi = A * coupling + coherence
phi = phi + eta_phi * dphi
phi = np.mod(phi, 2 * np.pi)
```

### Amplitude Update

```python
# Amplitude dynamics: dA/dt = -2λ A sin²(φ) - 2β(A - 1)
dA = -2 * lambda_coh * A * np.sin(phi)**2 - 2 * beta * (A - 1)
A = A + eta_A * dA
A = np.clip(A, 0.01, 2.0)  # Numerical stability
```

**Mathematical correctness:** Both coupling and coherence terms use **negative** signs to implement gradient descent on the energy function. This is essential for convergence to {0, π} phase attractors.

---

## 📁 Repository Structure

```
CONN-Associative-Memory/
├── CONN_VALIDATION_V4_FINAL.py   # Main implementation
├── README.md                     # This file
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── results/                      # Output CSVs
│   ├── capacity_results.csv
│   ├── noise_robustness.csv
│   └── ablation_results.csv
├── docs/                         # Documentation
│   ├── implementation_notes.md
│   └── reproduction_guide.md
└── papers/                       # Manuscript PDFs
    ├── CONN_main_paper.pdf
    └── CONN_supplementary.pdf
```

---

## 🔄 Reproducibility

### System Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- Matplotlib ≥ 3.3
- SciPy ≥ 1.6

### Reproducing Paper Results

1. **Modify hyperparameters** to optimal values (λ=1.0)
2. **Run full validation:** `python CONN_VALIDATION_V4_FINAL.py`
3. **Compare outputs** with published manuscript CSVs
4. **Expected runtime:** ~15-30 minutes for full validation

### Important Notes

- Current repository uses λ=4.0 (exploratory)
- Paper uses λ=1.0 (optimal, validated)
- Both implementations are mathematically correct
- Performance difference is due to hyperparameter choice only

---

## 📖 Citation

If you use this code, please cite:

```bibtex
@article{marku2025conn,
  title={Topological Phase Constraints and Amplitude Dynamics Improve 
         Associative Memory in Oscillatory Neural Networks},
  author={Marku, Labinot},
  journal={ResearchGate Preprint},
  year={2025},
  doi={10.13140/RG.2.2.21347.00801},
  note={Code: https://github.com/labinot-marku/CONN-Associative-Memory. 
        AI collaboration: Claude (Anthropic), ChatGPT (OpenAI)}
}
```

---

## 🤝 Contributing

This is a research implementation for validation and reproduction of published results.

**To contribute:**
1. Open an issue describing the problem
2. Provide a minimal reproducible example
3. Suggest a fix if you have one

**Note:** The primary purpose is validation of published claims, not active development.

---

## 👤 Author & Contact

**Labinot Marku, M.D.**  
Department of Neurosurgery  
KRH Klinikum Nordstadt Hannover, Germany

📧 Email: labinot.marku@krh.de  
🔬 ResearchGate: [Profile](https://www.researchgate.net/profile/Labinot-Marku)  
💻 GitHub: [Issues](https://github.com/labinot-marku/CONN-Associative-Memory/issues)

### AI Collaboration Acknowledgment

This research involved significant collaboration with AI assistants:
- **Claude** (Anthropic): Implementation, debugging, documentation
- **ChatGPT** (OpenAI): Validation, optimization, analysis

All AI contributions are fully disclosed in the manuscript acknowledgments.

---

## 📜 License

MIT License – see [LICENSE](LICENSE) file for details.

**Academic use:** Free for research and educational purposes  
**Commercial use:** Contact author for licensing

---

## ❓ FAQ

### Q: Why does the code use λ=4.0 instead of the optimal λ=1.0?

**A:** This repository represents exploratory research at λ=4.0. The published paper subsequently optimized hyperparameters through systematic grid search and found λ=1.0 to be optimal. Both implementations are mathematically valid—they differ only in parameter choice.

**Note:** "Quick mode" (`--quick` flag) reduces trial counts and network sizes for faster testing—it does not change the λ value. To use optimal parameters, you must manually modify the code.

### Q: How do I reproduce the exact paper results?

**A:** Change line 61 from `LAMBDA = 4.0` to `LAMBDA = 1.0`, then run the validation suite. All other parameters are already correct.

### Q: Are the dynamics mathematically correct?

**A:** Yes. The implementation correctly uses gradient descent with negative signs in both coupling and coherence terms. This has been verified against the paper's mathematical formulation.

### Q: What's the difference between "exploratory" and "validated" results?

**A:** 
- **Exploratory (λ=4.0):** This repository's current configuration, showing suboptimal capacity
- **Validated (λ=1.0):** Paper's optimized configuration with rigorous 30-trial validation

Both are scientifically valid explorations of the CONN mechanism.

### Q: Can I use this code for benchmarking?

**A:** Yes, but use λ=1.0 (optimal) for fair comparison. The current λ=4.0 configuration is 33% below optimal capacity.

---

**Last Updated:** January 18, 2026  
**Version:** 3.0 (Corrected documentation with accurate hyperparameter guidance)

---

## 🔍 Version History

- **v1.0** (Dec 2025): Initial implementation with λ=4.0
- **v2.0** (Jan 2026): Added validation suite and documentation
- **v3.0** (Jan 2026): **Corrected README** with accurate hyperparameter guidance and removal of misleading "bug fix" narrative
