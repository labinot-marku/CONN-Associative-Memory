# CONN Validation Suite - Correct Implementation

**Coherence Oscillatory Neural Networks (CONN)**

[![DOI](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.25288.99841-blue)](https://doi.org/10.13140/RG.2.2.25288.99841)

---

## ⚠️ CRITICAL IMPLEMENTATION NOTE

**This repository contains the CORRECTED implementation.**

During development, a sign error was discovered in the coherence gradient term:

- **WRONG (original bug):** `coherence = +2λ sin(φ)cos(φ)` ← Gradient **ascent** (diverges)
- **CORRECT (fixed):**  `coherence = -2λ sin(φ)cos(φ)` ← Gradient **descent** (converges)

All results in the manuscript use the corrected version. This is a standard debugging process in scientific software development.

---

## 📄 Manuscript Reference

**"Topological Phase Constraints and Amplitude Dynamics Improve Associative Memory in Oscillatory Neural Networks"**

- **Author:** Labinot Marku, M.D.
- **Institution:** KRH Klinikum Nordstadt Hannover, Germany
- **Date:** January 11, 2026
- **DOI:** [10.13140/RG.2.2.25288.99841](https://doi.org/10.13140/RG.2.2.25288.99841)
- **AI Collaboration:** Claude (Anthropic), ChatGPT (OpenAI)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/labinot-marku/CONN-Associative-Memory.git
cd CONN-Associative-Memory

# Install dependencies
pip install numpy matplotlib scipy
```

### Run Validation

```bash
# Full validation suite (reproduces all manuscript results)
python CONN_VALIDATION_V4_FINAL.py

# Quick test (N=32 only)
python CONN_VALIDATION_V4_FINAL.py --quick

# Single experiment
python CONN_VALIDATION_V4_FINAL.py --experiment capacity
python CONN_VALIDATION_V4_FINAL.py --experiment noise
python CONN_VALIDATION_V4_FINAL.py --experiment ablation
```

---

## 📊 Expected Results

### ⚠️ CRITICAL: Exploratory vs Validated Results

**This code provides EXPLORATORY validation of the CONN mechanism.**

Results at small N (e.g., N=32) in quick mode are **upper bounds** and should 
**NOT** be interpreted as rigorous capacity measurements.

---

### Published Claims (Validated - Use These) ✅

**From manuscript (DOI: 10.13140/RG.2.2.25288.99841):**

- **Capacity improvement: ~1.38×** (conservative estimate)
- Based on: Rigorous testing across multiple network sizes
- Status: ✅ **Validated - use for citations and claims**

---

### Repository Quick-Mode Results (Exploratory) ⚠️

**When running this code at N=32 in quick mode:**

- May show: **2-4× improvement**
- Status: ⚠️ **Exploratory upper bounds - NOT validated**
- Artifacts from:
  - Small network size (N=32 is non-asymptotic)
  - Quick-mode optimistic binary search
  - Limited trial counts
  - Theoretical Hopfield baseline (not experimentally matched)

**Do NOT use these numbers for performance claims.**

---

### Why the Difference?

The ablation study reveals the truth:

```
Full CONN:     89.6%
No λ:          89.6%
No Amplitude:  89.6%
Baseline:      89.6%
```

**All identical at M=8, N=32** → Task is too easy → No stress test → 
Any large capacity gain is an artifact, not a validated improvement.

**Key Point:** When all methods perform equally well, apparent capacity 
improvements are measurement artifacts, not real performance gains.

---

### The Scientific Truth

- **Paper's 1.38×** = Conservative, rigorous, validated ✅
- **Code's 4× at N=32** = Exploratory artifact, not validated ⚠️

**For any claims, use the published validated estimate (1.38×).**

---

### Experiment-Specific Expected Results

#### Experiment 1: Capacity Improvement
- **Published (validated):** ~1.38× improvement ✅
- **Repository (exploratory):** May show 2-4× at N=32 (artifact)
- **Use for claims:** Published value only

#### Experiment 2: Noise Robustness ✅
- Graceful degradation with increasing noise (validated)
- >80% recall at 30% noise for M=8
- This result is robust and validated

#### Experiment 3: Ablation Study
- At easy regime (M=8, N=32): Minimal separation (expected)
- Near capacity (M≈14-16): Component benefits emerge
- This demonstrates regime-dependent behavior

---

## 📋 Usage Guidelines

### ✅ DO Use This Code For:
- Exploring CONN mechanism
- Educational demonstrations
- Implementation verification
- Understanding gradient descent dynamics

### ❌ DO NOT Use This Code For:
- Claiming capacity improvements >1.5×
- Performance benchmarks
- Ignoring published conservative estimates
- Making numerical claims without validation

**For any performance claims, cite the published manuscript.**

---

## 🔬 Implementation Details

### Core CONN Dynamics (CORRECTED)

**Phase Update:**

```python
# Coupling term: Σᵢ Jⱼᵢ Aᵢ sin(φᵢ - φⱼ)
phi_diff = phi[:, np.newaxis] - phi[np.newaxis, :]  # (N, N) pairwise differences
coupling_matrix = J * A[np.newaxis, :] * np.sin(phi_diff)
coupling = np.sum(coupling_matrix, axis=1)

# *** CRITICAL FIX ***
# Coherence term: -2λ A² sin(φ)cos(φ)  [NEGATIVE for gradient descent]
coherence = -2 * lambda_coh * A**2 * np.sin(phi) * np.cos(phi)

# Phase dynamics
dphi = A * coupling + coherence
phi = phi + eta_phi * dphi
phi = np.mod(phi, 2 * np.pi)
```

**Amplitude Update:**

```python
# dA/dt = -2λ A sin²(φ)
dA = -2 * lambda_coh * A * np.sin(phi)**2
A = A + eta_A * dA
A = np.clip(A, 0.01, 2.0)
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lambda_coh` | 4.0 | Coherence strength |
| `eta_phi` | 0.005 | Phase learning rate |
| `eta_A` | 0.03 | Amplitude learning rate |
| `max_steps` | 150 | Integration steps |

---

## 🔬 Mathematical Foundation

### Energy Function

```
E(φ, A) = -½ Σᵢⱼ Jᵢⱼ Aᵢ Aⱼ cos(φᵢ - φⱼ) + λ Σⱼ Aⱼ² sin²(φⱼ)
          └─────────────┬──────────────┘   └────────┬──────────┘
                   Hebbian coupling         Topological prior
```

### Gradient Descent (CORRECT)

```
∂E/∂φⱼ = ½ Σᵢ Jᵢⱼ Aᵢ Aⱼ sin(φᵢ - φⱼ) + 2λ Aⱼ² sin(φⱼ)cos(φⱼ)

dφⱼ/dt = -∂E/∂φⱼ = -½ Σᵢ Jᵢⱼ Aᵢ Aⱼ sin(φᵢ - φⱼ) - 2λ Aⱼ² sin(φⱼ)cos(φⱼ)
         └─────────────────┬────────────────────┘   └────────────┬────────────┘
                      Coupling term                    Coherence term
                     (attracts to neighbors)         (attracts to {0,π})
```

**Both terms MUST be negative for gradient descent!**

---

## 📁 Repository Structure

```
CONN-Associative-Memory/
├── CONN_VALIDATION_V4_FINAL.py   # Main implementation (CORRECTED)
├── README.md                     # This file
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
│
├── results/                      # Output from validation
│   ├── capacity_results.csv
│   ├── noise_robustness.csv
│   └── ablation_results.csv
│
├── docs/                         # Documentation
│   ├── implementation_notes.md   # Detailed implementation guide
│   ├── bug_fix_history.md        # Sign error discovery and fix
│   └── reproduction_guide.md     # Step-by-step reproduction
│
└── papers/                       # Manuscript and supplementary
    ├── CONN_main_paper.pdf
    ├── CONN_supplementary.pdf
    └── code_availability.txt     # Citation instructions
```

---

## 🐛 Bug Fix History

### Sign Error in Coherence Gradient (Fixed)

**Discovery:** During validation testing, January 2026

**Problem:** Original implementation used **positive** sign for coherence term:
```python
coherence = +2 * lambda_coh * A**2 * np.sin(phi) * np.cos(phi)
```

This implements **gradient ascent** instead of **gradient descent**, causing:
- Divergence from {0, π} attractors
- Poor recall performance
- Instability with higher learning rates

**Fix:** Changed to **negative** sign (correct gradient descent):
```python
coherence = -2 * lambda_coh * A**2 * np.sin(phi) * np.cos(phi)
```

**Impact:**
- ✅ Proper convergence to {0, π} attractors
- ✅ Stable dynamics
- ✅ Achieves reported capacity improvements

**All manuscript results use the corrected version.**

---

## 🔄 Reproducibility

### System Requirements

- Python 3.8+
- NumPy 1.20+
- Matplotlib 3.3+
- SciPy 1.6+ (for statistical tests)

### Reproduction Steps

1. **Clone and install:**
   ```bash
   git clone https://github.com/labinot-marku/CONN-Associative-Memory.git
   cd CONN-Associative-Memory
   pip install -r requirements.txt
   ```

2. **Run full validation:**
   ```bash
   python CONN_VALIDATION_V4_FINAL.py
   ```

3. **Check results:**
   ```bash
   cat results/capacity_results.csv
   cat results/noise_robustness.csv
   cat results/ablation_results.csv
   ```

4. **Expected runtime:**
   - Quick mode (`--quick`): ~2-5 minutes
   - Full validation: ~15-30 minutes (depending on hardware)

### Verification

To verify your results match the manuscript:

```bash
# Run and save results
python CONN_VALIDATION_V4_FINAL.py > validation_log.txt

# Check capacity improvement (conservative estimate ~1.38×)
grep "improvement" results/capacity_results.csv

# Check ablation results (Full CONN should be best)
cat results/ablation_results.csv
```

**Note:** Quick mode may show higher numbers (2-4×) due to small-N artifacts. 
These are exploratory upper bounds, not validated measurements.

---

## 📖 Citation

If you use this code, please cite:

```bibtex
@article{marku2026conn,
  title={Topological Phase Constraints and Amplitude Dynamics Improve 
         Associative Memory in Oscillatory Neural Networks},
  author={Marku, Labinot},
  journal={ResearchGate Preprint},
  year={2026},
  doi={10.13140/RG.2.2.25288.99841},
  note={AI collaboration: Claude (Anthropic), ChatGPT (OpenAI)}
}
```

---

## 🤝 Contributing

This is a research implementation. If you find issues or have improvements:

1. Open an issue describing the problem
2. Provide minimal reproducible example
3. Suggest fix (if you have one)

**Note:** This code is primarily for validation and reproduction of published results.

---

## 📜 License

MIT License - see LICENSE file for details.

**Academic Use:** Free for research and educational purposes.  
**Commercial Use:** Contact author for licensing.

---

## 👤 Author

**Labinot Marku, M.D.**
- Institution: KRH Klinikum Nordstadt Hannover
- Email: labinot.marku@krh.de
- ResearchGate: [Profile](https://www.researchgate.net/profile/Labinot-Marku)

**AI Collaboration Acknowledgment:**  
This work involved collaboration with Claude (Anthropic) and ChatGPT (OpenAI) for implementation, debugging, and documentation.

---

## 🔍 Transparency Statement

### AI Involvement

This research involved significant AI collaboration:

- **Conceptualization:** Human (L.M.)
- **Mathematical formulation:** Human with AI assistance
- **Implementation:** Joint human-AI (iterative debugging)
- **Bug discovery:** Joint (sign error found during validation)
- **Documentation:** AI-assisted

All AI contributions are fully disclosed in the manuscript acknowledgments.

### Code Evolution

1. **Initial implementation:** Browser-based (TSX/React) for interactive exploration
2. **Bug discovery:** Sign error in coherence gradient found during testing
3. **Correction:** Fixed to proper gradient descent
4. **Python conversion:** This repository (CONN_VALIDATION_V4_FINAL.py)

**The corrected version is the one validated and reported in the manuscript.**

---

## ❓ FAQ

### Q: Why was there a sign error?

**A:** Common mistake when implementing gradient descent. The energy function has a positive coherence term, but the dynamics follow -∇E (gradient descent), requiring a negative sign. This was caught during validation testing.

### Q: Do the Python results match the paper?

**A:** Yes, this corrected Python implementation reproduces the browser-based (TSX) results that generated the manuscript figures. However, quick-mode exploratory results may show higher numbers than the validated conservative estimates reported in the paper.

### Q: Can I use the uncorrected version?

**A:** No. The uncorrected version implements gradient ascent and will not work. Only the corrected version (this repository) is valid.

### Q: How can I verify the fix is correct?

**A:** Run the validation suite. You should see:
- Stable convergence (phi converges to {0, π})
- Capacity improvement (exploratory may show 2-4×, validated is ~1.38×)
- Ablation: At easy regime, all similar; near capacity, Full CONN best

If you see divergence or poor performance, check you're using the corrected version.

### Q: Why does the code show 4× but the paper reports 1.38×?

**A:** The code's quick mode at N=32 shows exploratory upper bounds (artifacts from small network size and optimistic thresholds). The paper reports conservative, rigorously validated estimates. Always use the paper's 1.38× for claims.

---

## 📞 Contact

For questions about the implementation or results:

- Email: labinot.marku@krh.de
- ResearchGate: Message through platform
- GitHub Issues: For bug reports

---

**Last Updated:** January 2026  
**Version:** 2.0 (Corrected with exploratory disclaimer)
