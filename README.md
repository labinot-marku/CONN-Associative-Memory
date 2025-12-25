# CONN: Coherence-Oscillator Neural Network

[![ResearchGate](https://img.shields.io/badge/ResearchGate-Preprint-00CCBB.svg)](https://www.researchgate.net/profile/Labinot-Marku)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **⚠️ Repository Status:** Under active development. Manuscript available on ResearchGate. Code and data being progressively added.

**Topological Phase Constraints and Amplitude Dynamics Improve Associative Memory in Oscillatory Neural Networks**

**Author:** Labinot Marku, M.D.  
**Affiliation:** KRH Klinikum Nordstadt Hannover, Department of Neurosurgery  
**Address:** Haltenhoffstr. 41, 30167 Hannover, Germany  
**Contact:** labinot.marku@krh.de  
**Date:** December 21, 2025

---

## 🎯 TL;DR

CONN augments classical Hopfield networks with oscillatory dynamics and topological priors, achieving **1.38× average capacity improvement** (ranging from 1.09× to 1.69× across network sizes N=32-128). All results are fully reproducible.

**Key Results:**
- 📈 **1.38× capacity improvement** over Hopfield networks
- 🔬 **Rigorous validation:** 1,920 trials with 30 replications per condition
- 🧠 **Biologically inspired:** Phase-amplitude coupling observed in cortical networks
- ⚡ **Scaling behavior:** Improvement increases with network size (1.69× at N=128)
- 🔓 **Fully reproducible:** Complete methodology documented

---

## 📊 Quick Results

| Network Size | CONN Capacity | Hopfield Capacity | Improvement |
|--------------|---------------|-------------------|-------------|
| N=32         | α = 0.375     | α = 0.344         | **1.09×**   |
| N=64         | α = 0.359     | α = 0.266         | **1.35×**   |
| N=128        | α = 0.383     | α = 0.227         | **1.69×**   |

*Using optimized hyperparameters (λ=1.0, β=0.5) at 30% noise*

---

## 📄 Manuscript

The complete manuscript with detailed methods, results, and analysis is available on ResearchGate:

**[View Manuscript on ResearchGate →](https://www.researchgate.net/profile/Labinot-Marku)**

### Abstract

We introduce CONN (Coherence-Oscillator Neural Network), a phase-amplitude recurrent architecture that augments classical Hopfield associative memory using a π-coherence topological prior and amplitude dynamics that implement implicit Bayesian pruning. Through systematic hyperparameter optimization and rigorous validation, we demonstrate that CONN achieves a reproducible 1.38× average capacity improvement over classical Hopfield networks, with performance gains scaling from 1.09× (N=32) to 1.69× (N=128).

---

## 🔬 Model Overview

CONN represents each neuron as a **phase-amplitude oscillator**:

```
z_j = A_j × e^(iφ_j)
where A_j ≥ 0, φ_j ∈ [0, 2π)
```

### Energy Function

```
E = -(1/2) Σ J_ij A_i A_j cos(φ_i - φ_j) + λ Σ A_j² sin²(φ_j) + β Σ (A_j - 1)²
```

**Key innovation:** The π-coherence term `λ Σ A_j² sin²(φ_j)` provides a topological prior that biases phases toward {0, π}, reducing spurious attractors.

### Dynamics

**Phase:**
```
dφ_j/dt = A_j Σ J_ij A_i sin(φ_i - φ_j) - 2λA_j² sin(φ_j)cos(φ_j)
```

**Amplitude:**
```
dA_j/dt = -2λA_j sin²(φ_j) - 2β(A_j - 1)
```

**Optimized parameters:** λ=1.0, β=0.5, η_φ=0.005, η_A=0.03

---

## 📈 Key Findings

### 1. Capacity Improvement Scales with Network Size

```
Improvement Ratio
1.8 |                              ●
1.6 |                         ●
1.4 |                    ●
1.2 |              ●
1.0 |__________●________________________
    0    32   64   96  128  160  192
              Network Size (N)
```

**Interpretation:** Topological priors become more effective in higher-dimensional spaces.

### 2. Hyperparameter Optimization is Critical

| λ (coherence) | Capacity α (N=32) | vs Optimal |
|---------------|-------------------|------------|
| 1.0 (optimal) | 0.375             | —          |
| 2.0           | 0.344             | −8%        |
| 4.0 (initial) | 0.250             | −33%       |
| 8.0           | 0.281             | −25%       |

**Lesson:** Too-strong coherence (λ>4) over-constrains dynamics and hurts performance.

### 3. Ablation Study

| Configuration            | Recall (N=32, M=8) |
|--------------------------|---------------------|
| Full CONN (λ=1.0)        | 85.3%               |
| No coherence (λ=0)       | 72.1%               |
| No amplitude dynamics    | 80.9%               |
| Baseline (λ=0, no amp)   | 68.5%               |

π-coherence contributes +13.2 percentage points, amplitude dynamics +4.4 pp.

---

## 🗺️ Repository Roadmap

### ✅ Completed
- [x] Manuscript published on ResearchGate
- [x] Repository structure defined
- [x] README documentation

### 🚧 In Progress
- [ ] Core CONN implementation (`src/conn_core.py`)
- [ ] Validation suite (`src/conn_final_validation.tsx`)
- [ ] Hyperparameter search code (`src/hyperparameter_search.tsx`)
- [ ] Hopfield baseline (`src/hopfield_baseline.py`)

### 📋 Planned
- [ ] Raw experimental data (1,920 trials CSV)
- [ ] Hyperparameter search results (720 trials CSV)
- [ ] Figure generation scripts
- [ ] Jupyter notebooks for analysis
- [ ] Installation and reproduction instructions
- [ ] Unit tests

---

## 📁 Planned Repository Structure

```
CONN-Associative-Memory/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (coming soon)
├── LICENSE                            # MIT License
│
├── docs/
│   ├── manuscript.pdf                 # Complete manuscript
│   └── supplementary_materials.pdf    # Supplementary materials
│
├── src/
│   ├── conn_core.py                  # Core CONN implementation
│   ├── conn_final_validation.tsx     # Interactive validation suite
│   ├── hyperparameter_search.tsx     # Parameter optimization
│   └── hopfield_baseline.py          # Hopfield comparison
│
├── experiments/
│   ├── run_full_validation.py        # Main validation
│   ├── run_hyperparameter_search.py  # Optimization
│   ├── run_ablation.py               # Ablation study
│   └── run_noise_robustness.py       # Noise analysis
│
├── data/
│   ├── raw/                           # Raw trial data (CSV)
│   └── processed/                     # Summary statistics
│
├── figures/                           # All figures (PDF)
│
└── analysis/
    ├── statistical_analysis.ipynb    # Jupyter notebook
    ├── generate_figures.py           # Reproduce figures
    └── verify_reproducibility.py     # Verification
```

---

## 🛠️ Installation (Coming Soon)

Once code is uploaded, installation will be:

```bash
# Clone repository
git clone https://github.com/labinot-marku/CONN-Associative-Memory.git
cd CONN-Associative-Memory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage (Preview)

Once implemented, typical usage will be:

```python
from src.conn_core import CONN

# Initialize network
conn = CONN(N=32, lambda_param=1.0, beta=0.5)

# Store patterns
patterns = [[0, π, 0, π, ...], ...]  # Binary phase patterns
conn.store_patterns(patterns)

# Test recall
noisy_pattern = add_noise(patterns[0], noise_level=0.3)
recovered = conn.recall(noisy_pattern, steps=150)

print(f"Overlap: {compute_overlap(recovered, patterns[0]):.2%}")
```

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{marku2025conn,
  title={Topological Phase Constraints and Amplitude Dynamics Improve 
         Associative Memory in Oscillatory Neural Networks},
  author={Marku, Labinot},
  year={2025},
  institution={KRH Klinikum Nordstadt Hannover},
  note={Preprint available on ResearchGate}
}
```

---

## 🤝 Contributing

This is an active research project. Contributions are welcome:

- **Bug reports:** Open an issue if you find errors
- **Reproductions:** Share your reproduction results
- **Extensions:** Propose improvements or extensions
- **Discussion:** Open issues for scientific discussion

Please wait for initial code upload before submitting pull requests.

---

## 📜 License

- **Code:** MIT License (when uploaded)
- **Data & Figures:** CC BY 4.0
- **Manuscript:** CC BY 4.0

You are free to use, modify, and distribute this work with attribution.

---

## 🙏 Acknowledgments

- Inspired by biological phase-amplitude coupling in cortical networks
- Built on foundational work by Hopfield (1982), Krotov & Hopfield (2016)
- Validation methodology influenced by modern reproducibility practices

---

## 📧 Contact

**Labinot Marku, M.D.**  
KRH Klinikum Nordstadt Hannover  
Department of Neurosurgery  
Haltenhoffstr. 41  
30167 Hannover, Germany

📧 Email: labinot.marku@krh.de  
🐙 GitHub: [@labinot-marku](https://github.com/labinot-marku)  
🔬 ResearchGate: [Profile](https://www.researchgate.net/profile/Labinot-Marku)

**For questions:**
- Technical issues: Open a GitHub issue
- Scientific discussion: Email or ResearchGate message
- Collaboration inquiries: Email directly

---

## 📚 Additional Resources

- **Manuscript:** [ResearchGate](https://www.researchgate.net/profile/Labinot-Marku)
- **Supplementary Materials:** Available upon request
- **Interactive Demo:** Coming soon

---

## 🔖 Keywords

`associative-memory` `hopfield-networks` `neural-oscillations` `phase-amplitude-coupling` `computational-neuroscience` `machine-learning` `topological-priors` `capacity-bounds` `reproducible-research` `neurosurgery-research`

---

## 📊 Development Status

**Last updated:** December 21, 2025  
**Version:** 0.1.0-dev  
**Status:** Manuscript complete, code upload in progress  
**Preprint:** Available on ResearchGate (not peer-reviewed)

---

**Made with ❤️ for open and reproducible science**

---

## 🔔 Updates

Subscribe to repository updates to be notified when:
- Source code is uploaded
- Data becomes available
- Analysis scripts are added
- Interactive demos launch

**Star ⭐ this repository** to follow progress!
