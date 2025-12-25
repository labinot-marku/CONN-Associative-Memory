# CONN-Associative-Memory

### Source code and data for "Topological Phase Constraints and Amplitude Dynamics Improve Associative Memory in Oscillatory Neural Networks"

**CONN: Coherence-Oscillator Neural Network**

**Author:** Labinot Marku  
**Affiliation:** KRH Klinikum Nordstadt Hannover, Department of Neurosurgery  
**Contact:** [labinot.marku@krh.de](mailto:labinot.marku@krh.de)  
**Preprint:** arXiv:XXXX.XXXXX (cs.NE)  
**Date:** December 21, 2025

---

## 🎯 TL;DR
CONN augments classical Hopfield networks with oscillatory dynamics and topological priors, achieving a **1.38× average capacity improvement** (ranging from 1.09× to 1.69× across network sizes $N=32$ to $128$). All results are fully reproducible.

---

## 📊 Quick Results

| Network Size | CONN Capacity ($\alpha$) | Hopfield Capacity | Improvement |
| :--- | :--- | :--- | :--- |
| **N=32** | 0.375 | 0.344 | 1.09x |
| **N=64** | 0.359 | 0.266 | 1.35x |
| **N=128** | 0.383 | 0.227 | 1.69x |

*Results obtained using optimized hyperparameters ($\lambda=1.0, \beta=0.5$) at 30% noise.*

---

## 🚀 Quick Start (Local Python)

```bash
# Clone the repository
git clone [https://github.com/labinot-marku/CONN-Associative-Memory.git](https://github.com/labinot-marku/CONN-Associative-Memory.git)
cd CONN-Associative-Memory

# Install dependencies
pip install -r requirements.txt

# Run main validation (~20 minutes)
python experiments/run_full_validation.py
Expected output:✓ N=32 capacity: M=12 ($\alpha=0.375$) - MATCHES✓ N=64 capacity: M=23 ($\alpha=0.359$) - MATCHES✓ N=128 capacity: M=49 ($\alpha=0.383$) - MATCHES✓ All statistical tests pass🔬 Model OverviewCONN represents each neuron as a phase-amplitude oscillator:$$z_j = A_j e^{i\phi_j}$$Energy Function:The network minimizes a Lyapunov function that includes a $\pi$-coherence topological prior:$$E = -\frac{1}{2} \sum_{i \neq j} J_{ij} A_i A_j \cos(\phi_i - \phi_j) + \lambda \sum_j A_j^2 \sin^2(\phi_j) + \beta \sum_j (A_j - 1)^2$$🧠 Network DynamicsThe system evolves via coupled differential equations governing phase and amplitude:Phase Update:$$\frac{d\phi_j}{dt} = A_j \sum_i J_{ij} A_i \sin(\phi_i - \phi_j) - 2\lambda A_j \sin(\phi_j) \cos(\phi_j)$$Amplitude Update:$$\frac{dA_j}{dt} = -2\lambda A_j \sin^2(\phi_j) - 2\beta(A_j - 1)$$📈 Key FindingsScaling Behavior: Improvement ratio increases with network size (1.69x at $N=128$), suggesting that topological priors become more effective in higher-dimensional state spaces.Hyperparameter Sensitivity: Our grid search revealed that too-strong coherence ($\lambda > 4.0$) over-constrains the dynamics. The optimal balance for capacity was found at $\lambda=1.0$.Ablation Study:ConfigurationRecall Rate (N=32, M=8)Full CONN ($\lambda=1.0$)85.3%No Coherence ($\lambda=0$)72.1%No Amplitude Dynamics80.9%Baseline (Hopfield-like)68.5%📁 Repository StructurePlaintextconn-validation/
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── src/
│   ├── conn_core.py         # Core CONN implementation
│   └── hopfield_baseline.py # Hopfield comparison
├── experiments/
│   ├── run_full_validation.py
│   ├── run_ablation.py
│   └── run_noise_robustness.py
├── data/
│   └── raw/                 # 1,920 trial results (CSV)
├── figures/                 # PDF versions of plots
└── docs/
    ├── manuscript.pdf       # Full Preprint
    └── supplements.pdf      # Technical Appendix
📄 CitationCode snippet@misc{marku2025conn,
  title={Topological Phase Constraints and Amplitude Dynamics Improve Associative Memory in Oscillatory Neural Networks},
  author={Marku, Labinot},
  year={2025},
  url={[https://github.com/labinot-marku/CONN-Associative-Memory](https://github.com/labinot-marku/CONN-Associative-Memory)}
}
📜 LicenseCode: MIT LicenseData & Figures: CC BY 4.0📧 Contact & Clinical InquiriesLabinot Marku Department of Neurosurgery, KRH Klinikum Nordstadt HannoverWebsite: https://nordstadt.krh.de/kliniken-zentren/neurochirurgieStatus: Preprint (not peer-reviewed). Last updated: December 21, 2025.
