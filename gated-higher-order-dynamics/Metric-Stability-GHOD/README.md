# Metric-Induced Stability (GHOD)

This folder contains the official numerical validation for the research note: 
**"Metric-Induced Stability Beyond Spectral Criteria in Gated Higher-Order Dynamical Systems"**

### Overview
Standard neural models often rely on linear eigenvalues (spectral criteria) for stability. This code proves that by using a **Cubic Gated Governor**, we can achieve global stability and high memory capacity even when classical linear theory predicts divergence.

### Related Work & Foundations
This stability theory builds upon the **Gated Higher-Order Dynamics (GHOD)** mechanism. You can find the foundational links and permanent archives below:

* **ResearchGate (Foundational Preprint):** [Gated Higher-Order Dynamics with Guaranteed Return](https://www.researchgate.net/publication/388915855_Gated_Higher-Order_Dynamics_with_Guaranteed_Return_to_the_Manifold_Basin_Metric-Induced_Stability_and_High-Capacity_Associative_Memory)
* **Zenodo (Permanent Archive):** [https://zenodo.org/records/18792800]
* **ResearchGate (V5 - Stability Paper):** [Metric-Induced Stability Beyond Spectral Criteria](https://doi.org/10.13140/RG.2.2.15913.15209)

### Key Features
- **Theorem HG Validation:** Demonstrates exponential convergence under stochastic regularization.
- **Confinement Radius ($R^*$):** Visualizes how the system stays bounded regardless of initial noise levels.
- **Comparative Analysis:** Directly compares the GHOD mechanism against the classical Hopfield Network.

### Getting Started
To reproduce the stability plots (Figure A in the paper), run the following script:
`python ghod_v4_official.py`

---
*Exploratory research by Labinot Marku, MD (Department of Neurosurgery, Hannover).*
