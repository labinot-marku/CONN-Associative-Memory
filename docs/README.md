# A No-Go Theorem for Phase-Coded Associative Memory

**Author:** Labinot Marku, M.D.  
**Affiliation:** KRH Klinikum Nordstadt Hannover, Department of Neurosurgery  
**Date:** December 21, 2025  

---

## 🎯 Abstract
[cite_start]We formalize a structural impossibility result for phase-coded associative memory systems operating under conservative (gradient-based) dynamics[cite: 3]. [cite_start]We show that in any system admitting a Lyapunov energy function and requiring stable, noise-robust retrieval, information encoded exclusively in phase-like degrees of freedom cannot be preserved asymptotically[cite: 4]. [cite_start]The result explains persistent empirical failures of phase-coding schemes and establishes a sharp boundary between safe pairwise systems and unstable higher-order models[cite: 5].

---

## 🔬 The No-Go Theorem
[cite_start]**Theorem:** In a conservative gradient system with a bounded-below energy function $E(y, \phi)$, no information encoded exclusively in phase variables $\phi$ can be stably retrieved asymptotically if phase directions are flat and attractors are isolated[cite: 40, 41, 46].

### The Proof Logic
* [cite_start]**Lyapunov Descent**: Forces convergence to the critical set where $\nabla E = 0$[cite: 51].
* [cite_start]**Flat Phase Fibers**: Imply that the critical set contains continuous manifolds, leading to phase diffusion[cite: 52, 53].
* [cite_start]**Information Erasure**: Restoring noise robustness requires adding curvature, which collapses phase distinctions and erases the stored information[cite: 53, 54].



---

## 🗺️ The Universality Boundary
[cite_start]This theorem establishes a qualitative boundary that cannot be crossed smoothly[cite: 63, 66]:

| System Type | Stability & Safety | Phase-Coded Memory Support |
| :--- | :--- | :--- |
| **Pairwise ($p=2$)** | ✔️ Stable, Safe, Interpretable | [cite_start]❌ Cannot support phase-coding [cite: 64] |
| **Higher-Order ($p \ge 3$)** | ❌ Unstable, Unpredictable | [cite_start]✔️ Can encode relational structure [cite: 65] |

---

## 🧠 Interpretation & Significance
* [cite_start]**Geometric Invariants**: Failure is not due to poor optimization, but to geometric and dynamical constraints[cite: 68, 69].
* [cite_start]**The Curvature Law**: Only directions that generate curvature in the energy landscape can store information under gradient descent[cite: 70].
* [cite_start]**Guidance for CONN**: This work delineates what *cannot* work, providing the principled guidance needed for the **CONN architecture** to bypass these limits[cite: 73, 74].



---

## 📄 Documentation
* **Full Manuscript:** [Download No-Go Theorem PDF](./A_No-Go_Theorem_for_Phase-Coded_Associative_Memory.pdf)
* **Main Project:** [Back to CONN Repository](../README.md)

## 📋 Citation
```bibtex
@article{marku2025nogo,
  title={A No-Go Theorem for Phase-Coded Associative Memory under Conservative Dynamics},
  author={Marku, Labinot},
  institution={KRH Klinikum Nordstadt Hannover},
  year={2025},
  url={[https://github.com/labinot-marku/CONN-Associative-Memory](https://github.com/labinot-marku/CONN-Associative-Memory)}
}
