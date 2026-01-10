No-Go Theorem Validation Code
Purpose:
This code implements the validation experiments reported in Table 5.1 of "A No-Go Theorem forPhase-Coded Associative Memory under Conservative Dynamics" (Marku, 2026).
⚠️
Important:
This code is provided for reproducibility only and is
NOT required
for the proof of the No-Gotheorem, which holds independently of any specifi c architecture or parameterization.
🎯 What This Code Validates
The No-Go theorem proves that phase-only associative memory is fundamentally impossible under conservativegradient dynamics. This code demonstrates the prediction experimentally using CONN as a test architecture:
Experimental Results (Table 5.1):
✅
Spatial accuracy:
96.0% ± 2.0% (CONN architecture works correctly)
✅
Phase accuracy:
32.0% ± 2.5% vs 25.0% chance level (p=0.326)
✅
Conclusion:
Phase information cannot be encoded, exactly as theorem predicts
Key Finding:
Even with optimized implementation, phase discrimination remains at chance level while spatialretrieval succeeds. This validates the No-Go theorem's core claim: phase-shift symmetry prevents phaseencoding.
⚠️ What This Code Does NOT Claim
This validation code is
NOT
about:
❌ Optimizing CONN performance
❌ Architectural improvements or innovations
❌ Capacity maximization strategies
❌ Component ablation studies (see CONN architecture paper for those)
This code
ONLY
validates the No-Go theorem's theoretical predictions.
🚀 Quick Start
Requirements
bash
Python
37
+
Installation
Run Validation
Expected runtime:
~5 minutes
Expected output:
Results matching Table 5.1 from the manuscript
📊 Experiments
The validation suite runs three experiments:
Experiment 1: Capacity Validation
Purpose:
Verify CONN's capacity improvement over Hopfi eld networks
Results:
N=32: α = 0.281 (2.25× improvement over Hopfi eld α = 0.138)
N=64: α = 0.281 (2.25× improvement)
Interpretation:
Shows CONN architecture works correctly for spatial patterns.
Experiment 2: Phase-Coding Test (Main No-Go Validation)
Purpose:
Test whether phase variants can be distinguished
Setup:
Store M=3 spatial patterns with K phase variants each
For K=4: 3×4=12 total patterns
Python
3.7
+
NumPy
SciPy
bash
# Clone repository
git
clone https://github.com/labinot-marku/CONN-Associative-Memory.git
cd
CONN-Associative-Memory/validation
# Install dependencies
pip
install
numpy scipy
bash
python CONN_VALIDATION_V4_FINAL.py
Test: Can network distinguish phase variants after retrieval?
Results:
Interpretation:
Network correctly retrieves spatial patterns (96-100% accuracy)
Phase information collapses to chance level (p > 0.05)
Validates No-Go Theorem Outcome B (Symmetry Collapse)
Experiment 3: Information Decay
Purpose:
Measure how fast phase information degrades vs spatial
Results:
Spatial decay time: τ_spatial = 58 ± 8 steps
Phase decay time: τ_phase = 14 ± 3 steps
Ratio: τ_phase/τ_spatial = 0.24 (phase decays 4× faster)
Interpretation:
Phase information unstable, confi rming Theorem 4 predictions.
📈 Output
The code generates:
1.
Console output:
Real-time progress and results
2.
Summary statistics:
For each experiment
3.
CSV fi les (optional):
Detailed trial-by-trial data
Example output:
K=4: Spatial=96.0%, Phase=32.0% (chance=25.0%), p=0.326 ✓
K=8: Spatial=100.0%, Phase=12.0% (chance=12.5%), p=1.000 ✓
========================================
EXPERIMENT 2: Phase-Coding Test
========================================
Testing K=4 phase variants:
Spatial Accuracy: 96.0% ± 2.0%
Phase Accuracy: 32.0% ± 2.5%
🔬 Technical Details
Network Confi guration
Pattern Generation
Spatial patterns:
Random phase confi gurations ξ^μ ∈ [0, 2π)^N
Phase variants:
Deterministic perturbations with k·(2π/K) offsets
Noise:
10% phase noise during retrieval
Statistical Tests
Phase accuracy:
Two-tailed binomial test vs chance (1/K)
Signifi cance threshold:
α = 0.05
Trials:
50 replications per condition
📖 Related Publications
No-Go Theorem Paper (This Validation)
Chance Level: 25.0%
p-value: 0.326
✓ Phase accuracy at chance level (validates No-Go theorem)
python
N
=
32
# Network size
M_spatial
=
3
# Base spatial patterns
K
=
4
# Phase variants per pattern
λ
=
0.1
# Coherence penalty
β
=
0.5
# Amplitude regularization
η_φ
=
0.005
# Phase learning rate
η_A
=
0.03
# Amplitude learning rate
T
=
150
# Convergence steps
bibtex
@article{marku2026nogo,
title={A No-Go Theorem for Phase-Coded Associative Memory
under Conservative Dynamics},
author={Marku, Labinot},
CONN Architecture Paper
🤝 Citation
If you use this validation code in your research, please cite the No-Go Theorem paper:
Marku, L. (2026). A No-Go Theorem for Phase-Coded Associative Memory under Conservative Dynamics.ResearchGate Preprint. DOI: 10.13140/RG.2.2.16245.03041
📧 Contact
Labinot Marku, M.D.
Department of Neurosurgery
KRH Klinikum Nordstadt Hannover
Email:
labinot.marku@krh.de
For questions:
Technical issues:
Open a GitHub issue
Scientifi c discussion:
Email or ResearchGate
Reproducibility problems:
Open an issue with your output
📜 License
Code:
MIT License
Dt
CCBY40
year={2026},
month={January},
doi={10.13140/RG.2.2.16245.03041}
}
bibtex
@article{marku2025conn,
title={Topological Phase Constraints and Amplitude Dynamics
Improve Associative Memory in Oscillatory Neural Networks},
author={Marku, Labinot},
year={2025},
month={December},
doi={10.13140/RG.2.2.21347.00801}
}
Data:
CC BY 4.0
Documentation:
CC BY 4.0
Free to use, modify, and distribute with attribution.
✅ Reproducibility Checklist
To verify the No-Go theorem validation:
Install dependencies (NumPy, SciPy)
Run
python CONN_VALIDATION_V4_FINAL.py
Check Experiment 2 results
Verify phase accuracy ≈ chance level (p > 0.05)
Verify spatial accuracy > 90%
Compare with Table 5.1 in manuscript
Expected:
Phase at chance, spatial high → No-Go theorem validated ✓
Last Updated:
January 10, 2026
Version:
1.0.0
Status:
Production - Reproduces published results
