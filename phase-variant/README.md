# Phase-variant storage experiment

Code and results for Section 5.1 of *Geometry, Topology, and Universality Boundaries in Associative Memory Networks*, Version 2 (November 2026).

## Run

```
python phase_variant_experiment.py
```

This writes `phase_variant_results.csv` (Table in Section 5.1) and `phase_variant_per_k.csv` (variant accuracy for each θ_k). Runtime is about one minute on a laptop CPU.

Tested with Python 3.13, NumPy 2.5, SciPy 1.18.

## Design

- Spatial patterns: M = 10 random patterns ξ^μ ∈ {0, π}^N, with N ∈ {32, 128}.
- Variants: K ∈ {2, 4} global phase shifts ξ^μ + θ_k, with θ_k = 2πk/K.
- Weights: Hebbian, J = sᵀs / N with s = cos ξ and a zero diagonal. J is the same for every global phase variant.
- Recall: CONN v2 dynamics, copied unchanged from `validation/CONN_VALIDATION_V5.py`: η_φ = 0.02, η_A = 0.03, β = 1, 400 steps, λ annealed linearly from 0 to λ_final.
- Conditions: λ_final = 0.5 (standard CONN) and λ_final = 0 (no coherence term).
- Noise: identical to CONN v2, 30% π-flips plus Gaussian phase jitter with σ = 0.5 rad.
- Trials: 30 per variant, so 300·K per condition.
- Decoder: for every (μ, k), score = mean_j cos(φ_j − ξ^μ_j − θ_k) on the retrieved phases; the argmax gives the prediction. Spatial is correct if μ matches, the variant is correct if k matches, and a trial succeeds if both match.
- Statistics: 95% confidence intervals are Wilson score intervals; p-values are two-sided exact binomial tests against chance 1/K (`scipy.stats.binomtest`).
- Seeds: one `numpy.random.default_rng` per condition, with seed = 1000·N + 10·K + 10·λ_final. The run is deterministic.

## Implemented dynamics

The amplitude update omits the pairwise coupling term, so the implementation is not exactly the gradient flow of the CONN energy. It is equivariant under φ → φ + π, and with λ = 0 under every global phase shift, which is what Proposition 3 of the paper requires.

```
φ_j ← φ_j + η_φ [ A_j Σ_i J_ji A_i sin(φ_i − φ_j) − λ(t) A_j² sin 2φ_j ]
A_j ← clip( A_j + η_A [ −2λ(t) A_j sin²φ_j − 2β (A_j − 1) ], 0.01, 2 )
```

The amplitude term −2β(A_j − 1) corresponds to the energy term β Σ_j (A_j − 1)².

## Note on Version 1

The phase-variant results reported in Version 1 of the preprint (24.9% and 48.8%) could not be traced to any code and are withdrawn. This experiment replaces them.
