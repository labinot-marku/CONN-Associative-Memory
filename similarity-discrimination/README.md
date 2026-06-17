# Pairwise Phase-Difference Coupling Preserves Fine-Grained Pattern Discrimination

**Working manuscript + reproducible code**

Pairwise phase-difference coupling on a circular manifold preserves family-member identity information (MI ≈ 1.76–2.00 bits, 88–100% of maximum) where both binary and continuous Hopfield networks collapse to family prototypes (MI ≈ 0.05 bits), despite Hopfield achieving higher exact bit-level recall.

## Key Result

| System | Discrimination | Exact Recall | MI (bits) |
|--------|---------------|-------------|-----------|
| Phase-coupled (sin) | 92.5% | 74.7% | 1.764 |
| Modern Hopfield (β=1) | 92.9% | — | — |
| Continuous Hopfield (tanh) | 25.8% | 87.3% | 0.074 |
| Binary Hopfield (sign) | 25.8% | 87.7% | 0.049 |

Modern Hopfield matches phase coupling numerically but 99–100% of its outputs are identical to hard nearest-neighbor lookup (softmax saturation at N=80). Pairwise phase coupling achieves equivalent discrimination through genuine oscillatory convergence on the compressed Hebbian matrix.

## Mechanism

The critical ingredient is **pairwise phase-difference coupling**: any function of (φ_j − φ_i) discriminates, while any global field-based coupling f(Σ_j W_ij x_j) does not. The coupling function is secondary (sin ≈ linear ≈ cos); the coupling *structure* (pairwise difference vs global field) is essential.

## Progressive Ablation (12 Experiments)

| # | Experiment | Question | Result |
|---|-----------|----------|--------|
| 1 | Initial discovery | Does phase coupling discriminate? | 99.4% vs 25.6% |
| 2 | Component ablation | Amplitude? Coherence prior? | Neither matters |
| 3 | Similarity sweep | Robust across similarity? | 70%–95%, yes |
| 4 | Computational budget | More Hopfield attempts help? | No (10 attempts: 25%) |
| 5 | Matched conditions | Continuity or phase coupling? | Phase only (92.5% vs 25.8% vs 25.8%) |
| 6 | Phase decomposition | Which coupling function? | sin ≈ linear ≈ cos ≫ tanh ≈ binary |
| 7 | Mutual information | Information-theoretic? | 1.764 vs 0.049 bits (p=0.0003) |
| 8 | Scaling | Larger N? | Perfect at N≥192 |
| 9 | Load robustness | Higher α? | Robust to α=0.65 |
| 10 | Discrimination vs recall | Different objectives? | Hopfield wins bits, phase wins identity |
| 11 | Sparse connectivity | 25% connectivity at N=1024? | 100% vs 25% (exact Bernoulli) |
| 12 | Modern Hopfield baseline | Does modern Hopfield close the gap? | Matches numerically but = NN lookup |

## Requirements

- Python 3.8+
- NumPy
- SciPy

No GPU required. All scripts run on CPU in under 5 minutes (except sparse N=1024: ~25 minutes).

## Reproducing Results

Each script is self-contained and deterministic (fixed random seeds):

```bash
# Core experiments (N=80, ~1-3 minutes each)
python conn_similarity.py              # Experiment 1: Initial discovery
python conn_similarity_ablation.py     # Experiments 2-4: Ablation + sweep + budget
python conn_fair_comparison.py         # Experiment 5: Matched conditions + cont. Hopfield
python conn_phase_decomposition.py     # Experiments 6-7: Phase decomposition + sweep
python conn_final_validation.py        # Experiments 7-10: MI, scaling, capacity, disc vs recall

# Sparse scaling (N=1024, ~25 minutes)
python conn_sparse_validation.py       # Experiment 11: 25% sparse connectivity

# Modern Hopfield baseline (~1 minute)
python conn_modern_hopfield.py         # Experiment 12: Modern Hopfield + NN diagnostic
```

All results have been independently reproduced in Google Colab with identical numbers across multiple runs.

## Manuscript

- `CONN_similarity_manuscript.pdf` — Working manuscript (14 pages, 12 experiments)

## Limitations

- All families use prototype-flip generation; other correlation structures untested
- Fixed family structure (3×4) across scaling tests; simultaneous N/family scaling needed
- Modern Hopfield degenerates to NN lookup at N=80; soft-retrieval regime at larger N untested
- No formal mathematical proof of why pairwise coupling preserves identity
- Biological claims are speculative

## Multi-AI Collaboration

This work was developed through adversarial multi-AI review:
- **Claude (Anthropic)**: Primary computational collaborator; two independent instances designed experiments 1–10 and experiment 12 respectively
- **ChatGPT (OpenAI)**: Primary critical reviewer, identified confounds and requested controls
- **Gemini (Google)**: Code review, identified sparse implementation bug
- **Copilot (Microsoft)**: Generated scaling implementations (with bugs found by Gemini); detailed manuscript feedback
- **Grok (xAI)**: Independent assessment and hardware implementation analysis

The OMA architecture hypothesis was killed by ablation prompted by ChatGPT. The similarity-discrimination finding survived all critics.

## Citation

Marku, L. (2026). Pairwise Phase-Difference Coupling Preserves Fine-Grained Pattern Discrimination in Associative Memory: A Progressive Ablation Study. Working manuscript.

## Contact

Labinot Marku, MD — Department of Neurosurgery, KRH Klinikum Nordstadt Hannover — labinot.marku@krh.de

## License

MIT
