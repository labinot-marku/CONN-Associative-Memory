# Bit Accuracy Hides Prototype Collapse in Hebbian Associative Memory

**An integration-depth analysis of phase-coupled and Hopfield retrieval of correlated patterns: manuscript (Version 9) and reproducible code**

Preprint: ResearchGate, DOI [10.13140/RG.2.2.15884.07049](https://doi.org/10.13140/RG.2.2.15884.07049) (V9, October 2026; supersedes V4, "Pairwise Phase-Difference Coupling Preserves Fine-Grained Pattern Discrimination…").

> **Note on earlier versions.** Versions 1–4 of this work (and earlier versions of this README) reported that pairwise phase-difference coupling discriminates correlated patterns better than Hopfield dynamics. Version 9 withdraws that claim: the advantage is an integration-depth effect. The V1–V4 mutual-information estimator also had a sign error. See Appendices A–E of the manuscript for the full change log.

## Main findings

1. **Bit-accuracy inversion.** For families of correlated patterns, converged Hopfield retrieval reaches 83–92% bitwise accuracy while identifying the cued family member at chance (≈25%), because outputs converge toward the family prototype. A system that barely changes the noisy cue has lower bitwise accuracy (≈75%) but keeps ≈1.70 of 2 bits of member information.
2. **Integration depth, not coupling.** At its published parameters the phase network moves 0.2% of bits; the raw cue, random coupling and zero coupling give the same result. Hopfield networks stopped early preserve member identity equally well (continuous, 5 steps: 94.2%; asynchronous binary, 8 updates: 89.6%). Across 31 tested configurations, discrimination follows a common decreasing trend with output displacement from the cue.
3. **Robust to family geometry and readout.** The inversion holds with the prototype stored, with a hidden prototype, and with pairwise-equidistant members. A trained decoder tested on held-out seeds finds no more member information than nearest-neighbour readout, and a decoder-independent bound limits the information in binarised converged outputs (estimated 0.08–0.88 bits).

| System (Exp. 7, N=80, 85% similarity) | Discrimination | Bitwise accuracy | Decoder-level MI (bits) |
|---|---|---|---|
| Phase-coupled, published parameters | 94.4% | 75.4% | 1.703 [1.560, 1.881] |
| Continuous Hopfield, converged | 26.7% | 87.4% | 0.040 (p = 0.25) |
| Binary Hopfield, converged | 27.8% | 88.2% | 0.048 (p = 0.03, not replicated) |

MI: Miller–Madow corrected per entropy term, pooled over trials; 95% CIs from a seed-level bootstrap; p from seed-stratified permutation tests.

## Requirements

Python 3.8+, NumPy, SciPy (Spearman correlation only), Matplotlib (Figure 1 only). No GPU needed.

## Reproducing results

Run all scripts from inside this folder (the V9 scripts import `conn_stats.py` and `conn_family_geometry.py`). All are deterministic with fixed seeds.

```bash
# Experiments 1-12 (as in V4)
python conn_similarity.py              # Exp 1: initial observation
python conn_similarity_ablation.py     # Exps 2-4: ablation, similarity sweep, budget
python conn_fair_comparison.py         # Exp 5: matched conditions, continuous Hopfield
python conn_phase_decomposition.py     # Exp 6: coupling functions
python conn_sparse_validation.py       # Exp 11: N=1024, 25% connectivity (~25 min)
python conn_modern_hopfield.py         # Exp 12: modern Hopfield vs nearest neighbour
python conn_diagnostic_v3.py           # weight-matrix independence, family size, load

# Mutual information (corrected estimator, V5+)
python conn_mi_bootstrap.py            # Exp 7 (writes per_trial_predictions.csv)
python conn_mi_pooled.py               # Exps 8-10 (uses conn_final_validation.py)

# Integration depth
python conn_hopfield_depth.py          # Exp 13: Hopfield at controlled depth
python conn_parameter_sweep.py         # Exp 14: eta x steps sweep, matched-depth baselines
python fig1_collapse.py                # Figure 1 (needs the CSVs from Exps 13-14)

# Family geometry and readout (V6+)
python conn_family_geometry.py         # Exp 15: three family geometries
python conn_prototype_distance.py      # Exp 15b: distances to the prototype
python conn_decoder_cv.py              # Exp 15c: trained decoder + decoder-independent bound
```

Shared statistics (decoder-level MI, hierarchical bootstrap, seed-stratified permutation): `conn_stats.py`.

Result files: `per_trial_predictions.csv`, `parameter_sweep_results.csv`, `matched_depth_results.csv`, `hopfield_depth_results.csv`, `family_geometry_results.csv`, `prototype_distance_results.csv`, `decoder_cv_results.csv`.

## Manuscript

- `CONN_similarity_V9.pdf` — Version 9 (October 2026), 15 pages, Experiments 1–15c.

## Limitations

- Synthetic random binary patterns, N=80 (and N=1024 sparse); no real correlated data.
- Member identity is read out from the binarised output; the decoder-independent bound applies to binarised outputs, not to the continuous state.
- The displacement–discrimination relation is an empirical trend in the tested systems, not a general law.
- Modern Hopfield results hold only for the tested (saturated-softmax) regime.
- No formal proof of prototype collapse; biological statements are speculative.

## AI assistance

Developed with several AI systems through adversarial review: Claude (Anthropic) for implementation, analysis and drafting; ChatGPT (OpenAI) and Copilot (Microsoft) as critical reviewers; Gemini (Google) for code review. Details are in the manuscript's AI contribution statement. The author is responsible for all content.

## Citation

Marku, L. (2026). Bit Accuracy Hides Prototype Collapse in Hebbian Associative Memory: An Integration-Depth Analysis of Phase-Coupled and Hopfield Retrieval of Correlated Patterns. Preprint, Version 9. DOI: 10.13140/RG.2.2.15884.07049

Related: Marku, L. (2026). Phase Coherence Constraints and Amplitude Dynamics Improve Associative Memory in Oscillatory Neural Networks. Preprint. DOI: 10.13140/RG.2.2.21347.00801

## Contact

Labinot Marku, MD — Department of Neurosurgery, KRH Klinikum Nordstadt Hannover — labinot.marku@krh.de

## License

MIT
