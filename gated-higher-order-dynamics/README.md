# Gated Higher-Order Dynamics (GHOD)

Code accompanying the preprint

**Gated Higher-Order Dynamics: Regional Stability Certificates and the Timescale Condition for Bounded Excursions**, v2.0 (September 2026)
Labinot Marku, M.D. — ResearchGate, DOI of v1.0: 10.13140/RG.2.2.23260.24962 (v2.0 is uploaded as a new version of the same item)

v2.0 supersedes v1.0 (February 2026, "Gated Higher-Order Dynamics with Guaranteed Return"). Several v1.0 claims are withdrawn, including the "165.7× conservatism ratio", "guaranteed return", the three-regime classification and trajectory confinement. The full list is in the section *Changes from v1.0* of the v2.0 manuscript.

## What v2.0 establishes

- A regional Lyapunov certificate for gated cubic dynamics with closed-form certified radius R_V = 2(a_min + 2λu_max) / (u_max σ_T), and the Hessian radius R_H = R_V / 2.
- An exact radius × direction decomposition of the gap between worst-case and realised curvature.
- A certified shortfall of 3.4× to 28× between the certified basin and the realised operating region.
- A timescale condition: bounded excursion-and-return occurs only above a critical gate rate α*, close to ½κ(x₀) of the reduced one-mode model.

## Scripts

| Script | Reproduces | Status |
|---|---|---|
| `alpha_star.py` | §4.6 bisection table: n = 63, α*/(κ/2) median 1.158, α*/(ρ/2) median 0.802, κ/ρ median 0.689, log–log exponents 1.001 and 1.078 | verified; per-system data in `alpha_star.json` |
| `gated_dynamics_v3.py` | §4.1–4.5 headline sweep: 37/50 converged, decomposition, transient non-convexity | to be added |
| `grid_sweep3.py` | Appendix B, 16-point parameter grid | to be added |
| `excursion_complete.py` | §4.6 / Appendix C.1, initial-radius sweep at α = 0.3 | to be added |
| `alpha_sweep.py` | §4.6 / Appendix C.2, gate-rate sweep | to be added |
| `gated_dynamics.py` | **v1.0 code**, kept because the archived v1.0 cites it. Superseded; its headline ratios are withdrawn. | archived |

## Running

```bash
pip install -r requirements.txt     # numpy, scipy
python alpha_star.py                # full run, about 20-25 minutes
python alpha_star.py 0.2 0.3        # subset of radii; each radius has its own seed
```

In Colab: `!python alpha_star.py`

## AI assistance

This work was developed with AI assistance (ChatGPT/OpenAI, Claude/Anthropic, Gemini/Google DeepMind). Every quantitative claim in v2.0 is intended to be regenerable from the scripts in this folder.

## License

Code: MIT. Manuscript: CC BY 4.0 (v1.0 was CC BY-SA 4.0).
