"""
alpha_star.py -- per-system critical gate rate alpha* (GHOD v2.0, Section 4.6).

Exact radial equation (Euler's identity for the cubic E3):
    dr/dt = -r (xhat'A xhat + 2 lam u) + (1/2) u kappa(xhat) r^2,
    kappa(x) := -xhat' H3(x) xhat  <=  rho(H3(x)).
Reduced one-mode model (direction frozen at xhat0, linear terms dropped,
u = exp(-alpha t)) blows up iff alpha < alpha*_red, with

    alpha*_red = (1/2) kappa(x0)  <=  (1/2) rho(H3(x0)).

Both are per-system predictions with no free parameter. For each system the
true threshold is located by bisection in log-alpha, then compared with
(1/2) kappa(x0) and with the upper estimate (1/2) rho(H3(x0)).

Notes
- Systems are rejection-sampled to be expanding at t = 0.
- alpha* = None: system does not rise-and-return even at alpha = 400
  (excluded). alpha* = 0.1: returns already at the lower bracket (censored).
- Optional: pass radii on the command line to run a subset, e.g.
  python alpha_star.py 0.2 0.3 . Each radius uses its own RNG seed, so a
  subset reproduces the corresponding rows of the full run exactly.
"""
import json
import numpy as np
from numpy.linalg import eigvalsh, norm
from scipy.integrate import solve_ivp

N, LAM, TS, MINEIG = 10, 0.4, 2.0, 0.5
BETA, Q, R0G, T_FINAL = 0.5, 5.0, 0.5, 50.0
import sys as _sys
R0S = ([float(a) for a in _sys.argv[1:] if a.replace('.', '', 1).isdigit()]
       or [0.2, 0.3, 0.5, 0.75, 1.0, 1.5])
NSYS, NBIS = 15, 9


def grad(x, T):
    return 3.0 * np.einsum('ijk,j,k->i', T, x, x)


def hess(x, T):
    return 6.0 * np.einsum('ijk,k->ij', T, x)


def field(x, u, A, T):
    return -(A @ x) - u * grad(x, T) - 2.0 * LAM * u * x


def sys(t, s, A, T, al):
    x, u = s[:N], s[N]
    r = norm(x)
    gg = r**Q / (r**Q + R0G**Q) if r > 1e-300 else 0.0
    return np.concatenate([field(x, u, A, T), [-al * u + BETA * gg]])


def rose_and_returned(A, T, x0, al):
    sol = solve_ivp(sys, (0.0, T_FINAL), np.concatenate([x0, [1.0]]),
                    args=(A, T, al), rtol=1e-6, atol=1e-9)
    tx, tu = sol.y[:-1].T, sol.y[-1]
    ns = norm(tx, axis=1)
    if not (np.all(np.isfinite(ns)) and ns.max() < 1e6):
        return False, np.nan
    ok = (tu[-1] < 0.2) and (ns[-1] < 0.2) and (ns.max() > ns[0] * (1 + 1e-9))
    return bool(ok), float(ns.max() / ns[0])


def make(rng, r0):
    for _ in range(300):
        Qm = rng.standard_normal((N, N))
        A = Qm.T @ Qm
        A += (MINEIG - eigvalsh(A).min()) * np.eye(N)
        T = rng.standard_normal((N, N, N)) * TS
        T = (T + T.transpose(1, 0, 2) + T.transpose(2, 1, 0) + T.transpose(0, 2, 1)
             + T.transpose(1, 2, 0) + T.transpose(2, 0, 1)) / 6.0
        for _ in range(400):
            v = rng.standard_normal(N)
            x0 = r0 * v / norm(v)
            if float(x0 @ field(x0, 1.0, A, T)) > 0:
                return A, T, x0
    return None


out = []
for r0 in R0S:
    rng = np.random.default_rng(31337)
    rows = []
    for _ in range(NSYS):
        s = make(rng, r0)
        if s is None:
            continue
        A, T, x0 = s
        H0 = hess(x0, T)
        pred = 0.5 * float(np.max(np.abs(eigvalsh(H0))))           # (1/2) rho(H3(x0))
        xh = x0 / norm(x0)
        kap = 0.5 * float(-(xh @ H0 @ xh))                          # (1/2) kappa(x0)
        lo, hi = 0.1, 400.0
        if not rose_and_returned(A, T, x0, hi)[0]:
            rows.append(dict(pred=pred, kappa_half=kap, alpha_star=None)); continue
        if rose_and_returned(A, T, x0, lo)[0]:
            rows.append(dict(pred=pred, kappa_half=kap, alpha_star=lo)); continue
        for _ in range(NBIS):                       # bisect in log alpha
            mid = np.sqrt(lo * hi)
            if rose_and_returned(A, T, x0, mid)[0]:
                hi = mid
            else:
                lo = mid
        rows.append(dict(pred=pred, kappa_half=kap, alpha_star=float(np.sqrt(lo * hi))))
    got = [r for r in rows if r['alpha_star'] is not None]
    if got:
        a = np.array([r['alpha_star'] for r in got])
        p = np.array([r['pred'] for r in got])
        k = np.array([r['kappa_half'] for r in got])
        print(f"r0={r0:4.2f}  n={len(got):2d}/{len(rows):2d}  alpha* {np.median(a):6.2f}  "
              f"kappa/2 {np.median(k):6.2f}  a*/(kappa/2) {np.median(a/k):5.2f}  "
              f"rho/2 {np.median(p):6.2f}  a*/(rho/2) {np.median(a/p):5.2f}  "
              f"kappa/rho {np.median(k/p):5.2f}", flush=True)
    out.append(dict(r0=r0, rows=rows))
    json.dump(out, open('alpha_star.json' if len(R0S) == 6 else f'alpha_star_{"_".join(map(str, R0S))}.json', 'w'), indent=1)

rows = [r for o in out for r in o['rows'] if r['alpha_star']]
A = np.array([r['alpha_star'] for r in rows])
P = np.array([r['pred'] for r in rows])
K = np.array([r['kappa_half'] for r in rows])
q = lambda v: f"median {np.median(v):.3f}  IQR [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]"
print(f"\npooled n={len(A)}")
print(f"  alpha*/(kappa/2): {q(A / K)}")
print(f"  alpha*/(rho/2):   {q(A / P)}")
print(f"  kappa/rho:        {q(K / P)}")
bk = np.polyfit(np.log(K), np.log(A), 1)[0]
bp = np.polyfit(np.log(P), np.log(A), 1)[0]
print(f"  log-log exponent: {bk:.3f} against kappa/2,  {bp:.3f} against rho/2  (reduced model: 1)")
print("ALL DONE")
