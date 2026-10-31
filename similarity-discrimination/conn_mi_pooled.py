"""conn_mi_pooled.py - V5: Experiments 8-10 with MI pooled over all trials (corrected Miller-Madow).
Requires conn_final_validation.py (unchanged, plus returning per-trial indices) in the same folder."""
import numpy as np, conn_final_validation as fv
from conn_mi_bootstrap import mi_miller_madow as _mm
mi=_mm
def pool(args_list):
    T={k:([],[]) for k in ['phase','cont','binary']}; d={'pd':[],'bd':[],'pe':[],'be':[]}
    for a in args_list:
        r=fv.run_full_test(*a)
        for k in T: T[k][0].extend(r['idx'][k][0]); T[k][1].extend(r['idx'][k][1])
        d['pd'].append(r['phase_disc']); d['bd'].append(r['bin_disc']); d['pe'].append(r['phase_exact']); d['be'].append(r['bin_exact'])
    return {k:np.mean(v) for k,v in d.items()}, {k:mi(*T[k],4) for k in T}
S=15; noise=0.25
print("TEST2 scaling")
for N in [80,128,192,256]:
    nu=max(3,N//20); m,I=pool([(N,3,4,0.85,nu,noise,s*100+N) for s in range(S)])
    print(N,3*4+nu,f"{(12+nu)/N:.3f}",f"{m['pd']:.1%} {m['bd']:.1%} PhMI={I['phase']:.3f} BnMI={I['binary']:.3f} {m['pe']:.1%} {m['be']:.1%}")
print("TEST3 load")
for nu in [0,5,10,20,30,40]:
    m,I=pool([(80,3,4,0.85,nu,noise,s*100+nu) for s in range(S)])
    print(12+nu,f"{(12+nu)/80:.3f}",f"{m['pd']:.1%} {m['bd']:.1%} PhMI={I['phase']:.3f} BnMI={I['binary']:.3f} {m['pe']:.1%} {m['be']:.1%}")
print("TEST4 similarity")
for sim in [0.70,0.80,0.85,0.90,0.95]:
    m,I=pool([(80,3,4,sim,5,noise,s*100) for s in range(S)])
    print(sim,f"{m['pd']:.1%} {m['pe']:.1%} {m['bd']:.1%} {m['be']:.1%} PhMI={I['phase']:.3f} BnMI={I['binary']:.3f}")
