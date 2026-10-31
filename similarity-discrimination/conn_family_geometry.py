"""conn_family_geometry.py - V6 Experiment 15: robustness to family geometry.
Designs: A = V5 (prototype stored as member 0), B = prototype never stored,
C = disjoint flip sets so every member pair is exactly 85% similar.
Adds a prototype-only baseline, synchronous 2-cycle counts, hierarchical bootstrap
and seed-stratified permutation tests (conn_stats.py)."""
import csv
from conn_stats import mi_mm, hier_boot, perm_strat
import numpy as np
N=80; NOISE=0.25; SEEDS=20; K=4
def b2p(b): return np.where(b>0,0.0,np.pi)
def p2b(p): s=np.sign(np.cos(p)); s[s==0]=1; return s
def sgn(x): s=np.sign(x); s[s==0]=1; return s

def family(rng, proto, mode):
    if mode=="A_star_stored":            # V5 design: prototype stored as member 0
        k=int(0.15*N); fam=[proto.copy()]
        for _ in range(3):
            m=proto.copy(); m[rng.choice(N,k,replace=False)]*=-1; fam.append(m)
        return fam, proto
    if mode=="B_star_hidden":            # 4 perturbations of an UNSTORED prototype
        k=int(0.15*N); fam=[]
        for _ in range(4):
            m=proto.copy(); m[rng.choice(N,k,replace=False)]*=-1; fam.append(m)
        return fam, proto
    if mode=="C_equidistant85":          # disjoint flip sets: every pair exactly 85% similar
        k=6; idx=rng.permutation(N); fam=[]
        for i in range(4):
            m=proto.copy(); m[idx[i*k:(i+1)*k]]*=-1; fam.append(m)
        return fam, proto

def make(rng, mode):
    P=[]; fams=[]; protos=[]
    for f in range(3):
        proto=rng.choice([-1,1],N); fam,pr=family(rng,proto,mode)
        fams.append(list(range(len(P),len(P)+4))); P+=fam; protos.append(pr)
    for _ in range(5): P.append(rng.choice([-1,1],N))
    return np.array(P), fams, protos

def phase(J,phi,steps=150,eta=0.005):
    for _ in range(steps):
        phi=(phi+eta*np.sum(J*np.sin(phi[None,:]-phi[:,None]),axis=1))%(2*np.pi)
    return phi
def cont(W,x,k):
    for _ in range(k): x=x+0.1*(-x+np.tanh(2.0*(W@x)))
    return sgn(x)
def asyn(W,s,k,rng):
    s=s.copy()
    for i in rng.integers(0,N,k): s[i]=1 if W[i]@s>=0 else -1
    return s
def sync(W,s,stats):
    prev=None
    for t in range(100):
        s2=sgn(W@s)
        if np.array_equal(s2,s): stats["fixed"]+=1; return s
        if prev is not None and np.array_equal(s2,prev): stats["cycle2"]+=1; return s2
        prev=s; s=s2
    stats["none"]+=1; return s

CONDS=["raw cue","phase (published)","phase, J=0","cont. Hopfield 5 steps","cont. Hopfield converged",
       "async binary, 8 updates","sync binary, converged","prototype-only output"]
def run(mode):
    R={c:{"seed":[],"t":[],"o":[],"bit":[]} for c in CONDS}; st={"fixed":0,"cycle2":0,"none":0}
    for sd in range(SEEDS):
        rng=np.random.default_rng(42+sd); P,fams,protos=make(rng,mode)
        W=sum(np.outer(p,p) for p in P)/N; np.fill_diagonal(W,0); J=W.copy()   # cos(b2p(p)) = p
        for f in range(3):
            for mi,midx in enumerate(fams[f]):
                tgt=P[midx]; cue=tgt.copy(); cue[rng.random(N)<NOISE]*=-1
                phi0=b2p(cue)+rng.normal(0,0.15,N)
                outs={"raw cue":cue,"phase (published)":p2b(phase(J,phi0)),
                      "phase, J=0":p2b(phase(np.zeros_like(J),phi0)),
                      "cont. Hopfield 5 steps":cont(W,cue+rng.normal(0,0.15,N),5),
                      "cont. Hopfield converged":cont(W,cue+rng.normal(0,0.15,N),150),
                      "async binary, 8 updates":asyn(W,cue,8,rng),
                      "sync binary, converged":sync(W,cue,st),
                      "prototype-only output":protos[f]}
                for c,out in outs.items():
                    pr=int(np.argmax([out@P[j] for j in fams[f]]))
                    R[c]["seed"].append(sd); R[c]["t"].append(mi); R[c]["o"].append(pr); R[c]["bit"].append(np.mean(out==tgt))
    # similarity matrix check
    rng=np.random.default_rng(1); P,fams,_=make(rng,mode); F=P[fams[0]]
    S=(F@F.T/N+1)/2
    return R,st,S

OUT=[]
for mode in ["A_star_stored","B_star_hidden","C_equidistant85"]:
    R,st,S=run(mode)
    print(f"\n=== {mode} ===  sample family similarity matrix:\n{np.round(S,3)}")
    print(f"  sync Hopfield endings: {st}")
    print(f"  {'condition':<27}{'disc':>7}{'95%CI(hier)':>15}{'bit acc':>9}{'MI':>7}{'95%CI(hier)':>15}{'p_strat':>9}")
    for c in CONDS:
        r=R[c]; t=np.array(r['t']); o=np.array(r['o'])
        (dl,dh),(ml,mh)=hier_boot(r['seed'],t,o); p=perm_strat(r['seed'],t,o) if c!="prototype-only output" else float('nan')
        print(f"  {c:<27}{np.mean(t==o)*100:6.1f}%  [{dl*100:4.1f},{dh*100:5.1f}]{np.mean(r['bit'])*100:8.1f}%{mi_mm(t,o):7.3f}  [{ml:.3f},{mh:.3f}]{p:9.4f}")
        OUT.append([mode,c,np.mean(t==o)*100,dl*100,dh*100,np.mean(r['bit'])*100,mi_mm(t,o),ml,mh,p,st['cycle2'] if 'sync' in c else ''])

with open("family_geometry_results.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["design","condition","discrimination","disc_ci_low","disc_ci_high","bit_accuracy","mi_mm_bits","mi_ci_low","mi_ci_high","p_seed_stratified","sync_2cycles_of_240"])
    for row in OUT: w.writerow(row)
