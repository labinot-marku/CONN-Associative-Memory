"""conn_prototype_distance.py - V6 Experiment 15b: direct distance of converged outputs to the
(hidden or stored) family prototype vs. stored members. Uses the generators of conn_family_geometry.py."""
import numpy as np, csv, importlib.util, sys, io, contextlib
spec = importlib.util.spec_from_file_location("fg", "conn_family_geometry.py")
src = open("conn_family_geometry.py").read().split("OUT=[]")[0]          # helpers only, no main loop
fg = {}; exec(compile(src, "fg", "exec"), fg)
N=fg["N"]; rows=[]
for mode in ["A_star_stored","B_star_hidden","C_equidistant85"]:
    res={k:{"dp":[],"dt":[],"dn":[],"exact":[],"closer":[]} for k in ["raw cue","cont. converged","sync converged"]}
    for sd in range(fg["SEEDS"]):
        rng=np.random.default_rng(42+sd); P,fams,protos=fg["make"](rng,mode)
        W=sum(np.outer(p,p) for p in P)/N; np.fill_diagonal(W,0); st={"fixed":0,"cycle2":0,"none":0}
        for f in range(3):
            for mi,midx in enumerate(fams[f]):
                tgt=P[midx]; cue=tgt.copy(); cue[rng.random(N)<fg["NOISE"]]*=-1
                outs={"raw cue":cue,"cont. converged":fg["cont"](W,cue+rng.normal(0,0.15,N),150),"sync converged":fg["sync"](W,cue,st)}
                for k,o in outs.items():
                    pr=protos[f]
                    # distances in % bits; allow global sign flip for memories (spin symmetry)
                    d=lambda a,b: 100*min(np.mean(a!=b),np.mean(a!=-b))
                    dmem=[d(o,P[j]) for j in fams[f]]
                    others=[d(o,P[j]) for j in fams[f] if j!=fams[f][0] or mode!="A_star_stored"]
                    r=res[k]; r["dp"].append(d(o,pr)); r["dt"].append(d(o,tgt)); r["dn"].append(min(others))
                    r["exact"].append(d(o,pr)==0); r["closer"].append(d(o,pr)<min(others))
    for k,r in res.items():
        row=[mode,k]+[np.mean(r[x]) for x in ["dp","dt","dn"]]+[100*np.mean(r["exact"]),100*np.mean(r["closer"]),np.median(r["dp"]),np.percentile(r["dp"],90)]
        rows.append(row)
        print(f"{mode:<17}{k:<17} d(out,proto)={row[2]:5.1f}%  d(out,target)={row[3]:5.1f}%  d(out,nearest member)={row[4]:5.1f}%  "
              f"exactly proto={row[5]:5.1f}%  closer to proto than any member={row[6]:5.1f}%  median/90th d(proto)={row[7]:.1f}/{row[8]:.1f}")
with open("prototype_distance_results.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["design","condition","mean_d_proto_pct","mean_d_target_pct","mean_d_nearest_member_pct","pct_exactly_proto","pct_closer_to_proto_than_any_member","median_d_proto","p90_d_proto"])
    [w.writerow(r) for r in rows]
