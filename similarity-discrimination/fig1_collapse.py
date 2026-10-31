import csv, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rd=lambda f: list(csv.DictReader(open(f)))
ps=rd("parameter_sweep_results.csv"); md={r["condition"]:r for r in rd("matched_depth_results.csv")}
hd=rd("hopfield_depth_results.csv")
ink="#2b2b2a"; mut="#77766f"; grid="#e6e5e0"
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.edgecolor":mut,"axes.labelcolor":ink,"xtick.color":mut,"ytick.color":mut})
fig,ax=plt.subplots(figsize=(6.2,3.9),dpi=200)
ax.axhline(25,color=mut,lw=1,ls=(0,(4,3)),zorder=1); ax.text(47,21.2,"chance (25%)",ha="right",color=mut,fontsize=8)
ax.axvspan(0,8,color="#1baf7a",alpha=0.07,lw=0,zorder=0); ax.text(0.6,52,"preservation\nregime",color=ink,fontsize=8,va="center")
def S(x,y,c,m,lab,s=46): ax.scatter(x,y,c=c,marker=m,s=s,label=lab,edgecolors="white",linewidths=1.2,zorder=3)
S([float(r["bits_moved_pct"]) for r in ps],[float(r["discrimination"]) for r in ps],"#2a78d6","o","Phase coupling (η × steps sweep)")
S([float(md[k]["bits_moved_pct"]) for k in ["field_short","field_long"]],[float(md[k]["discrimination"]) for k in ["field_short","field_long"]],"#6250d6","D","Field coupling on circle",s=40)
c=[r for r in hd if r["condition"].startswith("Continuous")]
S([float(r["bits_moved_pct"]) for r in c],[float(r["discrimination"]) for r in c],"#eb6834","s","Continuous Hopfield (1–150 steps)",s=40)
a=[r for r in hd if r["condition"].startswith("Async")]+[r for r in hd if r["condition"].startswith("Sync")]
S([float(r["bits_moved_pct"]) for r in a],[float(r["discrimination"]) for r in a],"#1baf7a","^","Binary Hopfield (async k updates; sync)",s=50)
ax.scatter([0],[float(hd[0]["discrimination"])],marker="*",s=140,c=ink,edgecolors="white",linewidths=1,zorder=4,label="Raw noisy cue (no dynamics)")
ax.annotate("published phase parameters",xy=(0.21,93.3),xytext=(10,97),fontsize=8,va="center",color=ink,arrowprops=dict(arrowstyle="-",color=mut,lw=0.8))
ax.annotate("published Hopfield\n(converged)",xy=(30.2,27.1),xytext=(36,45),fontsize=8,color=ink,arrowprops=dict(arrowstyle="-",color=mut,lw=0.8))
ax.set_xlim(-1.5,53); ax.set_ylim(15,102)
ax.set_xlabel("State displacement from cue (% of bits flipped)"); ax.set_ylabel("Family-member discrimination (%)")
ax.grid(color=grid,lw=0.6); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.legend(frameon=False,fontsize=7.5,loc="upper right",bbox_to_anchor=(1.0,0.80),handletextpad=0.3)
fig.tight_layout(); fig.savefig("fig1_collapse.pdf"); fig.savefig("fig1_collapse.png")
