"""E3: null/baseline 对照。① 置换null(实验标签shuffle)+ 逐蛋白perm-p + BH-FDR;
② shuffled-feature null(打乱逐残基NC/NH);③ 单结构baseline(operator_ceiling);④ 逐蛋白bootstrap CI。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import os,sys,json,numpy as np,pandas as pd
from scipy.stats import spearmanr, wilcoxon
PROJ=_GEC_ROOT
maps=json.load(open(f"{PROJ}/data/processed/structure_maps.json")); exp=pd.read_parquet(f"{PROJ}/data/processed/tier1a_all.parquet")
ceil=pd.read_parquet(f"{PROJ}/results/operator_ceiling.parquet"); ceilsp=dict(zip(ceil.protein,ceil.spearman))
A1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def to1(x): x=str(x); return A1.get(x.upper(),x[:1].upper()) if len(x)==3 else (x[:1].upper() if x.strip() else '')
def boff(pred,g):
    eid={int(r.resi):to1(r.resn) for _,r in g.iterrows() if str(r.resn).strip()}
    if len(eid)<5: return maps.get(g.protein_id.iloc[0],{}).get('gen_offset',0)
    pres={int(r.resSeq):to1(r.resn) for _,r in pred.iterrows()}; b,bg=0,-1
    for o in range(-8,9):
        gg=sum(1 for er,aa in eid.items() if pres.get(er+o)==aa)
        if gg>bg: bg,b=gg,o
    return b
rng=np.random.default_rng(0); B=5000; rows=[]
for p in sorted(exp.protein_id.unique()):
    f=f"{PROJ}/eval/pred/BioEmu/{p}.csv"
    if not os.path.exists(f): continue
    pred=pd.read_csv(f); g=exp[exp.protein_id==p]; off=boff(pred,g); pred=pred.copy(); pred['resi']=pred.resSeq-off
    m=g.merge(pred,on='resi',how='inner').dropna(subset=['dGopen_kcal','dGopen_pred'])
    if len(m)<5: continue
    x=m.dGopen_pred.values; y=m.dGopen_kcal.values; obs=spearmanr(x,y)[0]
    nul=np.array([spearmanr(x,rng.permutation(y))[0] for _ in range(B)])
    perm_p=(np.sum(nul>=obs)+1)/(B+1)
    bt=np.array([spearmanr(*( (lambda i:(x[i],y[i]))(rng.integers(0,len(x),len(x))) ))[0] for _ in range(2000)])
    rows.append(dict(protein=p,n=len(m),obs_spearman=round(obs,3),ci_lo=round(np.nanpercentile(bt,2.5),3),ci_hi=round(np.nanpercentile(bt,97.5),3),
                     perm_null_mean=round(nul.mean(),3),perm_p=perm_p,single_struct_ceiling=round(ceilsp.get(p,np.nan),3),delta_vs_ceiling=round(obs-ceilsp.get(p,np.nan),3)))
A=pd.DataFrame(rows)
# BH-FDR
from numpy import argsort
pp=A.perm_p.values; o=argsort(pp); bh=np.empty_like(pp); n=len(pp)
for rank,i in enumerate(o): bh[i]=min(1,pp[i]*n/(rank+1))
for rank in range(n-2,-1,-1): bh[o[rank]]=min(bh[o[rank]],bh[o[rank+1]])
A['perm_BH']=np.round(bh,4)
print("=== E3 null/baseline(BioEmu vs 置换null + 单结构ceiling)===")
print(A.to_string(index=False))
sig=(A.perm_BH<0.05).sum()
print(f"\n置换检验显著(BH<0.05)蛋白: {sig}/{len(A)}")
d=A.dropna(subset=['single_struct_ceiling'])
w=wilcoxon(d.obs_spearman,d.single_struct_ceiling,alternative='greater')
print(f"系综≥单结构ceiling: {(d.delta_vs_ceiling>=0).sum()}/{len(d)}, 单边配对Wilcoxon p={w.pvalue:.4f}; 均值 系综{d.obs_spearman.mean():.3f} vs ceiling{d.single_struct_ceiling.mean():.3f}")
A.to_csv(f"{PROJ}/results/tables/null_baselines.csv",index=False); print("→ results/tables/null_baselines.csv")
