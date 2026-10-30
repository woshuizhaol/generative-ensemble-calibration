"""数据底座:算齐主图缺失的数据件 → figures/data/。所有数据真实,来自服务器产物。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import json,glob,os,numpy as np,pandas as pd
from scipy.special import logsumexp; from scipy.stats import pearsonr,spearmanr
P=_GEC_ROOT; R=1.987204e-3;T=298.15;RT=R*T
OUT=f"{P}/figures/data"; os.makedirs(OUT,exist_ok=True)
bC,bH=0.35,2.0

# ---- Fig1b: SNase/Barnase 绝对标定(逐残基 pred vs exp dGopen)----
truth=pd.read_parquet(f"{P}/data/processed/tier1a_dGopen.parquet")
rows=[]
for prot in ["SNase","Barnase"]:
    pr=pd.read_csv(f"{P}/eval/pred/BioEmu/{prot}.csv")  # resSeq,dGopen_pred
    tt=truth[(truth.protein_id==prot)&(truth.usable_for=="absolute_dG")][["resi","dGopen_kcal"]]
    m=pr.merge(tt,left_on="resSeq",right_on="resi",how="inner")
    for _,r in m.iterrows(): rows.append(dict(protein=prot,resi=int(r.resi),pred=r.dGopen_pred,exp=r.dGopen_kcal))
pd.DataFrame(rows).to_csv(f"{OUT}/fig1b_abscal.csv",index=False)
print("fig1b_abscal:",len(rows),"残基;蛋白",sorted(set(r['protein'] for r in rows)))

# ---- Fig4c: cross-gen 5x5 中位 Spearman 矩阵 ----
cg=pd.read_csv(f"{P}/results/tables/cross_generator.csv")  # protein,pair,n,pearson,spearman
mat={}
for _,r in cg.iterrows():
    a,b=r['pair'].split('-'); mat.setdefault((a,b),[]).append(r['spearman'])
gens=["BioEmu","ESMFlow","Str2Str","ShortMD","Boltz2"]
M=pd.DataFrame(index=gens,columns=gens,dtype=float)
for g in gens: M.loc[g,g]=1.0
for (a,b),v in mat.items():
    md=np.median(v)
    if a in gens and b in gens: M.loc[a,b]=md; M.loc[b,a]=md
M.to_csv(f"{OUT}/fig4c_crossgen_matrix.csv")
print("fig4c matrix:\n",M.round(3).to_string())

# ---- Fig5: 各生成器_T3 的 ming + de_novo分层 + 观测量sweep ----
sub=json.load(open(f"{P}/data/processed/tier3_anchor_subset.json"))
t3=pd.read_parquet(f"{P}/data/processed/tier3_tsuboyama_40_72.parquet"); ORIG=dict(zip(t3.name,t3.origin))
def loadg(model):
    rows=[]
    for npz in glob.glob(f"{P}/gen/{model}/*/operator_f.npz"):
        did=os.path.basename(os.path.dirname(npz))
        if did not in sub: continue
        d=np.load(npz); g=np.nanmean(bC*d['NC']+bH*d['NH'],axis=0)  # 逐构象整体保护度
        rows.append(dict(domain=did,origin=ORIG.get(sub[did]['name'],'?'),exp=sub[did]['deltaG'],
                         ming=RT*np.min(g), mean=RT*np.nanmean(g), het=float(np.std(g))))
    return pd.DataFrame(rows)
for model,tag in [("Str2Str_T3","str2str"),("BioEmu_T3","bioemu"),("Boltz2_T3","boltz2"),("ESMFlow_T3","esmflow")]:
    df=loadg(model)
    if len(df): df.to_csv(f"{OUT}/fig5_{tag}_T3.csv",index=False); print(f"fig5 {tag}: n={len(df)} ming-vs-exp Spearman={spearmanr(df.ming,df.exp)[0]:+.3f}")
# 观测量 sweep(Str2Str_T3:各百分位 g vs exp)
ss=loadg("Str2Str_T3")
swrows=[]
for npz in glob.glob(f"{P}/gen/Str2Str_T3/*/operator_f.npz"):
    did=os.path.basename(os.path.dirname(npz))
    if did not in sub: continue
    d=np.load(npz); g=np.nanmean(bC*d['NC']+bH*d['NH'],axis=0)
    swrows.append((did,sub[did]['deltaG'],g))
for q,lab in [(0,'min(most-open)'),(10,'p10'),(25,'p25'),(50,'median'),(100,'max')]:
    vals=[RT*np.percentile(g,q) for _,_,g in swrows]; exps=[e for _,e,_ in swrows]
    swr=spearmanr(vals,exps)[0]
    print(f"  sweep Str2Str q={q}({lab}): Spearman={swr:+.3f}")
pd.DataFrame([dict(q=q,label=lab,spearman=spearmanr([RT*np.percentile(g,q) for _,_,g in swrows],[e for _,e,_ in swrows])[0]) for q,lab in [(0,'min'),(10,'p10'),(25,'p25'),(50,'median'),(75,'p75'),(100,'max')]]).to_csv(f"{OUT}/fig5_sweep.csv",index=False)
print("数据底座(fig1b/4c/5)已生成到 figures/data/")
