"""E1: 全局配分函数代理 ΔG_global=-RT·ln⟨exp(-⟨lnPF⟩_res)⟩_conf(最开放构象主导=全局解折叠态)
vs Tsuboyama ΔG_fold。含 jackknife/LOO/偏相关/几何阶梯/β稳健性。复用 gen/BioEmu_T3/*/operator_f.npz。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.special import logsumexp
PROJ=_GEC_ROOT; R=1.987204e-3; T=298.15; RT=R*T
sub=json.load(open(f"{PROJ}/data/processed/tier3_anchor_subset.json"))
t3=pd.read_parquet(f"{PROJ}/data/processed/tier3_tsuboyama_40_72.parquet"); ORIG=dict(zip(t3.name,t3.origin))
def proxies(NC,NH,bC,bH):
    lnPF=bC*NC+bH*NH                       # [R,Nconf]
    g=np.nanmean(lnPF,axis=0)              # 每构象整体保护度
    ens=-RT*(logsumexp(-g)-np.log(len(g))) # 配分函数全局代理(kcal/mol)
    dgres=RT*np.nanmean(lnPF,axis=1)       # 每残基系综平均 ΔG
    dgres=dgres[~np.isnan(dgres)]
    return dict(ens_open=ens, mean_res=RT*np.nanmean(g), p90=np.percentile(dgres,90), mx=dgres.max())
def build(bC,bH):
    rows=[]
    for npz in sorted(glob.glob(f"{PROJ}/gen/BioEmu_T3/*/operator_f.npz")):
        did=npz.split("/")[-2]
        if did not in sub: continue
        d=np.load(npz); P=proxies(d["NC"],d["NH"],bC,bH)
        rows.append(dict(domain=did, origin=ORIG.get(sub[did]["name"],"?"), exp=sub[did]["deltaG"], L=sub[did]["length"], **P))
    return pd.DataFrame(rows)
A=build(0.35,2.0)
print(f"=== E1 全局配分函数代理(n={len(A)}, β=0.35/2.0)===")
print(f"exp ΔG_fold 范围 {A.exp.min():.2f}-{A.exp.max():.2f}; ens_open 范围 {A.ens_open.min():.2f}-{A.ens_open.max():.2f} kcal/mol")
print("--- 几何阶梯对照(代理 vs ΔG_fold)---")
for c in ["mean_res","p90","mx","ens_open"]:
    pr,pp=pearsonr(A[c],A.exp); sr,sp=spearmanr(A[c],A.exp)
    print(f"  {c:<10} Pearson={pr:+.3f}(p={pp:.1e}) Spearman={sr:+.3f}")
print("--- ens_open 分层(origin)---")
for o in ["natural","de_novo"]:
    g=A[A.origin==o]
    if len(g)>=5: pr,pp=pearsonr(g.ens_open,g.exp); print(f"  [{o}] n={len(g)} Pearson={pr:+.3f}(p={pp:.1e}) Spearman={spearmanr(g.ens_open,g.exp)[0]:+.3f}")
# jackknife
idx=list(A.index); jk=[pearsonr(A.ens_open[[j for j in idx if j!=i]],A.exp[[j for j in idx if j!=i]])[0] for i in idx]
print(f"--- 稳健性 ---\n  jackknife Pearson {np.mean(jk):.3f}±{np.std(jk):.3f} [{min(jk):.3f},{max(jk):.3f}]")
# LOO affine
preds=[np.polyval(np.polyfit(A.ens_open.drop(i),A.exp.drop(i),1),A.ens_open[i]) for i in idx]
print(f"  LOO affine 预测 RMSE={np.sqrt(np.mean((np.array(preds)-A.exp.values)**2)):.3f} kcal/mol; 全拟合 slope={np.polyfit(A.ens_open,A.exp,1)[0]:.2f}")
# partial corr controlling L
def partial(x,y,z):
    rxy,rxz,ryz=pearsonr(x,y)[0],pearsonr(x,z)[0],pearsonr(y,z)[0]
    return (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2))
print(f"  L vs ΔG_fold Pearson={pearsonr(A.L,A.exp)[0]:+.3f}; 控长度偏相关 r(ens_open,exp|L)={partial(A.ens_open.values,A.exp.values,A.L.values):+.3f}")
print("--- β 稳健性(ens_open Pearson)---")
for bc,bh in [(0.35,2.0),(0.2,4.2),(0.2,2.3),(0.1,2.0),(0.5,1.0)]:
    B=build(bc,bh); print(f"  β=({bc},{bh}): Pearson={pearsonr(B.ens_open,B.exp)[0]:+.3f}")
A.to_csv(f"{PROJ}/results/tables/global_partition.csv",index=False)
print(f"→ results/tables/global_partition.csv ({len(A)}域)")
