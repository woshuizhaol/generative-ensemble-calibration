"""Stage2 全局锚定(main-line③):对Tier-3小域系综,operator高保护残基ΔG_open(全局解折叠代理)
vs Tsuboyama全局ΔG_fold,跨域相关。代理用 top-k 残基 lnPF 的中位 ΔG。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, os, glob, json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
PROJ=_GEC_ROOT; R=1.987204e-3; T=298.15
sub=json.load(open(f"{PROJ}/data/processed/tier3_anchor_subset.json"))
_t3=pd.read_parquet(f"{PROJ}/data/processed/tier3_tsuboyama_40_72.parquet"); ORIG=dict(zip(_t3.name,_t3.origin))
rows=[]
for did,info in sub.items():
    npz=f"{PROJ}/gen/BioEmu_T3/{did}/operator_f.npz"
    if not os.path.exists(npz): continue
    d=np.load(npz); NC=d["NC"]; NH=d["NH"]
    dg=R*T*(0.35*np.nanmean(NC,1)+2.0*np.nanmean(NH,1))  # 每残基ΔG_open
    dg=dg[~np.isnan(dg)]
    if len(dg)<5: continue
    # 全局代理:最高保护残基(global unfolding决定),用top-10%中位 + max
    top=np.sort(dg)[::-1]; k=max(1,len(dg)//10)
    rows.append(dict(domain=did, origin=ORIG.get(info["name"],"?"), exp_dGfold=info["deltaG"], pred_top10med=float(np.median(top[:k])),
                     pred_max=float(top[0]), pred_p90=float(np.percentile(dg,90)), n_res=len(dg)))
A=pd.DataFrame(rows)
if len(A)>=5:
    A.to_csv(f"{PROJ}/results/tables/global_anchor.csv",index=False)
    for org in ["natural","de_novo"]:
        g=A[A.origin==org]
        if len(g)>=5:
            pr=pearsonr(g.pred_p90,g.exp_dGfold)[0]; sr=spearmanr(g.pred_p90,g.exp_dGfold)[0]
            print(f"  [{org}] n={len(g)}: p90-ΔG_open vs ΔG_fold Pearson={pr:.3f} Spearman={sr:.3f}")
    pr=pearsonr(A.pred_p90,A.exp_dGfold)[0]; print(f"  [全部] n={len(A)}: p90 Pearson={pr:.3f} Spearman={spearmanr(A.pred_p90,A.exp_dGfold)[0]:.3f}")
    print(f"→ results/tables/global_anchor.csv ({len(A)}域)")
else: print(f"锚定域不足({len(A)});Tier-3生成后运行")
