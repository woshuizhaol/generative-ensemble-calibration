"""E1 多无泄漏对照: 各生成器 _T3 系综全局配分代理 ens_open vs Tsuboyama ΔG_fold。
BioEmu(ΔG泄漏) vs Boltz2/Str2Str/ESMFlow(无ΔG标签泄漏)→ 判定双尺度=真物理 or 泄漏。"""
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
PROJ=_GEC_ROOT; R=1.987204e-3;T=298.15;RT=R*T
sub=json.load(open(f"{PROJ}/data/processed/tier3_anchor_subset.json"))
def prox(npz,bC=0.35,bH=2.0):
    d=np.load(npz);lnPF=bC*d['NC']+bH*d['NH'];g=np.nanmean(lnPF,axis=0)
    return -RT*(logsumexp(-g)-np.log(len(g))), RT*np.nanmean(g), float(np.std(g))
gens=[("BioEmu_T3","泄漏(训过MEGAscale ΔG)"),("Boltz2_T3","无泄漏(单序列,无异质性)"),
      ("Str2Str_T3","无泄漏(纯PDB,零ΔG/MD,有异质性)"),("ESMFlow_T3","无泄漏(MD训练,无ΔG标签)")]
data={}
for g,_ in gens:
    data[g]={}
    for npz in glob.glob(f"{PROJ}/gen/{g}/*/operator_f.npz"):
        did=os.path.basename(os.path.dirname(npz))
        if did in sub: data[g][did]=(*prox(npz), sub[did]["deltaG"])
print("=== E1 多无泄漏对照: 全局配分代理 ens_open vs ΔG_fold ===")
rows=[]
for g,desc in gens:
    if len(data[g])<10: print(f"{g}: n={len(data[g])} 跳过(不足)"); continue
    doms=list(data[g]); exp=np.array([data[g][d][3] for d in doms])
    ens=np.array([data[g][d][0] for d in doms]); mean=np.array([data[g][d][1] for d in doms]); het=np.array([data[g][d][2] for d in doms])
    pe=pearsonr(ens,exp)[0]; pm=pearsonr(mean,exp)[0]
    print(f"{g:12s} [{desc}] n={len(doms)}: ens_open={pe:+.3f} mean_res={pm:+.3f} 异质性={het.mean():.2f}")
    rows.append(dict(generator=g,desc=desc,n=len(doms),ens_pearson=round(pe,3),mean_pearson=round(pm,3),het_mean=round(het.mean(),3)))
pd.DataFrame(rows).to_csv(f"{PROJ}/results/tables/e1_multicontrol.csv",index=False)
print("→ results/tables/e1_multicontrol.csv")
print("判读: 若无泄漏+有异质性的 Str2Str_T3 ens_open 也≈0.6 → 双尺度真物理(救回);若≈0 → 信号依赖BioEmu(泄漏)")
