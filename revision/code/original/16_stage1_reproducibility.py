"""测算子确定性(pdbfixer加氢) + 晶体基标定 N=10 重复求均值±SD(解决复现性)。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, json, os, numpy as np, pandas as pd, mdtraj as md, warnings; warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
os.environ['OPENMM_CPU_THREADS']='1'
PROJ=_GEC_ROOT; sys.path.insert(0,f"{PROJ}/operator"); import forward
LN10=np.log(10); maps=json.load(open(f"{PROJ}/data/processed/structure_maps.json"))
PDB={'Barnase':'1A2P','RNaseA':'1FS3','SNase':'1SNO','CI2':'2CI2','cytochrome_c':'1HRC'}
def expm(p):
    if p=='cytochrome_c':
        c=pd.read_csv(f"{PROJ}/data/processed/tier1a_classics/cytc_Milne1998.csv"); c=c[c.flag!='unprot']
        return dict(zip(c.resi.astype(int),c.logP.astype(float)*LN10))
    d=pd.read_parquet(f"{PROJ}/data/processed/tier1a_dGopen.parquet"); d=d[d.protein_id==p]
    return dict(zip(d.resi.astype(int),d.log10PF.astype(float)*LN10))
exps={p:expm(p) for p in PDB}
def one_run(seed=None):
    if seed is not None: np.random.seed(seed)
    perres={}
    for p,pdb in PDB.items():
        t=md.load(f"{PROJ}/data/external/pdb/{pdb}.pdb"); t=t[0] if t.n_frames>1 else t
        nc=forward._nc_nh(forward._fix_frame(t,True),0.65,0.26,3,2); off=maps[p]['offset']
        rows=[(0.35*v[1]+2.0*v[2], exps[p][int(k)-off]) for k,v in nc.items() if (int(k)-off) in exps[p]]
        perres[p]=np.array(rows)
    return perres
# 确定性测试:同seed两次
r1=one_run(0); r2=one_run(0)
det=all(np.allclose(r1[p],r2[p]) for p in PDB)
print("固定seed确定性:", "确定(可复现)" if det else "仍非确定(openmm/pdbfixer内部)")
# N=10 重复
N=10; perp={p:[] for p in PDB}; pooled=[]; lopo=[]
for i in range(N):
    pr=one_run(seed=i)
    allr=[]
    for p in PDB:
        a=pr[p]; perp[p].append(pearsonr(a[:,0],a[:,1])[0]); allr.append((p,a))
    # 池化(含截距OLS)
    P=np.vstack([a for _,a in allr]); X=np.column_stack([P[:,0],np.ones(len(P))]); 
    import numpy.linalg as la; 
    # 简单池化Pearson(用pred=βC*NC... 这里a[:,0]已是lnPF_pred)
    pooled.append(pearsonr(P[:,0],P[:,1])[0])
    # LOPO
    ls=[]
    for p in PDB:
        tr=np.vstack([pr[q] for q in PDB if q!=p]); te=pr[p]
        b=np.polyfit(tr[:,0],tr[:,1],1); pred=np.polyval(b,te[:,0]); ls.append(pearsonr(pred,te[:,1])[0])
    lopo.append(np.mean(ls))
print("\n=== N=10 重复 均值±SD ===")
for p in PDB: print(f"  {p:<14} Pearson {np.mean(perp[p]):+.3f} ± {np.std(perp[p]):.3f}  [{min(perp[p]):+.2f},{max(perp[p]):+.2f}]")
print(f"  {'池化':<14} Pearson {np.mean(pooled):.3f} ± {np.std(pooled):.3f}")
print(f"  {'LOPO':<14} Pearson {np.mean(lopo):.3f} ± {np.std(lopo):.3f}")
out=pd.DataFrame([{'protein':p,'pearson_mean':round(np.mean(perp[p]),3),'pearson_sd':round(np.std(perp[p]),3),
                   'min':round(min(perp[p]),3),'max':round(max(perp[p]),3)} for p in PDB]+
                  [{'protein':'POOLED','pearson_mean':round(np.mean(pooled),3),'pearson_sd':round(np.std(pooled),3)},
                   {'protein':'LOPO','pearson_mean':round(np.mean(lopo),3),'pearson_sd':round(np.std(lopo),3)}])
out.to_csv(f"{PROJ}/results/stats/stage1_reproducibility.csv",index=False)
print("→ results/stats/stage1_reproducibility.csv")
