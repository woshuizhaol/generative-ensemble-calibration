"""Stage1 标定(晶体基底a):5绝对蛋白晶体结构→算子→vs实测lnPF;相关+OLS拟合β+LOPO-CV。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,(_GEC_ROOT + '/operator'))
import forward, mdtraj as md
from scipy.stats import pearsonr, spearmanr
PROJ=_GEC_ROOT
LN10=np.log(10)
# 5绝对蛋白:实验 log10PF 来源
def exp_log10PF(prot):
    if prot=='cytochrome_c':
        c=pd.read_csv(f"{PROJ}/data/processed/tier1a_classics/cytc_Milne1998.csv")
        c=c[c.flag!='unprot']; return dict(zip(c.resi.astype(int), c.logP.astype(float)))
    d=pd.read_parquet(f"{PROJ}/data/processed/tier1a_dGopen.parquet")
    d=d[d.protein_id==prot]; return dict(zip(d.resi.astype(int), d.log10PF.astype(float)))
PDB={'Barnase':'1A2P','RNaseA':'1FS3','SNase':'1SNO','CI2':'2CI2','cytochrome_c':'1HRC'}
maps=json.load(open(f"{PROJ}/data/processed/structure_maps.json"))
rows=[]; pooled=[]
for prot,pdb in PDB.items():
    off=maps[prot]['offset']
    df,_=forward.run_file(f"{PROJ}/data/external/pdb/{pdb}.pdb")  # resSeq=PDB编号
    expm=exp_log10PF(prot)
    # 实验resi+off=PDB resSeq → 用 resSeq-off 对回实验resi
    df['exp_resi']=df.resSeq-off
    df['lnPF_exp']=df.exp_resi.map(lambda r: expm.get(int(r),np.nan)*LN10 if r in expm else np.nan)
    m=df.dropna(subset=['lnPF_exp'])
    if len(m)<5: rows.append(dict(protein=prot,n=len(m),note='对齐<5')); continue
    rP=pearsonr(m.lnPF,m.lnPF_exp)[0]; rS=spearmanr(m.lnPF,m.lnPF_exp)[0]
    rows.append(dict(protein=prot,pdb=pdb,n=len(m),pearson_defaultβ=round(rP,3),spearman=round(rS,3),
                     NC_rng=f"{m.NC_mean.min():.0f}-{m.NC_mean.max():.0f}",NH_rng=f"{m.NH_mean.min():.0f}-{m.NH_mean.max():.0f}"))
    for _,r in m.iterrows(): pooled.append((prot,r.NC_mean,r.NH_mean,r.lnPF_exp))
rep=pd.DataFrame(rows); print("=== 各蛋白(默认β=0.35/2.0)===\n",rep.to_string(index=False))
P=pd.DataFrame(pooled,columns=['prot','NC','NH','lnPF_exp'])
# OLS拟合β(池化,含截距与不含)
def fit(df,intercept=True):
    X=np.column_stack([df.NC,df.NH]+([np.ones(len(df))] if intercept else []))
    c,*_=np.linalg.lstsq(X,df.lnPF_exp.values,rcond=None); return c
c_ni=fit(P,False); c_i=fit(P,True)
predP=P.NC*c_i[0]+P.NH*c_i[1]+c_i[2]
print(f"\n池化OLS(无截距): βC={c_ni[0]:.3f} βH={c_ni[1]:.3f}")
print(f"池化OLS(含截距): βC={c_i[0]:.3f} βH={c_i[1]:.3f} b={c_i[2]:.2f}; 池化Pearson={pearsonr(predP,P.lnPF_exp)[0]:.3f} Spearman={spearmanr(predP,P.lnPF_exp)[0]:.3f}")
# LOPO-CV(含截距)
print("\n=== LOPO-CV ===")
cv=[]
for p in P.prot.unique():
    tr=P[P.prot!=p]; te=P[P.prot==p]; c=fit(tr,True)
    pr=te.NC*c[0]+te.NH*c[1]+c[2]
    if len(te)>=5: cv.append((p,len(te),round(pearsonr(pr,te.lnPF_exp)[0],3),round(spearmanr(pr,te.lnPF_exp)[0],3)))
cvdf=pd.DataFrame(cv,columns=['heldout','n','pearson','spearman']); print(cvdf.to_string(index=False))
print(f"LOPO-CV 平均 Pearson={cvdf.pearson.mean():.3f} Spearman={cvdf.spearman.mean():.3f}")
# 保存参数
import os; os.makedirs(f"{PROJ}/operator/params",exist_ok=True)
json.dump(dict(basis='crystal',betaC_noint=float(c_ni[0]),betaH_noint=float(c_ni[1]),
               betaC_int=float(c_i[0]),betaH_int=float(c_i[1]),intercept=float(c_i[2]),
               pooled_pearson=float(pearsonr(predP,P.lnPF_exp)[0]),lopo_pearson=float(cvdf.pearson.mean()),
               n_proteins=int(P.prot.nunique()),n_residues=int(len(P)),
               default_betaC=0.35,default_betaH=2.0),
          open(f"{PROJ}/operator/params/A_crystal.json","w"),indent=1)
rep.to_csv(f"{PROJ}/results/stats/stage1_calibration_crystal.csv",index=False)
print("\n保存 operator/params/A_crystal.json + results/stats/stage1_calibration_crystal.csv")
