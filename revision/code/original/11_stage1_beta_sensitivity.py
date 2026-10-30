"""Stage1 β敏感性:5晶体prep一次,扫 cut_Nc/ij_Nc/cut_Nh,拟合β+池化Pearson,看稳健性。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,(_GEC_ROOT + '/operator'))
import forward, mdtraj as md
from scipy.stats import pearsonr
PROJ=_GEC_ROOT; LN10=np.log(10)
MAPS=json.load(open(f"{PROJ}/data/processed/structure_maps.json"))
PDB={'Barnase':'1A2P','RNaseA':'1FS3','SNase':'1SNO','CI2':'2CI2','cytochrome_c':'1HRC'}
def expm(prot):
    if prot=='cytochrome_c':
        c=pd.read_csv(f"{PROJ}/data/processed/tier1a_classics/cytc_Milne1998.csv"); c=c[c.flag!='unprot']
        return dict(zip(c.resi.astype(int),c.logP.astype(float)*LN10))
    d=pd.read_parquet(f"{PROJ}/data/processed/tier1a_dGopen.parquet"); d=d[d.protein_id==prot]
    return dict(zip(d.resi.astype(int),d.log10PF.astype(float)*LN10))
# prep 一次(pdbfixer 补全+加氢)
fixed={p:forward._fix_frame(md.load(f"{PROJ}/data/external/pdb/{pdb}.pdb")[0],True) for p,pdb in PDB.items()}
exps={p:expm(p) for p in PDB}
def sweep(cut_Nc,ij_Nc,cut_Nh,ij_Nh):
    pooled=[]
    for p in PDB:
        off=MAPS.get(p,{}).get('offset',0)
        nn=forward._nc_nh(fixed[p],cut_Nc,cut_Nh,ij_Nc,ij_Nh); e=exps[p]
        for rs,(resn,nc,nh) in nn.items():
            er=rs-off
            if er in e: pooled.append((nc,nh,e[er]))
    P=pd.DataFrame(pooled,columns=['NC','NH','lnPFexp'])
    X=np.column_stack([P.NC,P.NH,np.ones(len(P))]); c,*_=np.linalg.lstsq(X,P.lnPFexp,rcond=None)
    pred=X@c; return c[0],c[1],c[2],pearsonr(pred,P.lnPFexp)[0],len(P)
rows=[]
for cnc in [0.55,0.65,0.80]:
    for ij in [2,3]:
        for cnh in [0.24,0.26]:
            bc,bh,b0,r,n=sweep(cnc,ij,cnh,2)
            rows.append(dict(cut_Nc=cnc,ij_Nc=ij,cut_Nh=cnh,betaC=round(bc,3),betaH=round(bh,3),intercept=round(b0,2),pooled_pearson=round(r,3),n=n))
S=pd.DataFrame(rows); print(S.to_string(index=False))
print(f"\nβC 变异系数(CV)={S.betaC.std()/S.betaC.mean():.2f}; βH CV={S.betaH.std()/S.betaH.mean():.2f}; Pearson范围 {S.pooled_pearson.min()}-{S.pooled_pearson.max()}")
S.to_csv(f"{PROJ}/results/stats/stage1_beta_sensitivity.csv",index=False)
print("保存 results/stats/stage1_beta_sensitivity.csv")
