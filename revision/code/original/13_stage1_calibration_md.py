"""Stage1 基底b:算子作用于短MD系综→vs实测lnPF→拟合β→与晶体基底(a)对比。"""
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
from scipy.stats import pearsonr, spearmanr
PROJ=_GEC_ROOT; LN10=np.log(10)
PDB={'Barnase':'1A2P','RNaseA':'1FS3','SNase':'1SNO','CI2':'2CI2','cytochrome_c':'1HRC'}
def expm(prot):
    if prot=='cytochrome_c':
        c=pd.read_csv(f"{PROJ}/data/processed/tier1a_classics/cytc_Milne1998.csv"); c=c[c.flag!='unprot']
        return dict(zip(c.resi.astype(int),c.logP.astype(float)*LN10))
    d=pd.read_parquet(f"{PROJ}/data/processed/tier1a_dGopen.parquet"); d=d[d.protein_id==prot]
    return dict(zip(d.resi.astype(int),d.log10PF.astype(float)*LN10))
maps=json.load(open(f'{PROJ}/data/processed/structure_maps.json'))
rows=[]; pooled=[]
for prot in PDB:
    xtc=f"{PROJ}/gen/MD_short/{prot}/ensemble.xtc"; top=f"{PROJ}/gen/MD_short/{prot}/ensemble.pdb"
    import os
    if not os.path.exists(xtc): rows.append(dict(protein=prot,note='MD系综缺失')); continue
    t=md.load(xtc, top=top)
    df,_=forward.compute(t, add_missing_atoms=False)  # MD全原子,只加氢
    e=expm(prot); off=maps.get(prot,{}).get('offset',0)
    df['exp_resi']=df.resSeq-off
    df['lnPF_exp']=df.exp_resi.map(lambda r: e.get(int(r),np.nan)); m=df.dropna(subset=['lnPF_exp'])
    if len(m)<5: rows.append(dict(protein=prot,n=len(m),note='对齐<5')); continue
    rows.append(dict(protein=prot,n=len(m),frames=t.n_frames,pearson_defaultβ=round(pearsonr(m.lnPF,m.lnPF_exp)[0],3),spearman=round(spearmanr(m.lnPF,m.lnPF_exp)[0],3)))
    for _,r in m.iterrows(): pooled.append((prot,r.NC_mean,r.NH_mean,r.lnPF_exp))
rep=pd.DataFrame(rows); print("=== Stage1 基底b(短MD系综,默认β)===\n",rep.to_string(index=False))
if pooled:
    P=pd.DataFrame(pooled,columns=['prot','NC','NH','lnPF_exp'])
    X=np.column_stack([P.NC,P.NH,np.ones(len(P))]); c,*_=np.linalg.lstsq(X,P.lnPF_exp,rcond=None)
    pred=X@c
    print(f"\n基底b OLS(含截距): βC={c[0]:.3f} βH={c[1]:.3f} b={c[2]:.2f}; 池化Pearson={pearsonr(pred,P.lnPF_exp)[0]:.3f}")
    cr=json.load(open(f"{PROJ}/operator/params/A_crystal.json"))
    print(f"对比晶体基底a: βC={cr['betaC_int']:.3f} βH={cr['betaH_int']:.3f}; 池化Pearson={cr['pooled_pearson']:.3f}")
    json.dump(dict(basis='short_MD_2ns',betaC=float(c[0]),betaH=float(c[1]),intercept=float(c[2]),pooled_pearson=float(pearsonr(pred,P.lnPF_exp)[0]),n_residues=int(len(P))),
              open(f"{PROJ}/operator/params/B_shortMD.json","w"),indent=1)
    rep.to_csv(f"{PROJ}/results/stats/stage1_calibration_md.csv",index=False)
    print("保存 operator/params/B_shortMD.json + results/stats/stage1_calibration_md.csv")
