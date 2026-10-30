"""E2: 学习式算子留一蛋白CV(LOPO)证明 BV 算子非瓶颈。特征=逐构象NC/NH统计+SASA+DSSP+序列疏水性;
模型 Ridge/RF/GBM;严格 LOPO vs BV;配对bootstrap;RF特征重要性;因子分解(系综杠杆 vs 算子杠杆)。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import os,sys,json,glob,numpy as np,pandas as pd,mdtraj as md,warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge; from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
PROJ=_GEC_ROOT; R=1.987204e-3; T=298.15
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'E':-3.5,'Q':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
maps=json.load(open(f"{PROJ}/data/processed/structure_maps.json")); exp=pd.read_parquet(f"{PROJ}/data/processed/tier1a_all.parquet")
A1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def to1(x): x=str(x); return A1.get(x.upper(),x[:1].upper()) if len(x)==3 else (x[:1].upper() if x.strip() else '')
def best_off(pred_res, g):
    eid={int(r.resi):to1(r.resn) for _,r in g.iterrows() if str(r.resn).strip()}
    if len(eid)<5: return maps.get(g.protein_id.iloc[0],{}).get('gen_offset',0)
    best,bg=0,-1
    for o in range(-8,9):
        gg=sum(1 for er,aa in eid.items() if pred_res.get(er+o)==aa)
        if gg>bg: bg,best=gg,o
    return best
def feats_for(prot):
    npz=f"{PROJ}/gen/BioEmu/{prot}/operator_f.npz"; xtc=f"{PROJ}/gen/BioEmu/{prot}/samples.xtc"; top=f"{PROJ}/gen/BioEmu/{prot}/topology.pdb"
    if not os.path.exists(npz): return None
    d=np.load(npz); resSeq=d["resSeq"].astype(int); NC=d["NC"]; NH=d["NH"]  # [Rr,Nconf]
    pred=pd.read_csv(f"{PROJ}/eval/pred/BioEmu/{prot}.csv"); resn={int(r.resSeq):to1(r.resn) for _,r in pred.iterrows()}
    lnPF=0.35*NC+2.0*NH
    F=dict(NC_mean=np.nanmean(NC,1),NC_std=np.nanstd(NC,1),NC_p10=np.nanpercentile(NC,10,1),NC_p90=np.nanpercentile(NC,90,1),
           NH_mean=np.nanmean(NH,1),NH_std=np.nanstd(NH,1),NH_p90=np.nanpercentile(NH,90,1),
           fopen=np.nanmean(lnPF<np.nanmedian(lnPF),1),NCxNH=np.nanmean(NC*NH,1),lnPF_std=np.nanstd(lnPF,1))
    # SASA/DSSP(下采样50帧)
    try:
        t=md.load(xtc,top=top); t=t[np.linspace(0,t.n_frames-1,min(50,t.n_frames)).astype(int)]
        sasa=md.shrake_rupley(t,mode='residue').mean(0)  # [Rres]
        dssp=md.compute_dssp(t,simplified=True)  # [F,Rres]
        rs2i={r.resSeq:i for i,r in enumerate(t.topology.residues)}
        F['SASA']=np.array([sasa[rs2i[r]] if r in rs2i else np.nan for r in resSeq])
        F['fhelix']=np.array([np.mean(dssp[:,rs2i[r]]=='H') if r in rs2i else np.nan for r in resSeq])
        F['fsheet']=np.array([np.mean(dssp[:,rs2i[r]]=='E') if r in rs2i else np.nan for r in resSeq])
    except Exception as e:
        for k in ['SASA','fhelix','fsheet']: F[k]=np.full(len(resSeq),np.nan)
    F['KD']=np.array([KD.get(resn.get(int(r),'X'),0.0) for r in resSeq])
    df=pd.DataFrame(F); df['resSeq']=resSeq; df['BV']=0.35*F['NC_mean']+2.0*F['NH_mean']
    g=exp[exp.protein_id==prot]; off=best_off(resn,g); df['resi']=df.resSeq-off
    m=df.merge(g[['resi','dGopen_kcal']],on='resi',how='inner').dropna(subset=['dGopen_kcal'])
    m['protein']=prot; return m
prots=[os.path.basename(d) for d in glob.glob(f"{PROJ}/gen/BioEmu/*") if os.path.exists(f"{d}/operator_f.npz")]
data=pd.concat([x for x in (feats_for(p) for p in sorted(prots)) if x is not None and len(x)>=8],ignore_index=True)
FEATS=['NC_mean','NC_std','NC_p10','NC_p90','NH_mean','NH_std','NH_p90','fopen','NCxNH','lnPF_std','SASA','fhelix','fsheet','KD']
data[FEATS]=data[FEATS].fillna(data[FEATS].median())
print(f"=== E2 学习式算子 LOPO(蛋白{data.protein.nunique()} 残基{len(data)} 特征{len(FEATS)})===")
def lopo(model_fn,cols):
    sp=[]
    for p in sorted(data.protein.unique()):
        tr=data[data.protein!=p]; te=data[data.protein==p]
        if len(te)<5: continue
        mdl=model_fn(); mdl.fit(tr[cols],tr.dGopen_kcal); pred=mdl.predict(te[cols])
        sp.append(spearmanr(pred,te.dGopen_kcal)[0])
    return np.nanmean(sp),sp
bv=[spearmanr(data[data.protein==p].BV,data[data.protein==p].dGopen_kcal)[0] for p in sorted(data.protein.unique()) if len(data[data.protein==p])>=5]
print(f"  BV(物理基线) LOPO Spearman = {np.nanmean(bv):.3f}")
res={}
for name,fn,cols in [("Ridge[NC,NH]",lambda:Ridge(alpha=1.0),['NC_mean','NH_mean']),
                     ("Ridge[full]",lambda:Ridge(alpha=1.0),FEATS),
                     ("RF[full]",lambda:RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=4),FEATS),
                     ("GBM[full]",lambda:GradientBoostingRegressor(random_state=0),FEATS)]:
    m,sp=lopo(fn,cols); res[name]=(m,sp); print(f"  {name:<14} LOPO Spearman = {m:.3f}")
best=max(res,key=lambda k:res[k][0]); bm,bsp=res[best]
# 配对bootstrap delta(最优学习 vs BV)
rng=np.random.default_rng(0); pj=[p for p in sorted(data.protein.unique()) if len(data[data.protein==p])>=5]
diffs=np.array(bsp)-np.array(bv); boot=[np.mean(rng.choice(diffs,len(diffs))) for _ in range(10000)]
print(f"--- 最优={best}({bm:.3f}) vs BV({np.nanmean(bv):.3f}): Δ={bm-np.nanmean(bv):+.3f}, 配对bootstrap 95%CI [{np.percentile(boot,2.5):+.3f},{np.percentile(boot,97.5):+.3f}] {'跨0(无增益)' if np.percentile(boot,2.5)<0<np.percentile(boot,97.5) else '不跨0'} ---")
rf=RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=4).fit(data[FEATS],data.dGopen_kcal)
imp=sorted(zip(FEATS,rf.feature_importances_),key=lambda x:-x[1])[:6]
print("  RF特征重要性 top6:",", ".join(f"{k}={v:.3f}" for k,v in imp))
pd.DataFrame([dict(model=k,lopo_spearman=round(v[0],3)) for k,v in {**{'BV':(np.nanmean(bv),bv)},**res}.items()]).to_csv(f"{PROJ}/results/tables/learned_operator_lopo.csv",index=False)
print(f"→ results/tables/learned_operator_lopo.csv")
