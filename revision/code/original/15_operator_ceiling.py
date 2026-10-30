"""Stage1.5 算子上限:对全12个Tier-1a蛋白在纯实验结构上跑算子,与实验残基级ΔG_open相关。
含身份验证(operator resn vs 实验resn)以catch offset错位。这是生成系综可达的"天花板"。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, json, pandas as pd, numpy as np, mdtraj as md, warnings, os; warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr
PROJ=_GEC_ROOT; sys.path.insert(0,f"{PROJ}/operator"); import forward
A1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def to1(x): x=str(x); return A1.get(x.upper(),x[0].upper()) if len(x)==3 else x[:1].upper()
maps=json.load(open(f"{PROJ}/data/processed/structure_maps.json"))
ea=pd.read_parquet(f"{PROJ}/data/processed/tier1a_all.parquet")
rows=[]
for p in sorted(ea.protein_id.unique()):
    e=ea[ea.protein_id==p].copy(); e['resn1']=e.resn.map(to1)
    pdb=maps[p]['pdb_id']; off=maps[p].get('offset',0); fp=f"{PROJ}/data/external/pdb/{pdb}.pdb"
    if not os.path.exists(fp): rows.append(dict(protein=p,pdb=pdb,note="结构缺")); continue
    t=md.load(fp); t=t[0] if t.n_frames>1 else t
    t1=forward._fix_frame(t,True); nc=forward._nc_nh(t1,0.65,0.26,3,2)
    df=pd.DataFrame([(int(k),v[0],0.35*v[1]+2.0*v[2]) for k,v in nc.items()],columns=['resSeq','resn_s','lnPF'])
    df['resi']=df.resSeq-off
    m=df.merge(e[['resi','dGopen_kcal','resn1','grade','ex2']],on='resi',how='inner')
    if len(m)<4: rows.append(dict(protein=p,pdb=pdb,offset=off,n=len(m),note="重叠<4")); continue
    idpct=100*np.mean([a==b for a,b in zip(m.resn_s,m.resn1)])
    pr=pearsonr(m.lnPF,m.dGopen_kcal)[0]; sr=spearmanr(m.lnPF,m.dGopen_kcal)[0]
    rows.append(dict(protein=p,pdb=pdb,offset=off,n=len(m),identity_pct=round(idpct,1),
                     pearson=round(pr,3),spearman=round(sr,3),grade=str(e.grade.iloc[0]),ex2=str(e.ex2.iloc[0])))
R=pd.DataFrame(rows)
print(R.to_string(index=False))
R.to_parquet(f"{PROJ}/results/operator_ceiling.parquet")
ok=R[R.identity_pct>=80] if 'identity_pct' in R else R
print(f"\n身份≥80%的蛋白: {len(ok)}/{len(R)}; 这些蛋白 Spearman 中位数={ok.spearman.median():.3f}" if 'spearman' in ok and len(ok) else "")
print("→ results/operator_ceiling.parquet (已用正确CI2=2CI2/offset1重建)")
