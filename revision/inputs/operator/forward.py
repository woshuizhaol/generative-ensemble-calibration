"""Best-Vendruscolo 前向算子(production, pdbfixer 验证版)。
每帧:取最大蛋白链→pdbfixer(可选补缺失重原子[晶体用]+加真实氢)→真实氢键。
N_C=酰胺N 6.5Å内|i-j|>=ij_Nc 非氢重原子;N_H=酰胺H与主链O<cut_Nh、|i-j|>=ij_Nh;
lnPF_i=βC⟨N_C⟩+βH⟨N_H⟩(系综平均后);ΔG_open=RT·lnPF。返回逐残基表+逐构象f_i(供BME)。
经 Stage1.5 验证:晶体上 barnase~0.60/SNase~0.70/cyt c~0.55(与原型/文献一致)。
"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import mdtraj as md, numpy as np, pandas as pd, tempfile, os, warnings
warnings.filterwarnings("ignore")
from pdbfixer import PDBFixer
from openmm.app import PDBFile
R_KCAL=0.0019872041
T2O={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
_SHM="/dev/shm" if os.path.isdir("/dev/shm") else None

def _fix_frame(traj1, add_missing_atoms):
    """单帧:取最大蛋白链→pdbfixer→返回带H的mdtraj单帧。"""
    top=traj1.topology
    ch=max(top.chains,key=lambda c:sum(1 for r in c.residues if r.name in T2O))
    t=traj1.atom_slice([a.index for a in ch.atoms])
    tmp=tempfile.mktemp(suffix='.pdb',dir=_SHM); t.save_pdb(tmp)
    fx=PDBFixer(filename=tmp); fx.findMissingResidues()
    fx.missingResidues={}   # 不补链内缺失整残基(避免引入大段建模)
    fx.findNonstandardResidues(); fx.findMissingAtoms()
    if not add_missing_atoms: fx.missingAtoms={}; fx.missingTerminals={}
    fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    out=tempfile.mktemp(suffix='.pdb',dir=_SHM); PDBFile.writeFile(fx.topology,fx.positions,open(out,'w'))
    r=md.load(out); os.remove(tmp); os.remove(out); return r

def _nc_nh(t1, cut_Nc, cut_Nh, ij_Nc, ij_Nh):
    top=t1.topology; xyz=t1.xyz[0]
    heavy=np.array([a.index for a in top.atoms if a.element and a.element.symbol!='H'])
    hv_res=np.array([top.atom(i).residue.index for i in heavy])
    Oc=np.array([a.index for a in top.atoms if a.name=='O']); o_res=np.array([top.atom(i).residue.index for i in Oc])
    out={}
    for res in top.residues:
        if res.name=='PRO' or res.name not in T2O: continue
        Ns=[a for a in res.atoms if a.name=='N']; Hs=[a for a in res.atoms if a.name in ('H','HN','H1')]
        if not Ns: continue
        ri=res.index; npos=xyz[Ns[0].index]
        NC=int(((np.linalg.norm(xyz[heavy]-npos,axis=1)<cut_Nc)&(np.abs(hv_res-ri)>=ij_Nc)).sum())
        NH=0
        if Hs and len(Oc):
            NH=int(((np.linalg.norm(xyz[Oc]-xyz[Hs[0].index],axis=1)<cut_Nh)&(np.abs(o_res-ri)>=ij_Nh)).sum())
        out[res.resSeq]=(T2O[res.name],NC,NH)
    return out

def compute(traj, weights=None, betaC=0.35, betaH=2.0, cut_Nc=0.65, cut_Nh=0.26,
            ij_Nc=3, ij_Nh=2, add_missing_atoms=True):
    F=traj.n_frames
    w=np.ones(F)/F if weights is None else np.asarray(weights,float)/np.sum(weights)
    perframe=[]
    for k in range(F):
        perframe.append(_nc_nh(_fix_frame(traj[k],add_missing_atoms),cut_Nc,cut_Nh,ij_Nc,ij_Nh))
    allres=sorted(set().union(*[set(d) for d in perframe]))
    recs=[]; NCm=[]; NHm=[]
    for rs in allres:
        ncv=np.array([perframe[k][rs][1] if rs in perframe[k] else np.nan for k in range(F)],float)
        nhv=np.array([perframe[k][rs][2] if rs in perframe[k] else np.nan for k in range(F)],float)
        resn=next(perframe[k][rs][0] for k in range(F) if rs in perframe[k])
        # 仅在出现该残基的帧上做权重平均
        msk=~np.isnan(ncv); ww=w[msk]/w[msk].sum() if msk.sum() else w
        nc=float(np.sum(ww*ncv[msk])); nh=float(np.sum(ww*nhv[msk]))
        lnPF=betaC*nc+betaH*nh
        recs.append(dict(resSeq=rs,resn=resn,NC_mean=nc,NH_mean=nh,lnPF=lnPF,log10PF=lnPF/np.log(10),dGopen_kcal=R_KCAL*298.15*lnPF))
        NCm.append(ncv); NHm.append(nhv)
    df=pd.DataFrame(recs)
    NCmat=np.array(NCm) if NCm else np.zeros((0,F)); NHmat=np.array(NHm) if NHm else np.zeros((0,F))
    fA=R_KCAL*298.15*(betaC*NCmat+betaH*NHmat)  # 变体A逐构象f_i(n)
    return df, dict(resSeq=[r['resSeq'] for r in recs], NC=NCmat, NH=NHmat, fA=fA, weights=w)

def run_file(path, **kw):
    return compute(md.load(path), **kw)

if __name__=='__main__':
    import sys
    df,f=run_file(sys.argv[1], add_missing_atoms=True)
    print(df.to_string(index=False)); print("n_res",len(df))
