"""帧级并行打分:对 gen/<model>/<prot>/samples.xtc 用多进程池跑满CPU补侧链+N_C/N_H→
ensemble平均→eval/pred + operator_f.npz(逐构象)。等价 forward.compute 但并行。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, os, glob, numpy as np, pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OPENMM_CPU_THREADS"]="1"
import mdtraj as md, multiprocessing as mp
PROJ=_GEC_ROOT; sys.path.insert(0,f"{PROJ}/operator")
import forward
R=1.987204e-3; T=298.15
import signal
def _to(sig,frm): raise TimeoutError()
def score_frame(fr):
    try:
        signal.signal(signal.SIGALRM,_to); signal.alarm(int(os.environ.get('FRAMETO','40')))
        r=forward._nc_nh(forward._fix_frame(fr, True), 0.65,0.26,3,2); signal.alarm(0); return r
    except Exception: signal.alarm(0); return None
def agg(prot, model, d, perframe):
    perframe=[p for p in perframe if p]
    if not perframe: print(f"{prot}: 全帧失败"); return
    allres=sorted(set().union(*[set(p) for p in perframe])); Fp=len(perframe)
    recs=[]; NCm=[]; NHm=[]
    for rs in allres:
        ncv=np.array([perframe[k][rs][1] if rs in perframe[k] else np.nan for k in range(Fp)],float)
        nhv=np.array([perframe[k][rs][2] if rs in perframe[k] else np.nan for k in range(Fp)],float)
        resn=next(perframe[k][rs][0] for k in range(Fp) if rs in perframe[k])
        nc=float(np.nanmean(ncv)); nh=float(np.nanmean(nhv)); lnpf=0.35*nc+2.0*nh
        recs.append(dict(resSeq=rs,resn=resn,NC_mean=nc,NH_mean=nh,lnPF=lnpf,log10PF=lnpf/np.log(10),dGopen_pred=R*T*lnpf))
        NCm.append(ncv); NHm.append(nhv)
    df=pd.DataFrame(recs); os.makedirs(f"{PROJ}/eval/pred/{model}",exist_ok=True)
    df.to_csv(f"{PROJ}/eval/pred/{model}/{prot}.csv",index=False)
    NCmat=np.array(NCm); NHmat=np.array(NHm); fA=R*T*(0.35*NCmat+2.0*NHmat)
    np.savez(f"{d}/operator_f.npz", resSeq=[r['resSeq'] for r in recs], NC=NCmat, NH=NHmat, fA=fA)
    print(f"{prot}: {Fp}帧→{len(df)}残基 ✓", flush=True)
def main():
    model=sys.argv[1] if len(sys.argv)>1 else "BioEmu"
    nproc=int(sys.argv[2]) if len(sys.argv)>2 else 48
    only=sys.argv[3].split(",") if len(sys.argv)>3 else None
    todo=[]
    for d in sorted(glob.glob(f"{PROJ}/gen/{model}/*")):
        prot=os.path.basename(d)
        if only and prot not in only: continue
        if os.path.exists(f"{d}/samples.xtc") and not os.path.exists(f"{PROJ}/eval/pred/{model}/{prot}.csv"):
            todo.append((prot,d))
    print(f"待打分: {[t[0] for t in todo]} (nproc={nproc})", flush=True)
    pool=mp.Pool(nproc)
    for prot,d in todo:
        traj=md.load(f"{d}/samples.xtc", top=f"{d}/topology.pdb"); Ftot=traj.n_frames
        MX=int(os.environ.get("MAXFRAMES","150"))
        if Ftot>MX: traj=traj[np.linspace(0,Ftot-1,MX).astype(int)]
        F=traj.n_frames
        perframe=pool.map(score_frame, [traj[k] for k in range(F)], chunksize=2)
        agg(prot, model, d, perframe)
    pool.close(); pool.join(); print("[并行打分完成]", flush=True)
if __name__=="__main__": main()
