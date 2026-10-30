"""E1泄漏对照: Boltz2单序列(未训ΔG,无泄漏)对Tsuboyama域产系综+算子→gen/Boltz2_T3/<域>/operator_f.npz。
用法: python 43_boltz_t3_control.py <domain:T3_xxx> <n_samples> <gpu>"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, os, json, subprocess, numpy as np, glob as _g
PROJ=_GEC_ROOT
did, nsamp, gpu = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seq=json.load(open(f"{PROJ}/data/processed/tier3_anchor_subset.json"))[did]["seq"]
outdir=f"{PROJ}/gen/Boltz2_T3/{did}"; os.makedirs(outdir,exist_ok=True)
if os.path.exists(f"{outdir}/operator_f.npz"): print(f"{did}: 已存在,跳过"); sys.exit(0)
env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PROJ=PROJ, XDG_CACHE_HOME=f"{PROJ}/cache",
         HF_HOME=f"{PROJ}/cache/huggingface", HF_ENDPOINT="https://hf-mirror.com",
         TORCH_HOME=f"{PROJ}/cache/torch", MPLCONFIGDIR=f"{PROJ}/cache/mpl")
import mdtraj as md
bdir=f"{outdir}/_in"; os.makedirs(bdir,exist_ok=True)
open(f"{bdir}/{did}.a3m","w").write(f">query\n{seq}\n")
open(f"{bdir}/{did}.fasta","w").write(f">A|protein|{bdir}/{did}.a3m\n{seq}\n")
cmd=[f"{PROJ}/env/boltz/bin/boltz","predict",f"{bdir}/{did}.fasta","--diffusion_samples",str(nsamp),
     "--output_format","pdb","--out_dir",outdir,"--cache",f"{PROJ}/.cache/boltz","--override","--no_kernels","--num_workers","2"]
subprocess.run(cmd, env=env, check=True)
pdbs=sorted(_g.glob(f"{outdir}/**/*model_*.pdb", recursive=True))
if not pdbs: raise SystemExit("Boltz无输出")
frames=[md.load(p) for p in pdbs]; traj=frames[0].join(frames[1:]) if len(frames)>1 else frames[0]
traj.save_xtc(f"{outdir}/samples.xtc"); traj[0].save_pdb(f"{outdir}/topology.pdb")
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0,f"{PROJ}/operator"); import forward
t=md.load(f"{outdir}/samples.xtc", top=f"{outdir}/topology.pdb")
df,f=forward.compute(t, add_missing_atoms=True)
np.savez(f"{outdir}/operator_f.npz", resSeq=f['resSeq'], NC=f['NC'], NH=f['NH'], fA=f['fA'])
print(f"{did}: {t.n_frames}帧→{len(df)}残基 operator_f.npz")
