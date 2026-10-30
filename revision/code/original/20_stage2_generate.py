"""Stage2 生成驱动:对一个蛋白用指定生成器产系综→算子(add_missing_atoms补侧链)→eval/pred/<model>/<prot>.csv。
用法: python 20_stage2_generate.py <model:BioEmu|ColabFold|Boltz2> <protein> <n_samples> <gpu>
"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, os, json, subprocess, numpy as np, pandas as pd
PROJ=_GEC_ROOT
model, prot, nsamp, gpu = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
seqs=json.load(open(f"{PROJ}/data/processed/tier1a_sequences.json")); seq=seqs[prot]['seq']
outdir=f"{PROJ}/gen/{model}/{prot}"; os.makedirs(outdir,exist_ok=True)
env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PROJ=PROJ,
         XDG_CACHE_HOME=f"{PROJ}/cache", HF_HOME=f"{PROJ}/cache/huggingface", HF_ENDPOINT="https://hf-mirror.com",
         COLABFOLD_DIR=f"{PROJ}/cache/colabfold", TORCH_HOME=f"{PROJ}/cache/torch", MPLCONFIGDIR=f"{PROJ}/cache/mpl",
         all_proxy="socks5h://localhost:18099", https_proxy="socks5h://localhost:18099", http_proxy="socks5h://localhost:18099")
def gen_bioemu():
    bs=10  # bioemu默认;内部 batch=int(bs*(100/len)^2),显存安全且大蛋白≥1
    cmd=["python","-m","bioemu.sample","--sequence",seq,"--num_samples",str(nsamp),"--output_dir",outdir,
         "--batch_size_100",str(bs),"--model_name","bioemu-v1.1",
         "--cache_so3_dir",f"{PROJ}/cache/sampling_so3_cache","--cache_embeds_dir",f"{PROJ}/cache/colabfold/embeds_cache"]
    py=f"{PROJ}/env/bioemu/bin/python"; cmd[0]=py
    subprocess.run(cmd, env=env, check=True)
    return f"{outdir}/samples.xtc", f"{outdir}/topology.pdb"
def gen_boltz():
    import glob as _g, mdtraj as md
    bdir=f"{outdir}/_in"; os.makedirs(bdir,exist_ok=True)
    open(f"{bdir}/{prot}.a3m","w").write(f">query\n{seq}\n")   # 单序列(离线可靠)
    open(f"{bdir}/{prot}.fasta","w").write(f">A|protein|{bdir}/{prot}.a3m\n{seq}\n")
    benv={k:v for k,v in env.items() if k not in ('all_proxy','https_proxy','http_proxy')}
    cmd=[f"{PROJ}/env/boltz/bin/boltz","predict",f"{bdir}/{prot}.fasta",
         "--diffusion_samples",str(nsamp),"--output_format","pdb","--out_dir",outdir,
         "--cache",f"{PROJ}/.cache/boltz","--override","--no_kernels","--num_workers","2"]
    subprocess.run(cmd, env=benv, check=True)
    pdbs=sorted(_g.glob(f"{outdir}/**/*model_*.pdb", recursive=True))
    if not pdbs: raise SystemExit("Boltz无输出PDB")
    frames=[md.load(p) for p in pdbs]; traj=frames[0].join(frames[1:]) if len(frames)>1 else frames[0]
    traj.save_xtc(f"{outdir}/samples.xtc"); traj[0].save_pdb(f"{outdir}/topology.pdb")
    return f"{outdir}/samples.xtc", f"{outdir}/topology.pdb"
# (ColabFold/Boltz2 分支后续补)
if model=="BioEmu": xtc,top=gen_bioemu()
elif model=="Boltz2": xtc,top=gen_boltz()
else: raise SystemExit(f"{model} 分支待补")
if os.environ.get("GEN_ONLY"): print(f"{model}/{prot}: 生成完成(GEN_ONLY,跳过打分)"); sys.exit(0)
# 算子(env/operator, CPU防openmm泄漏;生成子进程已单独拿GPU)
os.environ["CUDA_VISIBLE_DEVICES"]=""  # 算子CPU
import mdtraj as md; sys.path.insert(0,f"{PROJ}/operator"); import forward
t=md.load(xtc, top=top)
df,f=forward.compute(t, add_missing_atoms=True)
os.makedirs(f"{PROJ}/eval/pred/{model}",exist_ok=True)
df.rename(columns={'dGopen_kcal':'dGopen_pred'}).to_csv(f"{PROJ}/eval/pred/{model}/{prot}.csv",index=False)
np.savez(f"{outdir}/operator_f.npz", resSeq=f['resSeq'], NC=f['NC'], NH=f['NH'], fA=f['fA'])
print(f"{model}/{prot}: {t.n_frames}帧→{len(df)}残基 pred CSV + operator_f.npz")
