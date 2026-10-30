"""Stage 5B · Boltz-2 补跑：D_new ∪ D_legacy65 中还没有 Boltz-2 系综的域（147 个）。设置同原 43_boltz_t3_control.py：单序列（a3m 只含查询序列，无 MSA、无模板）、
--diffusion_samples 50、--output_format pdb、--no_kernels、--num_workers 2；方案要求种子统一设为 11：--seed 11（原运行未设种子）。每个域单独调用一次（同原运行），各域随机数流相互独立。
缓存用 R/cache/boltz（boltz2_conf.ckpt、boltz2_aff.ckpt、ccd.pkl、mols.tar，均在服务器本地，运行时不联网）。多泳道：os.mkdir 原子领取。
用法：python s5_boltz_t3new.py <gpu> [n_samples] [domain_id …]；不带域名则处理全部待补域。输出 raw_outputs/s5/gen/Boltz2_T3new/<域>/{samples.xtc, topology.pdb}；运行表 raw_outputs/s5/runs/boltz_gpu<g>.tsv。"""
import os, sys, glob, time, datetime, subprocess
import pandas as pd
gpu = sys.argv[1]; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 50; only = sys.argv[3:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from lib import revio  # noqa: E402
P, R, _ = revio.load_paths(); os.chdir(R)   # 路径唯一来源：config/paths.yaml
out_root = os.environ.get("S5_BOLTZ_OUT", "raw_outputs/s5/gen/Boltz2_T3new"); claims = os.environ.get("S5_BOLTZ_CLAIMS", "raw_outputs/s5/claims/boltz")
for d in (out_root, claims, "raw_outputs/s5/runs", "logs/s5/boltz_t3new"):
    os.makedirs(d, exist_ok=True)
dom = pd.read_csv("tables/S5_domain_table.csv"); dom = dom[dom.need_Boltz2_T3].sort_values("length", ascending=False)
if only:
    dom = dom[dom.domain_id.isin(only)]
env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HOME=R + "/cache/huggingface", TORCH_HOME=R + "/cache/torch", MPLCONFIGDIR=R + "/cache/mpl", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1")
runs = f"raw_outputs/s5/runs/boltz_gpu{gpu}.tsv"
if not os.path.exists(runs):
    open(runs, "w").write("domain_id\tlength\tgpu\tseed\tn_samples\tstart\tend\twall_s\texit_code\tn_models\tcommand\n")
import mdtraj as md
for r in dom.itertuples():
    out = f"{out_root}/{r.domain_id}"
    try:
        os.mkdir(f"{claims}/{r.domain_id}")
    except FileExistsError:
        continue
    bdir = f"{out}/_in"; os.makedirs(bdir, exist_ok=True)
    open(f"{bdir}/{r.domain_id}.a3m", "w").write(f">query\n{r.seq}\n"); open(f"{bdir}/{r.domain_id}.fasta", "w").write(f">A|protein|{os.path.abspath(bdir)}/{r.domain_id}.a3m\n{r.seq}\n")
    cmd = [P + "/env/boltz/bin/boltz", "predict", f"{bdir}/{r.domain_id}.fasta", "--diffusion_samples", str(ns), "--output_format", "pdb", "--out_dir", out, "--cache", R + "/cache/boltz",
           "--override", "--no_kernels", "--num_workers", "2", "--seed", "11"]
    t0 = time.time(); s = datetime.datetime.now().isoformat(timespec="seconds")
    rc = subprocess.run(cmd, env=env, stdout=open(f"logs/s5/boltz_t3new/{r.domain_id}.log", "w"), stderr=subprocess.STDOUT).returncode
    nm = -1
    if rc == 0:
        pdbs = sorted(glob.glob(f"{out}/**/*model_*.pdb", recursive=True))
        if pdbs:
            fr = [md.load(p) for p in pdbs]; tr = fr[0].join(fr[1:]) if len(fr) > 1 else fr[0]; tr.save_xtc(f"{out}/samples.xtc"); tr[0].save_pdb(f"{out}/topology.pdb"); nm = len(pdbs)
    open(runs, "a").write("\t".join(map(str, [r.domain_id, r.length, gpu, 11, ns, s, datetime.datetime.now().isoformat(timespec="seconds"), round(time.time() - t0), rc, nm, "boltz predict <fasta> --diffusion_samples %d --seed 11 --no_kernels" % ns])) + "\n")
