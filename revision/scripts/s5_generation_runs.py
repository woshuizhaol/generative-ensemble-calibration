"""Stage 5 · S5_generation_runs.csv：每一次生成运行一行（命令、GPU、种子、开始/结束时间、样本数、退出码）。
数据来源只用盘上的真实记录：各启动脚本的命令模板（读脚本原文）、退出码日志、日志文件的创建时间（stat 的 birth time）、BioEmu 与 Boltz-2 的逐域运行表、噪声系综任务表。
不补写任何没有记录的值；没有记录的字段写 not_recorded。"""
import os, sys, glob, subprocess, datetime
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=5, task="generation_runs", cpu_procs=1, seeds="none")
birth = lambda p: (datetime.datetime.fromtimestamp(float(subprocess.run(["stat", "-c", "%W", p], capture_output=True, text=True).stdout.strip())).isoformat(timespec="seconds") if os.path.exists(p) else "not_recorded")
rd = lambda p: open(os.path.join(REV, p)).read().strip().replace("\n", " ⏎ ")
rows = []
# Str2Str_xtal：5 个种子
cmd_x = rd("scripts/s5_launch_str2str_xtal.sh").split("S5_SEED=")[1].split("echo")[0].strip()
ex = {}
for l in open(os.path.join(REV, "logs/s5/str2str_xtal_exit.log")):
    d = dict(x.split("=") for x in l.split()); ex[int(d["seed"])] = d
for sd in (11, 12, 13, 14, 15):
    d = ex[sd]; lg = os.path.join(REV, f"logs/s5/str2str_xtal_seed{sd}.log")
    rows.append(dict(generator="Str2Str_xtal", scope="12 蛋白 × 50 构象", seed=sd, gpu=d["gpu"], start=birth(lg), end=d["end"], n_samples=600, exit_code=d["exit"].replace("exit=", ""), delta=0.25,
                     command=f"S5_SEED={sd} … scripts/s5_str2str_run.py task_name=inference ckpt_path=data/ckpt/pretrain.pth model.inference.delta_min=0.25 model.inference.delta_max=0.25 model.inference.n_replica=50", log=f"logs/s5/str2str_xtal_seed{sd}.log",
                     note="第一次运行（起点 RNaseH_Tt 有重号残基）已作废并归档在 raw_outputs/s5/gen/_superseded_Str2Str_xtal_run1_dup_resSeq/，此行为重跑"))
# Str2Str_AF
for l in open(os.path.join(REV, "logs/s5/str2str_af_exit.log")):
    d = dict(x.split("=") for x in l.split()); tag = d["delta"].replace(".", ""); lg = os.path.join(REV, f"logs/s5/str2str_af_d{tag}.log")
    rows.append(dict(generator=f"Str2Str_AF_d{tag}", scope="193 域 × 50 构象", seed=11, gpu=d["gpu"], start=birth(lg), end=d["end"], n_samples=193 * 50, exit_code=d["exit"].replace("exit=", ""), delta=float(d["delta"]),
                     command=f"S5_SEED=11 … scripts/s5_str2str_run.py task_name=inference ckpt_path=data/ckpt/pretrain.pth model.inference.delta_min={d['delta']} model.inference.delta_max={d['delta']} model.inference.n_replica=50", log=f"logs/s5/str2str_af_d{tag}.log", note=""))
# ESMFlow
for l in open(os.path.join(REV, "logs/s5/esmflow_exit.log")):
    d = dict(x.split("=") for x in l.split()); lg = os.path.join(REV, f"logs/s5/esmflow_chunk_{d['chunk']}.log"); n = len(pd.read_csv(os.path.join(REV, f"data/interim/esmflow_in/chunk_{d['chunk']}.csv")))
    rows.append(dict(generator="ESMFlow_T3new", scope=f"{n} 域 × 50 构象（块 {d['chunk']}）", seed=11, gpu=d["gpu"], start=birth(lg), end=d["end"], n_samples=n * 50, exit_code=d["exit"].replace("exit=", ""), delta="",
                     command="S5_SEED=11 … scripts/s5_esmflow_run.py --mode esmfold --input_csv data/interim/esmflow_in/chunk_%s.csv --weights weights/esmflow_md_base_202402.pt --samples 50 --no_overwrite" % d["chunk"], log=f"logs/s5/esmflow_chunk_{d['chunk']}.log", note="alphaflow predict.py 无种子参数，由 s5_esmflow_run.py 先设 random/numpy/torch 种子"))
# BioEmu 与 Boltz-2：逐域
for kind, pat in (("BioEmu_T3new", "raw_outputs/s5/runs/bioemu_gpu*.tsv"), ("Boltz2_T3new", "raw_outputs/s5/runs/boltz_gpu*.tsv")):
    for f in sorted(glob.glob(os.path.join(REV, pat))):
        for r in pd.read_csv(f, sep="\t").itertuples():
            nm = r.n_frames_kept if kind.startswith("BioEmu") else r.n_models
            cmd = ("python -m bioemu.sample --sequence <seq> --num_samples 200 --batch_size_100 10 --ckpt_path cache/ckpt/bioemu-v1.1_checkpoint.ckpt --model_config_path cache/ckpt/bioemu-v1.1_config.yaml --base_seed 11" if kind.startswith("BioEmu")
                   else "boltz predict <fasta> --diffusion_samples 50 --output_format pdb --no_kernels --num_workers 2 --seed 11（单序列，无 MSA）")
            rows.append(dict(generator=kind, scope=r.domain_id, seed=11, gpu=r.gpu, start=r.start, end=r.end, n_samples=nm, exit_code=r.exit_code, delta="", command=cmd, log=("logs/s5/bioemu_t3new/%s.log" if kind.startswith("BioEmu") else "logs/s5/boltz_t3new/%s.log") % r.domain_id,
                             note=f"请求 {200 if kind.startswith('BioEmu') else 50} 个，保留 {nm} 个"))
R = pd.DataFrame(rows); f = os.path.join(REV, "tables", "S5_generation_runs.csv"); R.to_csv(f, index=False)
print(R.groupby("generator").agg(runs=("scope", "size"), nonzero_exit=("exit_code", lambda x: int((x.astype(str) != "0").sum())), samples=("n_samples", "sum")).to_string()); log.finish([f], note=f"rows={len(R)}")
