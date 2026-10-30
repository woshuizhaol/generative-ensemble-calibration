"""Stage 3 · BV 打分计时基准的“帧块并行”补跑与汇总（对 s3_benchmark.py 的补充，测量函数完全复用）。

为什么需要：s3_benchmark.py 对每个条件只起一个单线程进程串行跑完 3 次重复，UCH-L1 的 BioEmu 主链帧要补全全部侧链，
CPU 平台上每帧约 80–100 s，151 帧需要 3–4 小时，而服务器还有空闲核。本脚本把同一个条件拆成“重复 × 帧块”的独立任务并发执行：
  - 测量函数 `time_frame`、情形开关 `SCEN_FLAG`、汇总函数 `summarize` 全部从 s3_benchmark.py 导入，一个字不改；
  - 每个任务在自己的进程里先用 1 个不计时的帧预热，再逐帧计时，每帧三段（pdbfixer、计数、SASA）墙钟 + CPU 时间；
  - 每个重复的 50 帧被切成若干块，块之间并发；重复内的“每帧耗时均值”仍由 50 个帧的记录算出，与串行版口径相同；
  - 并发度与 s3_benchmark.py 首次运行相当（服务器负载 ≈ 50–60，核数 56，每核单线程），所以单帧耗时的争用环境可比。
子命令：
  run  --cond Prot:Gen:Scenario --reps 3 --chunks 4 --workers 12   并发补跑一个条件（已有分片自动跳过）
  agg                                                                把 bench_parts/ 下全部分片汇总成 benchmark_raw/summary/meta
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 打分是纯 CPU 流程；须在 import 任何 openmm/pdbfixer 之前设置，spawn 子进程会继承
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, time, json, argparse, platform, socket, subprocess
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402
import s3_benchmark as B  # noqa: E402  (只导入函数，不触发其 main)


def chunk_bounds(n, k):
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def run_chunk(job):
    prot, gen, scen, rep, k0, k1, outdir, gen_dir = job
    out = os.path.join(outdir, f"{prot}__{gen}__{scen}__r{rep}_f{k0:02d}-{k1:02d}.csv")
    if os.path.exists(out + ".done"):
        return out
    import mdtraj as md
    import openmm as mm
    from lib import bv
    _sys = mm.System(); _sys.addParticle(1.0)
    omm_platform = mm.Context(_sys, mm.VerletIntegrator(0.001)).getPlatform().getName()
    if omm_platform != "CPU":
        raise SystemExit(f"OpenMM 平台为 {omm_platform}，不是 CPU；中止以免占用他人的卡")
    d = os.path.join(gen_dir, gen, prot)
    chunk = next(md.iterload(os.path.join(d, "samples.xtc"), top=os.path.join(d, "topology.pdb"), chunk=50))
    flag = B.SCEN_FLAG[scen]
    try:
        B.time_frame(bv, md, chunk[k0], flag)   # 预热，不计入
    except Exception:
        pass
    rows = []
    for k in range(k0, k1):
        try:
            rec = B.time_frame(bv, md, chunk[k], flag)
        except Exception as e:
            rec = dict(t_pdbfixer=np.nan, t_count=np.nan, t_sasa=np.nan, cpu_pdbfixer=np.nan, cpu_count=np.nan, cpu_sasa=np.nan,
                       n_atoms_after=-1, n_res_scored=-1, status=f"fail:{type(e).__name__}:{str(e)[:80]}")
        rec.update(protein=prot, n_res=chunk.n_residues, generator=gen, scenario=scen, repeat=rep, frame_idx=k, n_atoms_in=chunk.n_atoms,
                   pid=os.getpid(), omm_platform=omm_platform)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out, index=False)
    open(out + ".done", "w").write("ok\n")
    return out


def cmd_run(a):
    PROJ, REV, INP = revio.load_paths()
    outdir = os.path.join(REV, "raw_outputs", "s3", "bench_parts_chunked"); os.makedirs(outdir, exist_ok=True)   # 与串行版的 bench_parts 分开，互不覆盖
    prot, gen, scen = a.cond.split(":")
    log = revio.RunLogger(stage=3, task="benchmark_chunked", cpu_procs=a.workers,
                          seeds="frames: first 50 (deterministic); hydrogen placement uses unseeded python random")
    reps = [r for r in range(a.rep_from, a.rep_from + a.reps)]
    jobs = [(prot, gen, scen, r, k0, k1, outdir, INP["gen_dir"]) for r in reps for k0, k1 in chunk_bounds(50, a.chunks)]
    print(f"{a.cond}: {len(jobs)} tasks, reps {reps}, workers {a.workers}, load {os.getloadavg()}", flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(run_chunk, jobs):
            print("done", os.path.basename(r), f"{time.time() - t0:.0f}s", flush=True)
    log.finish([outdir], note=f"{a.cond} reps={reps} chunks={a.chunks} wall={time.time() - t0:.0f}s")


def cmd_agg(a):
    PROJ, REV, INP = revio.load_paths()
    raw_dir = os.path.join(REV, "raw_outputs", "s3")
    log = revio.RunLogger(stage=3, task="benchmark_aggregate", cpu_procs=1, seeds="none")
    ser_dir, chk_dir = os.path.join(raw_dir, "bench_parts"), os.path.join(raw_dir, "bench_parts_chunked")
    ser = [os.path.join(ser_dir, f) for f in sorted(os.listdir(ser_dir)) if f.endswith(".csv") and os.path.exists(os.path.join(ser_dir, f + ".done"))]
    chk = ([os.path.join(chk_dir, f) for f in sorted(os.listdir(chk_dir)) if f.endswith(".csv") and os.path.exists(os.path.join(chk_dir, f + ".done"))]
           if os.path.isdir(chk_dir) else [])
    S = pd.concat([pd.read_csv(p) for p in ser], ignore_index=True)
    C = pd.concat([pd.read_csv(p) for p in chk], ignore_index=True) if chk else pd.DataFrame(columns=S.columns)
    key = ["protein", "generator", "scenario"]
    both = S[key].drop_duplicates().merge(C[key].drop_duplicates(), on=key) if len(C) else pd.DataFrame(columns=key)
    if len(both):   # 同一条件两处都有 → 以串行原版（协议原样）为准，分块版弃用并记录
        print("两处都有的条件，保留串行版、弃用分块版:", both.values.tolist(), flush=True)
        C = C.merge(both.assign(_drop=1), on=key, how="left"); C = C[C._drop.isna()].drop(columns="_drop")
    raw = pd.concat([S, C], ignore_index=True)
    parts = ser + chk
    assert set(raw.omm_platform) == {"CPU"}, f"存在非 CPU 平台的记录: {set(raw.omm_platform)}"
    dup = raw.duplicated(["protein", "generator", "scenario", "repeat", "frame_idx"]).sum()
    assert dup == 0, f"存在重复记录 {dup} 条"
    f_raw = os.path.join(raw_dir, "benchmark_raw.csv"); raw.to_csv(f_raw, index=False)
    summ = B.summarize(raw); f_sum = os.path.join(raw_dir, "benchmark_summary.csv"); summ.to_csv(f_sum, index=False)
    import mdtraj, openmm, pdbfixer
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    meta = dict(host=socket.gethostname(), cpu_model=[l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")][0],
                n_cpus=os.cpu_count(), loadavg_at_aggregation=os.getloadavg(), python=platform.python_version(), mdtraj=mdtraj.__version__,
                openmm=openmm.__version__, pdbfixer=getattr(pdbfixer, "__version__", "1.12.0 (dist-info)"),
                threads_env="OMP/MKL/OPENBLAS/OPENMM_CPU_THREADS=1", omm_platform_all_records="CPU (断言)", cuda_visible_devices="",
                cv_limit=B.CV_LIMIT, frames_per_repeat=50, warmup="1 untimed frame per condition (per task in chunked mode)",
                n_part_files=len(parts), n_records=len(raw),
                note="工作进程并发（每任务一个单线程进程），与服务器其他用户负载共享 CPU；墙钟含争用，CPU 时间列更接近 core-s。"
                     "多数条件由 s3_benchmark.py 串行跑完 3 次重复；UCH-L1×BioEmu×sidechain_plus_H 由 s3_benchmark_chunked.py 按帧块并发补跑，测量函数相同")
    f_meta = os.path.join(raw_dir, "benchmark_meta.json"); revio.write_json(meta, f_meta)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))
    log.finish([f_raw, f_sum, f_meta], note=f"conditions={len(summ)} records={len(raw)} failed={int((raw.status != 'ok').sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--cond", required=True); r.add_argument("--reps", type=int, default=3)
    r.add_argument("--rep-from", type=int, default=1); r.add_argument("--chunks", type=int, default=4); r.add_argument("--workers", type=int, default=12)
    sp.add_parser("agg")
    a = ap.parse_args()
    {"run": cmd_run, "agg": cmd_agg}[a.cmd](a)
