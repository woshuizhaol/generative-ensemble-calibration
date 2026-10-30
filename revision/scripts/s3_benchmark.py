"""Stage 3 · BV 打分计时基准（revision_proposal.md Stage 3 步骤 3；对应 R2-m8）。

对 Im7（86 aa）、SNase（136 aa）、UCH-L1（223 aa）各取生成器输出的前 50 帧（xtc 帧序 0–49，不抽样；帧选择无随机性，但 PDBFixer/OpenMM 放氢的初始位置用 Python 全局 random，计时不受影响、打分值本身带微小随机性），
在单核上逐帧计时 BV 打分的三段：
  t_pdbfixer  bv.fix_frame（取最大链 → PDBFixer 补原子 + pH 7.0 加氢 → 重读），即原稿打分的第一步；
  t_count     bv.nc_nh（N_C / N_H 计数）；
  t_sasa      mdtraj.shrake_rupley(probe_radius=0.14 nm, n_sphere_points=960, mode='residue')（S4 将用的 SASA 参数）。
条件（每个蛋白）：
  ESMFlow 全原子帧   × {H_only: add_missing_atoms=False, sidechain_plus_H: add_missing_atoms=True}
  BioEmu 主链 + CB 帧 × {sidechain_plus_H}；H_only 对主链帧在 PDBFixer 处必然报错（缺侧链原子），只取前 5 帧留下失败记录。
每个条件重复 3 次（每次 50 帧，串行，单线程）；若 3 次的「每帧总耗时」或「pdbfixer 段」变异系数 ≥ 20%，加测到 5 次。
每个条件正式计时前先做 1 帧不计入的预热（载入 OpenMM 力场模板等一次性开销）。
同时记录墙钟（time.perf_counter，方案规定）与进程 CPU 时间（time.process_time）；服务器有他人负载，CPU 时间更接近 core-s。

用法：python s3_benchmark.py [--workers 9] [--frames 50]
输出：raw_outputs/s3/benchmark_raw.csv（逐帧）、raw_outputs/s3/benchmark_summary.csv（逐条件）、raw_outputs/s3/benchmark_meta.json
可重入：每个条件一个分片文件 raw_outputs/s3/bench_parts/*.csv(+.done)，已完成的条件不重跑。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 打分是纯 CPU 流程；见 lib/bv.py 顶部说明。须在 import 任何 openmm/pdbfixer 之前设置，spawn 子进程会继承
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, time, json, argparse, platform, socket, subprocess
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROTEINS = [("Im7", 86), ("SNase", 136), ("UCH-L1", 223)]
SCEN_FLAG = {"H_only": False, "sidechain_plus_H": True}
CV_LIMIT = 0.20


def time_frame(bv, md, fr, add_missing):
    c0, w0 = time.process_time(), time.perf_counter()
    t1 = bv.fix_frame(fr, add_missing_atoms=add_missing)
    c1, w1 = time.process_time(), time.perf_counter()
    r = bv.nc_nh(t1)
    c2, w2 = time.process_time(), time.perf_counter()
    md.shrake_rupley(t1, probe_radius=0.14, n_sphere_points=960, mode="residue")
    c3, w3 = time.process_time(), time.perf_counter()
    return dict(t_pdbfixer=w1 - w0, t_count=w2 - w1, t_sasa=w3 - w2,
                cpu_pdbfixer=c1 - c0, cpu_count=c2 - c1, cpu_sasa=c3 - c2,
                n_atoms_after=t1.n_atoms, n_res_scored=len(r), status="ok")


def run_task(job):
    prot, gen, scen, nframes, outdir, gen_dir = job
    import mdtraj as md
    from lib import bv
    out = os.path.join(outdir, f"{prot}__{gen}__{scen}.csv")
    if os.path.exists(out + ".done"):
        return out
    import openmm as mm
    _sys = mm.System(); _sys.addParticle(1.0)
    omm_platform = mm.Context(_sys, mm.VerletIntegrator(0.001)).getPlatform().getName()   # OpenMM 实际选中的计算平台
    if omm_platform != "CPU":
        raise SystemExit(f"OpenMM 平台为 {omm_platform}，不是 CPU（GPU 未被隐藏？）；中止以免占用他人的卡")
    d = os.path.join(gen_dir, gen, prot)
    chunk = next(md.iterload(os.path.join(d, "samples.xtc"), top=os.path.join(d, "topology.pdb"), chunk=nframes))
    flag = SCEN_FLAG[scen]
    probe_only = (gen == "BioEmu" and scen == "H_only")   # 主链帧 + 只加氢：只留失败记录
    rows = []

    def one(rep, k):
        try:
            rec = time_frame(bv, md, chunk[k], flag)
        except Exception as e:  # 失败帧不丢弃，记录下来
            rec = dict(t_pdbfixer=np.nan, t_count=np.nan, t_sasa=np.nan, cpu_pdbfixer=np.nan, cpu_count=np.nan,
                       cpu_sasa=np.nan, n_atoms_after=-1, n_res_scored=-1, status=f"fail:{type(e).__name__}:{str(e)[:80]}")
        rec.update(protein=prot, n_res=chunk.n_residues, generator=gen, scenario=scen, repeat=rep, frame_idx=k,
                   n_atoms_in=chunk.n_atoms, pid=os.getpid(), omm_platform=omm_platform)
        return rec

    if probe_only:
        for k in range(5):
            rows.append(one(1, k))
    else:
        try:
            time_frame(bv, md, chunk[0], flag)   # 预热，不计入
        except Exception:
            pass
        rep = 0
        while True:
            rep += 1
            for k in range(chunk.n_frames):
                rows.append(one(rep, k))
            if rep >= 3:
                df = pd.DataFrame(rows)
                per = df[df.status == "ok"].groupby("repeat").agg(tot=("t_pdbfixer", "mean"), fx=("t_pdbfixer", "mean"),
                                                                 c=("t_count", "mean"), s=("t_sasa", "mean"))
                per["tot"] = per.fx + per.c + per.s
                cv_tot = per.tot.std(ddof=1) / per.tot.mean(); cv_fx = per.fx.std(ddof=1) / per.fx.mean()
                if (cv_tot < CV_LIMIT and cv_fx < CV_LIMIT) or rep >= 5:
                    break
    pd.DataFrame(rows).to_csv(out, index=False)
    open(out + ".done", "w").write("ok\n")
    return out


def summarize(raw):
    ok = raw[raw.status == "ok"].copy()
    ok["t_total"] = ok.t_pdbfixer + ok.t_count + ok.t_sasa
    ok["cpu_total"] = ok.cpu_pdbfixer + ok.cpu_count + ok.cpu_sasa
    keys = ["protein", "n_res", "generator", "scenario"]
    rep = ok.groupby(keys + ["repeat"]).agg(w=("t_total", "mean"), fx=("t_pdbfixer", "mean"), c=("t_count", "mean"),
                                           s=("t_sasa", "mean"), cpu=("cpu_total", "mean")).reset_index()
    def cv(x):
        return float(x.std(ddof=1) / x.mean()) if len(x) > 1 else np.nan
    g = rep.groupby(keys).agg(n_repeats=("repeat", "nunique"), wall_s_per_frame=("w", "mean"), cv_total=("w", cv),
                              pdbfixer_s=("fx", "mean"), cv_pdbfixer=("fx", cv), count_s=("c", "mean"), cv_count=("c", cv),
                              sasa_s=("s", "mean"), cv_sasa=("s", cv), cpu_core_s_per_frame=("cpu", "mean")).reset_index()
    nfail = raw[raw.status != "ok"].groupby(keys).size().rename("n_failed_frames").reset_index()
    nfr = raw.groupby(keys).size().rename("n_frame_records").reset_index()
    g = g.merge(nfr, on=keys, how="outer").merge(nfail, on=keys, how="left")
    g["n_failed_frames"] = g.n_failed_frames.fillna(0).astype(int)
    g["cv_criterion_pass"] = (g.cv_total < CV_LIMIT) & (g.cv_pdbfixer < CV_LIMIT)
    return g


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=9); ap.add_argument("--frames", type=int, default=50)
    a = ap.parse_args()
    PROJ, REV, INP = revio.load_paths()
    outdir = os.path.join(REV, "raw_outputs", "s3", "bench_parts"); os.makedirs(outdir, exist_ok=True)
    log = revio.RunLogger(stage=3, task="benchmark", cpu_procs=a.workers, seeds="frames: first 50 (deterministic); hydrogen placement uses unseeded python random")
    jobs = []
    for prot, _ in PROTEINS:
        jobs += [(prot, "ESMFlow", "H_only", a.frames, outdir, INP["gen_dir"]),
                 (prot, "ESMFlow", "sidechain_plus_H", a.frames, outdir, INP["gen_dir"]),
                 (prot, "BioEmu", "sidechain_plus_H", a.frames, outdir, INP["gen_dir"]),
                 (prot, "BioEmu", "H_only", a.frames, outdir, INP["gen_dir"])]
    load0 = os.getloadavg()
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(run_task, jobs):
            print("done", os.path.basename(r), flush=True)
    wall = time.time() - t0
    raw = pd.concat([pd.read_csv(os.path.join(outdir, f)) for f in sorted(os.listdir(outdir)) if f.endswith(".csv")], ignore_index=True)
    assert set(raw.omm_platform) == {"CPU"}, f"存在非 CPU 平台的记录: {set(raw.omm_platform)}"
    raw_dir = os.path.join(REV, "raw_outputs", "s3")
    f_raw = os.path.join(raw_dir, "benchmark_raw.csv"); raw.to_csv(f_raw, index=False)
    summ = summarize(raw); f_sum = os.path.join(raw_dir, "benchmark_summary.csv"); summ.to_csv(f_sum, index=False)
    import mdtraj, openmm, pdbfixer
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    meta = dict(host=socket.gethostname(), cpu_model=[l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")][0],
                n_cpus=os.cpu_count(), loadavg_start=load0, loadavg_end=os.getloadavg(), wall_total_s=round(wall, 1), workers=a.workers,
                python=platform.python_version(), mdtraj=mdtraj.__version__, openmm=openmm.__version__,
                pdbfixer=getattr(pdbfixer, "__version__", "1.12.0 (dist-info)"), threads_env="OMP/MKL/OPENBLAS/OPENMM_CPU_THREADS=1", omm_platform_all_records="CPU (断言)", cuda_visible_devices="",
                cv_limit=CV_LIMIT, frames_per_repeat=a.frames, warmup="1 untimed frame per condition",
                note="工作进程并发（每条件一个单线程进程），与服务器其他用户负载共享 CPU；墙钟含争用，CPU 时间列更接近 core-s")
    f_meta = os.path.join(raw_dir, "benchmark_meta.json"); revio.write_json(meta, f_meta)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))
    log.finish([f_raw, f_sum, f_meta], note=f"conditions={len(summ)} records={len(raw)} failed={int((raw.status != 'ok').sum())}")


if __name__ == "__main__":
    main()
