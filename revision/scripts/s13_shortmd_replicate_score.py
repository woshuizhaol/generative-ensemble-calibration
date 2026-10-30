"""Stage 13 · 把 ShortMD 副本（r2、r3 × 12 蛋白）按与现行 ShortMD 完全相同的流程打分（S4 的逐帧计数）。
流程：md_replicates/md/<protein>_r<k>/ensemble.xtc + ensemble.pdb（MD 脚本已按原脚本的方式切出蛋白原子，所有链）→ `s4_rescore.run_task`
  （`bv.fix_frame` 取最大蛋白链即 A 链；逐帧 PDBFixer 重新加氢 pH 7.0；N_C / N_H / rSASA；确定性种子 det_seed(generator, protein, frame, recon_k=-1, attempt)）→
  分片落盘 md_replicates/analysis/parts/<generator>/<protein>/（可重入，已完成分片自动跳过）→ 合并成
  raw_outputs/s4/counts/ShortMD_r<k>/<protein>.parquet 与 _frames.parquet（与现有 ShortMD 同格式；recon_k=-1，同 S4 对 xtc 的约定）。
所有 200 帧都打分；失败帧记入帧状态表，不丢弃。CPU 单线程每进程（OPENMM_CPU_THREADS=1 须在 OpenMM 加载前设置），spawn，每个任务一个新进程（max_tasks_per_child=1）。
用法：python s13_shortmd_replicate_score.py --workers 16 [--reps 2,3] [--proteins A,B] [--md-root DIR --counts-root DIR --parts-root DIR --max-frames N（仅测试）]"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, argparse, glob, json
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np, pandas as pd
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402
import s4_rescore as S4  # noqa: E402

PROJ, REV, INP = revio.load_paths()
ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--reps", default="2,3"); ap.add_argument("--proteins", default="")
ap.add_argument("--md-root", default=os.path.join(REV, "md_replicates", "md")); ap.add_argument("--counts-root", default=os.path.join(REV, "raw_outputs", "s4", "counts"))
ap.add_argument("--parts-root", default=os.path.join(REV, "md_replicates", "analysis", "parts")); ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--status-out", default=os.path.join(REV, "tables", "S13_scoring_status.csv")); a = ap.parse_args()
CHUNK = 25
log = revio.RunLogger(stage=13, task="shortmd_replicate_score", cpu_procs=a.workers, seeds="det_seed(generator, protein, frame, -1, attempt)")


def main():
    max_asa = os.path.join(REV, "data", "external", "tien2013_maxasa", "tien2013_theoretical_maxASA.csv")
    seqs = json.load(open(INP["tier1a_sequences"])); prots = [p for p in sorted(seqs) if (not a.proteins or p in a.proteins.split(","))]
    tasks, info = [], {}
    for k in [int(x) for x in a.reps.split(",")]:
        for p in prots:
            d = os.path.join(a.md_root, f"{p}_r{k}")
            if not os.path.exists(os.path.join(d, "md_meta.json")):
                raise SystemExit(f"{d}/md_meta.json 不存在：该副本 MD 尚未完成")
            g = f"ShortMD_r{k}"; xtc, top = os.path.join(d, "ensemble.xtc"), os.path.join(d, "ensemble.pdb")
            with md.open(xtc) as fh:
                nfr = len(fh)
            if a.max_frames:
                nfr = min(nfr, a.max_frames)
            info[(g, p)] = nfr
            for f0 in range(0, nfr, CHUNK):
                n_res = seqs[p]["length"]
                tasks.append(dict(group="s13", generator=g, protein=p, recon_k=-1, kind="xtc", xtc=xtc, top=top, f0=f0, f1=min(nfr, f0 + CHUNK), n_frames=nfr, n_res=n_res,
                                  est_s=S4.est_cost_s("ShortMD", n_res) * (min(nfr, f0 + CHUNK) - f0), parts_root=a.parts_root))
    tasks.sort(key=lambda t: -t["est_s"]); todo = [t for t in tasks if not (os.path.exists(S4.part_paths(REV, t)[0]) and os.path.exists(S4.part_paths(REV, t)[1]))]
    print(f"任务 {len(tasks)}，待执行 {len(todo)}，估计 {sum(t['est_s'] for t in todo) / 3600:.2f} core-h，workers={a.workers}，load={os.getloadavg()}", flush=True)
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx, max_tasks_per_child=1) as ex:
        futs = [ex.submit(S4.run_task, (t, REV, max_asa)) for t in todo]
        for n, f in enumerate(cf.as_completed(futs), 1):
            f.result()
            if n % 10 == 0 or n == len(futs):
                print(f"{n}/{len(futs)} 任务完成，load={os.getloadavg()[0]:.1f}", flush=True)
    rows, outs = [], []
    for (g, p), nfr in info.items():
        base = os.path.join(a.parts_root, g, p); rs = sorted(glob.glob(os.path.join(base, "*.parquet")))
        res_files = [f for f in rs if not f.endswith("_frames.parquet")]; frm_files = [f for f in rs if f.endswith("_frames.parquet")]
        R = pd.concat([pd.read_parquet(f) for f in res_files], ignore_index=True); F = pd.concat([pd.read_parquet(f) for f in frm_files], ignore_index=True)
        assert set(F.frame_idx) == set(range(nfr)) and not F.duplicated(["frame_idx", "recon_k"]).any(), (g, p, "帧不完整或重复")
        ok = F.status.isin(["ok", "ok_retry"]); okset = set(zip(R.frame_idx.astype(int), R.recon_k.astype(int)))
        assert all((int(r.frame_idx), int(r.recon_k)) in okset for r in F[ok].itertuples()), (g, p, "状态为 ok 的帧在残基表里没有行")
        d = os.path.join(a.counts_root, g); os.makedirs(d, exist_ok=True)
        R.sort_values(["frame_idx", "recon_k", "resSeq"]).to_parquet(os.path.join(d, f"{p}.parquet"), index=False); F.sort_values(["frame_idx", "recon_k"]).to_parquet(os.path.join(d, f"{p}_frames.parquet"), index=False)
        rows.append(dict(generator=g, protein=p, n_frames=len(F), n_ok=int((F.status == "ok").sum()), n_ok_retry=int((F.status == "ok_retry").sum()), n_failed=int((~ok).sum()), n_residues_scored=int(R.resSeq.nunique()),
                         rows_in_parquet=len(R), mean_t_fix_s=round(float(F.t_fix_s.mean()), 3), total_core_s=round(float(F.t_wall_all_attempts_s.sum()), 1), counts_file=os.path.relpath(os.path.join(d, f"{p}.parquet"), REV)))
        outs.append(os.path.join(d, f"{p}.parquet"))
    S = pd.DataFrame(rows); S.to_csv(a.status_out, index=False); pd.set_option("display.width", 220); print(S.to_string(index=False))
    log.finish(outs + [a.status_out], note=f"cells={len(S)} failed_frames={int(S.n_failed.sum())}")


if __name__ == "__main__":
    main()
