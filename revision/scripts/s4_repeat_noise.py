"""Stage 4a · C-parity 未达标项的归因实验（revision_proposal.md Stage 4"风险与处理"：C-parity 不达标 → 排除编号与链选择问题后，
把差异归因于 pdbfixer 的随机性，并在若干蛋白上做 3 次独立重打分量化这部分随机性）。

做法：对一批（生成器 × 蛋白）格子，只重打分"旧流程当年打过的那些帧"（S4_old_column_to_frame_map.csv 给出的帧），用与正式重打分不同的
确定性种子（det_seed 的 rep=1,2,3，rep=0 即正式结果）独立重复 3 次。这样每个格子有 4 份"新打分"（正式 + 3 次重复），帧集合完全相同，
彼此的差异只来自 PDBFixer/Modeller 的随机性。然后比较两类差异的分布：
  旧 vs 新（旧 operator_f.npz 与每一份新打分）与 新 vs 新（4 份新打分两两之间）。
如果旧 vs 新的差异和新 vs 新的差异同量级，就说明 C-parity 里 Spearman 差 >0.02 的格子只是 pdbfixer 随机性所致，而不是编号、链选择或打分口径的系统差异。

格子：
  BioEmu：CI2、Ubiquitin（旧列 = 全帧）与 RNaseA、SNase、UCH-L1、cytochrome_c（旧列只是 xtc 的子集）；完整 12 个蛋白的重复成本过高（约 50 core-h × 3），故取 6 个。
  ESMFlow、Boltz2、ShortMD、Str2Str：12 个蛋白全做。
  Str2Str_T3、Str2Str_T3_d035：65 个域全做（N_H Pearson 未达 0.99 的格子集中在这两个生成器）。
  BioEmu_T3：S4_parity 中 Pearson 未达 0.99 的域。
用法：python s4_repeat_noise.py plan|run|analyze [--reps 3] [--workers W]
输出：raw_outputs/s4/repeat_noise/rep{r}/parts/…；tables/S4_repeat_noise_cells.csv、S4_repeat_noise_summary.csv、S4_repeat_noise_cell_envelope.csv、S4_repeat_noise_rank.csv、S4_repeat_noise_rank_summary.csv
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, time, glob, json, argparse, itertools
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402
import s4_rescore as S4  # noqa: E402
import s4_postprocess as PP  # noqa: E402

BIOEMU_PROTEINS = ["CI2", "Ubiquitin", "RNaseA", "SNase", "UCH-L1", "cytochrome_c"]
FULL_GENS = ["ESMFlow", "Boltz2", "ShortMD", "Str2Str"]
RT = bv.R_GAS * bv.T_REF


def cell_list(REV, INP):
    seqs = json.load(open(INP["tier1a_sequences"]))
    cells = [("BioEmu", p) for p in BIOEMU_PROTEINS] + [(g, p) for g in FULL_GENS for p in seqs]
    L = pd.read_csv(os.path.join(REV, "tables", "S1_tier3_legacy65_provenance.csv")); U = L[L.in_legacy65 | L.in_Dnew].T3_id.tolist()
    P = pd.read_csv(os.path.join(REV, "tables", "S4_parity.csv"))
    cells += [(g, p) for g in ("Str2Str_T3", "Str2Str_T3_d035") for p in U if ((P.generator == g) & (P.protein == p)).any()]
    bad = P[(P.generator == "BioEmu_T3") & ~P["pass_pearson_0.99"]]
    cells += [("BioEmu_T3", p) for p in bad.protein]
    return cells


def build_tasks(REV, INP, reps):
    items = {(i["generator"], i["protein"]): i for i in S4.build_items(["tier1a", "t3"], INP, REV)}
    MP = pd.read_csv(os.path.join(REV, "tables", "S4_old_column_to_frame_map.csv")); tasks = []
    for g, p in cell_list(REV, INP):
        it = items[(g, p)]; fr = sorted(MP[(MP.generator == g) & (MP.protein == p)].frame_idx.unique().tolist())
        assert fr, (g, p)
        c = S4.est_cost_s(g, it["n_res"]); per = int(np.clip(S4.CHUNK_TARGET_S // max(c, 0.05), 1, 100))
        for r in range(1, reps + 1):
            root = os.path.join(REV, "raw_outputs", "s4", "repeat_noise", f"rep{r}", "parts")
            for ci, i0 in enumerate(range(0, len(fr), per)):
                sub = fr[i0:i0 + per]
                tasks.append(dict(it, recon_k=-1, frame_list=sub, chunk=ci, rep=r, parts_root=root, est_s=c * len(sub), f0=sub[0], f1=sub[-1] + 1))
    return sorted(tasks, key=lambda t: -t["est_s"])


def pending(REV, tasks):
    return [t for t in tasks if not (os.path.exists(S4.part_paths(REV, t)[0]) and os.path.exists(S4.part_paths(REV, t)[1]))]


def cmd_plan(a, INP, REV):
    tasks = build_tasks(REV, INP, a.reps); todo = pending(REV, tasks)
    df = pd.DataFrame([dict(generator=t["generator"], rep=t["rep"], est_s=t["est_s"], frames=len(t["frame_list"])) for t in tasks])
    print(df.groupby("generator").agg(tasks=("est_s", "size"), frames=("frames", "sum"), est_core_h=("est_s", lambda x: x.sum() / 3600)).round(2).to_string())
    print("总计 tasks", len(tasks), "frames", int(df.frames.sum()), "est core-h", round(df.est_s.sum() / 3600, 1), "| 待执行", len(todo))


def cmd_run(a, INP, REV):
    tasks = pending(REV, build_tasks(REV, INP, a.reps))
    log = revio.RunLogger(stage=4, task="repeat_noise_rescore", cpu_procs=a.workers, seeds="det_seed rep=1..%d" % a.reps)
    print(f"待执行任务 {len(tasks)}，估计 {sum(t['est_s'] for t in tasks) / 3600:.1f} core-h，workers={a.workers}，load={os.getloadavg()}", flush=True)
    max_asa_path = os.path.join(REV, "data", "external", "tien2013_maxasa", "tien2013_theoretical_maxASA.csv")
    t0 = time.time(); n_done = 0; fails = []
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        futs = {ex.submit(S4.run_task, (t, REV, max_asa_path)): t for t in tasks}
        for fu in cf.as_completed(futs):
            t = futs[fu]
            try:
                fu.result(); n_done += 1
            except Exception as e:
                fails.append((t["generator"], t["protein"], t["rep"], repr(e)[:200])); print("任务失败:", fails[-1], flush=True)
            if n_done % 25 == 0 or n_done == len(tasks):
                print(f"[{time.time() - t0:.0f}s] {n_done}/{len(tasks)} 任务完成，load={os.getloadavg()[0]:.1f}", flush=True)
    log.finish([os.path.join(REV, "raw_outputs", "s4", "repeat_noise")], exit_code=0 if not fails else 1, status="ok" if not fails else "partial",
               note=f"tasks={len(tasks)} failed={len(fails)} wall={time.time() - t0:.0f}s")
    if fails:
        raise SystemExit(f"{len(fails)} 个任务失败")


def load_rep(REV, r, g, p):
    base = os.path.join(REV, "raw_outputs", "s4", "repeat_noise", f"rep{r}", "parts", g, p)
    fs = sorted(glob.glob(os.path.join(base, "*.parquet"))); res = [f for f in fs if not f.endswith("_frames.parquet")]; frm = [f for f in fs if f.endswith("_frames.parquet")]
    R = pd.concat([pd.read_parquet(f) for f in res], ignore_index=True); F = pd.concat([pd.read_parquet(f) for f in frm], ignore_index=True)
    return R, F


def cmd_analyze(a, INP, REV):
    log = revio.RunLogger(stage=4, task="repeat_noise_analyze", cpu_procs=1, seeds="none")
    exp = pd.read_parquet(INP["tier1a_all"]); maps = json.load(open(INP["structure_maps"]))
    MP = pd.read_csv(os.path.join(REV, "tables", "S4_old_column_to_frame_map.csv")); PAR = pd.read_csv(os.path.join(REV, "tables", "S4_parity.csv")); rows = []
    pr = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    bad_status = 0; rank_rows = []
    for g, p in cell_list(REV, INP):
        old = PP.load_old(INP, g, p); new = PP.load_new(REV, g, p)
        fr = MP[(MP.generator == g) & (MP.protein == p)].sort_values("old_col").frame_idx.values
        common = [r for r in old["resSeq"] if r in new["NC"].index]; io = [old["resSeq"].index(r) for r in common]
        runs = {"new0": (new["NC"].loc[common, fr].values, new["NH"].loc[common, fr].values)}
        for r in range(1, a.reps + 1):
            R, F = load_rep(REV, r, g, p); bad_status += int((~F.status.isin(["ok", "ok_retry"])).sum())
            nc = R.pivot(index="resSeq", columns="frame_idx", values="NC"); nh = R.pivot(index="resSeq", columns="frame_idx", values="NH")
            assert set(fr) == set(nc.columns), (g, p, r)
            runs[f"rep{r}"] = (nc.loc[common, fr].values, nh.loc[common, fr].values)
        have_rho = g in PP.TIER1A
        if have_rho:
            ge = exp[exp.protein_id == p]; off = float(PAR[(PAR.generator == g) & (PAR.protein == p)].offset.iloc[0])
            def rho(NC, NH):
                x, y, _ = bv.align(common, RT * bv.aggregate(NC, NH, "count_avg"), ge, int(off)); return float(stats.spearmanr(x, y)[0])
            rhos = {"old": rho(old["NC"][io, :], old["NH"][io, :]), **{k: rho(*v) for k, v in runs.items()}}
        means = {"old": (np.nanmean(old["NC"][io, :], axis=1), np.nanmean(old["NH"][io, :], axis=1)), **{k: (np.nanmean(v[0], axis=1), np.nanmean(v[1], axis=1)) for k, v in runs.items()}}
        names = ["old"] + list(runs); n_fr = len(fr)
        # 排位检验：5 次运行（旧 + 正式 + 3 次重复）若可交换，则任一运行偏离"其余 4 次均值"的程度，旧结果在 5 次中的名次应均匀分布于 1–5。
        def loo(vals, kind):
            out = {}
            for i, k in enumerate(names):
                rest = [vals[j] for j in names if j != k]
                if kind == "scalar":
                    out[k] = abs(vals[k] - float(np.mean(rest)))
                else:
                    out[k] = float(np.sqrt(np.nanmean((vals[k] - np.nanmean(np.stack(rest), axis=0)) ** 2)))
            return out
        metrics = {"NC": ({k: means[k][0] for k in names}, "vec"), "NH": ({k: means[k][1] for k in names}, "vec")}
        if have_rho:
            metrics["rho"] = (rhos, "scalar")
        for mname, (vals, kind) in metrics.items():
            d = loo(vals, kind); arr = np.array([d[k] for k in names])
            det = bool(arr.max() - arr.min() < 1e-12)
            rank_rows.append(dict(generator=g, protein=p, metric=mname, deterministic=det, **{f"d_{k}": d[k] for k in names}, rank_old=(np.nan if det else int(1 + (arr[1:] < arr[0]).sum()))))
        for x, y in itertools.combinations(names, 2):
            kind = "old_vs_new" if x == "old" else "new_vs_new"
            rows.append(dict(generator=g, protein=p, kind=kind, run_a=x, run_b=y, n_frames=n_fr, n_res=len(common), pearson_NC=pr(means[x][0], means[y][0]), pearson_NH=pr(means[x][1], means[y][1]),
                             d_rho=(rhos[y] - rhos[x]) if have_rho else np.nan))
    T = pd.DataFrame(rows); T["abs_d_rho"] = T.d_rho.abs()
    f1 = os.path.join(REV, "tables", "S4_repeat_noise_cells.csv"); T.to_csv(f1, index=False)
    S = []
    for (g, kind), d in T.groupby(["generator", "kind"]):
        dr = d.abs_d_rho.dropna()
        S.append(dict(generator=g, kind=kind, n_cells=d.protein.nunique(), n_comparisons=len(d), min_pearson_NC=d.pearson_NC.min(), min_pearson_NH=d.pearson_NH.min(),
                      median_pearson_NH=d.pearson_NH.median(), n_NH_below_0p99=int((d.pearson_NH < 0.99).sum()),
                      median_abs_d_rho=dr.median() if len(dr) else np.nan, p95_abs_d_rho=dr.quantile(.95) if len(dr) else np.nan, max_abs_d_rho=dr.max() if len(dr) else np.nan,
                      frac_abs_d_rho_gt_0p02=float((dr > 0.02).mean()) if len(dr) else np.nan))
    S = pd.DataFrame(S)
    # 同一格子内：旧 vs 新的 |Δρ| 是否落在新 vs 新的分布内（每格取 max）
    c = []
    for (g, p), d in T[T.generator.isin(PP.TIER1A)].groupby(["generator", "protein"]):
        on = d[d.kind == "old_vs_new"].abs_d_rho; nn = d[d.kind == "new_vs_new"].abs_d_rho
        c.append(dict(generator=g, protein=p, max_old_vs_new=on.max(), max_new_vs_new=nn.max(), old_within_new_envelope=bool(on.max() <= nn.max() + 1e-12)))
    C = pd.DataFrame(c)
    for g, d in C.groupby("generator"):
        S.loc[S.generator == g, "cells_old_within_new_envelope"] = int(d.old_within_new_envelope.sum()); S.loc[S.generator == g, "cells_tier1a"] = len(d)
    f2 = os.path.join(REV, "tables", "S4_repeat_noise_summary.csv"); S.round(5).to_csv(f2, index=False)
    f3 = os.path.join(REV, "tables", "S4_repeat_noise_cell_envelope.csv"); C.round(5).to_csv(f3, index=False)
    RK = pd.DataFrame(rank_rows); f4 = os.path.join(REV, "tables", "S4_repeat_noise_rank.csv"); RK.to_csv(f4, index=False)
    def exact_p(n, obs):
        pm = np.array([1.0])
        for _ in range(n):
            pm = np.convolve(pm, np.r_[0, np.ones(5) / 5])
        s_ = np.arange(len(pm)); return float(pm[np.abs(s_ - 3 * n) >= abs(obs - 3 * n) - 1e-9].sum())
    rs = []
    for (g, m), d in RK.groupby(["generator", "metric"]):
        nd = d[~d.deterministic]
        rs.append(dict(generator=g, metric=m, n_cells=len(d), n_deterministic_skipped=int(d.deterministic.sum()), n_tested=len(nd),
                       mean_rank_old=float(nd.rank_old.mean()) if len(nd) else np.nan, share_rank_old_is_worst=float((nd.rank_old == 5).mean()) if len(nd) else np.nan,
                       p_exact_two_sided=exact_p(len(nd), float(nd.rank_old.sum())) if len(nd) else np.nan))
    RS = pd.DataFrame(rs); f5 = os.path.join(REV, "tables", "S4_repeat_noise_rank_summary.csv"); RS.round(5).to_csv(f5, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); print(S.round(4).to_string(index=False)); print("\n排位检验（H0：旧结果与 4 次新打分可交换 → 平均名次 3）\n", RS.round(4).to_string(index=False)); print("非 ok 帧数（重复实验）:", bad_status)
    log.finish([f1, f2, f3, f4, f5], note=f"rows={len(T)} bad_status={bad_status}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["plan", "run", "analyze"]); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--workers", type=int, default=60)
    a = ap.parse_args(); PROJ, REV, INP = revio.load_paths()
    {"plan": cmd_plan, "run": cmd_run, "analyze": cmd_analyze}[a.cmd](a, INP, REV)
