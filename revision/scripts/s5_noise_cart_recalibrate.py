"""Stage 5C · 笛卡尔噪声的重新校准（偏离 D5-4）。
背景：第一轮校准（s5_noise_ensembles.py：割线法、最多 8 次、每次换一个校准种子 101/102/103、工作容差 3%）下，笛卡尔噪声 60 个条件里 16 个的最终散布在 ±5% 之外（10 个 final_out_of_tolerance、
6 个 unreachable 且最终偏差 >5%，含 cyt c×ShortMD 实现 0.454 Å 对目标 0.758 Å）。`s5_noise_cart_diagnostic.py` 的 σ 网格显示：最小化后散布随 σ 单调上升但在 σ≈0.9–1.6 Å 附近出现台阶（压缩比 0.20→0.34），
目标散布是**可达**的，割线法在台阶处失效——是我的校准算法的局限，不是物理上不可达。
本脚本只对第一轮最终散布在 ±5% 之外的条件（仅由校准表决定，早于任何噪声系综的得分）重新校准：
  - 评估函数固定校准种子 101（每次评估 20 个构象），使 σ→散布成为确定性函数，消除“每次迭代换种子”带来的噪声；
  - 先用乘 1.4 / 除 1.4 的步进找到包住目标的区间（利用单调性），再在对数 σ 内用二分/反插值收敛到 |S/s_t−1| ≤ 3%（最多 20 次评估）；
  - 最终系综仍是 5 个种子（21–25）× 50 个构象，验收仍是预登记的“5 个种子的实现散布均值落在 s_G ± 5%”；不通过时用最终系综的实测做最多 2 次修正。
噪声类型、构造方式、约束最小化、种子、验收标准都不变（不改用别的噪声）。
子命令：archive（把第一轮这 16 个条件的 xtc、任务标记、校准/几何行、已打分分片移入 _superseded_cart_run1，不删除）| run [--workers W] | list
输出：raw_outputs/s5/noise_work/{xtc,tables}（新的 16 个条件）、tables/S5_noise_calibration_run1.csv（第一轮全表，archive 时备份）"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS"):
    os.environ.setdefault(_k, "1")
import sys, json, shutil, argparse, time
import concurrent.futures as cf, multiprocessing as mp
import numpy as np, pandas as pd, mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, geom, noise, bv  # noqa: E402
import s5_noise_ensembles as N  # noqa: E402

PROJ, REV, INP = revio.load_paths()
WORK = os.path.join(REV, "raw_outputs", "s5", "noise_work"); TDIR = os.path.join(WORK, "tables"); SUP = os.path.join(WORK, "_superseded_cart_run1")
PARTS = os.path.join(REV, "raw_outputs", "s4", "parts"); SUP_PARTS = os.path.join(PARTS, "_superseded_noise_cart_run1")
SEED_CAL = 101; N_CAL = 20; MAX_EVAL = 20; STEP = 1.4


def targets():
    """第一轮校准表里 cart 条件最终散布在 ±5% 之外者。优先读备份（archive 之后校准表可能已含新行）。"""
    f = os.path.join(REV, "tables", "S5_noise_calibration_run1.csv"); f = f if os.path.exists(f) else os.path.join(REV, "tables", "S5_noise_calibration.csv")
    C = pd.read_csv(f); c = C[(C.noise_type == "cart") & (C.status != "czero") & (C.within_5pct == 0)]
    return [(r.protein, r.condition, float(r.param_final)) for r in c.itertuples()]


def cmd_list(a):
    t = targets(); print(len(t), t)


def cmd_archive(a):
    f1 = os.path.join(REV, "tables", "S5_noise_calibration_run1.csv")
    if not os.path.exists(f1):
        shutil.copy(os.path.join(REV, "tables", "S5_noise_calibration.csv"), f1); shutil.copy(os.path.join(REV, "tables", "S5_noise_geometry_qc.csv"), os.path.join(REV, "tables", "S5_noise_geometry_qc_run1.csv"))
    n = 0
    for p, cond, _ in targets():
        key = f"{p}__cart__{cond}"
        for suf in ("__claim", "__done", "__cal.csv", "__geo.csv"):
            src = os.path.join(TDIR, key + suf)
            if os.path.exists(src):
                os.makedirs(os.path.join(SUP, "tables"), exist_ok=True); shutil.move(src, os.path.join(SUP, "tables", key + suf)); n += 1
        src = os.path.join(WORK, "xtc", p, "cart", cond)
        if os.path.isdir(src):
            dst = os.path.join(SUP, "xtc", p, "cart"); os.makedirs(dst, exist_ok=True); shutil.move(src, os.path.join(dst, cond)); n += 1
        for k in N.FINAL_SEEDS:
            src = os.path.join(PARTS, f"noise_cart_{cond}_s{k}", p)
            if os.path.isdir(src):
                dst = os.path.join(SUP_PARTS, f"noise_cart_{cond}_s{k}"); os.makedirs(dst, exist_ok=True); shutil.move(src, os.path.join(dst, p)); n += 1
    print("archived items:", n)


def recal_task(args):
    p, cond, param0 = args; key = f"{p}__cart__{cond}"
    if os.path.exists(os.path.join(TDIR, key + "__done")):
        return key, "skipped"
    try:
        os.mkdir(os.path.join(TDIR, key + "__claim"))
    except FileExistsError:
        return key, "claimed_by_other_pool"
    t_start = time.time(); MPmap = pd.read_csv(os.path.join(REV, "tables", "S4_old_column_to_frame_map.csv"))
    R = N.setup_ref(p, REV, INP); pos, ca_idx, mth = R["pos"], R["ca_idx"], R["mtop_heavy"]; M = noise.Minimizer(R["top"])
    xdir = os.path.join(WORK, "xtc", p); os.makedirs(xdir, exist_ok=True); tp = os.path.join(xdir, "topology_heavy.pdb")
    X, sq = N.load_target(cond, p, REV, INP, MPmap); a_idx, b_idx = N.common_idx(R["seq_ref"], sq); s_t = geom.mean_pairwise_rmsd(X[:, b_idx]) * 10; rmsf_t = N.rmsf(X[:, b_idx])

    def make(param, seed, n):
        rng = np.random.default_rng(seed); pre = [noise.cart_noise(pos, R["heavy_idx"], R["h_parent"], param / 10.0, rng) for _ in range(n)]
        return np.array([M.minimize(x) for x in pre])

    pts = []   # (σ, 校准种子 101 下的散布)

    def ev(param):
        S = geom.mean_pairwise_rmsd(make(param, SEED_CAL, N_CAL)[:, ca_idx][:, a_idx]) * 10; pts.append((param, S)); return S

    def ok(S):
        return abs(S / s_t - 1) <= N.TOL_WORK
    param = float(np.clip(param0, N.P_MIN, N.P_MAX)); S = ev(param); lo = hi = None
    if S < s_t:
        lo = (param, S)
    else:
        hi = (param, S)
    n_ev = 1
    while not ok(S) and n_ev < MAX_EVAL:
        if lo is None or hi is None:   # 先找区间
            param = float(np.clip(param * (STEP if hi is None else 1 / STEP), N.P_MIN, N.P_MAX)); S = ev(param); n_ev += 1
            if S < s_t:
                lo = (param, S) if (lo is None or param > lo[0]) else lo
            else:
                hi = (param, S) if (hi is None or param < hi[0]) else hi
            continue
        # 区间内：对数—对数反插值，并保证新点离两端至少 10% 的对数宽度（防止停滞）
        l0, l1 = np.log(lo[0]), np.log(hi[0]); y0, y1 = np.log(lo[1]), np.log(hi[1]); fr = (np.log(s_t) - y0) / (y1 - y0) if y1 != y0 else 0.5; fr = float(np.clip(fr, 0.1, 0.9))
        param = float(np.exp(l0 + fr * (l1 - l0))); S = ev(param); n_ev += 1
        if S < s_t:
            lo = (param, S)
        else:
            hi = (param, S)
    status_cal = "ok" if ok(S) else "unreachable"
    n_final = 0

    def generate(param_):
        sp, allx = [], []
        for sd in N.FINAL_SEEDS:
            Xs = make(param_, sd, 50); sp.append(geom.mean_pairwise_rmsd(Xs[:, ca_idx][:, a_idx]) * 10)
            od = os.path.join(xdir, "cart", cond); os.makedirs(od, exist_ok=True); md.Trajectory(Xs[:, R["keep"]], mth).save_xtc(os.path.join(od, f"seed{sd}.xtc")); allx.append(Xs[:, R["keep"]])
        return sp, np.concatenate(allx)
    if not os.path.exists(tp):
        md.Trajectory(pos[R["keep"]][None], mth).save_pdb(tp + f".tmp{os.getpid()}"); os.replace(tp + f".tmp{os.getpid()}", tp)
    sp, allx = generate(param); n_final = 1; mean_sp = float(np.mean(sp))
    while abs(mean_sp / s_t - 1) > N.TOL_FINAL and n_final < 3:   # 最终系综的实测作为一个新点，在已有点上反插值后再试
        near = sorted(pts, key=lambda q: abs(np.log(q[1] / s_t)))[:2]
        if len(near) == 2 and near[0][0] != near[1][0] and near[0][1] != near[1][1]:
            sl = np.log(near[0][1] / near[1][1]) / np.log(near[0][0] / near[1][0]); sl = sl if 0.2 < sl < 4 else 1.0
        else:
            sl = 1.0
        param = float(np.clip(param * (s_t / mean_sp) ** (1.0 / sl), N.P_MIN, N.P_MAX)); sp, allx = generate(param); n_final += 1; mean_sp = float(np.mean(sp))
    st = status_cal if abs(mean_sp / s_t - 1) > N.TOL_FINAL else ("ok" if status_cal == "ok" else "ok_final_within_5pct")
    rm_n = N.rmsf(allx[:50][:, R["ca_heavy"]][:, a_idx])
    row = dict(protein=p, noise_type="cart", condition=cond, param_unit="A sigma", target_spread_A=s_t, param_final=param, n_calibration_iter=len(pts), status=st, hit_bound=bool(param in (N.P_MIN, N.P_MAX)), realized_mean_A=mean_sp,
               realized_sd_across_seeds_A=float(np.std(sp, ddof=1)), realized_min_A=float(np.min(sp)), realized_max_A=float(np.max(sp)), rel_dev=mean_sp / s_t - 1, within_5pct=bool(abs(mean_sp / s_t - 1) <= N.TOL_FINAL), n_common_CA=len(a_idx),
               n_target_frames=len(X), rmsf_pearson_with_G=float(np.corrcoef(rm_n, rmsf_t)[0, 1]), calibration_round=2, n_final_generations=n_final, calibration_trace=json.dumps([[round(q[0], 4), round(q[1], 4)] for q in pts]))
    b_, om_, omf_, ram_ = N.geometry(allx, mth); n_bad = 0; sub = allx[:20]
    for k in range(len(sub)):
        try:
            bv.fix_frame(md.Trajectory(sub[k][None], mth), True, seed=N.S4.det_seed(f"noise_cart_{cond}", p, k, 0, 0))
        except Exception:
            n_bad += 1
    row["wall_s"] = round(time.time() - t_start)
    pd.DataFrame([row]).to_csv(os.path.join(TDIR, key + "__cal.csv"), index=False)
    pd.DataFrame([dict(protein=p, noise_type="cart", condition=cond, bond_rms_dev_A=b_, omega_dev_mean_deg=om_, omega_dev_gt30_frac=omf_, rama_outside_frac=ram_, hydrogen_fail_rate=n_bad / len(sub), n_conf=len(allx))]).to_csv(os.path.join(TDIR, key + "__geo.csv"), index=False)
    open(os.path.join(TDIR, key + "__done"), "w").write("ok"); return key, "done"


def cmd_run(a):
    seqs = json.load(open(INP["tier1a_sequences"])); T = sorted(targets(), key=lambda t: -seqs[t[0]]["length"]); log = revio.RunLogger(stage=5, task="noise_cart_recalibrate", cpu_procs=a.workers, seeds=f"cal {SEED_CAL}; final 21-25")
    t0 = time.time(); fails = []
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        futs = {ex.submit(recal_task, t): t for t in T}
        for fu in cf.as_completed(futs):
            try:
                print(fu.result(), f"[{time.time() - t0:.0f}s]", flush=True)
            except Exception as e:
                fails.append((futs[fu][:2], repr(e)[:300])); print("失败:", fails[-1], flush=True)
    log.finish([TDIR], exit_code=0 if not fails else 1, status="ok" if not fails else "partial", note=f"tasks={len(T)} failed={len(fails)}")
    if fails:
        raise SystemExit(f"{len(fails)} 个任务失败")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["list", "archive", "run"]); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    {"list": cmd_list, "archive": cmd_archive, "run": cmd_run}[a.cmd](a)
