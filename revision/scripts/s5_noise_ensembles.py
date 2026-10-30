"""Stage 5C · 噪声系综（revision_proposal.md S5 5C；定义以 tables/S5_noise_prereg.md 为准，已冻结）。

每个任务 = （蛋白 × 噪声类型 × 条件），一个工作进程，单线程；max_tasks_per_child=1，避免 OpenMM CPU 平台的线程累积。每个任务做的事：
  1. 参考结构 = 沉积结构 PDBFixer 重建 recon_k=0（bv.fix_frame_openmm，种子 det_seed("Crystal", 蛋白, 0, 0, 0)，与 S4a 的 recon_k=0 同一条流水线）。
  2. 5 个生成器 G 的目标散布 s_G：G 主分析帧的平均成对 CA-RMSD（按序列比对取与参考的公共 CA）。
     BioEmu/ShortMD = 口径 O 的 M=50 第 1 次抽样（旧 npz 列 → xtc 帧号经 S4_old_column_to_frame_map.csv），ESMFlow/Boltz-2 = 全部 50 帧，Str2Str_xtal = 种子 11 的 50 帧。
  3. 两类噪声（anm 主分析 / cart 次要分析）× 5 个匹配条件：在约束最小化之后的坐标上，用对数尺度的割线法迭代匹配幅度参数（校准种子 101-103，每次 20 个构象，
     工作容差 ±3%，最多 8 次；参数碰到上下界仍不收敛 → unreachable，如实记录，不改用别的噪声），再生成最终系综 5 个种子（21-25）× 50 个构象；
     验收 = 5 个种子实现散布的均值在 s_G ±5% 内；超出则按比例修正参数重生成一次，仍超出标 final_out_of_tolerance。
  4. 阴性对照 C-zero：幅度 0.01 Å，5 种子 × 50 构象。
  5. 几何质量（键长 RMS 偏差、ω、Ramachandran 粗略分区外比例、加氢失败率）与对 G 的逐残基 RMSF 的 Pearson（仅描述）。
输出（out_root，默认 raw_outputs/s5/noise_work）：
  xtc/<蛋白>/topology_heavy.pdb；xtc/<蛋白>/<类型>/<条件>/seed<k>.xtc（重原子；打分时 bv.fix_frame 重新加氢，与其他生成器一致）
  tables/<蛋白>__<类型>__<条件>__{cal.csv,geo.csv,done}（按（蛋白×类型×条件）切成任务；done 存在即跳过，可重入）
用法：python s5_noise_ensembles.py run [--proteins a,b] [--workers W] [--out_root DIR] [--quick] [--reverse] | merge [--out_root DIR]\n崩溃后重跑：先删掉没有 __done 的 __claim 目录（claim 存在但 done 不存在 = 上次中断）。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time, argparse, difflib, glob
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np, pandas as pd
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, geom, noise  # noqa: E402
from lib.ensemble import np_load_verified  # noqa: E402
import s4_rescore as S4  # noqa: E402  (只用 det_seed)

GENS = ["BioEmu", "ESMFlow", "Boltz2", "ShortMD", "Str2Str_xtal"]
TYPES = ["anm", "cart"]
CAL_SEEDS = (101, 102, 103); FINAL_SEEDS = (21, 22, 23, 24, 25)
TOL_WORK, TOL_FINAL, MAX_IT, P_MIN, P_MAX = 0.03, 0.05, 8, 0.02, 60.0
CZERO = 0.01
EH = {("N", "CA"): 1.458, ("CA", "C"): 1.525, ("C", "O"): 1.231}   # Engh & Huber 1991（Å）
EH_CN = 1.329


def load_target(g, p, REV, INP, MPmap, full=False):
    """返回 (CA 坐标 (n,N,3) nm, CA 残基单字母序列)；full=True 时返回 (全部原子 mdtraj 轨迹, None)，用于生成器自身的几何指标。"""
    idx = None
    if g in ("ESMFlow", "Boltz2"):
        d = os.path.join(INP["gen_dir"], g, p)
    elif g == "Str2Str_xtal":
        d = os.path.join(REV, "raw_outputs", "s5", "gen", "Str2Str_xtal", p, "seed11")
    else:
        d = os.path.join(INP["gen_dir"], g, p)
        z = np_load_verified("O", os.path.join(REV, "raw_outputs", "s4", "draw_indices", f"O_{g}_{p}_M50.npz")); cols = [int(c) for c in z["frames"][0]]
        m = MPmap[(MPmap.generator == g) & (MPmap.protein == p)].set_index("old_col").frame_idx; idx = [int(m.loc[c]) for c in cols]
    if full:
        t = md.load(os.path.join(d, "samples.xtc"), top=os.path.join(d, "topology.pdb"))
        return (t[idx] if idx is not None else t), None
    ca = md.load_topology(os.path.join(d, "topology.pdb")).select("name CA")
    t = md.load(os.path.join(d, "samples.xtc"), top=os.path.join(d, "topology.pdb"), atom_indices=ca)
    if idx is not None:
        t = t[idx]
    return t.xyz.copy(), "".join(bv.T2O.get(t.topology.atom(i).residue.name, "X") for i in range(t.n_atoms))


def common_idx(seq_ref, seq_g):
    a, b = [], []
    for blk in difflib.SequenceMatcher(None, seq_ref, seq_g, autojunk=False).get_matching_blocks():
        a += list(range(blk.a, blk.a + blk.size)); b += list(range(blk.b, blk.b + blk.size))
    return np.array(a, int), np.array(b, int)


def rmsf(X):
    """叠合到均值结构后的逐残基 RMSF（nm），迭代 3 次。X: (n,N,3)。"""
    ref = X[0].copy(); Y = X
    for _ in range(3):
        Y = np.array([x @ r.T + t for x in X for (r, t) in [noise.kabsch_rt(x, ref)]]); ref = Y.mean(0)
    return np.sqrt(((Y - ref) ** 2).sum(2).mean(0))


def geometry(xyz_heavy, mtop):
    """xyz_heavy: (n,Nh,3) nm；mtop: 重原子 mdtraj 拓扑。返回 (键长RMS偏差Å, ω偏离均值°, ω偏离>30°比例, Ramachandran 粗略分区外比例(非Gly))。"""
    t = md.Trajectory(xyz_heavy, mtop); dev = []; res = list(mtop.residues)
    for r in res:
        nm = {a.name: a.index for a in r.atoms}
        for (a_, b_), ref in EH.items():
            if a_ in nm and b_ in nm:
                dev.append(np.linalg.norm(xyz_heavy[:, nm[a_]] - xyz_heavy[:, nm[b_]], axis=1) * 10 - ref)
    for r0, r1 in zip(res[:-1], res[1:]):
        n0 = {a.name: a.index for a in r0.atoms}; n1 = {a.name: a.index for a in r1.atoms}
        if "C" in n0 and "N" in n1 and np.linalg.norm(xyz_heavy[0, n0["C"]] - xyz_heavy[0, n1["N"]]) < 0.2:
            dev.append(np.linalg.norm(xyz_heavy[:, n0["C"]] - xyz_heavy[:, n1["N"]], axis=1) * 10 - EH_CN)
    bond_rms = float(np.sqrt(np.mean(np.concatenate(dev) ** 2)))
    _, om = md.compute_omega(t); omd = np.abs(np.abs(np.degrees(om)) - 180.0)
    ip, phi = md.compute_phi(t); iq, psi = md.compute_psi(t); phi = np.degrees(phi); psi = np.degrees(psi)
    pk = {(int(ix[1]), int(ix[2])): j for j, ix in enumerate(ip)}; n_out = n_tot = 0
    for j, ix in enumerate(iq):
        k = (int(ix[0]), int(ix[1]))
        if k not in pk or t.topology.atom(int(ix[1])).residue.name == "GLY":
            continue
        ph, ps = phi[:, pk[k]], psi[:, j]
        ok = (((ph < -30) & ((ps > 90) | (ps < -150))) | ((ph < -20) & (ph > -160) & (ps > -120) & (ps < 50)) | ((ph > 30) & (ph < 100) & (ps > -30) & (ps < 90)))
        n_out += int((~ok).sum()); n_tot += len(ok)
    return bond_rms, float(omd.mean()), float((omd > 30).mean()), (n_out / n_tot if n_tot else float("nan"))


def setup_ref(p, REV, INP):
    """参考结构与 OpenMM 拓扑等（每个任务各自重建，约几秒；确定性种子，所以各任务得到逐位相同的参考）。"""
    nb = pd.read_csv(os.path.join(REV, "tables", "S1_deposited_structure_numbering.csv")); pdb = nb[nb.protein == p].pdb.iloc[0]
    fr = md.load(os.path.join(INP["pdb_dir"], f"{pdb}.pdb")); fr = fr[0] if fr.n_frames > 1 else fr
    top, pos = bv.fix_frame_openmm(fr, True, seed=S4.det_seed("Crystal", p, 0, 0, 0)); atoms = list(top.atoms())
    R = dict(top=top, pos=pos, ca_idx=np.array([a.index for a in atoms if a.name == "CA"]), res_atoms=[np.array([a.index for a in r.atoms()]) for r in top.residues()],
             heavy_idx=np.array([a.index for a in atoms if a.element is not None and a.element.symbol != "H"]), h_parent={})
    for b in top.bonds():
        for h, o in ((b.atom1, b.atom2), (b.atom2, b.atom1)):
            if h.element is not None and h.element.symbol == "H":
                R["h_parent"][h.index] = o.index
    mt = md.Topology.from_openmm(top); R["keep"] = mt.select("not element H"); assert len(R["keep"]) == len(R["heavy_idx"]); R["mtop_heavy"] = mt.subset(R["keep"])
    R["ca_heavy"] = np.array([i for i, a in enumerate(R["mtop_heavy"].atoms) if a.name == "CA"])
    R["seq_ref"] = "".join(bv.T2O.get(r.name, "X") for r in top.residues() if any(a.name == "CA" for a in r.atoms()))
    return R


def do_task(args):
    """一个任务 = 一个（蛋白, 类型, 条件）。类型 anm/cart 的条件 ∈ 5 个生成器 + czero；类型 generator 只算生成器自身的几何作对照。"""
    p, typ, cond, out_root, quick = args
    PROJ, REV, INP = revio.load_paths(); tdir = os.path.join(out_root, "tables"); os.makedirs(tdir, exist_ok=True); key = f"{p}__{typ}__{cond}"
    if os.path.exists(os.path.join(tdir, key + "__done")):
        return key, "skipped"
    try:   # 原子认领：允许同时开第二个任务池（--reverse）而不会重复跑同一个任务；每个任务都在全新的 spawn 子进程里读取本文件，所以运行中途加入此段对在跑的池同样生效
        os.mkdir(os.path.join(tdir, key + "__claim"))
    except FileExistsError:
        return key, "claimed_by_other_pool"
    t_start = time.time(); n_cal, n_fin, fseeds, max_it = (6, 5, FINAL_SEEDS[:2], 3) if quick else (20, 50, FINAL_SEEDS, MAX_IT)
    MPmap = pd.read_csv(os.path.join(REV, "tables", "S4_old_column_to_frame_map.csv"))
    if typ == "generator":
        tg, _ = load_target(cond, p, REV, INP, MPmap, full=True); b_, om_, omf_, ram_ = geometry(tg.xyz, tg.topology)
        pd.DataFrame([dict(protein=p, noise_type="generator", condition=cond, bond_rms_dev_A=b_, omega_dev_mean_deg=om_, omega_dev_gt30_frac=omf_, rama_outside_frac=ram_, hydrogen_fail_rate=np.nan, n_conf=tg.n_frames)]).to_csv(os.path.join(tdir, key + "__geo.csv"), index=False)
        open(os.path.join(tdir, key + "__done"), "w").write("ok"); return key, "done"
    R = setup_ref(p, REV, INP); pos, ca_idx, mth = R["pos"], R["ca_idx"], R["mtop_heavy"]; M = noise.Minimizer(R["top"])
    xdir = os.path.join(out_root, "xtc", p); os.makedirs(xdir, exist_ok=True); tp = os.path.join(xdir, "topology_heavy.pdb")
    if not os.path.exists(tp):
        md.Trajectory(pos[R["keep"]][None], mth).save_pdb(tp + f".tmp{os.getpid()}"); os.replace(tp + f".tmp{os.getpid()}", tp)

    def make(param, seed, n):
        if typ == "anm":
            cas = noise.anm_ca_samples(pos[ca_idx], n, param, seed); pre = [noise.displace_allatom(pos, R["res_atoms"], ca_idx, c) for c in cas]
        else:
            rng = np.random.default_rng(seed); pre = [noise.cart_noise(pos, R["heavy_idx"], R["h_parent"], param / 10.0, rng) for _ in range(n)]
        return np.array([M.minimize(x) for x in pre])

    def generate(param, a_idx):
        sp, allx = [], []
        for sd in fseeds:
            X = make(param, sd, n_fin); sp.append(geom.mean_pairwise_rmsd(X[:, ca_idx][:, a_idx]) * 10 if a_idx is not None else geom.mean_pairwise_rmsd(X[:, ca_idx]) * 10)
            od = os.path.join(xdir, typ, cond); os.makedirs(od, exist_ok=True); md.Trajectory(X[:, R["keep"]], mth).save_xtc(os.path.join(od, f"seed{sd}.xtc")); allx.append(X[:, R["keep"]])
        return sp, np.concatenate(allx)

    row = dict(protein=p, noise_type=typ, condition=cond, param_unit=("A rmsd" if typ == "anm" else "A sigma"))
    if cond == "czero":
        sp, allx = generate(CZERO, None)
        row.update(target_spread_A=np.nan, param_final=CZERO, n_calibration_iter=0, status="czero", hit_bound=False, realized_mean_A=float(np.mean(sp)), realized_sd_across_seeds_A=float(np.std(sp, ddof=1)),
                   realized_min_A=float(np.min(sp)), realized_max_A=float(np.max(sp)), rel_dev=np.nan, within_5pct=np.nan, n_common_CA=len(ca_idx), n_target_frames=0, rmsf_pearson_with_G=np.nan)
    else:
        X, sq = load_target(cond, p, REV, INP, MPmap); a_idx, b_idx = common_idx(R["seq_ref"], sq); s_t = geom.mean_pairwise_rmsd(X[:, b_idx]) * 10; rmsf_t = rmsf(X[:, b_idx])
        param = float(np.clip(s_t / np.sqrt(2) * (1.3 if typ == "anm" else 0.6), P_MIN, P_MAX)); pts = []; status = "unreachable"; hit = False
        for it in range(max_it):
            S = geom.mean_pairwise_rmsd(make(param, CAL_SEEDS[it % 3], n_cal)[:, ca_idx][:, a_idx]) * 10; pts.append((param, S))
            if abs(S / s_t - 1) <= TOL_WORK:
                status = "ok"; break
            slope = 0.8
            if len(pts) >= 2 and pts[-1][0] != pts[-2][0] and pts[-1][1] != pts[-2][1]:
                sl = np.log(pts[-1][1] / pts[-2][1]) / np.log(pts[-1][0] / pts[-2][0]); slope = sl if 0.2 < sl < 3 else 0.8
            new = float(np.clip(param * (s_t / S) ** (1.0 / slope), P_MIN, P_MAX))
            if new == param:
                hit = True; break
            hit = hit or new in (P_MIN, P_MAX); param = new
        sp, allx = generate(param, a_idx); mean_sp = float(np.mean(sp))
        if abs(mean_sp / s_t - 1) > TOL_FINAL and status == "ok":
            param = float(np.clip(param * (s_t / mean_sp) ** 1.25, P_MIN, P_MAX)); sp, allx = generate(param, a_idx); mean_sp = float(np.mean(sp))
        st = status if status != "ok" else ("ok" if abs(mean_sp / s_t - 1) <= TOL_FINAL else "final_out_of_tolerance")
        rm_n = rmsf(allx[:50][:, R["ca_heavy"]][:, a_idx])
        row.update(target_spread_A=s_t, param_final=param, n_calibration_iter=len(pts), status=st, hit_bound=hit, realized_mean_A=mean_sp, realized_sd_across_seeds_A=float(np.std(sp, ddof=1)), realized_min_A=float(np.min(sp)),
                   realized_max_A=float(np.max(sp)), rel_dev=mean_sp / s_t - 1, within_5pct=bool(abs(mean_sp / s_t - 1) <= TOL_FINAL), n_common_CA=len(a_idx), n_target_frames=len(X), rmsf_pearson_with_G=float(np.corrcoef(rm_n, rmsf_t)[0, 1]))
    b_, om_, omf_, ram_ = geometry(allx, mth); n_bad = 0; sub = allx[:20]
    for k in range(len(sub)):   # 加氢失败率：对前 20 个构象（重原子）调 bv.fix_frame，数抛错的帧
        try:
            bv.fix_frame(md.Trajectory(sub[k][None], mth), True, seed=S4.det_seed(f"noise_{typ}_{cond}", p, k, 0, 0))
        except Exception:
            n_bad += 1
    row["wall_s"] = round(time.time() - t_start)
    pd.DataFrame([row]).to_csv(os.path.join(tdir, key + "__cal.csv"), index=False)
    pd.DataFrame([dict(protein=p, noise_type=typ, condition=cond, bond_rms_dev_A=b_, omega_dev_mean_deg=om_, omega_dev_gt30_frac=omf_, rama_outside_frac=ram_, hydrogen_fail_rate=n_bad / len(sub), n_conf=len(allx))]).to_csv(os.path.join(tdir, key + "__geo.csv"), index=False)
    open(os.path.join(tdir, key + "__done"), "w").write("ok"); return key, "done"


def cmd_run(a, REV, INP):
    seqs = json.load(open(INP["tier1a_sequences"])); prots = a.proteins.split(",") if a.proteins else sorted(seqs)
    tasks = [(q, t, c, a.out_root, a.quick) for q in prots for t in TYPES for c in GENS + ["czero"]] + [(q, "generator", g, a.out_root, a.quick) for q in prots for g in GENS]
    tasks.sort(key=lambda x: -seqs[x[0]]["length"] * (0.05 if x[1] == "generator" else 1), reverse=a.reverse)   # 默认大蛋白先跑；--reverse 从小蛋白开始，用于第二个任务池，两头往中间走
    log = revio.RunLogger(stage=5, task="noise_ensembles" + ("_quick" if a.quick else ""), cpu_procs=a.workers, seeds="cal 101-103; final 21-25; ref det_seed(Crystal,p,0,0,0)")
    t0 = time.time(); fails = []; n = 0
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        futs = {ex.submit(do_task, t): t for t in tasks}
        for fu in cf.as_completed(futs):
            try:
                fu.result(); n += 1
            except Exception as e:
                fails.append((futs[fu][:3], repr(e)[:300])); print("失败:", fails[-1], flush=True)
            if n % 10 == 0:
                print(f"[{time.time() - t0:.0f}s] {n}/{len(tasks)} 任务完成，load={os.getloadavg()[0]:.1f}", flush=True)
    log.finish([a.out_root], exit_code=0 if not fails else 1, status="ok" if not fails else "partial", note=f"tasks={len(tasks)} failed={len(fails)} wall={time.time() - t0:.0f}s")
    if fails:
        raise SystemExit(f"{len(fails)} 个任务失败")


def cmd_merge(a, REV, INP):
    tdir = os.path.join(a.out_root, "tables")
    C = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(tdir, "*__cal.csv")))], ignore_index=True)
    G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(tdir, "*__geo.csv")))], ignore_index=True)
    if "calibration_round" not in C.columns:
        C["calibration_round"] = 1
    C["calibration_round"] = C["calibration_round"].fillna(1).astype(int)   # 第 2 轮 = s5_noise_cart_recalibrate.py 重新校准的笛卡尔条件（偏离 D5-4）
    C.to_csv(os.path.join(REV, "tables", "S5_noise_calibration.csv"), index=False); G.to_csv(os.path.join(REV, "tables", "S5_noise_geometry_qc.csv"), index=False)
    print(C.groupby(["noise_type", "status"]).size().to_string()); print("proteins:", C.protein.nunique(), "rows:", len(C), len(G))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "merge"]); ap.add_argument("--proteins", default=""); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_root", default=""); ap.add_argument("--quick", action="store_true"); ap.add_argument("--reverse", action="store_true"); a = ap.parse_args(); PROJ, REV, INP = revio.load_paths()
    a.out_root = a.out_root or os.path.join(REV, "raw_outputs", "s5", "noise_work")
    {"run": cmd_run, "merge": cmd_merge}[a.cmd](a, REV, INP)
