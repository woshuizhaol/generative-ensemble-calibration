"""Stage 13 (c) · ShortMD 每条轨迹（r1 = 原 ShortMD，r2、r3 = 新副本）的时间历程：势能、温度、盒子体积、Cα RMSD、回旋半径，残基层 ρ 在 0.5 ns 窗口与累积窗口上的值，前半 vs 后半，平衡性描述。
判据与口径：md_replicates/analysis/PREREG_replicates.md 第 5 节与补充 1。帧 i（0 基）的生产段时间 = (i+1)×10 ps。
  · r1 的 md.log 只有 100 ps 分辨率的势能与温度（20 行，原脚本只记录了这些）；盒子体积来自 ensemble.xtc 每帧晶胞（10 ps）。r2、r3 有 10 ps 的势能 / 温度 / 体积 / 动能，以及 1 ps 分辨率的 100 ps 平衡段日志。
  · Cα 为 A 链（打分链）的 Cα。RMSD 先做最优叠合：对生产段首帧；对沉积结构（按残基顺序 / 序列比对逐一对应）；r2、r3 另对 initial_system.pdb（溶剂化后、最小化前）。
  · ρ 同 S13_replicate_rho_tests.py（计数平均、口径 F、编号偏移取 r1）。窗口 ρ 两种口径：all = 窗口内全部帧一次计算；m50 = 窗口内无放回抽 50 帧、200 次的中位数（窗口帧数 ≤50 时与 all 相同）。
输出（tables/）：
  S13_shortmd_frame_series.csv   每条轨迹每帧一行（200 × 36）
  S13_shortmd_timecourse.csv     蛋白 × 副本 × 窗口（4 个独立 0.5 ns 窗口 + 4 个累积窗口）：窗口均值与 ρ
  S13_shortmd_half_split.csv     前半（0–1 ns）vs 后半（1–2 ns）的 ρ 与 ln PF 谱相关；row_type=summary 为跨蛋白汇总
  S13_shortmd_equilibration.csv  平衡段与生产段的漂移描述
用法：python s13_timecourse.py [--reps r1,r2,r3] [--out-dir DIR] [--proteins A,B]"""
import os, sys, json, argparse, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, ensemble as E, s13util as U  # noqa: E402

PROJ, REV, INP = revio.load_paths()
ap = argparse.ArgumentParser(); ap.add_argument("--reps", default="r1,r2,r3"); ap.add_argument("--out-dir", default=os.path.join(REV, "tables")); ap.add_argument("--proteins", default=""); a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True); REPS = a.reps.split(",")
log = revio.RunLogger(stage=13, task="timecourse", cpu_procs=1, seeds=f"window draws entropy={U.WIN_ENTROPY}+r, spawn_key=(cell_id_T, 50)")
PROTS = [p for p in sorted(json.load(open(INP["tier1a_sequences"]))) if not a.proteins or p in a.proteins.split(",")]
S3 = pd.read_csv(os.path.join(REV, "tables", "S3_md_protocol_per_protein.csv")).set_index("protein")


def rmsd_to(t_ca, ref_xyz):
    ref = md.Trajectory(ref_xyz[None], t_ca.topology); return md.rmsd(t_ca, ref, 0) * 10.0


def load_log_prod(path, rep):
    """生产段日志 → DataFrame(frame_idx, potential_kJ, kinetic_kJ, temperature_K, volume_nm3)，frame_idx 由生产段时间 (ps)/10 − 1 得到。"""
    L = U.read_md_log(path)
    t_prod = U.prod_time_ps_from_step(L.step) if rep == "r1" else (L.time_ps.values - 100.0)
    L["frame_idx"] = np.rint(np.asarray(t_prod, float) / U.FRAME_DT_PS).astype(int) - 1
    return L[L.frame_idx.between(0, 199)].drop_duplicates("frame_idx")


def lin_slope(t, y):
    r = stats.linregress(t, y); return float(r.slope)


def main():
    FS, TC, HS, EQ = [], [], [], []
    for p in PROTS:
        pdb_id = S3.loc[p, "pdb_id"]
        for rep in REPS:
            P = U.md_paths(rep, p); t = md.load(P["ens_xtc"], top=P["ens_pdb"]); ca = U.chain0_ca(t.topology); tca = t.atom_slice(ca); md_names = [at.residue.name for at in tca.topology.atoms]
            n = t.n_frames; assert n == 200, (p, rep, n)
            pos, dxyz, nmatch, ndep = U.deposited_ca(pdb_id, md_names)
            rm_first = md.rmsd(tca, tca, 0) * 10.0; rm_dep = md.rmsd(tca.atom_slice(pos), md.Trajectory(dxyz[None], tca.atom_slice(pos).topology), 0) * 10.0
            rm_init = np.full(n, np.nan)
            if P["initial"]:
                ini = md.load(P["initial"]); ica = U.chain0_ca(ini.topology); assert len(ica) == len(ca), (p, rep, len(ica), len(ca)); rm_init = rmsd_to(tca, ini.xyz[0][ica])
            rg = md.compute_rg(tca) * 10.0; vol = t.unitcell_volumes
            L = load_log_prod(P["md_log"], rep).set_index("frame_idx")
            fs = pd.DataFrame(dict(protein=p, replicate=rep, frame_idx=np.arange(n), time_ns=(np.arange(n) + 1) * U.FRAME_DT_PS / 1000.0, box_volume_nm3=vol, ca_rmsd_to_first_frame_A=rm_first, ca_rmsd_to_deposited_A=rm_dep,
                                   ca_rmsd_to_initial_system_A=rm_init, ca_rg_A=rg))
            for c in ("potential_kJ", "kinetic_kJ", "temperature_K"):
                fs[c] = fs.frame_idx.map(L[c]) if c in L else np.nan
            fs["on_100ps_grid"] = (fs.frame_idx % 10 == 9); FS.append(fs)
            cell, _ = U.load_rep_cell(rep, p); gen = U.gen_of(rep); assert cell["NC"].shape[1] == n, (p, rep, cell["NC"].shape)
            # ---------- 窗口表
            grid = fs[fs.on_100ps_grid]
            for lab, f0, f1, kind in U.windows():
                w = fs[(fs.frame_idx >= f0) & (fs.frame_idx < f1)]; g = grid[(grid.frame_idx >= f0) & (grid.frame_idx < f1)]; ids = list(range(f0, f1))
                r_all, nf = U.rho_subset_frames(cell, p, ids); r_m50, _, nd = U.rho_window_matched(cell, p, ids, gen, lab)
                TC.append(dict(protein=p, replicate=rep, window=lab, window_type=kind, t_start_ns=f0 * U.FRAME_DT_PS / 1000.0, t_end_ns=f1 * U.FRAME_DT_PS / 1000.0, n_frames=nf,
                               mean_box_volume_nm3=w.box_volume_nm3.mean(), mean_ca_rmsd_to_first_A=w.ca_rmsd_to_first_frame_A.mean(), sd_ca_rmsd_to_first_A=w.ca_rmsd_to_first_frame_A.std(ddof=1),
                               mean_ca_rmsd_to_deposited_A=w.ca_rmsd_to_deposited_A.mean(), sd_ca_rmsd_to_deposited_A=w.ca_rmsd_to_deposited_A.std(ddof=1), mean_ca_rg_A=w.ca_rg_A.mean(),
                               mean_potential_kJ_100ps_grid=g.potential_kJ.mean(), mean_temperature_K_100ps_grid=g.temperature_K.mean(), n_grid_rows=len(g),
                               rho_all_frames_in_window=r_all, rho_matched50_median=r_m50, n_matched50_draws=nd))
            # ---------- 前半 vs 后半
            h = {}
            for lab, f0, f1 in (("H1", 0, 100), ("H2", 100, 200)):
                ids = list(range(f0, f1)); r_all, _ = U.rho_subset_frames(cell, p, ids); r_m, _, _ = U.rho_window_matched(cell, p, ids, gen, lab); h[lab] = (r_all, r_m, U.lnpf_profile(cell, p, ids)[1])
            rho_lp = stats.spearmanr(h["H1"][2], h["H2"][2])[0]
            HS.append(dict(row_type="run", protein=p, replicate=rep, rho_first_half_all=h["H1"][0], rho_second_half_all=h["H2"][0], diff_second_minus_first_all=h["H2"][0] - h["H1"][0],
                           rho_first_half_m50=h["H1"][1], rho_second_half_m50=h["H2"][1], diff_second_minus_first_m50=h["H2"][1] - h["H1"][1], lnpf_profile_spearman_first_vs_second_half=float(rho_lp),
                           ca_rmsd_first_half_mean_A=fs.ca_rmsd_to_deposited_A[:100].mean(), ca_rmsd_second_half_mean_A=fs.ca_rmsd_to_deposited_A[100:].mean()))
            # ---------- 平衡性描述
            row = dict(protein=p, replicate=rep, equil_log_available=bool(P["equil_log"]))
            if P["equil_log"]:
                e = U.read_md_log(P["equil_log"]); e["t"] = e.time_ps.values; last = e[e.t > e.t.max() - 50.0]
                row.update(equil_n_rows=len(e), equil_last50ps_potential_slope_kJmol_per_ns=lin_slope(last.t / 1000.0, last.potential_kJ), equil_last50ps_potential_slope_pct_per_ns=100 * lin_slope(last.t / 1000.0, last.potential_kJ) / abs(last.potential_kJ.mean()),
                           equil_last50ps_volume_slope_pct_per_ns=100 * lin_slope(last.t / 1000.0, last.volume_nm3) / last.volume_nm3.mean(), equil_last50ps_mean_T_K=float(last.temperature_K.mean()), equil_mean_T_K=float(e.temperature_K.mean()),
                           equil_first_potential_kJ=float(e.potential_kJ.iloc[0]), equil_last_potential_kJ=float(e.potential_kJ.iloc[-1]), equil_first_volume_nm3=float(e.volume_nm3.iloc[0]), equil_last_volume_nm3=float(e.volume_nm3.iloc[-1]))
            tt = fs.time_ns.values
            for name, series in (("volume_nm3", fs.box_volume_nm3), ("ca_rmsd_to_deposited_A", fs.ca_rmsd_to_deposited_A), ("potential_kJ_100ps_grid", grid.potential_kJ), ("temperature_K_100ps_grid", grid.temperature_K)):
                if name.endswith("100ps_grid"):
                    tv, yv, fi = grid.time_ns.values, series.values, grid.frame_idx.values
                else:
                    tv, yv, fi = tt, series.values, fs.frame_idx.values
                blocks = [yv[(fi >= 50 * k) & (fi < 50 * (k + 1))].mean() for k in range(4)]; first = yv[fi < 100].mean(); second = yv[fi >= 100].mean(); bse = float(np.std(blocks, ddof=1) / 2.0)
                row.update({f"prod_{name}_mean": float(yv.mean()), f"prod_{name}_slope_per_ns": lin_slope(tv, yv), f"prod_{name}_first_half_mean": float(first), f"prod_{name}_second_half_mean": float(second),
                            f"prod_{name}_second_minus_first": float(second - first), f"prod_{name}_block_se": bse, f"prod_{name}_diff_over_block_se": float((second - first) / bse) if bse > 0 else np.nan})
            row.update(prod_potential_slope_pct_per_ns=100 * row["prod_potential_kJ_100ps_grid_slope_per_ns"] / abs(row["prod_potential_kJ_100ps_grid_mean"]), prod_volume_slope_pct_per_ns=100 * row["prod_volume_nm3_slope_per_ns"] / row["prod_volume_nm3_mean"],
                       prod_T_sd_K_100ps_grid=float(grid.temperature_K.std(ddof=1)), n_ca_matched_to_deposited=nmatch, n_ca_deposited_chain=ndep, n_ca_md_chain=len(ca))
            EQ.append(row); print(p, rep, "ok", flush=True)
    FS = pd.concat(FS, ignore_index=True); TC = pd.DataFrame(TC); HS = pd.DataFrame(HS); EQ = pd.DataFrame(EQ)
    # ---------- 前半 vs 后半的跨蛋白汇总
    summ = []
    rep_sd = np.nan
    f_rho = os.path.join(a.out_dir, "S13_shortmd_replicate_rho.csv")
    if os.path.exists(f_rho):
        Rt = pd.read_csv(f_rho); rep_sd = float(Rt[Rt.row_type == "protein_summary"].rho_sd_over_replicates.median())
    for rep, d in list(HS.groupby("replicate")) + ([("mean_over_replicates", HS.groupby("protein")[["diff_second_minus_first_all", "diff_second_minus_first_m50", "lnpf_profile_spearman_first_vs_second_half"]].mean().reset_index())] if len(REPS) > 1 else []):
        for col, lab in (("diff_second_minus_first_all", "all"), ("diff_second_minus_first_m50", "m50")):
            x = d[col].values; npos = int((x > 0).sum()); nnz = int((x != 0).sum())
            summ.append(dict(row_type="summary", replicate=rep, basis=lab, n_proteins=len(x), mean_diff=float(x.mean()), median_diff=float(np.median(x)), median_abs_diff=float(np.median(np.abs(x))), max_abs_diff=float(np.abs(x).max()), n_second_gt_first=npos,
                             sign_test_two_sided_p=float(stats.binomtest(npos, nnz, 0.5).pvalue) if nnz else np.nan, wilcoxon_signed_rank_two_sided_p=float(stats.wilcoxon(x).pvalue), median_lnpf_profile_spearman=float(d.lnpf_profile_spearman_first_vs_second_half.median()),
                             median_replicate_sd_of_rho=rep_sd))
    HS = pd.concat([HS, pd.DataFrame(summ)], ignore_index=True)
    for name, df in (("frame_series", FS), ("timecourse", TC), ("half_split", HS), ("equilibration", EQ)):
        df.to_csv(os.path.join(a.out_dir, f"S13_shortmd_{name}.csv"), index=False); print(name, df.shape)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(HS[HS.row_type == "summary"].round(4).to_string(index=False)); print(TC[TC.protein == PROTS[0]].round(3).head(24).to_string(index=False))
    log.finish([os.path.join(a.out_dir, f"S13_shortmd_{n}.csv") for n in ("frame_series", "timecourse", "half_split", "equilibration")])


if __name__ == "__main__":
    main()
