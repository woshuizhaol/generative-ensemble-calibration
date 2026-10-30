"""Stage 9 追加（U-S3-2）· 步骤 8：cyt c 带血红素 / 无血红素对照 MD 的分析（判据见 `md_cytc_heme/analysis/PREREG_heme_effect.md`，sha256 与时间戳在同目录 .sha256，早于任何带血红素的结果）。
输出（tables/）：
  S9e_cytc_heme_rho.csv          每条轨迹一行：cyt c 残基层 ρ（计数平均，M=50、200 次冻结抽样，口径 F，编号偏移同 ShortMD）的中位数与 2.5 / 97.5 百分位，另含原 ShortMD（口径 F 与口径 O）；
  S9e_cytc_heme_effect.csv       血红素效应 Δ、两组极差与是否重叠；
  S9e_cytc_heme_substitution.csv 把 S7 表里 ShortMD×cyt c 的 ρ 依次替换后，S9 的检验结果（ShortMD 对结构的 Wilcoxon / 符号检验、五类 Friedman、BioEmu 对 ShortMD）；
  S9e_cytc_heme_qc.csv           每条轨迹的 QC（温度、帧数、失败帧、血红素配位距离、骨架 RMSD）；
  S9e_cytc_heme_prereg_check.csv 预登记判据（PREREG_heme_effect.md 第 1–4 条与补充 1）逐条核对。
用法：python s9e_cytc_heme_analysis.py --apo cytc_apo_s101,cytc_apo_s102,cytc_apo_s103 --heme cytc_heme_s201,cytc_heme_s202,cytc_heme_s203"""
import os, sys, argparse, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
from scipy import stats
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, ensemble as E, revstats, stats_extra as X  # noqa: E402
import s7_invpf_form as S7  # noqa: E402

PROJ, REV, INP = revio.load_paths(); TAB = os.path.join(REV, "tables"); MDD = os.path.join(REV, "md_cytc_heme", "md")
ap = argparse.ArgumentParser(); ap.add_argument("--apo", required=True); ap.add_argument("--heme", default=""); a = ap.parse_args()
log = revio.RunLogger(stage=9, task="cytc_heme_analysis", cpu_procs=1, seeds="frozen draws F_ShortMD_cytochrome_c_M50")
PROTS = S7.PROTS; P = "cytochrome_c"


def rho_of(g, force_offset):
    cell = E.load_cell(g, P, "F"); cell["offset"] = force_offset
    fr, meta = E.load_draws("F", "ShortMD", P, 50)   # 原 ShortMD × cyt c 的冻结抽样（帧号 0–199）
    cols = E.frames_to_cols(cell, fr); r = E.rho_draws(cell, P, cols, "count_avg")
    return r, cell


def between_(x, lo, hi):
    return lo <= x <= hi


def main():
    runs = [(g, "apo") for g in a.apo.split(",") if g] + [(g, "heme") for g in a.heme.split(",") if g]
    orig = E.load_cell("ShortMD", P, "F"); off = orig["offset"]; rows = []
    r0, _ = rho_of("ShortMD", off); Q = pd.read_csv(os.path.join(TAB, "S7_invpf_vs_countavg_per_protein.csv")); Qc = Q[(Q.scenario == "primary") & (Q.form == "count_avg")]
    qO = float(Qc[(Qc.generator == "ShortMD") & (Qc.protein == P)].rho_median.iloc[0])
    rows.append(dict(run="ShortMD (original, no heme, basis F)", group="original", rho_median=float(np.nanmedian(r0)), rho_p2_5=float(np.nanpercentile(r0, 2.5)), rho_p97_5=float(np.nanpercentile(r0, 97.5)), n_draws=int(np.isfinite(r0).sum())))
    rows.append(dict(run="ShortMD (original, no heme, basis O = S7 table)", group="original", rho_median=qO, n_draws=200))
    per = {}
    for g, grp in runs:
        gname = "ShortMD_" + g; r, cell = rho_of(gname, off); per[g] = float(np.nanmedian(r)); best = E.bv.best_offset({int(x): E.bv.to1(cell["resn"][int(x)]) for x in cell["res"]}, E.exp_residues(P), 0)
        rows.append(dict(run=g, group=grp, rho_median=per[g], rho_p2_5=float(np.nanpercentile(r, 2.5)), rho_p97_5=float(np.nanpercentile(r, 97.5)), n_draws=int(np.isfinite(r).sum()), offset_used=off, offset_by_identity=int(best)))
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "S9e_cytc_heme_rho.csv"), index=False)
    ap_, he_ = [per[g] for g, k in runs if k == "apo"], [per[g] for g, k in runs if k == "heme"]
    Qs = {g: d_.set_index("protein").rho_median.reindex(PROTS) for g, d_ in Qc.groupby("generator")}; base = Qc.drop_duplicates("protein").set_index("protein").baseline_rho.reindex(PROTS)
    eff = [dict(quantity="no-heme control seeds: mean / min / max / SD", value=f"{np.mean(ap_):.4f} / {min(ap_):.4f} / {max(ap_):.4f} / {np.std(ap_, ddof=1):.4f}", n=len(ap_))]
    if he_:
        d = np.mean(he_) - np.mean(ap_); overlap = not (min(he_) > max(ap_) or max(he_) < min(ap_))
        allv = np.array(ap_ + he_); lab = np.array([0] * len(ap_) + [1] * len(he_)); import itertools
        obs = abs(allv[lab == 1].mean() - allv[lab == 0].mean()); cnt = tot = 0
        for comb in itertools.combinations(range(len(allv)), len(he_)):
            m = np.zeros(len(allv), bool); m[list(comb)] = True; tot += 1; cnt += abs(allv[m].mean() - allv[~m].mean()) >= obs - 1e-12
        eff += [dict(quantity="heme seeds: mean / min / max / SD", value=f"{np.mean(he_):.4f} / {min(he_):.4f} / {max(he_):.4f} / {np.std(he_, ddof=1):.4f}", n=len(he_)), dict(quantity="heme effect Δ = mean(heme) − mean(no-heme control)", value=f"{d:+.4f}", n=len(he_)),
                dict(quantity="ranges overlap", value=str(overlap), n=len(he_)), dict(quantity="exact permutation p (two-sided, difference of means)", value=f"{cnt / tot:.3f}", n=tot),
                dict(quantity="95% CI of Δ (pooled-variance t, df = n1 + n2 − 2)", value="[%+.3f, %+.3f]" % ((d - stats.t.ppf(0.975, len(he_) + len(ap_) - 2) * np.sqrt(((len(he_) - 1) * np.var(he_, ddof=1) + (len(ap_) - 1) * np.var(ap_, ddof=1)) / (len(he_) + len(ap_) - 2)) * np.sqrt(1 / len(he_) + 1 / len(ap_))), (d + stats.t.ppf(0.975, len(he_) + len(ap_) - 2) * np.sqrt(((len(he_) - 1) * np.var(he_, ddof=1) + (len(ap_) - 1) * np.var(ap_, ddof=1)) / (len(he_) + len(ap_) - 2)) * np.sqrt(1 / len(he_) + 1 / len(ap_)))), n=len(he_)),
                dict(quantity="cyt c: deposited-structure rho / median rho of the other four generators (S9_shortmd_outlier_check)", value="%.3f / %.3f" % (float(base.loc[P]), float(np.median([Qs[g].loc[P] for g in ("BioEmu", "ESMFlow", "Boltz2", "Str2Str_xtal")]))), n=1),
                dict(quantity="original ShortMD (basis F) minus no-heme control mean", value=f"{float(np.nanmedian(r0)) - np.mean(ap_):+.4f}", n=1)]
    pd.DataFrame(eff).to_csv(os.path.join(TAB, "S9e_cytc_heme_effect.csv"), index=False)
    # ---- 替换检验
    five = ["BioEmu", "ESMFlow", "Boltz2", "Str2Str_xtal", "ShortMD"]; sub = []

    def run_case(label, rho_cyt):
        sm = Qs["ShortMD"].copy(); sm.loc[P] = rho_cyt; d = (sm - base).values; pos, nn, ps = X.sign_test_greater(d)
        M = np.stack([(sm if g == "ShortMD" else Qs[g]).reindex(PROTS).values for g in five], axis=1); fm = stats.friedmanchisquare(*[M[:, k] for k in range(5)])
        sub.append(dict(case=label, rho_cytc_ShortMD=float(rho_cyt), n=12, wins=int((d > 0).sum()), wilcoxon_greater_p=revstats.wilcoxon_greater(sm.values, base.values)[1], sign_test_p=ps, median_delta=float(np.median(d)),
                        friedman_p=float(fm.pvalue), mean_rank_ShortMD=float(stats.rankdata(-M, axis=1).mean(0)[five.index("ShortMD")]),
                        bioemu_vs_shortmd_wilcoxon_two_sided_p=float(stats.wilcoxon(Qs["BioEmu"].values, sm.values).pvalue)))
    run_case("as published (S7 table, basis O, no heme)", qO); run_case("original ShortMD re-scored (basis F, no heme)", float(np.nanmedian(r0))); run_case("no-heme control seeds, mean", float(np.mean(ap_)))
    if he_:
        run_case("heme seeds, mean", float(np.mean(he_)))
    for g, grp in runs:   # 逐条轨迹单独代入（单条轨迹的随机性足以让 ShortMD 对沉积结构的 p 在 0.05 附近来回）
        run_case(f"single trajectory: {g} ({grp})", per[g])
    pd.DataFrame(sub).to_csv(os.path.join(TAB, "S9e_cytc_heme_substitution.csv"), index=False)
    # ---- QC
    qc = []
    for g, grp in runs:
        d = os.path.join(MDD, g); lg = pd.read_csv(os.path.join(d, "md.log")); lg.columns = [c.strip('#" ') for c in lg.columns]; T = lg.iloc[:, 2]
        F = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "ShortMD_" + g, f"{P}_frames.parquet")); top = md.load(os.path.join(d, "final.pdb")).topology
        t = md.load(os.path.join(d, "full.xtc"), top=os.path.join(d, "final.pdb")); ca = top.select("protein and name CA"); rm = md.rmsd(t, t, 0, atom_indices=ca) * 10.0
        ref = t[0]; sv = md.load(os.path.join(REV, "md_cytc_heme", "param", ("cytc_heme" if grp == "heme" else "cytc_apo") + "_solv.pdb")); assert sv.n_atoms == t.n_atoms, (sv.n_atoms, t.n_atoms)   # tleap 起始结构（加水前后原子顺序同 prmtop）
        ref.xyz[0] = sv.xyz[0]; rs_ = md.rmsd(t, ref, 0, atom_indices=ca) * 10.0
        row = dict(run=g, group=grp, md_log_mean_T_K=float(T.mean()), frames=int(t.n_frames), failed_scoring_frames=int((~F.status.isin(["ok", "ok_retry"])).sum()), ca_rmsd_to_first_frame_mean_A=float(rm.mean()), ca_rmsd_max_A=float(rm.max()),
                   ca_rmsd_to_tleap_start_mean_A=float(rs_.mean()), ca_rmsd_to_tleap_start_max_A=float(rs_.max()))
        if grp == "heme":
            fe = top.select("name FE and resname FE")[0]; ne2 = [at.index for at in top.atoms if at.residue.resSeq == 18 and at.name == "NE2"][0]; sd = [at.index for at in top.atoms if at.residue.resSeq == 80 and at.name == "SD"][0]
            sg14 = [at.index for at in top.atoms if at.residue.resSeq == 14 and at.name == "SG"][0]; cab = [at.index for at in top.atoms if at.residue.name == "HEC" and at.name == "CAB"][0]
            sg17 = [at.index for at in top.atoms if at.residue.resSeq == 17 and at.name == "SG"][0]; cac = [at.index for at in top.atoms if at.residue.name == "HEC" and at.name == "CAC"][0]
            dd = md.compute_distances(t, [[fe, ne2], [fe, sd], [sg14, cab], [sg17, cac]]) * 10.0
            row.update(fe_NE2_His18_mean_A=float(dd[:, 0].mean()), fe_NE2_max_A=float(dd[:, 0].max()), fe_SD_Met80_mean_A=float(dd[:, 1].mean()), fe_SD_max_A=float(dd[:, 1].max()), sg14_cab_mean_A=float(dd[:, 2].mean()), sg17_cac_mean_A=float(dd[:, 3].mean()))
        qc.append(row)
    QC = pd.DataFrame(qc); QC.to_csv(os.path.join(TAB, "S9e_cytc_heme_qc.csv"), index=False)
    # ---- 预登记判据逐条核对（PREREG_heme_effect.md 与 PREREG_heme_addendum1.md）
    chk = []
    H = QC[QC.group == "heme"]
    if len(H):
        add = lambda cid, crit, ok, ev: chk.append(dict(criterion=cid, text=crit, passed=bool(ok), evidence=ev))
        add("1", "平均温度 300±5 K；200 帧；打分失败帧 0（带血红素的 3 条）", ((H.md_log_mean_T_K - 300).abs() < 5).all() and (H.frames == 200).all() and (H.failed_scoring_frames == 0).all(), f"T = {H.md_log_mean_T_K.min():.2f}–{H.md_log_mean_T_K.max():.2f} K；帧数 {sorted(set(H.frames))}；失败帧 {int(H.failed_scoring_frames.sum())}")
        add("2", "Fe–NE2 均值 1.8–2.5 Å；Fe–SD 均值 2.0–3.0 Å；两个硫醚键不断", H.fe_NE2_His18_mean_A.between(1.8, 2.5).all() and H.fe_SD_Met80_mean_A.between(2.0, 3.0).all() and (H.sg14_cab_mean_A < 2.2).all() and (H.sg17_cac_mean_A < 2.2).all(),
            f"Fe–NE2 {H.fe_NE2_His18_mean_A.min():.2f}–{H.fe_NE2_His18_mean_A.max():.2f}（最大 {H.fe_NE2_max_A.max():.2f}）Å；Fe–SD {H.fe_SD_Met80_mean_A.min():.2f}–{H.fe_SD_Met80_mean_A.max():.2f}（最大 {H.fe_SD_max_A.max():.2f}）Å；SG–CAB {H.sg14_cab_mean_A.min():.2f}、SG–CAC {H.sg17_cac_mean_A.min():.2f}–{H.sg17_cac_mean_A.max():.2f} Å")
        add("3", "蛋白 CA 对起始结构的 RMSD 均值 <0.4 nm（预登记写“起始结构”；这里两种参照都报，判定取相对 tleap 起始结构）", (H.ca_rmsd_to_tleap_start_mean_A < 4.0).all() and (H.ca_rmsd_to_first_frame_mean_A < 4.0).all(), f"相对生产段首帧 {H.ca_rmsd_to_first_frame_mean_A.min():.2f}–{H.ca_rmsd_to_first_frame_mean_A.max():.2f} Å（无血红素对照 {QC[QC.group == 'apo'].ca_rmsd_to_first_frame_mean_A.min():.2f}–{QC[QC.group == 'apo'].ca_rmsd_to_first_frame_mean_A.max():.2f}）；相对 tleap 起始结构 {H.ca_rmsd_to_tleap_start_mean_A.min():.2f}–{H.ca_rmsd_to_tleap_start_mean_A.max():.2f} Å（无血红素对照 {QC[QC.group == 'apo'].ca_rmsd_to_tleap_start_mean_A.min():.2f}–{QC[QC.group == 'apo'].ca_rmsd_to_tleap_start_mean_A.max():.2f}）")
        sem = pd.read_csv(os.path.join(REV, "md_cytc_heme", "param", "seminario_fe_site.tsv"), sep="\t"); b = sem[sem.kind == "bond"]
        kN = b[b.atoms.str.contains("HEC.N")].K_amber; kNE2 = b[b.atoms.str.contains("NE2")].K_amber.iloc[0]; kS = b[b.atoms.str.contains("SD")].K_amber.iloc[0]
        add("4a", "补充 1：Seminario 的 Fe–N 键力常数在 Giammona/Bayly 值（50–60）的 2 倍以内（25–120），Fe–S 在 20–100 kcal/mol/Å²", kN.between(25, 120).all() and between_(kNE2, 25, 120) and between_(kS, 20, 100), f"Fe–N(卟啉) {kN.min():.1f}–{kN.max():.1f}；Fe–NE2 {kNE2:.1f}；Fe–SD {kS:.1f}")
        fq = [float(x) for l in open(os.path.join(REV, "md_cytc_heme", "qm", "small.log")) if "Frequencies --" in l for x in l.split("--")[1].split()]
        add("4b", "补充 1：小模型优化收敛、频率分析没有虚频", min(fq) > 0, f"最低频率 {min(fq):.1f} cm⁻¹，虚频 {sum(1 for x in fq if x < 0)} 个")
        cr = {}
        for l in open(os.path.join(REV, "md_cytc_heme", "prep", "cytc_heme_start.pdb")):
            if l.startswith(("ATOM", "HETATM")):
                cr[(l[17:20].strip(), int(l[22:26]), l[12:16].strip())] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        fe0 = cr[("FE", 105, "FE")]; dc = {n: float(np.linalg.norm(cr[("HEC", 106, n)] - fe0)) for n in ("NA", "NB", "NC", "ND")}; dc["NE2"] = float(np.linalg.norm(cr[("HID", 18, "NE2")] - fe0)); dc["SD"] = float(np.linalg.norm(cr[("MET", 80, "SD")] - fe0))
        qmg = {r.atoms.split("-")[1].replace("HEC.", "").replace("HID18.", "").replace("MET80.", ""): r.geom for r in b.itertuples()}
        dN = {n: abs(qmg[n] - dc[n]) for n in ("NA", "NB", "NC", "ND")}
        add("4c", "补充 1：优化几何与晶体的 Fe 配位相符——Fe–N(卟啉) 每个键差 ≤0.06 Å、Fe–NE2 差 ≤0.1 Å、Fe–S 差 ≤0.15 Å", all(v <= 0.06 for v in dN.values()) and abs(qmg["NE2"] - dc["NE2"]) <= 0.1 and abs(qmg["SD"] - dc["SD"]) <= 0.15,
            "Fe–N 逐键差 " + "、".join(f"{n} {v:.3f}" for n, v in dN.items()) + f" Å；Fe–NE2 差 {abs(qmg['NE2'] - dc['NE2']):.3f} Å；Fe–SD 差 {abs(qmg['SD'] - dc['SD']):.3f} Å（晶体 Fe–N 坐标误差在 1.9 Å 分辨率下约 0.05–0.1 Å）")
        rr = pd.read_csv(os.path.join(REV, "md_cytc_heme", "param", "resp_charges.tsv"), sep="\t"); add("4d", "补充 1：HEC + FE 总电荷 = −1", abs(rr.charge.sum() - (-1.0)) < 1e-4, f"RESP 六位小数的和 {rr.charge.sum():+.6f}（写入 mol2 时把舍入残差加到 Fe 上，tleap 报告体系净电荷 8.000000）")
    pd.DataFrame(chk).to_csv(os.path.join(TAB, "S9e_cytc_heme_prereg_check.csv"), index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    for n in ("rho", "effect", "substitution", "qc", "prereg_check"):
        print("==", n); print(pd.read_csv(os.path.join(TAB, f"S9e_cytc_heme_{n}.csv")).round(4).to_string(index=False))
    log.finish([os.path.join(TAB, f"S9e_cytc_heme_{n}.csv") for n in ("rho", "effect", "substitution", "qc", "prereg_check")])


if __name__ == "__main__":
    main()
