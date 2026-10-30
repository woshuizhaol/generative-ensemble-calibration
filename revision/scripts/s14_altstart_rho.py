"""Stage 14 · 从 BioEmu 构象出发的短 MD（r4 = 第 25 百分位起点，r5 = 第 75 百分位起点）的残基层 ρ，与从沉积结构出发的 r1–r3 比较。
口径与 S13 相同：计数平均形式（Eq. 1）、口径 F、M_eff = min(50, n) 的 200 次冻结抽样中位数；r4、r5 的编号偏移按残基身份推得（MD 输出的残基号由 1 重新编号；核对：对齐到的实验残基数与 BioEmu 系综相同）。
分析与判读规则：docs/PREREG_s14_altstart.md（写于任何 r4、r5 的 ρ 被计算之前）。
输出（tables/）：
  S14_altstart_rho.csv       蛋白 × 起点一行（r4、r5）：ρ、与沉积结构基线之差、与 r1–r3 均值之差及其与副本间 s.d. 之比
  S14_altstart_tests.csv     相对沉积结构的检验：r1–r3 均值、r4、r5、r4 与 r5 的平均（单侧配对 Wilcoxon、符号检验、中位 Δρ、改善蛋白数）
  S14_altstart_summary.csv   预登记指标与判读规则的逐项结果；Cα RMSD 与平衡性的汇总
用法：python s14_altstart_rho.py"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, ensemble as E, revstats, stats_extra as X, s13util as U  # noqa: E402

PROJ, REV, INP = revio.load_paths()
TAB = os.path.join(REV, "tables")
log = revio.RunLogger(stage=14, task="altstart_rho", cpu_procs=1, seeds="frozen draws F_ShortMD_r<4|5>_<protein>_M50 (tables/S14_draw_indices_manifest.csv)")
PROTS = sorted(json.load(open(INP["tier1a_sequences"])))
NEW = ["r4", "r5"]


def main():
    R13 = pd.read_csv(os.path.join(TAB, "S13_shortmd_replicate_rho.csv")); R13 = R13[R13.row_type == "replicate"]
    rho13 = R13.pivot(index="protein", columns="replicate", values="rho_median").reindex(PROTS); base = R13.drop_duplicates("protein").set_index("protein").baseline_rho_deposited.reindex(PROTS)
    Q7 = pd.read_csv(os.path.join(TAB, "S7_invpf_vs_countavg_per_protein.csv")); Q7 = Q7[(Q7.scenario == "primary") & (Q7.form == "count_avg")]
    Qs = {g: d_.set_index("protein").rho_median.reindex(PROTS) for g, d_ in Q7.groupby("generator")}; Q = Qs["BioEmu"]
    sd13 = rho13.std(axis=1, ddof=1); mean13 = rho13.mean(axis=1)
    rows = []
    for p in PROTS:
        for rep in NEW:
            cell, off_id = U.load_rep_cell(rep, p); fr, meta = U.load_frozen_draws(rep, p, 50); r = U.rho_frozen(cell, p, fr); n_ok = cell["NC"].shape[1]
            rho = float(np.nanmedian(r))
            n_al = int(len(E.align_idx(cell, p)[1])); n_ref = int(len(E.align_idx(E.load_cell("BioEmu", p, "F"), p)[1])); assert n_al == n_ref, (p, rep, n_al, n_ref)   # 与 BioEmu 系综对齐到同一批实验残基
            rows.append(dict(protein=p, replicate=rep, start=("25th percentile" if rep == "r4" else "75th percentile"), n_frames_ok=n_ok, M_eff=meta["M_eff"], n_draws=int(np.isfinite(r).sum()), n_exp_residues_aligned=n_al, n_exp_residues_aligned_BioEmu=n_ref,
                             rho_median=rho, rho_p2_5=float(np.nanpercentile(r, 2.5)), rho_p97_5=float(np.nanpercentile(r, 97.5)), rho_all_frames=U.rho_all_frames(cell, p), offset_used=cell["offset"], offset_by_identity=off_id,
                             baseline_rho_deposited=float(base.loc[p]), rho_BioEmu=float(Q.loc[p]), rho_mean_r1_r3=float(mean13.loc[p]), rho_sd_r1_r3=float(sd13.loc[p])))
    T = pd.DataFrame(rows); T["delta_vs_deposited"] = T.rho_median - T.baseline_rho_deposited; T["diff_vs_mean_r1_r3"] = T.rho_median - T.rho_mean_r1_r3
    T["abs_diff_over_sd_r1_r3"] = T.diff_vs_mean_r1_r3.abs() / T.rho_sd_r1_r3
    sel = pd.read_csv(os.path.join(TAB, "S14_altstart_selection.csv")).assign(replicate=lambda d: "r" + d.replicate.astype(str))[["protein", "replicate", "frame_idx", "calpha_rmsd_to_deposited_A"]].rename(columns={"frame_idx": "start_frame_idx", "calpha_rmsd_to_deposited_A": "start_ca_rmsd_to_deposited_A"})
    T = T.merge(sel, on=["protein", "replicate"], how="left"); T.to_csv(os.path.join(TAB, "S14_altstart_rho.csv"), index=False)
    A = {rep: T[T.replicate == rep].set_index("protein").rho_median.reindex(PROTS) for rep in NEW}
    # ---------------- 相对沉积结构的检验（与 S13 同口径）
    def case(label, v):
        sm = pd.Series(v, index=PROTS).astype(float); d = (sm - base).values; pos, nn, ps = X.sign_test_greater(d); wst, wp = revstats.wilcoxon_greater(sm.values, base.values)
        # 与 S13 同口径的五类生成器比较（短 MD 的 ρ 换成该组；其余四个生成器不变）。该部分在看到逐蛋白结果之后加入，属事后分析。
        FIVE = ["BioEmu", "ESMFlow", "Boltz2", "Str2Str_xtal", "ShortMD"]
        M = np.stack([(sm if g == "ShortMD" else Qs[g]).reindex(PROTS).values for g in FIVE], axis=1); fm = stats.friedmanchisquare(*[M[:, k] for k in range(5)]); rk = stats.rankdata(-M, axis=1).mean(0)
        return dict(case=label, n_proteins=len(PROTS), wins=int((d > 0).sum()), n_nonzero=int(nn), median_delta_rho=float(np.median(d)), mean_delta_rho=float(d.mean()), wilcoxon_greater_stat=wst, wilcoxon_greater_p=wp, sign_test_greater_p=ps, wilcoxon_p_below_0_05=bool(wp < 0.05),
                    friedman_chi2_post_hoc=float(fm.statistic), friedman_p_post_hoc=float(fm.pvalue), mean_rank_ShortMD_post_hoc=float(rk[4]), mean_rank_BioEmu_post_hoc=float(rk[0]),
                    bioemu_vs_shortmd_wilcoxon_two_sided_p_post_hoc=float(stats.wilcoxon(Qs["BioEmu"].reindex(PROTS).values, sm.values).pvalue), n_proteins_shortmd_below_bioemu_post_hoc=int((sm.values < Qs["BioEmu"].reindex(PROTS).values).sum()))
    C = pd.DataFrame([case("r1-r3 mean (deposited-structure start)", mean13.values), case("r4 (25th-percentile BioEmu start)", A["r4"].values), case("r5 (75th-percentile BioEmu start)", A["r5"].values),
                      case("mean of r4 and r5", ((A["r4"] + A["r5"]) / 2).values)])
    C.to_csv(os.path.join(TAB, "S14_altstart_tests.csv"), index=False)
    # ---------------- 汇总与判读规则
    S = []
    add = lambda item, value, note="": S.append(dict(item=item, value=value, note=note))
    add("n_proteins", len(PROTS)); add("n_trajectories", len(T)); add("n_frames_ok per trajectory: min", int(T.n_frames_ok.min())); add("aligned experimental residues equal to those of the BioEmu ensemble in all trajectories", bool((T.n_exp_residues_aligned == T.n_exp_residues_aligned_BioEmu).all()), "offsets used (by residue identity): " + ";".join(f"{r.protein}_{r.replicate}:{r.offset_used}" for r in T.itertuples()))
    ratio = T.abs_diff_over_sd_r1_r3.values
    add("median |rho(start) - mean(r1-r3)| / SD(r1-r3), 24 trajectories", float(np.median(ratio)), "描述量，不设门槛"); add("  r4 only", float(np.median(T[T.replicate == "r4"].abs_diff_over_sd_r1_r3))); add("  r5 only", float(np.median(T[T.replicate == "r5"].abs_diff_over_sd_r1_r3)))
    add("fraction of trajectories with |diff| / SD(r1-r3) <= 2", float((ratio <= 2).mean())); add("median |rho(start) - mean(r1-r3)|", float(T.diff_vs_mean_r1_r3.abs().median())); add("max |rho(start) - mean(r1-r3)|", float(T.diff_vs_mean_r1_r3.abs().max()), str(T.loc[T.diff_vs_mean_r1_r3.abs().idxmax(), ["protein", "replicate"]].tolist()))
    add("median SD(r1-r3) over proteins", float(sd13.median()))
    c0 = C.iloc[0]
    add("median delta-rho > 0 for r1-r3 mean, r4 and r5", bool(all(C.iloc[i].median_delta_rho > 0 for i in (0, 1, 2))))
    add("one-sided Wilcoxon p of r1-r3 mean, r4, r5 versus deposited structure", "; ".join(f"{C.iloc[i].wilcoxon_greater_p:.4g}" for i in (0, 1, 2)))
    # 标准一致性统计：组内相关、Lin 一致性相关系数、Bland–Altman、系统性偏移检验
    allr = pd.concat([rho13, A["r4"].rename("r4"), A["r5"].rename("r5")], axis=1)[["r1", "r2", "r3", "r4", "r5"]]
    k_ = allr.shape[1]; n_ = allr.shape[0]; gm = allr.values.mean(); msb = k_ * ((allr.mean(1) - gm) ** 2).sum() / (n_ - 1); msw = ((allr.values - allr.values.mean(1, keepdims=True)) ** 2).sum() / (n_ * (k_ - 1))
    icc = (msb - msw) / (msb + (k_ - 1) * msw); F = msb / msw; Fl = F / stats.f.ppf(0.975, n_ - 1, n_ * (k_ - 1)); Fu = F * stats.f.ppf(0.975, n_ * (k_ - 1), n_ - 1)
    add("ICC(1) of rho across r1-r5 (12 proteins as groups, 5 trajectories each)", float(icc), "95%% CI %.3f to %.3f; MSB=%.4f MSW=%.4f" % ((Fl - 1) / (Fl + k_ - 1), (Fu - 1) / (Fu + k_ - 1), msb, msw))
    add("between-protein SD of the mean rho over r1-r5", float(allr.mean(1).std(ddof=1)))
    def ccc(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float); return float(2 * np.cov(x, y, ddof=0)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))
    for nm, x in (("r4", A["r4"]), ("r5", A["r5"])):
        d_ = (x - mean13).values; sd_ = d_.std(ddof=1)
        add(f"Lin concordance correlation coefficient, {nm} versus mean(r1-r3)", ccc(x, mean13)); add(f"Bland-Altman bias {nm} - mean(r1-r3)", float(d_.mean()), "limits of agreement %.3f to %.3f (bias +/- 1.96 SD)" % (d_.mean() - 1.96 * sd_, d_.mean() + 1.96 * sd_))
        add(f"two-sided paired Wilcoxon p, {nm} versus mean(r1-r3)", float(stats.wilcoxon(x.values, mean13.values).pvalue))
    add("Lin concordance correlation coefficient, r4 versus r5", ccc(A["r4"], A["r5"]))
    d45_ = (A["r4"] - A["r5"]).values; add("Bland-Altman bias r4 - r5", float(d45_.mean()), "limits of agreement %.3f to %.3f" % (d45_.mean() - 1.96 * d45_.std(ddof=1), d45_.mean() + 1.96 * d45_.std(ddof=1)))
    add("two-sided paired Wilcoxon p, r4 versus r5", float(stats.wilcoxon(A["r4"].values, A["r5"].values).pvalue))
    # 起点之间
    d45 = (A["r4"] - A["r5"]).abs(); pair13 = pd.concat([(rho13[x] - rho13[y]).abs() for x, y in (("r1", "r2"), ("r1", "r3"), ("r2", "r3"))])
    add("median |rho(r4) - rho(r5)| over proteins", float(d45.median())); add("max |rho(r4) - rho(r5)|", float(d45.max()), str(d45.idxmax())); add("median |rho_i - rho_j| over r1-r3 pairs and proteins", float(pair13.median())); add("max |rho_i - rho_j| over r1-r3 pairs", float(pair13.max()))
    add("Spearman across proteins: rho(r4) vs mean(r1-r3)", float(stats.spearmanr(A["r4"], mean13)[0])); add("Spearman across proteins: rho(r5) vs mean(r1-r3)", float(stats.spearmanr(A["r5"], mean13)[0])); add("Spearman across proteins: rho(r4) vs rho(r5)", float(stats.spearmanr(A["r4"], A["r5"])[0]))
    add("Pearson across proteins: rho(r4) vs mean(r1-r3)", float(np.corrcoef(A["r4"], mean13)[0, 1])); add("Pearson across proteins: rho(r5) vs mean(r1-r3)", float(np.corrcoef(A["r5"], mean13)[0, 1]))
    add("proteins with rho > deposited in r4", int((A["r4"] > base).sum())); add("proteins with rho > deposited in r5", int((A["r5"] > base).sum())); add("proteins with rho > deposited in r4, r5 and mean(r1-r3)", int(((A["r4"] > base) & (A["r5"] > base) & (mean13 > base)).sum()))
    add("proteins whose sign of (rho - deposited) is the same in r1, r2, r3, r4 and r5", int(((np.sign(rho13.sub(base, axis=0)).eq(np.sign(A["r4"] - base), axis=0)).all(axis=1) & (np.sign(A["r4"] - base) == np.sign(A["r5"] - base))).sum()))
    add("median over proteins of |rho(start) - rho(BioEmu)|", float((T.rho_median - T.rho_BioEmu).abs().median())); add("median over proteins of |mean(r1-r3) - rho(BioEmu)|", float((mean13 - Q).abs().median()))
    # Cα RMSD 与平衡性（若 s13_timecourse 已对 r4、r5 运行）
    f14 = os.path.join(TAB, "S14_altstart_frame_series.csv"); f13 = os.path.join(TAB, "S13_shortmd_frame_series.csv")
    if os.path.exists(f14) and os.path.exists(f13):
        F = pd.concat([pd.read_csv(f13), pd.read_csv(f14)], ignore_index=True); F["group"] = np.where(F.replicate.isin(NEW), F.replicate, "r1-r3")
        g = F.groupby(["group", "protein", "replicate"]).agg(rm_dep=("ca_rmsd_to_deposited_A", "mean"), rm_first=("ca_rmsd_to_first_frame_A", "mean"), rm_init=("ca_rmsd_to_initial_system_A", "mean"),
                                                             rm_dep_end=("ca_rmsd_to_deposited_A", lambda x: x.iloc[-20:].mean()), rm_first_end=("ca_rmsd_to_first_frame_A", lambda x: x.iloc[-20:].mean())).reset_index()
        for grp, d in g.groupby("group"):
            add(f"{grp}: median over trajectories of mean Cα RMSD to deposited structure (A)", float(d.rm_dep.median()), "n=%d" % len(d))
            add(f"{grp}: median over trajectories of mean Cα RMSD to first production frame (A)", float(d.rm_first.median()))
            add(f"{grp}: median over trajectories of Cα RMSD to deposited structure, last 0.2 ns (A)", float(d.rm_dep_end.median()))
            if grp in NEW + ["r1-r3"]:
                add(f"{grp}: median over trajectories of mean Cα RMSD to initial system (A)", float(d.rm_init.median()) if d.rm_init.notna().any() else np.nan)
        # 起点相对沉积结构的偏差 vs 轨迹末端相对沉积结构的偏差（是否向沉积结构回归或远离）
        st = F[F.frame_idx == 0].set_index(["protein", "replicate"]).ca_rmsd_to_deposited_A; en = F[F.frame_idx >= 180].groupby(["protein", "replicate"]).ca_rmsd_to_deposited_A.mean()
        for rep in NEW:
            ix = [(p, rep) for p in PROTS if (p, rep) in st.index]
            if ix:
                add(f"{rep}: median Cα RMSD to deposited at first production frame (A)", float(st.loc[ix].median())); add(f"{rep}: median Cα RMSD to deposited, last 0.2 ns (A)", float(en.loc[ix].median()))
    V = pd.DataFrame(S); V.to_csv(os.path.join(TAB, "S14_altstart_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 100)
    print(T.pivot(index="protein", columns="replicate", values="rho_median").assign(mean_r1_r3=mean13, sd_r1_r3=sd13, deposited=base).round(4).to_string()); print(C.round(4).T.to_string()); print(V.to_string(index=False))
    log.finish([os.path.join(TAB, f"S14_altstart_{n}.csv") for n in ("rho", "tests", "summary")])


if __name__ == "__main__":
    main()
