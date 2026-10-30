"""Stage 13 (a)(b) · ShortMD 副本的残基层 ρ、副本间变异，以及 ShortMD 相对沉积结构的检验随副本的变化（判据：md_replicates/analysis/PREREG_replicates.md 第 4 节，sha256 在同目录 .sha256）。
ρ：计数平均形式（原稿 Eq. 1）、口径 F、M_eff = min(50, n) 的 200 次冻结抽样的中位数；另给不抽样的“全部 200 帧”ρ。编号偏移取 r1（S4_parity）。
输出（tables/）：
  S13_shortmd_replicate_rho.csv              蛋白 × 副本一行（row_type=replicate）+ 每蛋白的副本汇总行（row_type=protein_summary，replicate='r1-r3'）
  S13_shortmd_replicate_tests.csv            ShortMD 对沉积结构的检验（r1 发表值、r1、r2、r3、三副本均值）与五类 Friedman
  S13_shortmd_replicate_variance_summary.csv 副本间变异的跨蛋白汇总、参照量、检验结果的副本间范围
用法：python s13_replicate_rho_tests.py [--reps r1,r2,r3] [--out-dir DIR]"""
import os, sys, json, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, ensemble as E, revstats, stats_extra as X, s13util as U  # noqa: E402

PROJ, REV, INP = revio.load_paths()
ap = argparse.ArgumentParser(); ap.add_argument("--reps", default="r1,r2,r3"); ap.add_argument("--out-dir", default=os.path.join(REV, "tables")); a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True); REPS = a.reps.split(",")
log = revio.RunLogger(stage=13, task="replicate_rho_tests", cpu_procs=1, seeds="frozen draws F_ShortMD[_r<k>]_<protein>_M50 (S4 / S13 manifests)")
PROTS = sorted(json.load(open(INP["tier1a_sequences"])))
Q = pd.read_csv(os.path.join(REV, "tables", "S7_invpf_vs_countavg_per_protein.csv")); Qc = Q[(Q.scenario == "primary") & (Q.form == "count_avg")]
Qs = {g: d_.set_index("protein").rho_median.reindex(PROTS) for g, d_ in Qc.groupby("generator")}; base = Qc.drop_duplicates("protein").set_index("protein").baseline_rho.reindex(PROTS)
FIVE = ["BioEmu", "ESMFlow", "Boltz2", "Str2Str_xtal", "ShortMD"]


def build_rho_table():
    rows = []
    for p in PROTS:
        for rep in REPS:
            cell, off_id = U.load_rep_cell(rep, p); fr, meta = U.load_frozen_draws(rep, p, 50); r = U.rho_frozen(cell, p, fr); n_ok = cell["NC"].shape[1]
            rows.append(dict(row_type="replicate", protein=p, replicate=rep, generator=U.gen_of(rep), n_frames_ok=n_ok, M_eff=meta["M_eff"], n_draws=int(np.isfinite(r).sum()), n_exp_residues_aligned=int(len(E.align_idx(cell, p)[1])),
                             rho_median=float(np.nanmedian(r)), rho_p2_5=float(np.nanpercentile(r, 2.5)), rho_p97_5=float(np.nanpercentile(r, 97.5)), rho_all_frames=U.rho_all_frames(cell, p),
                             offset_used=cell["offset"], offset_by_identity=off_id if off_id is not None else np.nan,
                             rho_S7_basisO=float(Qs["ShortMD"].loc[p]) if rep == "r1" else np.nan, baseline_rho_deposited=float(base.loc[p])))
    T = pd.DataFrame(rows); T["delta_vs_deposited"] = T.rho_median - T.baseline_rho_deposited
    T["bridge_abs_diff_vs_S7"] = np.where(T.replicate == "r1", (T.rho_median - T.rho_S7_basisO).abs(), np.nan)
    summ = []
    for p, d in T.groupby("protein"):
        v = d.rho_median.values; w = d.rho_all_frames.values
        summ.append(dict(row_type="protein_summary", protein=p, replicate="r1-r3" if len(REPS) == 3 else "+".join(REPS), generator="ShortMD (replicate summary)", n_frames_ok=int(d.n_frames_ok.sum()), M_eff=int(d.M_eff.iloc[0]),
                         rho_median=float(v.mean()), rho_all_frames=float(w.mean()), rho_sd_over_replicates=float(v.std(ddof=1)) if len(v) > 1 else np.nan, rho_min_over_replicates=float(v.min()), rho_max_over_replicates=float(v.max()),
                         rho_range_over_replicates=float(v.max() - v.min()), rho_all_frames_sd_over_replicates=float(w.std(ddof=1)) if len(w) > 1 else np.nan,
                         rho_all_frames_range_over_replicates=float(w.max() - w.min()), baseline_rho_deposited=float(base.loc[p]), delta_vs_deposited=float(v.mean() - base.loc[p]), n_replicates=len(v)))
    return pd.concat([T, pd.DataFrame(summ)], ignore_index=True)


def run_case(label, rho_vec):
    sm = pd.Series(rho_vec, index=PROTS).astype(float); d = (sm - base).values; pos, nn, ps = X.sign_test_greater(d)
    M = np.stack([(sm if g == "ShortMD" else Qs[g]).reindex(PROTS).values for g in FIVE], axis=1); fm = stats.friedmanchisquare(*[M[:, k] for k in range(5)])
    rk = stats.rankdata(-M, axis=1).mean(0); wst, wp = revstats.wilcoxon_greater(sm.values, base.values)
    return dict(case=label, n_proteins=len(PROTS), wins=int((d > 0).sum()), n_nonzero=int(nn), median_delta_rho=float(np.median(d)), mean_delta_rho=float(d.mean()), wilcoxon_greater_stat=wst, wilcoxon_greater_p=wp, sign_test_greater_p=ps,
                friedman_chi2=float(fm.statistic), friedman_p=float(fm.pvalue), **{f"mean_rank_{g}": float(rk[i]) for i, g in enumerate(FIVE)},
                bioemu_vs_shortmd_wilcoxon_two_sided_p=float(stats.wilcoxon(Qs["BioEmu"].values, sm.values).pvalue), wilcoxon_p_below_0_05=bool(wp < 0.05), sign_p_below_0_05=bool(ps < 0.05), friedman_p_below_0_05=bool(fm.pvalue < 0.05))


def main():
    T = build_rho_table(); T.to_csv(os.path.join(a.out_dir, "S13_shortmd_replicate_rho.csv"), index=False)
    R = T[T.row_type == "replicate"].pivot(index="protein", columns="replicate", values="rho_median").reindex(PROTS)
    RA = T[T.row_type == "replicate"].pivot(index="protein", columns="replicate", values="rho_all_frames").reindex(PROTS)
    cases = [run_case("r1 as published (S7 table, basis O)", Qs["ShortMD"].values)] + [run_case(f"{rep} (basis F, frozen draws)", R[rep].values) for rep in REPS] + ([run_case("mean of r1-r3 (per protein)", R.mean(axis=1).values)] if len(REPS) > 1 else [])
    C = pd.DataFrame(cases); C.to_csv(os.path.join(a.out_dir, "S13_shortmd_replicate_tests.csv"), index=False)
    # ---------------- 变异汇总与参照量
    S = []
    add = lambda item, value, note="": S.append(dict(item=item, value=value, note=note))
    sd = R.std(axis=1, ddof=1) if len(REPS) > 1 else pd.Series(np.nan, index=PROTS); rg = R.max(axis=1) - R.min(axis=1); sda = RA.std(axis=1, ddof=1) if len(REPS) > 1 else pd.Series(np.nan, index=PROTS); rga = RA.max(axis=1) - RA.min(axis=1)
    add("replicates", "+".join(REPS)); add("n_proteins", len(PROTS))
    add("per-protein SD of rho_median over replicates: median", float(sd.median())); add("per-protein SD of rho_median over replicates: mean", float(sd.mean())); add("per-protein SD of rho_median over replicates: max", float(sd.max()), str(sd.idxmax()))
    add("per-protein range of rho_median over replicates: median", float(rg.median())); add("per-protein range of rho_median over replicates: max", float(rg.max()), str(rg.idxmax()))
    add("per-protein SD of rho_all_frames over replicates: median", float(sda.median())); add("per-protein range of rho_all_frames over replicates: median", float(rga.median()))
    rr = T[T.row_type == "replicate"]; wd = (rr.rho_p97_5 - rr.rho_p2_5)
    add("width of 2.5-97.5 percentile interval of the 200 frozen draws: median over (protein, replicate)", float(wd.median()), "抽样（M=50 帧）造成的不确定度，同一条轨迹内")
    mean_sm = R.mean(axis=1); gens = {g: Qs[g] for g in FIVE if g != "ShortMD"}; gens["ShortMD"] = mean_sm
    add("median over proteins of |ShortMD(r-mean) - BioEmu|", float((mean_sm - Qs["BioEmu"]).abs().median())); add("median over proteins of |ShortMD(r-mean) - deposited structure|", float((mean_sm - base).abs().median()))
    pair = [(g1, g2) for i, g1 in enumerate(FIVE) for g2 in FIVE[i + 1:]]; add("median over proteins and generator pairs of |rho_i - rho_j| (5 generators, ShortMD = replicate mean)", float(np.median([abs(gens[g1] - gens[g2]).median() for g1, g2 in pair])))
    if len(REPS) > 1:
        k = len(REPS); x = R.values; gm = x.mean(); msb = k * ((x.mean(1) - gm) ** 2).sum() / (len(PROTS) - 1); msw = ((x - x.mean(1, keepdims=True)) ** 2).sum() / (len(PROTS) * (k - 1))
        add("one-way ANOVA ICC(1) of rho_median (proteins as groups, replicates as repeats)", float((msb - msw) / (msb + (k - 1) * msw)), "蛋白间差异占总变异的比例；MSB=%.4f MSW=%.4f" % (msb, msw))
        add("between-protein SD of replicate-mean rho", float(x.mean(1).std(ddof=1))); add("within-protein (replicate) SD pooled, sqrt(MSW)", float(np.sqrt(msw)))
        D = R.sub(base, axis=0); sg = np.sign(D.values); consist = ((sg == sg[:, [0]]).all(axis=1))
        add("proteins whose sign of (rho - deposited) is the same in all replicates", int(consist.sum()), "不一致的蛋白: " + ",".join(np.array(PROTS)[~consist]))
        add("proteins with rho > deposited in all replicates", int((D.values > 0).all(axis=1).sum())); add("proteins with rho < deposited in all replicates", int((D.values < 0).all(axis=1).sum()))
        cc = C[C.case.str.contains("basis F")]   # 只取各副本的口径 F 行（不含“发表值（口径 O）”和“三副本均值”）；行数应等于副本数
        assert len(cc) == len(REPS), (len(cc), REPS)
        for col in ("wilcoxon_greater_p", "sign_test_greater_p", "friedman_p", "median_delta_rho", "wins", "mean_rank_ShortMD", "bioemu_vs_shortmd_wilcoxon_two_sided_p"):
            add(f"{col}: min over {'/'.join(REPS)}", float(cc[col].min())); add(f"{col}: max over {'/'.join(REPS)}", float(cc[col].max()))
        add("replicates with wilcoxon_greater_p < 0.05", int(cc.wilcoxon_p_below_0_05.sum()), "共 %d 条副本（口径 F）" % len(cc)); add("replicates with sign_test_greater_p < 0.05", int(cc.sign_p_below_0_05.sum())); add("replicates with friedman_p < 0.05", int(cc.friedman_p_below_0_05.sum()))
        add("pairwise Spearman correlation across proteins of rho_median between replicates", "; ".join(f"{x1}-{x2}: {stats.spearmanr(R[x1], R[x2])[0]:.3f}" for i, x1 in enumerate(REPS) for x2 in REPS[i + 1:]))
        add("pairwise Pearson correlation across proteins of rho_median between replicates", "; ".join(f"{x1}-{x2}: {np.corrcoef(R[x1], R[x2])[0, 1]:.3f}" for i, x1 in enumerate(REPS) for x2 in REPS[i + 1:]))
    V = pd.DataFrame(S); V.to_csv(os.path.join(a.out_dir, "S13_shortmd_replicate_variance_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 90)
    print(T[T.row_type == "replicate"].pivot(index="protein", columns="replicate", values="rho_median").round(4).to_string()); print(C.round(4).T.to_string()); print(V.to_string(index=False))
    log.finish([os.path.join(a.out_dir, f"S13_shortmd_replicate_{n}.csv") for n in ("rho", "tests", "variance_summary")])


if __name__ == "__main__":
    main()
