"""Stage 8d · 噪声系综对照的预登记检验（revision_proposal.md S8 步骤 2；R2-C6b）。检验的定义冻结在 tables/S5_noise_prereg.md §8（sha256 在 S5_noise_prereg.sha256，冻结于任何噪声得分之前）。
数据：S4b 的噪声系综打分 raw_outputs/s4/counts/noise_<anm|cart>_<条件>_s<种子>/；每个（蛋白, 噪声类型, 条件, 种子）的 50 个构象作为一个系综，计数平均与 1/PF 平均两种形式各算一个 Spearman ρ。
对齐：噪声系综是沉积结构 recon_0 加扰动，残基编号与 `Crystal` 的 parquet 同源，所以对齐用沉积结构基线用的偏移（operator_ceiling.offset），不按残基身份重新找（否则与 ρ_deposited 的口径不一致）。
参照量：ρ_G = S7 主分析的 ρ 中位数（同一生成器、口径 O、M_eff=min(50,n)；Str2Str_xtal 为种子 11 的 50 帧）；ρ_deposited = 原稿单结构基线（operator_ceiling.spearman，与 S7 相同），另报 10 次重建均值作敏感性。
统计（预登记）：
  主分析（ANM）：Δ=ρ_G−ρ_noise(匹配到 G)，12 个蛋白配对；单侧 Wilcoxon（G>noise），5 个生成器的 p 做 BH（报告校正前后）；TOST 等价界 ±0.05（配对 t，与原稿一致）；蛋白层 bootstrap（seed 7，B=10⁴）给 Δ 的 95% CI。
  次要分析（笛卡尔）：方法同主分析；同时报告“全部条件”与“仅第 1 轮即达标的条件（排除重新校准的 16 个）”两种集合。
  补充对比：ρ_noise 对 ρ_deposited，单侧 Wilcoxon（noise>structure），5 个条件 BH。
  C-zero：幅度 0.01 Å 的系综得分与沉积结构 10 次重建的 ρ 相比，差值在 10 次重建 SD 以内；SD=0 的蛋白（Barstar、CheY，沉积结构自带氢，重建无随机性）按绝对差报告并单独标注。
事后敏感性（C-zero 验收未全部通过后加入，**明确标为事后**，不替代上面的预登记主检验）：(a) czero_corrected：ρ_noise 减去该蛋白的 C-zero 偏移（ρ_czero − ρ_recon_mean），去掉“去氢—约束最小化—重新加氢”流程本身造成的逐蛋白位移后再做 G 对噪声的检验；
  (b) 留一蛋白：G 对匹配噪声的 Wilcoxon p 在 12 次留一中的范围。
子命令：python s8d_noise_tests.py run [--types anm,cart] [--seeds 21,22,23,24,25]
输出：tables/S8_noise_per_seed.csv、S8_noise_per_protein.csv、S8_noise_tests.csv、S8_noise_czero.csv（有 --suffix 时加后缀）"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys, json, argparse, hashlib
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, revstats, ensemble as E, stats_extra as X  # noqa: E402

PROJ, REV, INP = revio.load_paths(); TAB = os.path.join(REV, "tables")
PROTS = sorted(json.load(open(INP["tier1a_sequences"]))); GENS = ["BioEmu", "ESMFlow", "Boltz2", "ShortMD", "Str2Str_xtal"]; FORMS = ["count_avg", "invpf_avg"]; B = 10000; EPS = 0.05
ceil = pd.read_parquet(INP["operator_ceiling"]); OFF_S = dict(zip(ceil.protein, ceil.offset.astype(int))); BASE = dict(zip(ceil.protein, ceil.spearman))


NMR_HYDROGEN = ["Barstar", "CheY"]   # 沉积结构自带氢（NMR）的蛋白：10 次重建的 N_C/N_H 逐位相同，重建 SD=0


def rng_for(*tags):
    return np.random.default_rng([7, int(hashlib.sha256("|".join(map(str, tags)).encode()).hexdigest()[:8], 16)])


def per_seed(types, seeds):
    rows = []
    for typ in types:
        for cond in GENS + ["czero"]:
            for p in PROTS:
                for sd in seeds:
                    g = f"noise_{typ}_{cond}_s{sd}"; f = os.path.join(REV, "raw_outputs", "s4", "counts", g, f"{p}.parquet")
                    if not os.path.exists(f):
                        rows.append(dict(protein=p, noise_type=typ, condition=cond, seed=sd, status="not_scored")); continue
                    F = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", g, f"{p}_frames.parquet")); nb = int((~F.status.isin(["ok", "ok_retry"])).sum())
                    cell = E.load_cell(g, p, "F"); cell["offset"] = OFF_S[p]; cols = np.arange(cell["NC"].shape[1])[None, :]
                    for form in FORMS:
                        r = E.rho_draws(cell, p, cols, form)
                        rows.append(dict(protein=p, noise_type=typ, condition=cond, seed=sd, form=form, rho=float(r[0]), n_frames=len(F), n_failed_frames=nb, n_residues_compared=int(len(E.align_idx(cell, p)[0])), status="scored"))
    return pd.DataFrame(rows)


def refs():
    Q = pd.read_csv(os.path.join(TAB, "S7_invpf_vs_countavg_per_protein.csv"), float_precision="round_trip"); Q = Q[Q.scenario == "primary"]
    rg = {(r.generator, r.protein, r.form): r.rho_median for r in Q.itertuples()}
    rc = pd.read_csv(os.path.join(TAB, "S8_recon_sd.csv"), float_precision="round_trip").set_index("protein"); pr = pd.read_csv(os.path.join(TAB, "S8_recon_pairs.csv"), float_precision="round_trip")
    rec0 = pr[pr.recon_i == 0].drop_duplicates("protein").set_index("protein").rho_i
    cal = pd.read_csv(os.path.join(TAB, "S5_noise_calibration.csv")); cal = cal.set_index(["protein", "noise_type", "condition"])
    return rg, rc, rec0, cal


def paired_stats(a, b, tag):
    """a = ρ_G（或 ρ_noise），b = 参照；Δ=a−b。返回 dict。"""
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b; n = len(d)
    w = revstats.wilcoxon_greater(a, b)[1] if np.any(d != 0) else np.nan; hl = X.hodges_lehmann(d); md = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n))
    p_tost = float(max(stats.t.sf((md + EPS) / se, n - 1), stats.t.cdf((md - EPS) / se, n - 1))) if se > 0 else np.nan
    rng = rng_for("noise", tag); idx = rng.integers(0, n, size=(B, n)); bm = d[idx].mean(1); bmed = np.median(d[idx], axis=1)
    return dict(n=n, wins_a_gt_b=int((d > 0).sum()), wilcoxon_greater_p=w, median_delta=float(np.median(d)), mean_delta=md, hl_estimate=hl[0], hl_ci_lo=hl[1], hl_ci_hi=hl[2],
                boot_median_lo=float(np.percentile(bmed, 2.5)), boot_median_hi=float(np.percentile(bmed, 97.5)), boot_mean_lo=float(np.percentile(bm, 2.5)), boot_mean_hi=float(np.percentile(bm, 97.5)),
                tost_p_pm0p05=p_tost, equivalent_within_pm0p05=bool(p_tost < 0.05) if np.isfinite(p_tost) else False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run"]); ap.add_argument("--types", default="anm,cart"); ap.add_argument("--seeds", default="21,22,23,24,25"); ap.add_argument("--suffix", default=""); a = ap.parse_args()
    types = a.types.split(","); seeds = [int(x) for x in a.seeds.split(",")]; log = revio.RunLogger(stage=8, task="noise_tests" + a.suffix, cpu_procs=1, seeds="noise seeds 21-25; bootstrap [7, hash]")
    prereg_ok = open(os.path.join(TAB, "S5_noise_prereg.sha256")).read().split()[0] == revio.sha256(os.path.join(TAB, "S5_noise_prereg.md")); assert prereg_ok, "预登记文件的 sha256 与冻结时不一致"
    S = per_seed(types, seeds); S.to_csv(os.path.join(TAB, f"S8_noise_per_seed{a.suffix}.csv"), index=False)
    sc = S[S.status == "scored"]; rg, rc, rec0, cal = refs(); rows = []
    for (typ, cond, p, form), d in sc.groupby(["noise_type", "condition", "protein", "form"]):
        c = cal.loc[(p, typ, cond)] if (p, typ, cond) in cal.index else None
        rows.append(dict(protein=p, noise_type=typ, condition=cond, form=form, n_seeds=len(d), rho_noise_mean=d.rho.mean(), rho_noise_sd_over_seeds=d.rho.std(ddof=1) if len(d) > 1 else np.nan, rho_noise_min=d.rho.min(), rho_noise_max=d.rho.max(),
                         rho_G=rg.get((cond, p, form), np.nan) if cond != "czero" else np.nan, rho_deposited_original=BASE[p], rho_deposited_recon_mean=float(rc.loc[p, "rho_recon_mean"]), rho_deposited_recon0=float(rec0.loc[p]),
                         target_spread_A=(np.nan if c is None else c.target_spread_A), realized_spread_A=(np.nan if c is None else c.realized_mean_A), calibration_round=(np.nan if c is None else c.get("calibration_round", 1)),
                         rmsf_pearson_with_G=(np.nan if c is None else c.rmsf_pearson_with_G), n_failed_frames=int(d.n_failed_frames.sum())))
    P = pd.DataFrame(rows); P["calibration_round"] = P.calibration_round.fillna(1)
    czm = P[P.condition == "czero"].set_index(["protein", "noise_type", "form"]).rho_noise_mean   # 事后敏感性 (a)：该蛋白的 C-zero 偏移 = ρ_czero − ρ_recon_mean
    P["czero_shift"] = [float(czm.get((r.protein, r.noise_type, r.form), np.nan)) - r.rho_deposited_recon_mean for r in P.itertuples()]; P["rho_noise_czero_corrected"] = P.rho_noise_mean - P.czero_shift
    P.to_csv(os.path.join(TAB, f"S8_noise_per_protein{a.suffix}.csv"), index=False)
    out = []
    for typ in types:
        for form in FORMS:
            for cond in GENS:
                d = P[(P.noise_type == typ) & (P.condition == cond) & (P.form == form)].set_index("protein")
                for subset, sel in (("all_conditions", d), ("round1_conditions_only", d[d.calibration_round == 1])):
                    if typ == "anm" and subset != "all_conditions":
                        continue
                    complete = len(d) == len(PROTS) and (d.n_seeds == len(seeds)).all()
                    if len(sel) < 5:
                        continue
                    r1 = paired_stats(sel.rho_G, sel.rho_noise_mean, f"{typ}|{form}|{cond}|{subset}|G_vs_noise"); r1.update(analysis="G_vs_matched_noise", noise_type=typ, form=form, condition=cond, protein_subset=subset, complete=bool(complete)); out.append(r1)
                    loo = [revstats.wilcoxon_greater(sel.rho_G.drop(q).values, sel.rho_noise_mean.drop(q).values)[1] for q in sel.index]; r1.update(loo_p_min=float(np.nanmin(loo)), loo_p_max=float(np.nanmax(loo)), loo_n_p_lt_0p05=int(np.sum(np.array(loo) < 0.05)))
                    r1c = paired_stats(sel.rho_G, sel.rho_noise_czero_corrected, f"{typ}|{form}|{cond}|{subset}|G_vs_noise_corrected"); r1c.update(analysis="G_vs_matched_noise_czero_corrected_POST_HOC", noise_type=typ, form=form, condition=cond, protein_subset=subset, complete=bool(complete))
                    loo = [revstats.wilcoxon_greater(sel.rho_G.drop(q).values, sel.rho_noise_czero_corrected.drop(q).values)[1] for q in sel.index]; r1c.update(loo_p_min=float(np.nanmin(loo)), loo_p_max=float(np.nanmax(loo)), loo_n_p_lt_0p05=int(np.sum(np.array(loo) < 0.05))); out.append(r1c)
                    sel10 = sel.drop(index=[q for q in NMR_HYDROGEN if q in sel.index])   # 事后 (b)：去掉自带氢的 NMR 结构蛋白（C-zero 的偏差主要出在它们身上），只留 10 个 X 射线蛋白
                    if subset == "all_conditions" and len(sel10) >= 5:
                        for an10, col10 in (("G_vs_matched_noise_xray10_POST_HOC", "rho_noise_mean"), ("G_vs_matched_noise_czero_corrected_xray10_POST_HOC", "rho_noise_czero_corrected")):
                            r10 = paired_stats(sel10.rho_G, sel10[col10], f"{typ}|{form}|{cond}|xray10|{an10}"); r10.update(analysis=an10, noise_type=typ, form=form, condition=cond, protein_subset="xray10_only", complete=bool(complete))
                            loo = [revstats.wilcoxon_greater(sel10.rho_G.drop(q).values, sel10[col10].drop(q).values)[1] for q in sel10.index]; r10.update(loo_p_min=float(np.nanmin(loo)), loo_p_max=float(np.nanmax(loo)), loo_n_p_lt_0p05=int(np.sum(np.array(loo) < 0.05))); out.append(r10)
                    for bname, bcol in (("original_single_structure", "rho_deposited_original"), ("recon_mean", "rho_deposited_recon_mean")):
                        r2 = paired_stats(sel.rho_noise_mean, sel[bcol], f"{typ}|{form}|{cond}|{subset}|noise_vs_{bname}"); r2.update(analysis=f"noise_vs_deposited_{bname}", noise_type=typ, form=form, condition=cond, protein_subset=subset, complete=bool(complete)); out.append(r2)
    T = pd.DataFrame(out)
    if len(T):
        for (an, typ, form, subset), g in T.groupby(["analysis", "noise_type", "form", "protein_subset"]):   # BH：5 个生成器/条件为一族
            T.loc[g.index, "wilcoxon_greater_bh_q_5conditions"] = X.bh(g.wilcoxon_greater_p.fillna(1.0).values)
    T.to_csv(os.path.join(TAB, f"S8_noise_tests{a.suffix}.csv"), index=False)
    # ---- C-zero
    cz = []
    for typ in types:
        for form in FORMS:
            d = P[(P.noise_type == typ) & (P.condition == "czero") & (P.form == form)].set_index("protein")
            for p, r in d.iterrows():
                sdv = float(rc.loc[p, "rho_sd_over_10_recon"]); diff_mean = r.rho_noise_mean - r.rho_deposited_recon_mean; diff0 = r.rho_noise_mean - r.rho_deposited_recon0
                cz.append(dict(protein=p, noise_type=typ, form=form, rho_czero_mean=r.rho_noise_mean, rho_czero_sd_over_seeds=r.rho_noise_sd_over_seeds, rho_recon_mean=r.rho_deposited_recon_mean, rho_recon0=r.rho_deposited_recon0, sd_10_recon=sdv,
                               diff_vs_recon_mean=diff_mean, diff_vs_recon0=diff0, within_10recon_sd=bool(abs(diff_mean) <= sdv) if sdv > 1e-9 else False, deterministic_recon_sd_zero=bool(sdv <= 1e-9), abs_diff_if_sd_zero=(abs(diff_mean) if sdv <= 1e-9 else np.nan)))   # Barstar、CheY 的 SD 是 ~1e-17 的浮点残留，用容差判断
    Z = pd.DataFrame(cz); Z.to_csv(os.path.join(TAB, f"S8_noise_czero{a.suffix}.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("per-seed scored:", int((S.status == "scored").sum()), "not scored:", int((S.status == "not_scored").sum()), "failed frames:", int(sc.n_failed_frames.sum()) if len(sc) else 0)
    if len(T):
        print(T[(T.analysis == "G_vs_matched_noise") & (T.form == "count_avg")][["noise_type", "condition", "protein_subset", "n", "wins_a_gt_b", "median_delta", "mean_delta", "wilcoxon_greater_p", "wilcoxon_greater_bh_q_5conditions", "tost_p_pm0p05", "boot_median_lo", "boot_median_hi", "complete"]].round(4).to_string(index=False))
    if len(Z):
        zz = Z[Z.form == "count_avg"]; print("C-zero count_avg:", {t: f"{int(g.within_10recon_sd.sum())}/{len(g)} within SD; sd=0 proteins: {list(g[g.deterministic_recon_sd_zero].protein)}" for t, g in zz.groupby("noise_type")})
    log.finish([os.path.join(TAB, f"S8_noise_tests{a.suffix}.csv")], note="noise tests")


if __name__ == "__main__":
    main()
