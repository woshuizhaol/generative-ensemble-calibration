"""Stage 9b · S9 步骤 5–8：训练集重叠敏感性、cyt c 无血红素、五类生成器可区分性、全帧口径敏感性、标签置换阴性对照。
口径 Q = S7 主分析（口径 O，M_eff=min(50,n)；BioEmu、ShortMD 有 200 次冻结抽样，其余生成器 1 次）。口径 P = 原稿 null_baselines（仅 BioEmu）。
输出：tables/S9_overlap_sensitivity.csv、S9_cytc_md_sensitivity.csv、S9_paradigm_separability.csv、S9_paradigm_separability_draws.csv、S9_fullframe_sensitivity.csv、S9_perm_negative_control.csv。
Str2Str 一栏：--str2str xtal（默认，主表，新版 Str2Str_xtal，起点为沉积结构）；--str2str legacy（旧版，起点为 BioEmu 首帧，表名带后缀 _str2str_legacy）。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, itertools, argparse
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, revstats, ensemble as E, stats_extra as X  # noqa: E402
import s7_invpf_form as S7  # noqa: E402

PROJ, REV, INP = revio.load_paths()
PROTS = S7.PROTS; B = 10000
FIVE = {"BioEmu": "BioEmu", "ESMFlow": "ESMFlow", "Boltz2": "Boltz2", "Str2Str": "Str2Str_legacy", "ShortMD": "ShortMD"}   # 报告名 → S7 里的生成器名；main() 按 --str2str 把 "Str2Str" 换成 Str2Str_xtal（新版，沉积结构起点）或 Str2Str_legacy（旧版，BioEmu 首帧起点）
SFX = ""   # 变体依赖型输出表的后缀：新版为主（无后缀），旧版并列为 _str2str_legacy
ceil = pd.read_parquet(INP["operator_ceiling"]); OFF_S = dict(zip(ceil.protein, ceil.offset.astype(int))); NB = revio.correct_null_baselines(pd.read_csv(os.path.join(INP["results_tables"], "null_baselines.csv")), INP).set_index("protein")


def draws_rho(dname, p, form="count_avg", basis="O"):
    """口径 Q（或 F）：返回 (R,) ρ 抽样。"""
    if dname.startswith("Str2Str_xtal"):   # S4b：每个种子 50 个构象，没有冻结的口径 O 索引；与 S7 主分析一致，取种子 11 的全部 ok 帧（一次“抽样”）
        cell = E.load_cell(dname, p, "F"); cols = np.arange(cell["NC"].shape[1])[None, :]; return E.rho_draws(cell, p, cols, form), dict(M_eff=cols.shape[1], n_available=cols.shape[1], n_lt_M=cols.shape[1] < 50)
    cell = E.load_cell(dname, p, basis); fr, meta = E.load_draws(basis, dname, p, 50)
    return E.rho_draws(cell, p, (fr if basis == "O" else E.frames_to_cols(cell, fr)), form), meta


def tests(ens, base, names):
    d = (ens - base).loc[names].values; n = len(d); pos, nn, ps = X.sign_test_greater(d)
    return dict(n=n, wins=int((d > 0).sum()), wilcoxon_greater_p=revstats.wilcoxon_greater(ens.loc[names].values, base.loc[names].values)[1], sign_test_p=ps, median_delta=float(np.median(d)), mean_delta=float(d.mean()))


def main():
    global FIVE, SFX
    ap = argparse.ArgumentParser(); ap.add_argument("--str2str", choices=["auto", "xtal", "legacy"], default="auto"); a = ap.parse_args()
    Q0 = pd.read_csv(os.path.join(REV, "tables", "S7_invpf_vs_countavg_per_protein.csv")); has_x = "Str2Str_xtal" in set(Q0.generator)
    variant = ("xtal" if has_x else "legacy") if a.str2str == "auto" else a.str2str
    assert variant != "xtal" or has_x, "S7 表里还没有 Str2Str_xtal：先合并 S4b 并重跑 s7_invpf_form.py"
    FIVE = dict(FIVE, Str2Str=("Str2Str_xtal" if variant == "xtal" else "Str2Str_legacy")); SFX = "" if (variant == "xtal" or not has_x) else "_str2str_legacy"
    print(f"[s9b] 五类比较中的 Str2Str = {FIVE['Str2Str']}；变体依赖型输出表后缀 '{SFX}'", flush=True)
    log = revio.RunLogger(stage=9, task="overlap_separability" + SFX, cpu_procs=1, seeds="permutation seed 0 (numpy default_rng); no bootstrap in this script")
    Q = pd.read_csv(os.path.join(REV, "tables", "S7_invpf_vs_countavg_per_protein.csv")); Qc = Q[(Q.scenario == "primary") & (Q.form == "count_avg")]
    ens_q = {g: d.set_index("protein").rho_median.reindex(PROTS) for g, d in Qc.groupby("generator")}; base = Qc.drop_duplicates("protein").set_index("protein").baseline_rho.reindex(PROTS)
    # ---- 5. 重叠敏感性（名单取自 S2_overlap_exclusion_lists.csv）
    EX = pd.read_csv(os.path.join(REV, "tables", "S2_overlap_exclusion_lists.csv")).set_index("generator"); rows = []
    for gq, gs2 in (("BioEmu", "BioEmu"), ("ESMFlow", "ESMFlow"), ("Boltz2", "Boltz2"), ("Str2Str_legacy", "Str2Str"), ("ShortMD", "ShortMD")) + ((("Str2Str_xtal", "Str2Str"),) if has_x else ()):
        for lvl in ("none", "strong", "strict"):
            ex = [] if lvl == "none" else ([x for x in str(EX.loc[gs2, f"exclude_{lvl}"]).split(";") if x and x != "nan"])
            names = [p for p in PROTS if p not in ex]
            rows.append(dict(generator=gq, basis="Q", exclusion_level=lvl, excluded=";".join(ex), s2_list_verbatim=("" if lvl == "none" else str(EX.loc[gs2, f"exclude_{lvl}"])), **tests(ens_q[gq], base, names)))
            if gq == "BioEmu":
                rows.append(dict(generator=gq, basis="P", exclusion_level=lvl, excluded=";".join(ex), s2_list_verbatim=("" if lvl == "none" else str(EX.loc[gs2, f"exclude_{lvl}"])), **tests(NB.obs_spearman.reindex(PROTS), NB.single_struct_ceiling.reindex(PROTS), names)))
    pd.DataFrame(rows).to_csv(os.path.join(REV, "tables", "S9_overlap_sensitivity.csv"), index=False)
    # ---- 7. 五类生成器可区分性（每次抽样 r：BioEmu、ShortMD 取第 r 次抽样，其余生成器固定）
    R = 200; mat = {}
    for rep, dn in FIVE.items():
        dname = S7.GENS[dn]
        arr = np.full((R, len(PROTS)), np.nan)
        for j, p in enumerate(PROTS):
            r, meta = draws_rho(dname, p)
            arr[:, j] = r[:R] if len(r) >= R else np.resize(r, R)   # 只有 1 个"抽样"（n≤50 或固定帧）的生成器重复 R 次
        mat[rep] = arr
    names5 = list(FIVE); fr_p = []; pair_p = {pair: [] for pair in itertools.combinations(names5, 2)}; ranks = []
    for r in range(R):
        M = np.stack([mat[g][r] for g in names5], axis=1)                                   # (12, 5)
        fr = stats.friedmanchisquare(*[M[:, k] for k in range(5)]); fr_p.append((fr.statistic, fr.pvalue)); ranks.append(stats.rankdata(-M, axis=1).mean(0))
        ps = [stats.wilcoxon(M[:, i], M[:, j]).pvalue for (i, j) in itertools.combinations(range(5), 2)]; q = X.bh(ps)
        for (pair, qv) in zip(pair_p, q):
            pair_p[pair].append(qv)
    fr_p = np.array(fr_p); ranks = np.array(ranks)
    D = pd.DataFrame(dict(draw=np.arange(R), friedman_chi2=fr_p[:, 0], friedman_p=fr_p[:, 1], **{f"meanrank_{g}": ranks[:, k] for k, g in enumerate(names5)})); D.to_csv(os.path.join(REV, "tables", "S9_paradigm_separability_draws" + SFX + ".csv"), index=False)
    sep = [dict(item="Friedman", quantity="share of draws with p<0.05", value=float((fr_p[:, 1] < 0.05).mean())), dict(item="Friedman", quantity="median chi2", value=float(np.median(fr_p[:, 0]))), dict(item="Friedman", quantity="median p", value=float(np.median(fr_p[:, 1])))]
    for k, g in enumerate(names5):
        sep.append(dict(item=f"mean rank (1=best) {g}", quantity="median", value=float(np.median(ranks[:, k])), p2_5=float(np.percentile(ranks[:, k], 2.5)), p97_5=float(np.percentile(ranks[:, k], 97.5))))
    for pair, qs in pair_p.items():
        sep.append(dict(item=f"pair {pair[0]} vs {pair[1]}", quantity="share of draws with BH q<0.05", value=float((np.array(qs) < 0.05).mean()), median_q=float(np.median(qs))))
    # 用 Q 口径的中位数复算 TOST（ESMFlow 对 BioEmu，±0.05）与 BioEmu 对未用稳定性数据训练的生成器的 Wilcoxon
    med = {g: ens_q[FIVE[g]] for g in names5}; d = (med["ESMFlow"] - med["BioEmu"]).values; n = len(d); se = d.std(ddof=1) / np.sqrt(n); t1 = (d.mean() + 0.05) / se; t2 = (d.mean() - 0.05) / se
    p_tost = max(1 - stats.t.cdf(t1, n - 1), stats.t.cdf(t2, n - 1)); sep.append(dict(item="TOST ESMFlow vs BioEmu (±0.05, paired t, Q medians)", quantity="p", value=float(p_tost), mean_diff=float(d.mean())))
    for g in ("ESMFlow", "Boltz2", "Str2Str"):
        sep.append(dict(item=f"Wilcoxon BioEmu vs {g} (two-sided, Q medians)", quantity="p", value=float(stats.wilcoxon(med["BioEmu"].values, med[g].values).pvalue), median_diff=float(np.median(med["BioEmu"].values - med[g].values))))
    fm = stats.friedmanchisquare(*[med[g].values for g in names5]); sep.append(dict(item="Friedman on Q medians (5 generators)", quantity="chi2 / p", value=float(fm.pvalue), chi2=float(fm.statistic)))
    pd.DataFrame(sep).to_csv(os.path.join(REV, "tables", "S9_paradigm_separability" + SFX + ".csv"), index=False)
    # ---- 6. cyt c（ShortMD 无血红素）
    cy = []
    for lab, names in (("with cytochrome_c", PROTS), ("without cytochrome_c", [p for p in PROTS if p != "cytochrome_c"])):
        cy.append(dict(analysis=f"ShortMD vs structure, {lab}", **tests(ens_q["ShortMD"], base, names)))
        M = np.stack([med[g].reindex(names).values for g in names5], axis=1); fm = stats.friedmanchisquare(*[M[:, k] for k in range(5)])
        cy.append(dict(analysis=f"Friedman 5 generators on Q medians, {lab}", n=len(names), friedman_chi2=float(fm.statistic), friedman_p=float(fm.pvalue), mean_rank_ShortMD=float(stats.rankdata(-M, axis=1).mean(0)[names5.index("ShortMD")])))
        cy.append(dict(analysis=f"BioEmu vs ShortMD Wilcoxon two-sided, {lab}", n=len(names), wilcoxon_p=float(stats.wilcoxon(med["BioEmu"].reindex(names).values, med["ShortMD"].reindex(names).values).pvalue)))
    pd.DataFrame(cy).to_csv(os.path.join(REV, "tables", "S9_cytc_md_sensitivity" + SFX + ".csv"), index=False)
    # ---- cyt c 无血红素：逐抽样口径（方案步骤 7 指定的指标，同时报 Q 中位数）
    cyi = PROTS.index("cytochrome_c"); pc_rows = []; isg = names5.index("ShortMD")
    for lab, keep in (("with cytochrome_c", list(range(len(PROTS)))), ("without cytochrome_c", [i for i in range(len(PROTS)) if i != cyi])):
        fp, pos, sp_ = [], [], []
        for r in range(R):
            Mx = np.stack([mat[g][r][keep] for g in names5], axis=1); fp.append(stats.friedmanchisquare(*[Mx[:, k] for k in range(5)]).pvalue)
            mr = stats.rankdata(-Mx, axis=1).mean(0); pos.append(int(np.sum(mr < mr[isg])) + 1); sp_.append(revstats.wilcoxon_greater(mat["ShortMD"][r][keep], base.values[keep])[1])
        pos = np.array(pos); fp = np.array(fp); sp_ = np.array(sp_)
        pc_rows.append(dict(analysis=lab, n_proteins=len(keep), n_draws=R, share_friedman_p_lt_0p05=float((fp < 0.05).mean()), median_friedman_p=float(np.median(fp)), share_shortmd_position_4=float((pos == 4).mean()), share_shortmd_position_5=float((pos == 5).mean()),
                            share_shortmd_position_other=float(((pos != 4) & (pos != 5)).mean()), share_shortmd_vs_structure_p_lt_0p05=float((sp_ < 0.05).mean()), median_shortmd_vs_structure_p=float(np.median(sp_))))
    pd.DataFrame(pc_rows).to_csv(os.path.join(REV, "tables", "S9_cytc_per_draw" + SFX + ".csv"), index=False)
    outl = []   # cyt c 是否是 ShortMD 特有的离群点：逐蛋白看 ShortMD 与其余四个生成器的 ρ，以及有多少个生成器输给沉积结构
    for p in PROTS:
        v = {g: float(med[g].loc[p]) for g in names5}; others = [v[g] for g in names5 if g != "ShortMD"]
        outl.append(dict(protein=p, rho_ShortMD=v["ShortMD"], median_rho_other_four=float(np.median(others)), shortmd_minus_median_other_four=v["ShortMD"] - float(np.median(others)), shortmd_rank_among_five_1best=int(stats.rankdata([-v[g] for g in names5])[isg]),
                         shortmd_delta_vs_structure=v["ShortMD"] - float(base.loc[p]), n_generators_of_five_below_structure=int(sum(v[g] < float(base.loc[p]) for g in names5))))
    pd.DataFrame(outl).to_csv(os.path.join(REV, "tables", "S9_shortmd_outlier_check" + SFX + ".csv"), index=False)
    # ---- 重叠剔除后的五类比较（方案步骤 5）：Friedman 需要共同的蛋白集合，所以剔除五个生成器各自名单的并集
    ov_rows = []
    for lvl in ("none", "strong", "strict"):
        ex = set() if lvl == "none" else set(x for g in EX.index for x in str(EX.loc[g, f"exclude_{lvl}"]).split(";") if x and x != "nan"); keep = [i for i, p in enumerate(PROTS) if p not in ex]
        Mq = np.stack([med[g].reindex(PROTS).values[keep] for g in names5], axis=1); fm = stats.friedmanchisquare(*[Mq[:, k] for k in range(5)]); mrq = stats.rankdata(-Mq, axis=1).mean(0)
        fpd = np.array([stats.friedmanchisquare(*[np.stack([mat[g][r][keep] for g in names5], axis=1)[:, k] for k in range(5)]).pvalue for r in range(R)])
        ov_rows.append(dict(exclusion_level=lvl, excluded_union=";".join(sorted(ex)), n_proteins=len(keep), friedman_chi2_Q_medians=float(fm.statistic), friedman_p_Q_medians=float(fm.pvalue), share_draws_friedman_p_lt_0p05=float((fpd < 0.05).mean()), **{f"meanrank_Qmedians_{g}": float(mrq[k]) for k, g in enumerate(names5)}))
    pd.DataFrame(ov_rows).to_csv(os.path.join(REV, "tables", "S9_fiveclass_overlap_exclusion" + SFX + ".csv"), index=False)
    # cyt c 是否特殊？——逐个留出每个蛋白（11 个蛋白）的 Friedman p 和 ShortMD 对结构的检验；只去掉 cyt c 的结果要放在这个背景下读
    loo = []
    for out in PROTS:
        names = [p for p in PROTS if p != out]; M = np.stack([med[g].reindex(names).values for g in names5], axis=1); fm = stats.friedmanchisquare(*[M[:, k] for k in range(5)]); t = tests(ens_q["ShortMD"], base, names)
        loo.append(dict(left_out=out, friedman_chi2=float(fm.statistic), friedman_p=float(fm.pvalue), friedman_p_lt_0p05=bool(fm.pvalue < 0.05), shortmd_wins=t["wins"], shortmd_vs_structure_p=t["wilcoxon_greater_p"], has_heme_in_experiment=(out == "cytochrome_c")))
    L = pd.DataFrame(loo); L.to_csv(os.path.join(REV, "tables", "S9_friedman_leave_one_out" + SFX + ".csv"), index=False)
    print(pd.DataFrame(pc_rows).round(4).to_string(index=False)); print(pd.DataFrame(outl).round(3).to_string(index=False)); print(pd.DataFrame(ov_rows).round(4).to_string(index=False))
    print("leave-one-protein-out Friedman p<0.05 in", int(L.friedman_p_lt_0p05.sum()), "of 12 leave-outs; ShortMD-vs-structure p<0.05 in", int((L.shortmd_vs_structure_p < 0.05).sum()), "of 12")
    print(L.round(4).to_string(index=False))
    # ---- 全帧口径 F（S_all）：取自 S7 的 s2（M=50）与 sec_b（全部帧）
    FF = Q[(Q.form == "count_avg") & Q.scenario.isin(["s2", "sec_b"])]; ff = []
    for (scn, g), d in FF.groupby(["scenario", "generator"]):
        e = d.set_index("protein").rho_median.reindex(PROTS); b = d.set_index("protein").baseline_rho.reindex(PROTS); ff.append(dict(scenario=scn, generator=g, basis="F", **tests(e, b, PROTS), **{f"no_Im7_{k}": v for k, v in tests(e, b, [p for p in PROTS if p != "Im7"]).items() if k in ("wins", "wilcoxon_greater_p")}))
    pd.DataFrame(ff).to_csv(os.path.join(REV, "tables", "S9_fullframe_sensitivity.csv"), index=False)
    # ---- 8. 标签置换阴性对照（口径 P：BioEmu 全部旧列；B=1000）
    rngp = np.random.default_rng(0); Bp = 1000; ens_rank = {}; str_rank = {}; ys = {}
    for p in PROTS:
        c = E.load_cell("BioEmu", p, "O"); idx, y, resi = E.align_idx(c, p); e = np.nanmean(c["NC"][idx], axis=1) * E.BC + np.nanmean(c["NH"][idx], axis=1) * E.BH
        Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet")); sl = []
        for k in range(10):
            d = Rr[Rr.recon_k == k].set_index("resSeq"); sl.append([E.BC * float(d.NC.get(int(r) + OFF_S[p], np.nan)) + E.BH * float(d.NH.get(int(r) + OFF_S[p], np.nan)) for r in resi])
        sl = np.array(sl); ok = np.isfinite(sl).all(0); e = e[ok]; sl = sl[:, ok]; y = y[ok]; n = len(y)
        rc = lambda v: (lambda r: r - r.mean())(stats.rankdata(v)); ens_rank[p] = rc(e); str_rank[p] = np.array([rc(s) for s in sl]); ys[p] = y
    deltas = np.zeros((Bp, len(PROTS))); rho_e = np.zeros((Bp, len(PROTS))); rho_s = np.zeros((Bp, len(PROTS)))
    for j, p in enumerate(PROTS):
        y = ys[p]; n = len(y); perms = np.array([rngp.permutation(n) for _ in range(Bp)]); ry = stats.rankdata(y); ry = ry - ry.mean(); RY = ry[perms]                      # (Bp, n)
        den_y = np.sqrt((RY ** 2).sum(1)); re = ens_rank[p]; rho_e[:, j] = (RY @ re) / (den_y * np.sqrt((re ** 2).sum()))
        rho_s[:, j] = np.mean([(RY @ s) / (den_y * np.sqrt((s ** 2).sum())) for s in str_rank[p]], axis=0)
    deltas = rho_e - rho_s; med_delta = np.median(deltas, axis=1); wp = np.array([revstats.wilcoxon_greater(rho_e[b], rho_s[b])[1] for b in range(Bp)])
    # observed=0.074 / 0.0046 取自原单结构基线（口径 P）；零分布用 10 次重建均值基线，见工作文档 Stage 9 结果 6
    perm = pd.DataFrame([dict(statistic="median delta rho over 12 proteins (null)", null_median=float(np.median(med_delta)), null_mean=float(med_delta.mean()), null_p2_5=float(np.percentile(med_delta, 2.5)), null_p97_5=float(np.percentile(med_delta, 97.5)), observed=0.074, share_ge_observed=float((med_delta >= 0.074).mean())),
                         dict(statistic="one-sided paired Wilcoxon p (null)", null_median=float(np.median(wp)), null_mean=float(wp.mean()), null_p2_5=float(np.percentile(wp, 2.5)), null_p97_5=float(np.percentile(wp, 97.5)), observed=0.0046, share_ge_observed=float((wp <= 0.0046).mean())),
                         dict(statistic="mean rho_ens under shuffled labels (all proteins)", null_median=float(np.median(rho_e.mean(1))), null_mean=float(rho_e.mean()), null_p2_5=float(np.percentile(rho_e.mean(1), 2.5)), null_p97_5=float(np.percentile(rho_e.mean(1), 97.5)), observed=np.nan, share_ge_observed=np.nan)])
    perm["C_perm_pass_abs_median_lt_0p03"] = abs(float(np.median(med_delta))) < 0.03; perm.to_csv(os.path.join(REV, "tables", "S9_perm_negative_control.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    ov = pd.read_csv(os.path.join(REV, "tables", "S9_overlap_sensitivity.csv")); print(ov[ov.generator.isin(["BioEmu", "ESMFlow"])][["generator", "basis", "exclusion_level", "excluded", "n", "wins", "wilcoxon_greater_p", "sign_test_p", "median_delta"]].round(4).to_string(index=False))
    print(pd.DataFrame(sep).round(4).to_string(index=False)); print(pd.DataFrame(cy).round(4).to_string(index=False)); print(perm.round(4).to_string(index=False)); print(pd.DataFrame(ff).round(4).iloc[:, :9].to_string(index=False))
    log.finish([os.path.join(REV, "tables", f) for f in ("S9_overlap_sensitivity.csv", "S9_paradigm_separability.csv", "S9_cytc_md_sensitivity.csv", "S9_perm_negative_control.csv")], note="overlap/sep/cytc/perm")


if __name__ == "__main__":
    main()
