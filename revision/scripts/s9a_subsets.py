"""Stage 9a · 残基层主结论的稳健性（revision_proposal.md S9 步骤 1–4；R1-C2、R2-C5、R2-m2、R3-C15、R3-C16/17）。
口径 P：原稿忠实复现——results/tables/null_baselines.csv 的 BioEmu 系综 ρ（obs_spearman，计数平均形式，旧 npz 全部列）与原单结构基线（null_baselines.csv 的 single_struct_ceiling 列，即原项目的单结构基线），12 个蛋白。
口径 Q：S7 主分析（口径 O，M_eff=min(50,n)，200 次抽样的中位数）。各生成器各做一遍；子集分析在 BioEmu 上细做（审稿人质疑的是 BioEmu 的增益），其余生成器报告 S_all 与 S_noIm7。
子集：S_all；S_noIm7；S_noWeak4（去 Barstar、CI2、CheY、UCH-L1）；S_noCI2barstar；S_noIm7_noWeak4；S_noSNase（只用于 BH 重计数）。
每个子集：胜出数/n、单侧配对 Wilcoxon 精确 p、精确二项符号检验 p（单侧）、Δρ 中位数与均值、HL 估计与 95% CI、蛋白 bootstrap 95% CI（B=10⁴, seed 7）；BH 重计数（口径 P：用原稿 null_baselines 的逐蛋白置换 p）。
分组：G1（实验 ΔG_open 的 IQR 与 SD，以 12 个蛋白的中位数为界，高/低离散）、G2（单结构置换显著性：10 次重建各算 B=5000 的置换 p，取中位数，BH q<0.05）、G3（原稿：系综置换 q<0.05，仅作对照）；判据表写出每个蛋白的数值。
影响力：留一蛋白 jackknife；n_res 与 n_res−3 加权增益；Fisher-z 随机效应荟萃（DerSimonian–Laird；方差 = 残基层配对 bootstrap 的 Δz 方差，B=10⁴，两种预测在同一次重抽中针对同一组残基）与 Zou (2007) 备选。
残基层表导出（R3-C15）：S9_residue_table.parquet 与 S9_residue_table_README.md（写明 ΔSpearman 的定义与单位）。
输出：tables/S9_subset_tests.csv、S9_bh_recount.csv、S9_grouping_criteria.csv、S9_grouped_tests.csv、S9_jackknife.csv、S9_weighted_gain.csv、S9_meta_analysis.csv、S9_residue_table.parquet。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, hashlib
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, revstats, ensemble as E, stats_extra as X  # noqa: E402

PROJ, REV, INP = revio.load_paths()
PROTS = sorted(json.load(open(INP["tier1a_sequences"]))); B = 10000
SUBSETS = {"S_all": set(PROTS), "S_noIm7": set(PROTS) - {"Im7"}, "S_noWeak4": set(PROTS) - {"Barstar", "CI2", "CheY", "UCH-L1"}, "S_noCI2barstar": set(PROTS) - {"CI2", "Barstar"},
           "S_noIm7_noWeak4": set(PROTS) - {"Im7", "Barstar", "CI2", "CheY", "UCH-L1"}, "S_noSNase": set(PROTS) - {"SNase"}}
NB = revio.correct_null_baselines(pd.read_csv(os.path.join(INP["results_tables"], "null_baselines.csv")), INP).set_index("protein"); ceil = pd.read_parquet(INP["operator_ceiling"]); OFF_S = dict(zip(ceil.protein, ceil.offset.astype(int)))


def rng_for(*tags):
    """每个格子独立的随机流：种子 = [7, sha256(tags)]，不依赖运行顺序（共用一条 rng 时，新增格子会移动其后所有 bootstrap 区间）。"""
    return np.random.default_rng([7, int(hashlib.sha256('|'.join(map(str, tags)).encode()).hexdigest()[:8], 16)])


def subset_stats(label, basis, gen, ens, base, names):
    d = (ens - base).loc[names].values; n = len(d); pos, nn, ps = X.sign_test_greater(d)
    w = revstats.wilcoxon_greater(ens.loc[names].values, base.loc[names].values)[1] if np.any(d != 0) else np.nan
    est, lo, hi, _ = X.hodges_lehmann(d); rng = rng_for('subset', label, basis, gen); idx = rng.integers(0, n, size=(B, n)); bm = d[idx].mean(1); bmed = np.median(d[idx], axis=1)
    return dict(subset=label, basis=basis, generator=gen, n=n, wins=int((d > 0).sum()), wilcoxon_greater_p=w, sign_test_p=ps, median_delta=float(np.median(d)), mean_delta=float(d.mean()), hl_estimate=est, hl_ci_lo=lo, hl_ci_hi=hi,
                boot_mean_lo=float(np.percentile(bm, 2.5)), boot_mean_hi=float(np.percentile(bm, 97.5)), boot_median_lo=float(np.percentile(bmed, 2.5)), boot_median_hi=float(np.percentile(bmed, 97.5)))


def load_basis():
    """返回 {(basis, gen): (ens Series, base Series)}。P 只有 BioEmu。"""
    out = {("P", "BioEmu"): (NB.obs_spearman.reindex(PROTS), NB.single_struct_ceiling.reindex(PROTS))}
    Q0 = pd.read_csv(os.path.join(REV, "tables", "S7_invpf_vs_countavg_per_protein.csv")); Q0 = Q0[Q0.form == "count_avg"]
    for scn, lab in (("primary", "Q"), ("s1", "Q_s1")):   # Q_s1 = S7 的共同样本量 M=36 敏感性（方案步骤 5 要求的口径之一）
        for g, d in Q0[Q0.scenario == scn].groupby("generator"):
            d = d.set_index("protein"); out[(lab, g)] = (d.rho_median.reindex(PROTS), d.baseline_rho.reindex(PROTS))
    return out


def main():
    log = revio.RunLogger(stage=9, task="subsets", cpu_procs=1, seeds="bootstrap seed 7; permutation seed 0")
    BAS = load_basis(); rows = []
    for (basis, gen), (ens, base) in BAS.items():
        for lab, names in SUBSETS.items():
            if gen != "BioEmu" and lab not in ("S_all", "S_noIm7"):
                continue
            nm = [p for p in PROTS if p in names and np.isfinite(ens.get(p, np.nan))]
            if len(nm) >= 5:
                rows.append(subset_stats(lab, basis, gen, ens, base, nm))
    S = pd.DataFrame(rows); S.to_csv(os.path.join(REV, "tables", "S9_subset_tests.csv"), index=False)
    # ---- BH 重计数（口径 P：原稿逐蛋白置换 p）
    bh_rows = []
    for lab, names in SUBSETS.items():
        nm = [p for p in PROTS if p in names]; pp = NB.perm_p.reindex(nm).values; q = X.bh(pp); bh_rows.append(dict(subset=lab, n=len(nm), n_q_lt_0p05=int((q < 0.05).sum()), proteins_q_lt_0p05=";".join(p for p, v in zip(nm, q) if v < 0.05)))
    BHdf = pd.DataFrame(bh_rows); BHdf.to_csv(os.path.join(REV, "tables", "S9_bh_recount.csv"), index=False)
    orig_sig = set(NB.index[NB.perm_BH < 0.05]); weak_equal = orig_sig == SUBSETS["S_noWeak4"]
    # ---- 分组判据
    e_all = {p: E.exp_residues(p).dGopen_kcal.values.astype(float) for p in PROTS}
    crit = pd.DataFrame(dict(protein=PROTS, n_res=[len(e_all[p]) for p in PROTS], exp_dG_sd=[np.std(e_all[p], ddof=1) for p in PROTS], exp_dG_iqr=[np.subtract(*np.percentile(e_all[p], [75, 25])) for p in PROTS]))
    crit["G1_high_dispersion"] = (crit.exp_dG_iqr > crit.exp_dG_iqr.median()) & (crit.exp_dG_sd > crit.exp_dG_sd.median())
    # G2：单结构置换显著性
    rngp = np.random.default_rng(0); g2p = {}
    for p in PROTS:
        Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet")); e = E.exp_residues(p); y = dict(zip(e.resi.astype(int), e.dGopen_kcal.astype(float))); ps = []
        perms = None
        for k in range(10):
            d = Rr[Rr.recon_k == k].set_index("resSeq"); keep = [int(r) for r in e.resi.astype(int) if (int(r) + OFF_S[p]) in d.index]; yy = np.array([y[r] for r in keep]); x = (E.BC * d.NC.astype(float) + E.BH * d.NH.astype(float)).loc[[r + OFF_S[p] for r in keep]].values
            if perms is None:
                perms = np.array([rngp.permutation(len(yy)) for _ in range(5000)])
            rx = stats.rankdata(x); rx = rx - rx.mean(); ry = stats.rankdata(yy); ry = ry - ry.mean(); obs = float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
            nul = (ry[perms] @ rx) / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()); ps.append((np.sum(nul >= obs - 1e-12) + 1) / (len(nul) + 1))   # 全部 5000 次置换；标签打乱，结构预测不变
        g2p[p] = float(np.median(ps))
    crit["G2_struct_perm_p_median_over_recon"] = [g2p[p] for p in PROTS]; crit["G2_struct_perm_q"] = X.bh(crit.G2_struct_perm_p_median_over_recon.values); crit["G2_struct_significant"] = crit.G2_struct_perm_q < 0.05
    crit["G3_ens_perm_q_manuscript"] = NB.perm_BH.reindex(PROTS).values; crit["G3_ens_significant"] = crit.G3_ens_perm_q_manuscript < 0.05
    crit["original_weak4"] = crit.protein.isin(["Barstar", "CI2", "CheY", "UCH-L1"]); crit.to_csv(os.path.join(REV, "tables", "S9_grouping_criteria.csv"), index=False)
    gt = []
    for basis, gen in (("P", "BioEmu"), ("Q", "BioEmu")):
        ens, base = BAS[(basis, gen)]
        for gname, col in (("G1_high_dispersion", "G1_high_dispersion"), ("G2_struct_significant", "G2_struct_significant"), ("G3_ens_significant", "G3_ens_significant")):
            for val in (True, False):
                nm = list(crit.protein[crit[col] == val])
                if len(nm) >= 3:
                    r = subset_stats(f"{gname}={val}", basis, gen, ens, base, nm); r["members"] = ";".join(nm); gt.append(r)
    pd.DataFrame(gt).to_csv(os.path.join(REV, "tables", "S9_grouped_tests.csv"), index=False)
    gd = []   # 组间差异：分组结果只是描述，要说“增益在某组更大”必须检验两组的 Δρ 是否不同
    for basis, gen in (("P", "BioEmu"), ("Q", "BioEmu")):
        ens, base = BAS[(basis, gen)]; dl = (ens - base)
        for col in ("G1_high_dispersion", "G2_struct_significant", "G3_ens_significant"):
            a = dl[list(crit.protein[crit[col]])].values; b = dl[list(crit.protein[~crit[col]])].values; mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            gd.append(dict(basis=basis, grouping=col, n_true=len(a), n_false=len(b), median_delta_true=float(np.median(a)), median_delta_false=float(np.median(b)), mannwhitney_two_sided_p=float(mw.pvalue), note=("grouping defined from the ensemble results (descriptive only)" if col == "G3_ens_significant" else "")))
    pd.DataFrame(gd).to_csv(os.path.join(REV, "tables", "S9_group_difference_tests.csv"), index=False)
    # ---- jackknife、加权增益
    jk = []; wg = []
    for basis in ("P", "Q"):
        ens, base = BAS[(basis, "BioEmu")]
        for out in PROTS:
            nm = [p for p in PROTS if p != out]; d = (ens - base).loc[nm].values; w = revstats.wilcoxon_greater(ens.loc[nm].values, base.loc[nm].values)[1]
            jk.append(dict(basis=basis, left_out=out, n=11, wins=int((d > 0).sum()), wilcoxon_greater_p=w, median_delta=float(np.median(d)), mean_delta=float(d.mean())))
        d = (ens - base).reindex(PROTS).values; nres = np.array([len(e_all[p]) for p in PROTS], float)
        for wname, w in (("unweighted", np.ones(len(PROTS))), ("n_res", nres), ("n_res-3 (Fisher)", nres - 3)):
            o = np.argsort(d); cw = np.cumsum(w[o]) / w.sum(); wmed = float(d[o][np.searchsorted(cw, 0.5)])
            wg.append(dict(basis=basis, weighting=wname, weighted_mean_delta=float((w * d).sum() / w.sum()), weighted_median_delta=wmed, weight_share_Im7=float(w[PROTS.index("Im7")] / w.sum()), weight_share_SNase=float(w[PROTS.index("SNase")] / w.sum())))
    pd.DataFrame(jk).to_csv(os.path.join(REV, "tables", "S9_jackknife.csv"), index=False); pd.DataFrame(wg).to_csv(os.path.join(REV, "tables", "S9_weighted_gain.csv"), index=False)
    # ---- 荟萃分析与残基层表（口径 P：BioEmu 全部旧列；结构：10 次重建）
    meta_rows = []; res_rows = []; bs_se = {}
    for p in PROTS:
        c = E.load_cell("BioEmu", p, "O"); idx, y, resi = E.align_idx(c, p); ens = np.nanmean(c["NC"][idx], axis=1) * E.BC + np.nanmean(c["NH"][idx], axis=1) * E.BH
        Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet")); structs = []
        for k in range(10):
            d = Rr[Rr.recon_k == k].set_index("resSeq"); structs.append([E.BC * float(d.NC.get(int(r) + OFF_S[p], np.nan)) + E.BH * float(d.NH.get(int(r) + OFF_S[p], np.nan)) for r in resi])
        structs = np.array(structs); ok = np.isfinite(structs).all(0); resi, y, ens, structs = resi[ok], y[ok], ens[ok], structs[:, ok]; st_mean = structs.mean(0); n = len(y)
        r_e = stats.spearmanr(ens, y)[0]; r_s_each = np.array([stats.spearmanr(s, y)[0] for s in structs]); r_s = float(r_s_each.mean())
        # 配对 bootstrap：同一组残基重抽；结构取 10 次重建 ρ 的均值（对每次重抽，对每个重建求 ρ 再平均）
        rng = rng_for('meta', p); idxb = rng.integers(0, n, size=(B, n))
        def spear_boot(xv):
            xb = xv[idxb]; yb = y[idxb]; rx = stats.rankdata(xb, axis=1); ry = stats.rankdata(yb, axis=1); rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True); return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
        rb_e = spear_boot(ens); rb_s = np.mean([spear_boot(s) for s in structs], axis=0); dz = np.arctanh(np.clip(rb_e, -0.999999, 0.999999)) - np.arctanh(np.clip(rb_s, -0.999999, 0.999999)); dz = dz[np.isfinite(dz)]
        dz0 = np.arctanh(r_e) - np.arctanh(r_s); r12 = float(np.corrcoef(stats.rankdata(ens), stats.rankdata(st_mean))[0, 1]); zlo, zhi = X.zou_diff_ci(r_e, r_s, r12, n); zblo, zbhi = X.zou_diff_ci(r_e, r_s, r12, n, bw=True)
        meta_rows.append(dict(protein=p, n_res=n, rho_ens=r_e, rho_struct_mean10recon=r_s, delta_rho=r_e - r_s, delta_z=dz0, delta_z_boot_se=float(dz.std(ddof=1)), delta_z_ci_lo=float(np.percentile(dz, 2.5)), delta_z_ci_hi=float(np.percentile(dz, 97.5)),
                              ci_excludes_0=bool(np.percentile(dz, 2.5) > 0 or np.percentile(dz, 97.5) < 0), zou_delta_lo=zlo, zou_delta_hi=zhi, zou_excludes_0=bool(zlo > 0 or zhi < 0), zou_bw_lo=zblo, zou_bw_hi=zbhi, zou_bw_excludes_0=bool(zblo > 0 or zbhi < 0), r12_ens_struct=r12))
        rk = lambda v: stats.rankdata(v)
        T = float(E.exp_residues(p).T_C.iloc[0]) + 273.15
        res_rows.append(pd.DataFrame(dict(protein=p, resSeq=resi, dG_open_exp=y, pred_ens=ens * bv.R_GAS * T, pred_str=st_mean * bv.R_GAS * T, **{f"pred_str_recon{k}": structs[k] * bv.R_GAS * T for k in range(10)}, rank_exp=rk(y), rank_ens=rk(ens), rank_str=rk(st_mean), rank_str_recon0=rk(structs[0]))))
    M = pd.DataFrame(meta_rows); dl = X.dersimonian_laird(M.delta_z.values, M.delta_z_boot_se.values ** 2); zbar = float(np.arctanh(M.rho_struct_mean10recon).mean())
    summ = dict(protein="POOLED (DerSimonian–Laird)", delta_z=dl["mu"], delta_z_ci_lo=dl["lo"], delta_z_ci_hi=dl["hi"], tau2=dl["tau2"], I2=dl["I2"], Q=dl["Q"], delta_rho_backtransformed=float(np.tanh(zbar + dl["mu"]) - np.tanh(zbar)), delta_rho_lo=float(np.tanh(zbar + dl["lo"]) - np.tanh(zbar)), delta_rho_hi=float(np.tanh(zbar + dl["hi"]) - np.tanh(zbar)),
                n_proteins_ci_excludes_0=int(M.ci_excludes_0.sum()), n_proteins_zou_excludes_0=int(M.zou_excludes_0.sum()), n_proteins_zou_bw_excludes_0=int(M.zou_bw_excludes_0.sum()), note="back-transform: tanh(mean z_struct + dz) - tanh(mean z_struct)")
    pd.concat([M, pd.DataFrame([summ])], ignore_index=True).to_csv(os.path.join(REV, "tables", "S9_meta_analysis.csv"), index=False)
    RT = pd.concat(res_rows, ignore_index=True); RT.to_parquet(os.path.join(REV, "tables", "S9_residue_table.parquet"), index=False)
    open(os.path.join(REV, "tables", "S9_residue_table_README.md"), "w").write(
        "# S9_residue_table（R3-C15）\n\n每行一个有实验 ΔG_open 的残基（12 个蛋白，共 522 行）。列：protein；resSeq（实验编号；生成器/沉积结构的编号 = resSeq + offset，offset：CI2 +1、RNaseH_Tt −5，其余 0，来源 operator_ceiling.offset）；dG_open_exp（kcal/mol）；"
        "pred_ens（BioEmu 系综，旧 npz 全部列，计数平均形式，ΔG_pred = R·T_exp·ln PF，kcal/mol）；pred_str（沉积结构 10 次重建的 ln PF **先平均**再换算的 ΔG_pred）；"
        "pred_str_recon0 … pred_str_recon9（10 次重建各自的 ΔG_pred）；rank_exp、rank_ens、rank_str、rank_str_recon0（蛋白内秩，平均秩处理并列）。\n\n"
        "**ΔSpearman 的定义与单位**：Δρ = ρ(系综, 实验) − ρ(沉积结构, 实验)，其中 ρ 是**蛋白内**、在该表的残基上算的 Spearman 秩相关系数（逐蛋白一个数）；无量纲；取值范围 [−2, 2]。\n\n"
        "**沉积结构的 ρ 有三种取法，必须说明用的是哪一种**：(1) 原稿的单结构基线（原项目 `results/operator_ceiling.parquet` 的 spearman 列，一次运行的标量，当年没有留存逐残基预测，**本表里没有对应的预测列，不能由本表复算**；Im7 为 0.174）；"
        "(2) 本阶段 10 次重建各自的 ρ 取均值（用 `pred_str_recon0…9` 各算一个 ρ 再平均；Im7 为 0.327）；(3) 10 次重建先平均 ln PF 再算 ρ（用 `pred_str` 或 `rank_str`；Im7 为 0.359）。"
        "`S9_meta_analysis.csv` 与荟萃分析用 (2)：Im7 的 Δρ = 0.542 − 0.327 = +0.214；原稿与补充表里的 +0.368 = 0.542 − 0.174 用的是 (1)。可由本表复算的是 (2) 和 (3)。"
        "蛋白层数值由残基层比较得出，没有引入新的统计方法。\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S[S.generator == "BioEmu"][["subset", "basis", "n", "wins", "wilcoxon_greater_p", "sign_test_p", "median_delta", "mean_delta", "hl_estimate", "hl_ci_lo", "hl_ci_hi"]].round(4).to_string(index=False)); print(BHdf.to_string(index=False))
    print("original BH-significant set == S_noWeak4:", weak_equal, sorted(orig_sig)); print(M[["protein", "n_res", "rho_ens", "rho_struct_mean10recon", "delta_rho", "ci_excludes_0", "zou_excludes_0"]].round(3).to_string(index=False)); print(pd.DataFrame([summ]).round(4).T.to_string())
    log.finish([os.path.join(REV, "tables", f) for f in ("S9_subset_tests.csv", "S9_meta_analysis.csv", "S9_residue_table.parquet")], note="subsets")


if __name__ == "__main__":
    main()
