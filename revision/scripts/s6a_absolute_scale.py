"""Stage 6a · 绝对标度：逐蛋白指标与池化相关的分解（revision_proposal.md S6 步骤 1–2）。
数据：EX2 子集 5 个蛋白（Barnase、RNase A、SNase、CI2、cyt c；41+25+101+14+48 = 229 个残基；原稿的集合，不含 RNase H_Tt）。
预测：沉积结构 10 次重建（S4a 的 Crystal parquet，recon_k 0–9），ln PF_pred = 0.35·N_C + 2.0·N_H（固定系数，单结构，无系综）。
实验：ln PF_exp = dGopen_kcal / (R·T_exp)（S1 终表；对 10 个 Nguyen 蛋白与 tier1a_dGopen 的 log10PF·ln10 精确相等，cyt c 来自 Milne 表，脚本里逐项核对）。
编号偏移：口径 O 沿用原稿（cyt c 为 0）；offset_mode=identity 时 cyt c 用身份核对后的偏移 +1（S1_alignment_map.json），作敏感性分析。
温度：ΔG 空间两种方案——dG_pred_298 = R·298.15·lnPF_pred 与 dG_pred_Texp = R·T_exp·lnPF_pred；实验 ΔG 始终用各自的 T_exp。
不确定度：残基 bootstrap B=10⁴、seed 7；每次重抽对 10 次重建各算一遍统计量再取均值（得到"10 次重建均值"的 CI）。
输出：tables/S6_per_protein_absolute.csv、tables/S6_pooling_decomposition.csv、tables/S6_a_reproduction_checks.csv。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402
from lib.abs_metrics import metrics  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=6, task="absolute_scale_a", cpu_procs=1, seeds="bootstrap seed 7")
EX2 = ["Barnase", "RNaseA", "SNase", "CI2", "cytochrome_c"]; B = 10000; BC, BH = bv.BETA_C, bv.BETA_H; R_GAS = bv.R_GAS; LN10 = np.log(10)
S1 = pd.read_parquet(os.path.join(REV, "tables", "S1_tier1a_residues_final.parquet"))
ceil = pd.read_parquet(INP["operator_ceiling"]); OFF_MS = dict(zip(ceil.protein, ceil.offset.astype(int)))
OFF_ID = dict(OFF_MS); OFF_ID["cytochrome_c"] = 1   # 身份核对后的偏移：对沉积结构 1HRC 的残基名，offset 0 时 3/48 一致，+1 时 48/48（S1_deposited_structure_numbering.csv）；仅用于身份偏移敏感性


def load_exp(p):
    g = S1[S1.protein_id == p].dropna(subset=["dGopen_kcal"]); T = float(g.T_C.iloc[0]) + 273.15
    return pd.DataFrame(dict(resi=g.resi.astype(int).values, dG_exp=g.dGopen_kcal.astype(float).values, lnPF_exp=g.dGopen_kcal.astype(float).values / (R_GAS * T))), T


def load_pred(p):
    Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet"))
    return {k: g.set_index("resSeq") for k, g in Rr.groupby("recon_k")}


def build(mode):
    """返回 {蛋白: dict(exp=DataFrame, T, pred=array(10,n) lnPF_pred, resi)}（按残基号对齐到实验表）。"""
    out = {}; off = OFF_ID if mode == "identity" else OFF_MS
    for p in EX2:
        e, T = load_exp(p); P = load_pred(p); rows = []
        for k in range(10):
            d = P[k]; lnpf = BC * d.NC.astype(float) + BH * d.NH.astype(float)
            rows.append([lnpf.get(int(r) + off[p], np.nan) for r in e.resi])
        A = np.array(rows, float); keep = ~np.isnan(A).any(0)
        out[p] = dict(exp=e[keep].reset_index(drop=True), T=T, pred=A[:, keep], n_dropped=int((~keep).sum()))
    return out


def spearman_boot(pred, obs, idx):
    """pred: (10, n) 10 次重建；obs: (n,)；idx: (B, n) 残基重抽。返回 (B,)：每次重抽中 10 次重建 Spearman 的均值（尺度不变，与温度方案无关）。"""
    ro = stats.rankdata(obs[idx], axis=-1); rp = stats.rankdata(pred[:, idx], axis=-1)
    ro = ro - ro.mean(-1, keepdims=True); rp = rp - rp.mean(-1, keepdims=True)
    return ((rp * ro).sum(-1) / np.sqrt((rp ** 2).sum(-1) * (ro ** 2).sum(-1))).mean(0)


def main():
    rng = np.random.default_rng(7); per_rows, pool_rows, checks, rec_rows = [], [], [], []
    for mode in ("manuscript", "identity"):
        D = build(mode)
        # ---------------- 步骤 1：逐蛋白指标（ΔG 空间两种温度 + ln PF 空间），10 次重建均值 ± SD 与 bootstrap CI
        for p in EX2:
            e = D[p]["exp"]; n = len(e); idx = rng.integers(0, n, size=(B, n)); sp_lo, sp_hi = np.percentile(spearman_boot(D[p]["pred"], e.lnPF_exp.values, idx), [2.5, 97.5])   # Spearman 对任何正的尺度变换不变，三个空间共用
            for space, scale_pred, obs in (("lnPF", 1.0, e.lnPF_exp.values), ("dG_298", R_GAS * 298.15, e.dG_exp.values), ("dG_Texp", R_GAS * D[p]["T"], e.dG_exp.values)):
                pred = D[p]["pred"] * scale_pred; per_k = [metrics(pred[k], obs) for k in range(10)]
                rho = [stats.spearmanr(pred[k], obs)[0] for k in range(10)]
                bs = [metrics(pred[:, idx], np.broadcast_to(obs[idx], (10, B, n)))]   # (10,B)
                bsm = {m: bs[0][m].mean(0) for m in bs[0]}   # 每次重抽取 10 次重建均值 → (B,)
                for m in per_k[0]:
                    v = np.array([x[m] for x in per_k]); lo, hi = np.percentile(bsm[m], [2.5, 97.5])
                    per_rows.append(dict(offset_mode=mode, protein=p, space=space, metric=m, mean_over_10_recon=v.mean(), sd_over_10_recon=v.std(ddof=1), boot_ci_lo=lo, boot_ci_hi=hi, n_residues=n, n_dropped_unmatched=D[p]["n_dropped"]))
                per_rows.append(dict(offset_mode=mode, protein=p, space=space, metric="spearman", mean_over_10_recon=np.mean(rho), sd_over_10_recon=np.std(rho, ddof=1), boot_ci_lo=sp_lo, boot_ci_hi=sp_hi, n_residues=n, n_dropped_unmatched=D[p]["n_dropped"]))
        # ---------------- 步骤 2：池化分解（ln PF 空间，与原稿一致）
        pr = {p: (D[p]["pred"], D[p]["exp"].lnPF_exp.values) for p in EX2}
        pooled, within, pk, lopo, mean_r, zmean, drop = [], [], [], [], [], [], []
        for k in range(10):
            X = np.concatenate([pr[p][0][k] for p in EX2]); Y = np.concatenate([pr[p][1] for p in EX2]); g = np.concatenate([[i] * len(pr[p][1]) for i, p in enumerate(EX2)])
            pooled.append(np.corrcoef(X, Y)[0, 1])
            Xc = X - np.array([X[g == i].mean() for i in range(5)])[g]; Yc = Y - np.array([Y[g == i].mean() for i in range(5)])[g]; within.append(np.corrcoef(Xc, Yc)[0, 1])
            cov_tot = np.cov(X, Y, ddof=0)[0, 1]; cov_in = (Xc * Yc).mean(); cov_bt = cov_tot - cov_in; pk.append((cov_tot, cov_in, cov_bt, cov_bt / cov_tot))
            rs = np.array([np.corrcoef(pr[p][0][k], pr[p][1])[0, 1] for p in EX2]); ns = np.array([len(pr[p][1]) for p in EX2]); mean_r.append(rs.mean()); zmean.append(np.tanh(np.sum((ns - 3) * np.arctanh(rs)) / np.sum(ns - 3)))
            ls = []
            for p in EX2:
                tr = np.concatenate([np.column_stack([pr[q][0][k], pr[q][1]]) for q in EX2 if q != p]); b = np.polyfit(tr[:, 0], tr[:, 1], 1); ls.append(np.corrcoef(np.polyval(b, pr[p][0][k]), pr[p][1])[0, 1])
            lopo.append(np.mean(ls))
            keep = np.isin(g, [i for i, p in enumerate(EX2) if p not in ("Barnase", "SNase")])
            drop.append((np.corrcoef(X[keep], Y[keep])[0, 1], np.corrcoef((X - np.array([X[g == i].mean() for i in range(5)])[g])[keep], (Y - np.array([Y[g == i].mean() for i in range(5)])[g])[keep])[0, 1]))
        # 蛋白间成分（n=5，只作描述）：5 个蛋白的均值预测对均值实验
        bt = [np.corrcoef([pr[p][0][k].mean() for p in EX2], [pr[p][1].mean() for p in EX2])[0, 1] for k in range(10)]
        def add(name, v, note=""):
            v = np.asarray(v, float); rec_rows.extend([dict(offset_mode=mode, quantity=name, recon_k=k, value=float(x)) for k, x in enumerate(v)]); pool_rows.append(dict(offset_mode=mode, quantity=name, mean_over_10_recon=float(v.mean()), sd_over_10_recon=float(v.std(ddof=1)), note=note))
        add("(a) pooled Pearson r", pooled, "原稿 0.581±0.007"); add("(b) within-protein-centered pooled r", within); add("(c) between-protein r of protein means (n=5, descriptive)", bt)
        pk = np.array(pk); add("(d) total pooled covariance", pk[:, 0]); add("(d) within-protein covariance", pk[:, 1]); add("(d) between-protein covariance", pk[:, 2]); add("(d) between-protein share of total covariance", pk[:, 3])
        add("(e) simple mean of per-protein r", mean_r, "审稿人给出 0.330"); add("(e) Fisher-z weighted (n-3) mean of per-protein r", zmean); add("(f) LOPO coefficient transfer, mean held-out r", lopo, "原稿 0.327±0.013")
        drop = np.array(drop); add("(g) pooled r without Barnase, SNase (3 proteins)", drop[:, 0]); add("(g) within-protein-centered r without Barnase, SNase", drop[:, 1])
        if mode == "manuscript":
            for nm, vals, want, tol in (("pooled r", pooled, 0.581, 0.005), ("pooled r SD", None, 0.007, 0.003), ("simple mean of per-protein r (reviewer 0.330)", mean_r, 0.330, 0.005), ("LOPO mean r (manuscript 0.327)", lopo, 0.327, 0.005)):
                if vals is None:
                    got = float(np.std(pooled)); checks.append(dict(check=nm, got=got, expected=want, tol=tol, abs_diff=abs(got - want), passed_plan_tolerance=bool(abs(got - want) <= tol), se_of_10recon_mean=np.nan, z_vs_expected=np.nan, expected_within_10recon_range=np.nan)); continue
                v = np.asarray(vals, float); got = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(len(v)))
                sd_ms = {"pooled r": 0.007, "LOPO mean r (manuscript 0.327)": 0.013}.get(nm)   # 原稿的 0.581±0.007 与 0.327±0.013 本身就是 10 次重复重建的均值±SD（16_stage1_reproducibility.py，np.std ddof=0；氢放置用 Python 全局 random，脚本里的 np.random.seed 管不到，所以是独立随机重复）
                z2 = (got - want) / np.sqrt(se ** 2 + (sd_ms / np.sqrt(10)) ** 2) if sd_ms else np.nan   # 两个均值之差的 z
                checks.append(dict(check=nm, got=got, expected=want, tol=tol, abs_diff=abs(got - want), passed_plan_tolerance=bool(abs(got - want) <= tol), se_of_10recon_mean=se, z_vs_expected=(got - want) / se, z_two_means=z2, expected_within_10recon_range=bool(v.min() <= want <= v.max())))
    # 逐蛋白逐次重建的 Pearson（ln PF 空间，口径 O），并与审稿人引用的补充表 1 单次值（0.717、0.624、0.500、0.294、−0.483；顺序按其数值与本表蛋白对应）比较
    D0 = build("manuscript"); per_recon = [dict(protein=p, recon_k=k, pearson=float(np.corrcoef(D0[p]["pred"][k], D0[p]["exp"].lnPF_exp.values)[0, 1])) for p in EX2 for k in range(10)]
    PR = pd.DataFrame(per_recon); PR.to_csv(os.path.join(REV, "tables", "S6_per_protein_per_recon_pearson.csv"), index=False)
    # 补充表 1 的单次值直接取自原项目的源文件（results/stats/stage1_calibration_crystal.csv，列 pearson_defaultβ），蛋白对应由文件给出，不再按数值最接近来指派；并核对与审稿人引用的五个数一致
    src1 = pd.read_csv(os.path.join(INP["results_stats"], "stage1_calibration_crystal.csv")); st1 = dict(zip(src1.protein, src1["pearson_defaultβ"].astype(float)))
    assert sorted(round(v, 3) for v in st1.values()) == sorted([0.717, 0.624, 0.5, 0.294, -0.483]), st1
    cmp_ = []
    for p, v in st1.items():
        x = PR[PR.protein == p].pearson.values; cmp_.append(dict(protein=p, st1_value_cited_by_reviewer=v, min_10recon=x.min(), mean_10recon=x.mean(), max_10recon=x.max(), within_range=bool(x.min() <= v <= x.max()), z=(v - x.mean()) / x.std(ddof=1)))
    pd.DataFrame(cmp_).to_csv(os.path.join(REV, "tables", "S6_st1_vs_10recon.csv"), index=False); print(pd.DataFrame(cmp_).round(3).to_string(index=False))
    print("mean of the 5 ST1 values = %.4f" % np.mean(list(st1.values())))
    P = pd.DataFrame(per_rows); Q = pd.DataFrame(pool_rows); C = pd.DataFrame(checks)
    # 恒等检验（方案步骤 1）：斜率的 95% CI 是否含 1、截距的 95% CI 是否含 0；两个回归方向都报（回归方向的选择见工作文档：复现审稿人 0.22 的是 slope_exp_on_pred）。ΔG 空间，实验温度。
    idt = []
    for (mode, p), d in P[P.space == "dG_Texp"].groupby(["offset_mode", "protein"]):
        d = d.set_index("metric")
        for direction in ("pred_on_exp", "exp_on_pred"):
            sl = d.loc[f"slope_{direction}"]; ic = d.loc[f"intercept_{direction}"]
            idt.append(dict(offset_mode=mode, protein=p, direction=direction, slope=sl.mean_over_10_recon, slope_ci_lo=sl.boot_ci_lo, slope_ci_hi=sl.boot_ci_hi, slope_ci_contains_1=bool(sl.boot_ci_lo <= 1 <= sl.boot_ci_hi),
                            intercept_kcal_mol=ic.mean_over_10_recon, intercept_ci_lo=ic.boot_ci_lo, intercept_ci_hi=ic.boot_ci_hi, intercept_ci_contains_0=bool(ic.boot_ci_lo <= 0 <= ic.boot_ci_hi), n_residues=int(sl.n_residues)))
    pd.DataFrame(idt).to_csv(os.path.join(REV, "tables", "S6_identity_tests.csv"), index=False)
    f1 = os.path.join(REV, "tables", "S6_per_protein_absolute.csv"); f2 = os.path.join(REV, "tables", "S6_pooling_decomposition.csv"); f3 = os.path.join(REV, "tables", "S6_a_reproduction_checks.csv")
    P.to_csv(f1, index=False); Q.to_csv(f2, index=False); C.to_csv(f3, index=False); pd.DataFrame(rec_rows).to_csv(os.path.join(REV, "tables", "S6_pooling_per_recon.csv"), index=False)
    pd.set_option("display.width", 250); print(C.round(4).to_string(index=False)); print(Q[Q.offset_mode == "manuscript"].round(4).to_string(index=False))
    log.finish([f1, f2, f3], note=f"plan_tolerance_passed={int(C.passed_plan_tolerance.sum())}/{len(C)}")


main()
