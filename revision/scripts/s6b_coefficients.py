"""Stage 6b · 系数报告与不确定度（revision_proposal.md S6 步骤 4）。
数据同 s6a：EX2 子集 5 个蛋白、229 个残基；结构 = 沉积结构的 10 次确定性重建（与 S4a 逐位相同的种子 det_seed("Crystal", 蛋白, 0, k, 0)）。
对每个几何组合重新计数 N_C、N_H（cut_Nc ∈ {0.55, 0.65, 0.8} nm × ij_Nc ∈ {2, 3} × cut_Nh ∈ {0.24, 0.26} nm，其余同 bv.nc_nh），
拟合 M1：ln PF_exp = a + β_C·N_C + β_H·N_H（带截距 OLS，池化 229 个残基）；报告点估计、HC3 稳健 SE、拟合值与实验的池化 Pearson（= 多重相关 R，与原表同定义）。
(a) 12 种组合：逐次重建的值与 10 次均值 ± SD；与原稿补充表 6（results/stats/stage1_beta_sensitivity.csv）逐格比较；汇总行：范围、样本 CV（ddof=1）、N_C 与 N_H 的 Pearson、VIF。
(b) 生产几何组合（0.65 / 3 / 0.26）：估计量 = 10 次重建各自 OLS 系数的均值；残基 bootstrap 95% CI（B=10⁴，seed 7，同一组重抽下标用于 10 次重建）、
    留一蛋白 jackknife SE（5 折）、10 次重建拟合的 SD、bootstrap 分布内 corr(β̂_C, β̂_H)；M0（不带截距）只在此几何组合上拟合，供来源追溯。
输出：tables/S6_coef_grid.csv、tables/S6_coef_production_uncertainty.csv、tables/S6_coef_vs_manuscript_table6.csv。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, itertools, json
import numpy as np, pandas as pd
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402
import s4_rescore as S4  # noqa: E402  (只用 det_seed)

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=6, task="coefficients", cpu_procs=1, seeds="det_seed(Crystal,p,0,k,0); bootstrap seed 7")
EX2 = ["Barnase", "RNaseA", "SNase", "CI2", "cytochrome_c"]; B = 10000; R_GAS = bv.R_GAS
S1 = pd.read_parquet(os.path.join(REV, "tables", "S1_tier1a_residues_final.parquet"))
ceil = pd.read_parquet(INP["operator_ceiling"]); OFF = dict(zip(ceil.protein, ceil.offset.astype(int)))
nb = pd.read_csv(os.path.join(REV, "tables", "S1_deposited_structure_numbering.csv"))
GRID = [(c, ij, h) for c in (0.55, 0.65, 0.8) for ij in (2, 3) for h in (0.24, 0.26)]
PROD = (0.65, 3, 0.26)


def exp_of(p):
    g = S1[S1.protein_id == p].dropna(subset=["dGopen_kcal"]); T = float(g.T_C.iloc[0]) + 273.15
    return g.resi.astype(int).values, g.dGopen_kcal.astype(float).values / (R_GAS * T)


def features():
    """F[(geom, k)] = (N_C, N_H, y, protein_index)，池化 229 个残基。"""
    F = {}
    for k in range(10):
        per = {}
        for p in EX2:
            pdb = nb[nb.protein == p].pdb.iloc[0]; fr = md.load(os.path.join(INP["pdb_dir"], f"{pdb}.pdb")); fr = fr[0] if fr.n_frames > 1 else fr
            t1 = bv.fix_frame(fr, True, seed=S4.det_seed("Crystal", p, 0, k, 0)); per[p] = t1
        for geom in GRID:
            cut_nc, ij, cut_nh = geom; X = []
            for gi, p in enumerate(EX2):
                resi, y = exp_of(p); nn = bv.nc_nh(per[p], cut_nc, cut_nh, ij, 2)
                for r_, y_ in zip(resi, y):
                    key = int(r_) + OFF[p]
                    if key in nn:
                        X.append((nn[key][1], nn[key][2], y_, gi))
            F[(geom, k)] = np.array(X, float)
    return F


def ols(X, y, intercept=True):
    A = np.column_stack([np.ones(len(X)), X]) if intercept else X
    beta, *_ = np.linalg.lstsq(A, y, rcond=None); res = y - A @ beta; h = np.einsum("ij,jk,ik->i", A, np.linalg.inv(A.T @ A), A)
    meat = (A * (res / (1 - h))[:, None]).T @ (A * (res / (1 - h))[:, None]); cov = np.linalg.inv(A.T @ A) @ meat @ np.linalg.inv(A.T @ A)
    return beta, np.sqrt(np.diag(cov)), A @ beta


def fit_m1(D):
    beta, se, fit = ols(D[:, :2], D[:, 2]); r = float(np.corrcoef(fit, D[:, 2])[0, 1])
    return dict(intercept=beta[0], betaC=beta[1], betaH=beta[2], se_intercept=se[0], se_betaC=se[1], se_betaH=se[2], pooled_r=r, n=len(D))


def main():
    F = features(); rows = []
    for geom in GRID:
        fits = [fit_m1(F[(geom, k)]) for k in range(10)]
        for k, f in enumerate(fits):
            rows.append(dict(cut_Nc=geom[0], ij_Nc=geom[1], cut_Nh=geom[2], recon=str(k), **f))
        mean = {m: float(np.mean([f[m] for f in fits])) for m in fits[0]}; sd = {m: float(np.std([f[m] for f in fits], ddof=1)) for m in fits[0]}
        rows.append(dict(cut_Nc=geom[0], ij_Nc=geom[1], cut_Nh=geom[2], recon="mean10", **mean)); rows.append(dict(cut_Nc=geom[0], ij_Nc=geom[1], cut_Nh=geom[2], recon="sd10", **sd))
    G = pd.DataFrame(rows)
    # 与原稿补充表 6 逐格比较
    orig = pd.read_csv(os.path.join(INP["results_stats"], "stage1_beta_sensitivity.csv")); cmp_rows = []
    for o in orig.itertuples():
        sel = G[(G.cut_Nc == o.cut_Nc) & (G.ij_Nc == o.ij_Nc) & (G.cut_Nh == o.cut_Nh)]; m = sel[sel.recon == "mean10"].iloc[0]; sd = sel[sel.recon == "sd10"].iloc[0]; each = sel[~sel.recon.isin(["mean10", "sd10"])]
        r = dict(cut_Nc=o.cut_Nc, ij_Nc=o.ij_Nc, cut_Nh=o.cut_Nh, n_orig=o.n, n_new=int(m.n))
        for q, oq in (("betaC", o.betaC), ("betaH", o.betaH), ("intercept", o.intercept), ("pooled_r", o.pooled_pearson)):
            r[f"{q}_orig"] = oq; r[f"{q}_new_mean10"] = m[q]; r[f"{q}_new_sd10"] = sd[q]; r[f"{q}_diff"] = oq - m[q]
            r[f"{q}_orig_in_10recon_range"] = bool(each[q].min() <= oq <= each[q].max()); r[f"{q}_abs_diff_le_0.005"] = bool(abs(oq - m[q]) <= 0.005)
        cmp_rows.append(r)
    C = pd.DataFrame(cmp_rows)
    # 汇总行：12 个原表值的范围与 CV；N_C-N_H 相关与 VIF（生产几何、recon 0 与 10 次均值）
    summ = []
    for q in ("betaC", "betaH", "intercept", "pooled_r"):
        v = C[f"{q}_orig"].values; w = C[f"{q}_new_mean10"].values
        summ.append(dict(quantity=q, orig_min=v.min(), orig_max=v.max(), orig_cv_ddof1=v.std(ddof=1) / abs(v.mean()), new_min=w.min(), new_max=w.max(), new_cv_ddof1=w.std(ddof=1) / abs(w.mean())))
    corr = {}
    for geom in GRID:
        D = F[(geom, 0)]; r = np.corrcoef(D[:, 0], D[:, 1])[0, 1]; corr[geom] = (r, 1 / (1 - r ** 2))
    S = pd.DataFrame(summ); S["corr_NC_NH_production_recon0"] = corr[PROD][0]; S["VIF_production_recon0"] = corr[PROD][1]
    S["corr_NC_NH_range_over_12_geometries"] = f"{min(v[0] for v in corr.values()):.3f}–{max(v[0] for v in corr.values()):.3f}"; S["VIF_range_over_12_geometries"] = f"{min(v[1] for v in corr.values()):.2f}–{max(v[1] for v in corr.values()):.2f}"
    # (b) 生产几何组合的完整不确定度
    Ds = [F[(PROD, k)] for k in range(10)]; n = len(Ds[0]); assert all(len(d) == n for d in Ds) and all(np.array_equal(d[:, 3], Ds[0][:, 3]) for d in Ds)
    fits = [ols(d[:, :2], d[:, 2])[0] for d in Ds]; est = np.mean(fits, axis=0); sd_rec = np.std(fits, axis=0, ddof=1)
    hc3 = np.mean([ols(d[:, :2], d[:, 2])[1] for d in Ds], axis=0)
    rng = np.random.default_rng(7); idx = rng.integers(0, n, size=(B, n)); bs = np.zeros((B, 3))
    for d in Ds:   # 同一组重抽下标用于 10 次重建；估计量 = 10 次重建系数的均值
        Y = d[:, 2][idx]; X1 = np.stack([np.ones((B, n)), d[:, 0][idx], d[:, 1][idx]], axis=2)
        XtX = np.einsum("bni,bnj->bij", X1, X1); Xty = np.einsum("bni,bn->bi", X1, Y); bs += np.linalg.solve(XtX, Xty[..., None])[..., 0] / 10
    ci = np.percentile(bs, [2.5, 97.5], axis=0); rb = float(np.corrcoef(bs[:, 1], bs[:, 2])[0, 1])
    prot = Ds[0][:, 3].astype(int); jk = []
    for q in range(5):
        keep = prot != q; jk.append(np.mean([ols(d[keep][:, :2], d[keep][:, 2])[0] for d in Ds], axis=0))
    jk = np.array(jk); jk_se = np.sqrt((5 - 1) / 5 * ((jk - jk.mean(0)) ** 2).sum(0))
    # 留一蛋白的系数本身（核查发现：留一蛋白 SE 远大于残基 bootstrap SE，要让读者看到是哪个蛋白驱动）
    pd.DataFrame([dict(left_out=EX2[q], n_residues_fit=int((prot != q).sum()), intercept=jk[q, 0], betaC=jk[q, 1], betaH=jk[q, 2]) for q in range(5)] + [dict(left_out="(none: all 5 proteins)", n_residues_fit=n, intercept=est[0], betaC=est[1], betaH=est[2])]).to_csv(os.path.join(REV, "tables", "S6_coef_production_leave_one_protein.csv"), index=False)
    names = ["intercept", "betaC", "betaH"]
    U = pd.DataFrame([dict(model="M1 (with intercept)", coefficient=names[i], estimate_mean10recon=est[i], hc3_se_mean10recon=hc3[i], boot_ci_lo=ci[0, i], boot_ci_hi=ci[1, i], jackknife_se_leave_one_protein=jk_se[i],
                           sd_over_10_recon_fits=sd_rec[i], corr_betaC_betaH_in_bootstrap=rb, geometry="cut_Nc=0.65 nm, |i-j|>=3; cut_Nh=0.26 nm, |i-j|>=2", n_residues=n, B=B, bootstrap_seed=7) for i in range(3)])
    m0 = [ols(d[:, :2], d[:, 2], intercept=False)[0] for d in Ds]; m0m = np.mean(m0, axis=0); m0s = np.std(m0, axis=0, ddof=1)
    U = pd.concat([U, pd.DataFrame([dict(model="M0 (no intercept; provenance only)", coefficient=nm, estimate_mean10recon=m0m[i], sd_over_10_recon_fits=m0s[i], geometry="production", n_residues=n) for i, nm in enumerate(["betaC", "betaH"])])], ignore_index=True)
    # ---- 原稿补充表 6 与"单次重建"的相容性检验：原稿对每个蛋白只做了一次未设种子的重建，12 行来自同一次运行（所以彼此高度相关）。
    # 模拟这一过程：每个蛋白独立地从 10 次重建里抽一次（20,000 个组合），对 12 种几何各做 M1 拟合（用充分统计量向量化），得到 36 个量的联合分布；
    # 看原稿的 36 个值落在哪里：逐个量的分位，以及联合统计量 T=Σ((值−组合均值)/组合 SD)² 的单侧 p 值（对原稿值与每个组合同样计算）。
    rng2 = np.random.default_rng(7); NCOMB = 20000; ks = rng2.integers(0, 10, size=(NCOMB, 5)); sims = {}
    for geom in GRID:
        XtX = np.zeros((10, 5, 3, 3)); Xty = np.zeros((10, 5, 3)); yy = np.zeros((10, 5)); ny = np.zeros((10, 5)); sy = np.zeros((10, 5)); nn_ = np.zeros(5)
        for k in range(10):
            D = F[(geom, k)]
            for q in range(5):
                d = D[D[:, 3] == q]; A = np.column_stack([np.ones(len(d)), d[:, 0], d[:, 1]]); XtX[k, q] = A.T @ A; Xty[k, q] = A.T @ d[:, 2]; yy[k, q] = (d[:, 2] ** 2).sum(); sy[k, q] = d[:, 2].sum(); nn_[q] = len(d)
        sel = np.arange(5)[None, :]; XtXc = XtX[ks, sel].sum(1); Xtyc = Xty[ks, sel].sum(1); yyc = yy[ks, sel].sum(1); syc = sy[ks, sel].sum(1); nt = nn_.sum()
        beta = np.linalg.solve(XtXc, Xtyc[..., None])[..., 0]; rss = yyc - (beta * Xtyc).sum(1); tss = yyc - syc ** 2 / nt
        sims[geom] = dict(betaC=beta[:, 1], betaH=beta[:, 2], pooled_r=np.sqrt(1 - rss / tss))
    quants = ("betaC", "betaH", "pooled_r"); oq = {"betaC": "betaC", "betaH": "betaH", "pooled_r": "pooled_pearson"}; zs_c = np.zeros((NCOMB, 0)); zs_o = []; pr = []
    for geom in GRID:
        o = orig[(orig.cut_Nc == geom[0]) & (orig.ij_Nc == geom[1]) & (orig.cut_Nh == geom[2])].iloc[0]
        for q in quants:
            v = sims[geom][q]; m, sd = v.mean(), v.std(ddof=1); zs_c = np.column_stack([zs_c, (v - m) / sd]); zs_o.append((o[oq[q]] - m) / sd)
            pr.append(dict(cut_Nc=geom[0], ij_Nc=geom[1], cut_Nh=geom[2], quantity=q, orig=o[oq[q]], combo_mean=m, combo_sd=sd, z_orig=(o[oq[q]] - m) / sd, orig_percentile=float((v < o[oq[q]]).mean() * 100)))
    T_orig = float(np.sum(np.square(zs_o))); T_c = (zs_c ** 2).sum(1); p_joint = float((T_c >= T_orig).mean())
    PR = pd.DataFrame(pr); PR["joint_T_orig"] = T_orig; PR["joint_p_value"] = p_joint; PR["n_combinations"] = NCOMB
    PR.to_csv(os.path.join(REV, "tables", "S6_coef_table6_combo_consistency.csv"), index=False)
    print("原稿表 6 与\"每个蛋白一次随机重建\"的相容性：逐个量的 z 范围 %.2f 到 %.2f；落在组合分布 2.5–97.5 百分位内的量 %d/36；联合统计量 T=%.1f，单侧 p=%.3f（%d 个组合）" % (min(zs_o), max(zs_o), int(((PR.orig_percentile > 2.5) & (PR.orig_percentile < 97.5)).sum()), T_orig, p_joint, NCOMB))
    f1 = os.path.join(REV, "tables", "S6_coef_grid.csv"); f2 = os.path.join(REV, "tables", "S6_coef_production_uncertainty.csv"); f3 = os.path.join(REV, "tables", "S6_coef_vs_manuscript_table6.csv")
    G.to_csv(f1, index=False); U.to_csv(f2, index=False); C.to_csv(f3, index=False); S.to_csv(os.path.join(REV, "tables", "S6_coef_grid_summary.csv"), index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(C[["cut_Nc", "ij_Nc", "cut_Nh", "betaC_orig", "betaC_new_mean10", "betaC_new_sd10", "betaH_orig", "betaH_new_mean10", "betaH_new_sd10", "pooled_r_orig", "pooled_r_new_mean10", "pooled_r_new_sd10"]].round(3).to_string(index=False))
    print("table-6 values within the 10-recon range: betaC %d/12, betaH %d/12, pooled_r %d/12 | within 0.005 of mean: betaC %d, betaH %d, pooled_r %d" % tuple(int(C[c].sum()) for c in ("betaC_orig_in_10recon_range", "betaH_orig_in_10recon_range", "pooled_r_orig_in_10recon_range", "betaC_abs_diff_le_0.005", "betaH_abs_diff_le_0.005", "pooled_r_abs_diff_le_0.005")))
    print(S.round(3).to_string(index=False)); print(U.round(3).to_string(index=False))
    log.finish([f1, f2, f3], note=f"grid=12 recon=10 B={B}")


main()
