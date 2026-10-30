"""Stage 6f · Bland–Altman 分析与蛋白内中心化拟合（revision_proposal.md 总览表 R2-C2 与 R2-C3 两行所列，Stage 6 步骤清单未单列）。
(A) Bland–Altman（R2-C2）：EX2 子集 5 个蛋白（229 个残基）。预测 = 沉积结构 10 次重建的逐残基均值，固定系数 BV（0.35·N_C + 2.0·N_H），ΔG_pred = R·T_exp·lnPF_pred；
    实验 ΔG_open 取 S1 终表；编号偏移沿用原稿（cyt c 为 0）。差值 d = ΔG_pred − ΔG_exp，均值 m = (ΔG_pred + ΔG_exp)/2。
    每个蛋白和池化各报告：偏差（d 的均值）、d 的 SD、95% 一致性界限（偏差 ± 1.96 SD）、d 对 m 的回归斜率（比例偏差）与 Pearson r；
    全部带残基 bootstrap 95% CI（B=10⁴，seed 7；池化行在 229 个残基上重抽，蛋白行在各自残基上重抽）。
(B) 蛋白内中心化拟合（R2-C3 的第三种拟合）：与 s6b 同样的 12 种几何组合 × 10 次重建，模型 ln PF_exp = β_C·N_C + β_H·N_H + 蛋白固定效应
    （等价于各蛋白内部减去蛋白均值后无截距拟合；用蛋白哑变量的设计矩阵算 HC3 SE，使被吸收的 5 个蛋白均值计入自由度）。
    12 种组合报告 10 次重建的均值与 SD；生产几何组合（0.65/3/0.26）另给：分层（按蛋白内）残基 bootstrap 95% CI（B=10⁴，seed 7，同一组重抽用于 10 次重建）、
    留一蛋白 jackknife SE、10 次重建 SD、bootstrap 内 corr(β_C, β_H)、蛋白内 r（拟合值与中心化实验值的 Pearson）。
本脚本取数与拟合函数复制自 s6a（load_exp、load_pred）与 s6b（features、ols），因为这两个脚本在模块层直接运行；运行时断言：
    生产几何组合的 M1 估计（10 次重建系数均值）与 tables/S6_coef_production_uncertainty.csv 一致，保证两处口径相同。
(C) 无截距拟合 M0（ln PF_exp = β_C·N_C + β_H·N_H，池化 229 个残基）：12 种几何组合 × 10 次重建的均值与 SD，以及拟合值与实验的池化 Pearson（方案总览表 R2-C3 要求三种拟合，Stage 6 步骤 4(c) 原先只在生产几何上做 M0；这里补到 12 种几何，用来检验“β_H 2.5–3.1”的候选来源）。
输出：tables/S6_bland_altman.csv、tables/S6_coef_centered_grid.csv、tables/S6_coef_centered_production.csv、tables/S6_coef_noint_grid.csv。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
import numpy as np, pandas as pd
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402
import s4_rescore as S4  # noqa: E402  (只用 det_seed)

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=6, task="bland_altman_centered_fit", cpu_procs=1, seeds="bootstrap seed 7")
EX2 = ["Barnase", "RNaseA", "SNase", "CI2", "cytochrome_c"]; B = 10000; BC, BH = bv.BETA_C, bv.BETA_H; R_GAS = bv.R_GAS
TAB = os.path.join(REV, "tables")
S1 = pd.read_parquet(os.path.join(TAB, "S1_tier1a_residues_final.parquet"))
ceil = pd.read_parquet(INP["operator_ceiling"]); OFF = dict(zip(ceil.protein, ceil.offset.astype(int)))
nb = pd.read_csv(os.path.join(TAB, "S1_deposited_structure_numbering.csv"))
GRID = [(c, ij, h) for c in (0.55, 0.65, 0.8) for ij in (2, 3) for h in (0.24, 0.26)]
PROD = (0.65, 3, 0.26)


def exp_of(p):
    g = S1[S1.protein_id == p].dropna(subset=["dGopen_kcal"]); T = float(g.T_C.iloc[0]) + 273.15
    return g.resi.astype(int).values, g.dGopen_kcal.astype(float).values, T


def ba_inputs():
    """{蛋白: (dG_pred, dG_exp)}：预测为沉积结构 10 次重建的逐残基均值（S4a 的 Crystal 计数），ΔG 空间、实验温度。"""
    out = {}
    for p in EX2:
        resi, dg, T = exp_of(p)
        Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet")); per = {k: g.set_index("resSeq") for k, g in Rr.groupby("recon_k")}
        rows = []
        for k in range(10):
            d = per[k]; lnpf = BC * d.NC.astype(float) + BH * d.NH.astype(float); rows.append([lnpf.get(int(r) + OFF[p], np.nan) for r in resi])
        A = np.array(rows, float); keep = ~np.isnan(A).any(0)
        out[p] = (A[:, keep].mean(0) * R_GAS * T, dg[keep])
    return out


def ba_stats(pred, obs, idx):
    """pred, obs: (n,)；idx: (B, n)。返回点估计字典与 bootstrap 数组字典。"""
    def calc(pr, ob):
        d = pr - ob; m = (pr + ob) / 2; dm = d - d.mean(-1, keepdims=True); mm = m - m.mean(-1, keepdims=True)
        sd = d.std(-1, ddof=1); slope = (dm * mm).sum(-1) / (mm ** 2).sum(-1); r = (dm * mm).sum(-1) / np.sqrt((dm ** 2).sum(-1) * (mm ** 2).sum(-1))
        bias = d.mean(-1); return dict(bias=bias, sd_diff=sd, loa_lo=bias - 1.96 * sd, loa_hi=bias + 1.96 * sd, slope_d_on_mean=slope, pearson_d_mean=r)
    return calc(pred, obs), calc(pred[idx], obs[idx])


def features():
    """F[(geom, k)] = (N_C, N_H, y=lnPF_exp, protein_index)，池化 229 个残基（复制自 s6b）。"""
    F = {}
    for k in range(10):
        per = {}
        for p in EX2:
            pdb = nb[nb.protein == p].pdb.iloc[0]; fr = md.load(os.path.join(INP["pdb_dir"], f"{pdb}.pdb")); fr = fr[0] if fr.n_frames > 1 else fr
            per[p] = bv.fix_frame(fr, True, seed=S4.det_seed("Crystal", p, 0, k, 0))
        for geom in GRID:
            cut_nc, ij, cut_nh = geom; X = []
            for gi, p in enumerate(EX2):
                resi, dg, T = exp_of(p); y = dg / (R_GAS * T); nn = bv.nc_nh(per[p], cut_nc, cut_nh, ij, 2)
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


def center(D, groups):
    X = D[:, :2].copy(); y = D[:, 2].copy(); g = D[:, 3].astype(int)
    for q in groups:
        m = g == q; X[m] -= X[m].mean(0); y[m] -= y[m].mean()
    return X, y, g


def fit_fe(D, groups=range(5)):
    """蛋白固定效应拟合：返回 (β_C, β_H)、HC3 SE（含蛋白哑变量）、蛋白内 r。"""
    groups = list(groups); sel = np.isin(D[:, 3].astype(int), groups); D = D[sel]
    dummies = np.column_stack([(D[:, 3] == q).astype(float) for q in groups])
    A = np.column_stack([D[:, :2], dummies]); beta, se, _ = ols(A, D[:, 2], intercept=False)
    X, y, _ = center(D, groups); b2 = np.linalg.lstsq(X, y, rcond=None)[0]; assert np.allclose(b2, beta[:2], atol=1e-8), (b2, beta[:2])   # Frisch–Waugh–Lovell 核对
    r = float(np.corrcoef(X @ b2, y)[0, 1]); return b2, se[:2], r


def main():
    # ------------------------------------------------------------------ (A) Bland–Altman
    inp = ba_inputs(); rng = np.random.default_rng(7); rows = []
    allp = np.concatenate([inp[p][0] for p in EX2]); allo = np.concatenate([inp[p][1] for p in EX2]); prot = np.concatenate([[p] * len(inp[p][1]) for p in EX2])
    for name, pr, ob in [(p, inp[p][0], inp[p][1]) for p in EX2] + [("pooled (5 proteins)", allp, allo)]:
        n = len(ob); idx = rng.integers(0, n, size=(B, n)); pt, bs = ba_stats(pr, ob, idx)
        for q in pt:
            lo, hi = np.percentile(bs[q], [2.5, 97.5]); rows.append(dict(protein=name, quantity=q, estimate=float(pt[q]), boot_ci_lo=float(lo), boot_ci_hi=float(hi), n_residues=n, space="dG_Texp (kcal/mol)", B=B, bootstrap_seed=7,
                                                                     prediction="mean of 10 deposited-structure reconstructions, fixed-coefficient BV"))
    BA = pd.DataFrame(rows); f1 = os.path.join(TAB, "S6_bland_altman.csv"); BA.to_csv(f1, index=False)
    # ------------------------------------------------------------------ (B) 蛋白内中心化拟合
    F = features(); Dp = [F[(PROD, k)] for k in range(10)]; n = len(Dp[0]); assert n == 229 and all(len(d) == n for d in Dp) and all(np.array_equal(d[:, 3], Dp[0][:, 3]) for d in Dp)
    ref = pd.read_csv(os.path.join(TAB, "S6_coef_production_uncertainty.csv")); ref = ref[ref.model == "M1 (with intercept)"].set_index("coefficient")
    m1 = np.mean([ols(d[:, :2], d[:, 2])[0] for d in Dp], axis=0)
    assert abs(m1[1] - ref.loc["betaC", "estimate_mean10recon"]) < 1e-9 and abs(m1[2] - ref.loc["betaH", "estimate_mean10recon"]) < 1e-9, (m1, ref)   # 与 s6b 同口径
    grid = []
    for geom in GRID:
        fits = [fit_fe(F[(geom, k)]) for k in range(10)]
        for q, nm in enumerate(("betaC", "betaH")):
            v = np.array([f[0][q] for f in fits]); se = np.array([f[1][q] for f in fits])
            grid.append(dict(cut_Nc=geom[0], ij_Nc=geom[1], cut_Nh=geom[2], coefficient=nm, mean_over_10_recon=v.mean(), sd_over_10_recon=v.std(ddof=1), hc3_se_mean10recon=se.mean(), n_residues=len(F[(geom, 0)])))
        rr = np.array([f[2] for f in fits]); grid.append(dict(cut_Nc=geom[0], ij_Nc=geom[1], cut_Nh=geom[2], coefficient="within_protein_r", mean_over_10_recon=rr.mean(), sd_over_10_recon=rr.std(ddof=1), n_residues=len(F[(geom, 0)])))
    G = pd.DataFrame(grid); f2 = os.path.join(TAB, "S6_coef_centered_grid.csv"); G.to_csv(f2, index=False)
    # (C) 无截距 M0 在 12 种几何上
    m0rows = []
    for geom in GRID:
        fits = [ols(F[(geom, k)][:, :2], F[(geom, k)][:, 2], intercept=False) for k in range(10)]; rr = np.array([np.corrcoef(f_[2], F[(geom, k)][:, 2])[0, 1] for k, f_ in enumerate(fits)])
        for q, nm in enumerate(("betaC", "betaH")):
            v = np.array([f_[0][q] for f_ in fits]); se = np.array([f_[1][q] for f_ in fits])
            m0rows.append(dict(cut_Nc=geom[0], ij_Nc=geom[1], cut_Nh=geom[2], coefficient=nm, mean_over_10_recon=v.mean(), sd_over_10_recon=v.std(ddof=1), hc3_se_mean10recon=se.mean(), n_residues=len(F[(geom, 0)])))
        m0rows.append(dict(cut_Nc=geom[0], ij_Nc=geom[1], cut_Nh=geom[2], coefficient="pooled_r", mean_over_10_recon=rr.mean(), sd_over_10_recon=rr.std(ddof=1), n_residues=len(F[(geom, 0)])))
    M0 = pd.DataFrame(m0rows); f4 = os.path.join(TAB, "S6_coef_noint_grid.csv"); M0.to_csv(f4, index=False)
    refm0 = pd.read_csv(os.path.join(TAB, "S6_coef_production_uncertainty.csv")); refm0 = refm0[refm0.model.str.startswith("M0")].set_index("coefficient")
    pg = M0[(M0.cut_Nc == PROD[0]) & (M0.ij_Nc == PROD[1]) & (M0.cut_Nh == PROD[2])].set_index("coefficient")
    assert abs(pg.loc["betaC", "mean_over_10_recon"] - refm0.loc["betaC", "estimate_mean10recon"]) < 1e-9 and abs(pg.loc["betaH", "mean_over_10_recon"] - refm0.loc["betaH", "estimate_mean10recon"]) < 1e-9, (pg, refm0)   # 与 s6b 的 M0 同口径
    # 生产几何组合：分层（蛋白内）残基 bootstrap、留一蛋白 jackknife、10 次重建 SD
    est_each = [fit_fe(d) for d in Dp]; est = np.mean([f[0] for f in est_each], axis=0); sd_rec = np.std([f[0] for f in est_each], axis=0, ddof=1); hc3 = np.mean([f[1] for f in est_each], axis=0); rw = np.array([f[2] for f in est_each])
    rng2 = np.random.default_rng(7); g0 = Dp[0][:, 3].astype(int); XtX = np.zeros((B, 2, 2)); Xty = np.zeros((B, 2)); bs = np.zeros((B, 2))
    idx_q = {q: rng2.integers(0, int((g0 == q).sum()), size=(B, int((g0 == q).sum()))) for q in range(5)}
    for d in Dp:
        XtX[:] = 0; Xty[:] = 0
        for q in range(5):
            dq = d[g0 == q]; Xq = dq[:, :2][idx_q[q]]; yq = dq[:, 2][idx_q[q]]; Xq = Xq - Xq.mean(1, keepdims=True); yq = yq - yq.mean(1, keepdims=True)
            XtX += np.einsum("bni,bnj->bij", Xq, Xq); Xty += np.einsum("bni,bn->bi", Xq, yq)
        bs += np.linalg.solve(XtX, Xty[..., None])[..., 0] / 10
    ci = np.percentile(bs, [2.5, 97.5], axis=0); rb = float(np.corrcoef(bs[:, 0], bs[:, 1])[0, 1])
    jk = np.array([np.mean([fit_fe(d, [x for x in range(5) if x != q])[0] for d in Dp], axis=0) for q in range(5)]); jk_se = np.sqrt((5 - 1) / 5 * ((jk - jk.mean(0)) ** 2).sum(0))
    names = ["betaC", "betaH"]
    P = pd.DataFrame([dict(model="within-protein-centered (protein fixed effects, no intercept)", coefficient=names[i], estimate_mean10recon=est[i], hc3_se_mean10recon=hc3[i], boot_ci_lo=ci[0, i], boot_ci_hi=ci[1, i],
                           jackknife_se_leave_one_protein=jk_se[i], sd_over_10_recon_fits=sd_rec[i], corr_betaC_betaH_in_bootstrap=rb, within_protein_r_mean10recon=float(rw.mean()), geometry="cut_Nc=0.65 nm, |i-j|>=3; cut_Nh=0.26 nm, |i-j|>=2",
                           n_residues=n, B=B, bootstrap_seed=7, bootstrap_scheme="residues resampled within each protein") for i in range(2)])
    for q in range(5):
        P = pd.concat([P, pd.DataFrame([dict(model=f"leave out {EX2[q]}", coefficient=names[i], estimate_mean10recon=jk[q, i], n_residues=int((g0 != q).sum())) for i in range(2)])], ignore_index=True)
    f3 = os.path.join(TAB, "S6_coef_centered_production.csv"); P.to_csv(f3, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(BA[BA.quantity.isin(["bias", "sd_diff", "loa_lo", "loa_hi", "slope_d_on_mean"])].pivot(index="protein", columns="quantity", values="estimate").round(3).to_string())
    print(M0.groupby("coefficient").mean_over_10_recon.agg(["min", "max"]).round(3).to_string()); print(G[G.coefficient != "within_protein_r"].groupby("coefficient").mean_over_10_recon.agg(["min", "max"]).round(3).to_string()); print(P.round(3).to_string(index=False))
    log.finish([f1, f2, f3, f4], note=f"BA rows={len(BA)} centered grid=12x10 noint grid=12x10 B={B}")


main()
