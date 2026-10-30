"""Stage 8b · LOPO 中显式加入重拟合系数的线性模型（revision_proposal.md S8 步骤 4；R3-C6：灵活的机器学习回归模型是不是就是重拟合系数的线性模型）。
沿用原 LOPO 框架（code/analysis/40_learned_operator_lopo.py）：BioEmu 全部旧列（口径 O）的逐残基系综平均计数 NC_mean、NH_mean；12 个蛋白；严格留一蛋白；
指标 = 留出蛋白上的 Spearman，对 12 个蛋白取均值（留出蛋白残基数 ≥5）。训练残基 = 其余 11 个蛋白的全部实验残基（ΔG_open, kcal/mol）。
新增两个模型（均为 OLS）：
  bv_refit_noint   ΔG ≈ b_C·NC_mean + b_H·NH_mean（不带截距）
  bv_refit_int     ΔG ≈ a + b_C·NC_mean + b_H·NH_mean（带截距）
与 bv_fixed（0.35 NC_mean + 2.0 NH_mean，不拟合）以及原有 4 个机器学习模型并列。原有模型的数字直接读自 results/tables/learned_operator_lopo.csv（BV 0.390，Ridge[NC,NH] 0.382，
Ridge[full] 0.392，RF[full] 0.387，GBM[full] 0.403），不重跑、不改动。原有 4 个模型的特征清单（14 个）登记在输出里：
  NC_mean NC_std NC_p10 NC_p90 NH_mean NH_std NH_p90 fopen NCxNH lnPF_std SASA fhelix fsheet KD；Ridge[NC,NH] 只用 NC_mean、NH_mean。
附：每折的拟合系数与比值 b_H/b_C（固定模型为 2.0/0.35 = 5.71），配对 bootstrap（B=10⁴, seed 7，对蛋白重抽）比较各模型对 bv_fixed 的 Δ。
阳性对照：bv_fixed 的均值应复现 0.390（容差：leaderboard 三位小数的舍入）。输出 tables/S8_lopo_with_linear.csv、S8_lopo_folds.csv、S8_lopo_features_registry.csv。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, ensemble as E  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=8, task="lopo_linear", cpu_procs=1, seeds="bootstrap seed 7")
PROTS = sorted(json.load(open(INP["tier1a_sequences"]))); B = 10000
FEATS14 = "NC_mean NC_std NC_p10 NC_p90 NH_mean NH_std NH_p90 fopen NCxNH lnPF_std SASA fhelix fsheet KD".split()


def design():
    rows = []
    for p in PROTS:
        c = E.load_cell("BioEmu", p, "O"); idx, y, resi = E.align_idx(c, p); cols = np.arange(c["NC"].shape[1])
        nc = np.nanmean(c["NC"][idx], axis=1); nh = np.nanmean(c["NH"][idx], axis=1)
        rows += [dict(protein=p, resi=int(r), NC_mean=a, NH_mean=b, dG=v) for r, a, b, v in zip(resi, nc, nh, y)]
    return pd.DataFrame(rows)


def ols(X, y, intercept):
    A = np.column_stack([np.ones(len(X)), X]) if intercept else X; b, *_ = np.linalg.lstsq(A, y, rcond=None); return b


def main():
    D = design(); D = D.dropna(subset=["NC_mean", "NH_mean", "dG"]); folds = []; rho = {"bv_fixed": [], "bv_refit_noint": [], "bv_refit_int": []}; plist = [p for p in PROTS if (D.protein == p).sum() >= 5]
    for p in plist:
        tr = D[D.protein != p]; te = D[D.protein == p]; X = tr[["NC_mean", "NH_mean"]].values; y = tr.dG.values; Xt = te[["NC_mean", "NH_mean"]].values
        fixed = 0.35 * te.NC_mean.values + 2.0 * te.NH_mean.values; r_fixed = stats.spearmanr(fixed, te.dG)[0]
        b0 = ols(X, y, False); r0 = stats.spearmanr(Xt @ b0, te.dG)[0]; b1 = ols(X, y, True); r1 = stats.spearmanr(b1[0] + Xt @ b1[1:], te.dG)[0]
        rho["bv_fixed"].append(r_fixed); rho["bv_refit_noint"].append(r0); rho["bv_refit_int"].append(r1)
        folds.append(dict(held_out=p, n_test=len(te), n_train=len(tr), rho_bv_fixed=r_fixed, rho_bv_refit_noint=r0, rho_bv_refit_int=r1, noint_bC=b0[0], noint_bH=b0[1], noint_ratio_bH_over_bC=b0[1] / b0[0], int_a=b1[0], int_bC=b1[1], int_bH=b1[2], int_ratio_bH_over_bC=b1[2] / b1[1], fixed_ratio=2.0 / 0.35))
    F = pd.DataFrame(folds); orig = pd.read_csv(os.path.join(INP["results_tables"], "learned_operator_lopo.csv")).set_index("model").lopo_spearman
    rng = np.random.default_rng(7); n = len(plist); idx = rng.integers(0, n, size=(B, n)); fx = np.array(rho["bv_fixed"]); rows = []
    for name in ("bv_fixed", "bv_refit_noint", "bv_refit_int"):
        v = np.array(rho[name]); d = v - fx; bm = d[idx].mean(1)
        rows.append(dict(model=name, source="this stage (LOPO, 11 train proteins)", lopo_spearman_mean=float(np.nanmean(v)), lopo_spearman_median=float(np.nanmedian(v)), n_proteins=n, delta_vs_bv_fixed=float(np.mean(d)), delta_boot_lo=float(np.percentile(bm, 2.5)), delta_boot_hi=float(np.percentile(bm, 97.5)),
                         n_better_than_bv_fixed=int((d > 1e-12).sum()), n_worse_than_bv_fixed=int((d < -1e-12).sum())))
    for name in ("BV", "Ridge[NC,NH]", "Ridge[full]", "RF[full]", "GBM[full]"):
        rows.append(dict(model="manuscript:" + name, source="results/tables/learned_operator_lopo.csv (unchanged)", lopo_spearman_mean=float(orig[name]), delta_vs_bv_fixed=float(orig[name] - orig["BV"])))
    T = pd.DataFrame(rows); chk = abs(float(np.nanmean(rho["bv_fixed"])) - float(orig["BV"]))
    T["positive_control_bv_fixed_reproduces_0.390"] = chk <= 0.0006
    reg = pd.DataFrame([dict(model=m, n_features=len(f), features=" ".join(f)) for m, f in (("Ridge[NC,NH]", ["NC_mean", "NH_mean"]), ("Ridge[full]", FEATS14), ("RF[full]", FEATS14), ("GBM[full]", FEATS14), ("bv_refit_noint", ["NC_mean", "NH_mean"]), ("bv_refit_int", ["NC_mean", "NH_mean"]))])
    f1 = os.path.join(REV, "tables", "S8_lopo_with_linear.csv"); f2 = os.path.join(REV, "tables", "S8_lopo_folds.csv"); f3 = os.path.join(REV, "tables", "S8_lopo_features_registry.csv")
    T.to_csv(f1, index=False); F.to_csv(f2, index=False); reg.to_csv(f3, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(T.round(4).to_string(index=False)); print("bv_fixed reproduces 0.390: diff %.4f" % chk); print(F[["held_out", "n_test", "rho_bv_fixed", "rho_bv_refit_noint", "rho_bv_refit_int", "noint_ratio_bH_over_bC", "int_ratio_bH_over_bC"]].round(3).to_string(index=False))
    log.finish([f1, f2, f3], note="lopo linear")


if __name__ == "__main__":
    main()
