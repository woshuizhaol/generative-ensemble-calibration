"""Stage 6h · 绝对标度分析对"下界值 / 近似值"的敏感性（细胞色素 c 的 8 个非精确实验值）。

背景：Milne 等的 cyt c 表里有 6 个下界值（lower_bound）和 2 个近似值（approximate），主分析把它们当作精确值使用（S1 终表 censor_flag 列）。
本脚本把这 8 个残基从 EX2 子集的绝对标度分析中去掉，重算与 S6a 相同的量，并与全部 229 个残基的结果并排报告。
预测：沉积结构 10 次重建（S4a 的 Crystal parquet），固定系数 0.35 / 2.0；编号偏移取修正后的单结构基线表（cyt c 为 +1）。
指标（各取 10 次重建的均值）：逐蛋白 Pearson、斜率（实验对预测）、CCC、偏差；池化 Pearson；蛋白内中心化池化 r；逐蛋白 r 的简单均值；
池化 Bland–Altman 偏差与一致性界限（ΔG 空间，各蛋白实验温度）。
输出：tables/S6_censored_sensitivity.csv
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402
from lib.abs_metrics import metrics  # noqa: E402

PROJ, REV, INP = revio.load_paths()
EX2 = ["Barnase", "RNaseA", "SNase", "CI2", "cytochrome_c"]
BC, BH, R_GAS = bv.BETA_C, bv.BETA_H, bv.R_GAS
S1 = pd.read_parquet(os.path.join(REV, "tables", "S1_tier1a_residues_final.parquet"))
ceil = pd.read_parquet(INP["operator_ceiling"])
OFF = dict(zip(ceil.protein, ceil.offset.astype(int)))


def load(p):
    g = S1[S1.protein_id == p].dropna(subset=["dGopen_kcal"]).reset_index(drop=True)
    T = float(g.T_C.iloc[0]) + 273.15
    Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet"))
    P = {k: d.set_index("resSeq") for k, d in Rr.groupby("recon_k")}
    rows = []
    for k in range(10):
        lnpf = BC * P[k].NC.astype(float) + BH * P[k].NH.astype(float)
        rows.append([lnpf.get(int(r) + OFF[p], np.nan) for r in g.resi])
    A = np.array(rows, float)
    keep = ~np.isnan(A).any(0)
    g = g[keep].reset_index(drop=True)
    return g, A[:, keep], T


def stats_for(data, exclude_censored):
    per = {}
    pool_pred, pool_exp, pool_dg_diff, labels = [], [], [], []
    for p in EX2:
        g, A, T = data[p]
        m = np.ones(len(g), bool) if not exclude_censored else (g.censor_flag.fillna("") == "").values
        y_lnpf = (g.dGopen_kcal.values / (R_GAS * T))[m]
        y_dg = g.dGopen_kcal.values[m]
        pred = A[:, m]
        met = metrics(pred, np.broadcast_to(y_lnpf, pred.shape))
        dg_pred = pred * (R_GAS * T)
        met_dg = metrics(dg_pred, np.broadcast_to(y_dg, dg_pred.shape))
        per[p] = dict(n=int(m.sum()), pearson=float(met["pearson"].mean()), slope_exp_on_pred=float(met_dg["slope_exp_on_pred"].mean()),
                      ccc=float(met_dg["ccc"].mean()), bias=float(met_dg["bias"].mean()))
        pool_pred.append(pred); pool_exp.append(y_lnpf); pool_dg_diff.append(dg_pred - y_dg); labels += [p] * int(m.sum())
    P = np.concatenate(pool_pred, axis=1); Y = np.concatenate(pool_exp)
    pooled_r = np.array([np.corrcoef(P[k], Y)[0, 1] for k in range(P.shape[0])]).mean()
    lab = np.array(labels)
    Pc = P.copy(); Yc = Y.copy()
    for p in EX2:
        s = lab == p
        Pc[:, s] -= Pc[:, s].mean(1, keepdims=True); Yc[s] -= Yc[s].mean()
    centred = np.array([np.corrcoef(Pc[k], Yc)[0, 1] for k in range(P.shape[0])]).mean()
    D = np.concatenate(pool_dg_diff, axis=1)
    bias = D.mean(); sd = D.std(1, ddof=1).mean()
    return per, dict(pooled_r=float(pooled_r), centred_pooled_r=float(centred), mean_per_protein_r=float(np.mean([per[p]["pearson"] for p in EX2])),
                     ba_bias=float(bias), ba_loa_lo=float(bias - 1.96 * sd), ba_loa_hi=float(bias + 1.96 * sd), n_total=int(len(Y)))


def main():
    data = {p: load(p) for p in EX2}
    out = []
    res = {}
    for name, ex in (("all_values", False), ("exclude_lower_bound_and_approximate", True)):
        res[name] = stats_for(data, ex)
    for key in ("n_total", "pooled_r", "centred_pooled_r", "mean_per_protein_r", "ba_bias", "ba_loa_lo", "ba_loa_hi"):
        out.append(dict(quantity=f"pooled: {key}", all_values=res["all_values"][1][key], exclude_lower_bound_and_approximate=res["exclude_lower_bound_and_approximate"][1][key]))
    for p in EX2:
        for key in ("n", "pearson", "slope_exp_on_pred", "ccc", "bias"):
            out.append(dict(quantity=f"{p}: {key}", all_values=res["all_values"][0][p][key], exclude_lower_bound_and_approximate=res["exclude_lower_bound_and_approximate"][0][p][key]))
    T = pd.DataFrame(out)
    T.to_csv(os.path.join(REV, "tables", "S6_censored_sensitivity.csv"), index=False)
    pd.set_option("display.width", 200)
    print(T.round(4).to_string(index=False))


main()
