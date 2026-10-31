"""Stage 6g · 池化相关中各蛋白的权重（n·SD_x·SD_y 占总和的比例）。

背景：池化 Pearson r 可分解为各蛋白蛋白内协方差的加权和，权重 ∝ n_p · SD(pred)_p · SD(exp)_p。
本脚本把这个权重逐蛋白算出来，供 SI 表 1 引用（正文“SNase 占池化权重 68%”的出处）。
数据与 s6a_absolute_scale.py 相同：EX2 子集 5 个蛋白（229 残基），预测为沉积结构 10 次重建的固定系数 ln PF，
实验为 ln PF_exp；口径 O（cyt c 偏移 0）。每次重建各算一遍权重份额，再取 10 次的均值与标准差。
输出：tables/S6_pooling_weights.csv。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
EX2 = ["Barnase", "RNaseA", "SNase", "CI2", "cytochrome_c"]
BC, BH, R_GAS = bv.BETA_C, bv.BETA_H, bv.R_GAS
S1 = pd.read_parquet(os.path.join(REV, "tables", "S1_tier1a_residues_final.parquet"))
ceil = pd.read_parquet(INP["operator_ceiling"])
OFF_MS = dict(zip(ceil.protein, ceil.offset.astype(int)))


def load_exp(p):
    g = S1[S1.protein_id == p].dropna(subset=["dGopen_kcal"])
    T = float(g.T_C.iloc[0]) + 273.15
    return pd.DataFrame(dict(resi=g.resi.astype(int).values, lnPF_exp=g.dGopen_kcal.astype(float).values / (R_GAS * T)))


def load_pred(p):
    Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet"))
    return {k: g.set_index("resSeq") for k, g in Rr.groupby("recon_k")}


def main():
    data = {}
    for p in EX2:
        e = load_exp(p)
        P = load_pred(p)
        rows = []
        for k in range(10):
            d = P[k]
            lnpf = BC * d.NC.astype(float) + BH * d.NH.astype(float)
            rows.append([lnpf.get(int(r) + OFF_MS[p], np.nan) for r in e.resi])
        A = np.array(rows, float)
        keep = ~np.isnan(A).any(0)
        data[p] = (A[:, keep], e.lnPF_exp.values[keep])
    shares = []
    for k in range(10):
        w = {p: len(y) * np.std(A[k], ddof=0) * np.std(y, ddof=0) for p, (A, y) in data.items()}
        tot = sum(w.values())
        shares.append({p: w[p] / tot for p in EX2})
    S = pd.DataFrame(shares)
    out = pd.DataFrame({"protein": EX2, "n_residues": [len(data[p][1]) for p in EX2],
                        "weight_share_mean_over_10_recon": [S[p].mean() for p in EX2],
                        "weight_share_sd_over_10_recon": [S[p].std(ddof=1) for p in EX2]})
    out.to_csv(os.path.join(REV, "tables", "S6_pooling_weights.csv"), index=False)
    print(out.round(4).to_string(index=False))


main()
