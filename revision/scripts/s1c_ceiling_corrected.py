"""Stage 1c · 单结构基线表的编号修正（细胞色素 c 的沉积结构编号偏移 0 → +1）。

背景：1HRC 的沉积结构以乙酰帽（ACE）起头，PDBFixer 重新编号后蛋白残基整体后移一位；残基名与实验编号只在偏移 +1 时 48/48 对齐，
偏移 0 时只有 3/48（tables/S1_cytc_offset_check.csv、S1_deposited_structure_numbering.csv）。原稿的 原项目的单结构基线表 对
cyt c 记 offset=0，spearman=0.508（一次未设种子的重建；当年没有留存逐残基预测，无法在 +1 下复算）。

本脚本生成修正后的 单结构基线 副本，只改 cytochrome_c 一行：
  offset   → 1
  spearman / pearson → 沉积结构十次种子重建（S4a 的 Crystal parquet，recon_k = 0–9）在偏移 +1 下的均值（固定系数 0.35 / 2.0，ln PF 对实验 ln PF）
  identity_pct → 100.0，n → 匹配残基数
其余 11 个蛋白逐值不变。输出：
  tables/S1_deposited_structure_baseline_corrected.parquet
  tables/S1c_ceiling_corrected_check.csv（修正前后逐蛋白对照，以及 cyt c 十次重建在 0 与 +1 下的 ρ）
revio.load_paths 在该文件存在时把 INP["operator_ceiling"] 指向它，原文件保留为 INP["structure_baseline_original"]。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
ORIG = INP.get("structure_baseline_original", INP["operator_ceiling"])
BC, BH, R_GAS = bv.BETA_C, bv.BETA_H, bv.R_GAS
P = "cytochrome_c"


def main():
    ceil = pd.read_parquet(ORIG)
    S1 = pd.read_parquet(os.path.join(REV, "tables", "S1_tier1a_residues_final.parquet"))
    g = S1[S1.protein_id == P].dropna(subset=["dGopen_kcal"])
    T = float(g.T_C.iloc[0]) + 273.15
    resi = g.resi.astype(int).values
    y = g.dGopen_kcal.astype(float).values / (R_GAS * T)
    Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{P}.parquet"))
    rows = []
    for k, d in Rr.groupby("recon_k"):
        d = d.set_index("resSeq")
        lnpf = BC * d.NC.astype(float) + BH * d.NH.astype(float)
        for off in (0, 1):
            pred = np.array([lnpf.get(int(r) + off, np.nan) for r in resi], float)
            keep = ~np.isnan(pred)
            rows.append(dict(recon_k=int(k), offset=off, n=int(keep.sum()),
                             spearman=float(stats.spearmanr(pred[keep], y[keep])[0]),
                             pearson=float(stats.pearsonr(pred[keep], y[keep])[0])))
    R = pd.DataFrame(rows)
    m0 = R[R.offset == 0].spearman.mean(); m1 = R[R.offset == 1].spearman.mean()
    n1 = int(R[R.offset == 1].n.iloc[0])
    new = ceil.copy()
    i = new.index[new.protein == P][0]
    old = new.loc[i].to_dict()
    new.loc[i, "offset"] = 1
    new.loc[i, "n"] = n1
    new.loc[i, "identity_pct"] = 100.0
    new.loc[i, "spearman"] = float(R[R.offset == 1].spearman.mean())
    new.loc[i, "pearson"] = float(R[R.offset == 1].pearson.mean())
    new.to_parquet(os.path.join(REV, "tables", "S1_deposited_structure_baseline_corrected.parquet"), index=False)
    chk = []
    for c in ("offset", "n", "identity_pct", "spearman", "pearson"):
        chk.append(dict(protein=P, field=c, original=old[c], corrected=new.loc[i, c]))
    others = ceil[ceil.protein != P].reset_index(drop=True).equals(new[new.protein != P].reset_index(drop=True))
    chk.append(dict(protein="other 11 proteins", field="all columns identical", original=True, corrected=others))
    for off in (0, 1):
        s = R[R.offset == off].spearman
        chk.append(dict(protein=P, field=f"ten reconstructions, offset {off:+d}: mean ± s.d. Spearman", original=f"{s.mean():.4f}", corrected=f"{s.std(ddof=1):.4f}"))
    pd.DataFrame(chk).to_csv(os.path.join(REV, "tables", "S1c_ceiling_corrected_check.csv"), index=False)
    print(pd.DataFrame(chk).to_string(index=False))
    print(f"mean Spearman of the ten reconstructions: offset 0 = {m0:.4f}, offset +1 = {m1:.4f}; matched residues at +1 = {n1}")
    assert others and n1 == 48


main()
