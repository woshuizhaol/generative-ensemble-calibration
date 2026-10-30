"""Stage 9c · Str2Str_xtal 的种子间变异对五类生成器可区分性的影响。
五类比较（count_avg，口径 Q 的逐蛋白中位数 ρ；BioEmu、ESMFlow、ShortMD、Boltz-2 取 S7 主分析的值）中，把 Str2Str_xtal 一列依次换成种子 11–15 的值
（S7_xtal_seed_variation.csv，一个种子 = 50 帧一次“抽样”），重做 Friedman 检验、平均秩，并给出 Str2Str_xtal 对沉积结构（原稿单结构基线）的胜出数与单侧 Wilcoxon p。
输出 tables/S9_xtal_seed_friedman.csv。"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths(); TAB = os.path.join(REV, "tables")
log = revio.RunLogger(stage=9, task="xtal_seed_friedman", cpu_procs=1, seeds="none")
P = pd.read_csv(os.path.join(TAB, "S7_invpf_vs_countavg_per_protein.csv"), float_precision="round_trip"); V = pd.read_csv(os.path.join(TAB, "S7_xtal_seed_variation.csv"), float_precision="round_trip")
Pp = P[(P.scenario == "primary") & (P.form == "count_avg")]; wide = Pp.pivot(index="protein", columns="generator", values="rho_median"); base = Pp[Pp.generator == "BioEmu"].set_index("protein").baseline_rho
CLS = ["BioEmu", "ESMFlow", "Str2Str_xtal", "ShortMD", "Boltz2"]; rows = []
for k in (11, 12, 13, 14, 15):
    W = wide[[c for c in CLS if c != "Str2Str_xtal"]].copy(); xv = V[(V.seed == k) & (V.form == "count_avg")].set_index("protein").rho_median; W["Str2Str_xtal"] = xv; W = W[CLS].dropna()
    chi, p = stats.friedmanchisquare(*[W[c].values for c in CLS]); ranks = W.rank(axis=1, ascending=False).mean()
    d = (W.Str2Str_xtal - base.loc[W.index]).values; wil = stats.wilcoxon(d, alternative="greater") if np.any(d != 0) else None
    rows.append(dict(seed=k, n_proteins=len(W), friedman_chi2=float(chi), friedman_p=float(p), **{f"mean_rank_{c}": float(ranks[c]) for c in CLS}, xtal_wins_vs_structure=int((d > 0).sum()), xtal_vs_structure_wilcoxon_greater_p=(float(wil.pvalue) if wil is not None else np.nan)))
T = pd.DataFrame(rows); f = os.path.join(TAB, "S9_xtal_seed_friedman.csv"); T.to_csv(f, index=False); pd.set_option("display.width", 220); print(T.round(4).to_string(index=False)); log.finish([f], note="xtal seed friedman")
