"""Stage 8 · 噪声检验 TOST（±0.05）的粗估功效：假定真实差值为 0，用观测到的逐蛋白差值 SD 估计 se，
功效 ≈ 2Φ((ε − t_{0.95, n−1}·se)/se) − 1（均值近似正态；ε=0.05；ε ≤ t·se 时为 0）。这是事后的描述性估计，不是预先计划的功效分析。
输入 tables/S8_noise_per_protein.csv；输出 tables/S8_noise_tost_power.csv。"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths(); TAB = os.path.join(REV, "tables"); EPS = 0.05
log = revio.RunLogger(stage=8, task="tost_power", cpu_procs=1, seeds="none")
P = pd.read_csv(os.path.join(TAB, "S8_noise_per_protein.csv"), float_precision="round_trip"); rows = []
for (typ, form, cond), d in P[P.condition != "czero"].groupby(["noise_type", "form", "condition"]):
    for tag, col in (("uncorrected", "rho_noise_mean"), ("czero_corrected_POST_HOC", "rho_noise_czero_corrected")):
        dd = (d.rho_G - d[col]).dropna().values; n = len(dd)
        if n < 5:
            continue
        se = dd.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.95, n - 1); power = float(max(2 * stats.norm.cdf((EPS - tc * se) / se) - 1, 0.0)) if se > 0 else np.nan
        rows.append(dict(noise_type=typ, form=form, condition=cond, version=tag, n=n, sd_diff=float(dd.std(ddof=1)), se=float(se), power_at_zero_diff=power))
T = pd.DataFrame(rows); f = os.path.join(TAB, "S8_noise_tost_power.csv"); T.to_csv(f, index=False); pd.set_option("display.width", 200)
print(T.groupby(["noise_type", "form", "version"]).power_at_zero_diff.agg(["min", "median", "max"]).round(2).to_string())
print(T[T.condition == "ShortMD"].round(3).to_string(index=False))
log.finish([f], note="tost power")
