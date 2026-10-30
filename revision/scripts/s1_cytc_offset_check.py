"""Stage 1 补充：cyt c 沉积结构编号错位（U-S1-1）对 K1 的影响核算，结果落盘。

1HRC 第 0 号残基为乙酰帽 ACE；fix_frame 经 OpenMM 写出时重编号为 1..N，蛋白残基整体后移 1 位，
因此原稿 structure_maps offset=0 在重编号后的帧上错位，身份最优 offset 为 +1（S1 编号核对：0 → 3/48，+1 → 48/48）。
本脚本做 10 次独立 PDBFixer 重建（与 S4a 规模一致；pdbfixer 加氢非确定），分别在 offset 0 / +1 下计算
cyt c 单结构 Spearman 与 Pearson，并把每个值代入 K1（单侧配对 Wilcoxon，系综 ρ 取原稿 null_baselines.csv）。
S4a 会对 12 个蛋白各做 10 次重建，届时本表由 S4a 结果替代、两者应一致到重建随机性以内。
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402
import mdtraj as md

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=1, task="cytc_offset_check", cpu_procs=1, seeds="pdbfixer nondeterministic; N=10")
exp = pd.read_parquet(INP["tier1a_all"]); g = exp[exp.protein_id == "cytochrome_c"]
nb = pd.read_csv(os.path.join(INP["results_tables"], "null_baselines.csv"))
o = nb.obs_spearman.values; c = nb.single_struct_ceiling.values; i = list(nb.protein).index("cytochrome_c")
raw = md.load(os.path.join(INP["pdb_dir"], "1HRC.pdb")); raw = raw[0] if raw.n_frames > 1 else raw
rows = []
for k in range(10):
    tf = bv.fix_frame(raw, True); nn = bv.nc_nh(tf)
    for off in (0, 1):
        x, y, ok = [], [], 0
        for a in g.itertuples():
            key = int(a.resi) + off
            if key in nn:
                x.append(bv.BETA_C * nn[key][1] + bv.BETA_H * nn[key][2]); y.append(float(a.dGopen_kcal))
                ok += int(nn[key][0] == bv.to1(a.resn))
        rho = stats.spearmanr(x, y)[0]; r = stats.pearsonr(x, y)[0]
        c2 = c.copy(); c2[i] = round(rho, 3); d = o - c2
        rows.append(dict(recon_k=k, offset=off, n=len(x), n_resname_match=ok, spearman=rho, pearson=r,
                         K1_wilcoxon_p=stats.wilcoxon(o, c2, alternative="greater").pvalue, K1_n_pos=int((d > 0).sum()),
                         K1_median_delta=float(np.median(d)), K1_mean_delta=float(d.mean()), cytc_delta=float(o[i] - round(rho, 3))))
T = pd.DataFrame(rows)
f = os.path.join(REV, "tables", "S1_cytc_offset_check.csv"); T.to_csv(f, index=False)
S = T.groupby("offset").agg(spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"), spearman_min=("spearman", "min"),
                            spearman_max=("spearman", "max"), pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
                            K1_p_min=("K1_wilcoxon_p", "min"), K1_p_max=("K1_wilcoxon_p", "max"), K1_npos_min=("K1_n_pos", "min"),
                            K1_npos_max=("K1_n_pos", "max"), n_resname_match=("n_resname_match", "first")).reset_index()
# 临界值：cyt c 单结构 ρ 超过多少时 K1 的 Wilcoxon 统计量会改变（扫描）
crit = None
for v in np.arange(0.40, 0.70, 0.001):
    c2 = c.copy(); c2[i] = round(v, 3)
    if stats.wilcoxon(o, c2, alternative="greater").pvalue > 0.0047:
        crit = round(v, 3); break
S["K1_p_changes_when_cytc_single_rho_exceeds"] = crit
fs = os.path.join(REV, "tables", "S1_cytc_offset_summary.csv"); S.to_csv(fs, index=False)
pd.set_option("display.width", 220); print(T.to_string(index=False)); print(S.to_string(index=False))
log.finish([f, fs], note=f"crit={crit}")
