"""Stage 10 · 起点结构诊断（revision_proposal.md S10 风控：Str2Str_AF 与 Str2Str_legacy 差异很大时，额外报告两组起点之间的差异，把差异定位到起点上）。
对 D_legacy65 里既有旧版 Str2Str（起点 = BioEmu 首帧）又有 Str2Str_AF（起点 = AlphaFold 模型）的域（55 个，逐域配对）：
  1. 起点差异：AF 模型与 BioEmu 首帧的 CA-RMSD（S5_gen_qc_Str2Str_AF_*.csv 的 ca_rmsd_start_to_bioemu_nm）；
  2. g(c) 分布：两个系综的 obs_min 与 g(c) 的中位数、SD（ln PF）；
  3. 单个 BioEmu 首帧的 g（RT·g(frame 0)）本身与 ΔG_fold 的 Spearman ρ，以及旧版 Str2Str 的 obs_min 与它的 ρ（若旧版信号主要继承自起点，二者应高度相关）；
  4. 同一组 55 个域上 ρ(旧版) − ρ(AF) 的配对 bootstrap Δρ（B=10⁴）。
观测量与 S10 主分析相同（lib.globalstats / s10_global_stability 的函数），不另换观测量。输出 tables/S10_start_structure_diagnostic.csv（逐域）与 tables/S10_start_structure_summary.csv。"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, globalstats as GS  # noqa: E402
import s10_global_stability as S10  # noqa: E402

PROJ, REV, INP = revio.load_paths(); TAB = os.path.join(REV, "tables"); RT = S10.RT; B = 10000
log = revio.RunLogger(stage=10, task="start_diagnostic", cpu_procs=1, seeds="bootstrap/permutation via lib.globalstats.seeded_rng")
DT = S10.domain_table(); spec = {s["key"]: s for s in S10.SPECS}
KEYS = dict(leg35="Str2Str_legacy_d035|S4|native50", leg25="Str2Str_legacy_d025|S4|native50", af35="Str2Str_AF_d035|S4|native50", af25="Str2Str_AF_d025|S4|native50")
qc = {d: pd.read_csv(os.path.join(TAB, f"S5_gen_qc_Str2Str_AF_{d}.csv")).set_index("domain_id").ca_rmsd_start_to_bioemu_nm for d in ("d035", "d025")}
rows = []; GB = {}
for r in DT.itertuples():
    if not (r.in_legacy65 and r.need_Str2Str_AF):
        continue
    rec = dict(domain_id=r.domain_id, WT_name=r.WT_name, origin=r.origin, dG=r.dG, ca_rmsd_AF_start_to_BioEmu_frame0_nm=float(qc["d035"].get(r.domain_id, np.nan)))
    ok = True
    for k, key in KEYS.items():
        sp = spec[key]; gen = sp["gen"]; gdir = S10.source_name(gen, r, sp["scoring"])
        G = None if gdir is None else S10.ens_G(sp, r, gdir)
        if G is None:
            ok = False; break
        o = S10.obs_from_G(G); g = G[0]
        rec.update({f"obs_min_{k}": float(o["obs_min"][0]), f"g_median_{k}": float(np.median(g)), f"g_sd_{k}": float(np.std(g)), f"g_min_{k}": float(g.min()), f"n_frames_{k}": int(G.shape[1])})
    if not ok:
        continue
    bdir = S10.source_name("BioEmu", r, "S4_rescored"); gb = S10.g_new(bdir, r.domain_id) if bdir else None
    rec["bioemu_frame0_g_RT"] = float(RT * gb.loc[0]) if gb is not None and 0 in gb.index else np.nan; rows.append(rec); GB[r.domain_id] = gb
D = pd.DataFrame(rows); f1 = os.path.join(TAB, "S10_start_structure_diagnostic.csv"); D.to_csv(f1, index=False)
y = D.dG.values; out = []


def stat(name, x, extra=None):
    m = np.isfinite(x) & np.isfinite(y); s = GS.corr_stats(x[m], y[m], B_perm=100000, B_boot=B, tag="startdiag|" + name); out.append(dict(item=name, n=int(m.sum()), spearman_rho_with_dG=s["spearman_rho"], p_perm_two_sided=s["spearman_p_perm_two_sided"], boot_lo=s["spearman_boot_lo"], boot_hi=s["spearman_boot_hi"], **(extra or {})))


for k in KEYS:
    stat(f"obs_min {k} vs dG_fold", D[f"obs_min_{k}"].values)
stat("RT*g(BioEmu frame 0) vs dG_fold", D.bioemu_frame0_g_RT.values)


def inherit(name, a, b):
    m = np.isfinite(a) & np.isfinite(b); r_, p_ = stats.spearmanr(a[m], b[m]); out.append(dict(item=name, n=int(m.sum()), spearman_rho_with_dG=np.nan, p_perm_two_sided=float(p_), boot_lo=np.nan, boot_hi=np.nan, spearman_between_the_two=float(r_)))


inherit("Spearman between domains: obs_min(legacy d035) vs RT*g(BioEmu frame 0) [p = scipy two-sided]", D.obs_min_leg35.values, D.bioemu_frame0_g_RT.values)
inherit("Spearman between domains: obs_min(AF d035) vs RT*g(BioEmu frame 0) [p = scipy two-sided]", D.obs_min_af35.values, D.bioemu_frame0_g_RT.values)
inherit("Spearman between domains: obs_min(legacy d035) vs obs_min(AF d035) [p = scipy two-sided]", D.obs_min_leg35.values, D.obs_min_af35.values)
n = len(D)
for a_, b_, lab in (("obs_min_leg35", "obs_min_af35", "delta rho (legacy d035 - AF d035)"), ("obs_min_leg25", "obs_min_af25", "delta rho (legacy d025 - AF d025)"), ("obs_min_leg35", "bioemu_frame0_g_RT", "delta rho (legacy d035 - BioEmu frame 0)")):
    va, vb = D[a_].values, D[b_].values; m = np.isfinite(va) & np.isfinite(vb); va, vb, yy = va[m], vb[m], y[m]; nn = len(yy)
    rng = GS.seeded_rng("startdiag", lab, nn); idx = rng.integers(0, nn, size=(B, nn)); da = GS._spearman_rows(va[idx], yy[idx]) - GS._spearman_rows(vb[idx], yy[idx]); da = da[np.isfinite(da)]
    out.append(dict(item=lab, n=nn, spearman_rho_with_dG=float(stats.spearmanr(va, yy)[0] - stats.spearmanr(vb, yy)[0]), p_perm_two_sided=np.nan, boot_lo=float(np.percentile(da, 2.5)), boot_hi=float(np.percentile(da, 97.5))))
# 单个 BioEmu 帧的 ρ 分布：各域共有的帧 0..K-1，逐帧算 RT·g(frame k) 与 ΔG 的 Spearman ρ，看首帧在其中的位置
K = min(len(GB[d]) for d in D.domain_id); arr = np.array([RT * GB[d].sort_index().values[:K] for d in D.domain_id]); rho_k = np.array([stats.spearmanr(arr[:, k], y)[0] for k in range(K)])
out.append(dict(item=f"single BioEmu frame rho distribution (frames 0..{K - 1} common to the 55 domains; boot_lo/hi = 2.5/97.5 percentiles)", n=K, spearman_rho_with_dG=float(np.median(rho_k)), p_perm_two_sided=np.nan, boot_lo=float(np.percentile(rho_k, 2.5)), boot_hi=float(np.percentile(rho_k, 97.5)),
                spearman_between_the_two=float(np.mean(rho_k <= rho_k[0])), frame0_rho=float(rho_k[0])))   # spearman_between_the_two 列在此行里 = 首帧 ρ 在这 K 个帧的 ρ 中的百分位（≤ 首帧的比例）
desc = dict(item="descriptive medians over the 55 domains", n=n, median_ca_rmsd_AF_start_to_BioEmu_frame0_nm=float(D.ca_rmsd_AF_start_to_BioEmu_frame0_nm.median()), min_ca_rmsd_nm=float(D.ca_rmsd_AF_start_to_BioEmu_frame0_nm.min()), max_ca_rmsd_nm=float(D.ca_rmsd_AF_start_to_BioEmu_frame0_nm.max()))
for k in KEYS:
    desc[f"median_g_median_{k}"] = float(D[f"g_median_{k}"].median()); desc[f"median_g_sd_{k}"] = float(D[f"g_sd_{k}"].median()); desc[f"median_g_min_{k}"] = float(D[f"g_min_{k}"].median())
desc["median_bioemu_frame0_g_RT"] = float(D.bioemu_frame0_g_RT.median()); out.append(desc)
S = pd.DataFrame(out); f2 = os.path.join(TAB, "S10_start_structure_summary.csv"); S.to_csv(f2, index=False); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 90)
print(S[["item", "n", "spearman_rho_with_dG", "p_perm_two_sided", "boot_lo", "boot_hi", "spearman_between_the_two"]].to_string(index=False, float_format=lambda v: f"{v:.4f}")); print(S.iloc[-1].dropna().to_string())
log.finish([f1, f2], note=f"domains={n}")
