"""Stage 4a · K1 与 K5 对"重打分噪声"的敏感性（描述性，不替代 S6-S10 的正式重算）。

目的：S4a 的 C-parity 只证明新旧打分在逐残基系综平均上一致；这里直接问：把旧打分换成新打分，
结论级统计量 K1（残基层：系综 vs 单结构）与 K5（全局：obs_min vs ΔG_fold）会不会改变。
全部使用真实的新打分与真实的重复打分，不加人为噪声。

K1（BioEmu × 12 蛋白，单侧配对 Wilcoxon，系综 ρ > 单结构 ρ）
  系综 ρ：旧 npz（复现原稿）；新打分按映射帧（与旧同帧）；新打分全部 xtc 帧。
  单结构 ρ：原稿的沉积结构基线（原项目 operator_ceiling.parquet 的 spearman 列，表中标签 deposited_structure_manuscript）；以及沉积结构 10 次独立重建各自的 ρ（k=0..9）。
  对每个"新系综 × 单结构"的组合报告 p、改善蛋白数、Δρ 中位数；对 10 次重建报告 p 的范围。
  编号偏移沿用原稿口径（口径 O，§0.3 规则 5），cyt c 的沉积结构偏移问题见 U-S1-1，这里不改。
K5（Str2Str_T3、Str2Str_T3_d035、BioEmu_T3 × 65 个锚点域）
  obs_min = RT · min_frames( mean_residues(0.35·N_C + 2·N_H) )；与 ΔG_fold 的 Spearman（双侧 p，与原稿一致）。
  对 旧 npz（全部旧列，复现原稿）/ 旧 npz（与新打分公共残基）/ 新打分（映射帧）/ 重复打分 1-3（映射帧；仅 Str2Str 两类有重复）报告。
用法：python s4_k_sensitivity.py
输出：tables/S4_k1_sensitivity.csv、tables/S4_k5_sensitivity.csv
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, revstats  # noqa: E402
import s4_postprocess as PP  # noqa: E402
import s4_repeat_noise as RN  # noqa: E402

RT = bv.R_GAS * bv.T_REF
PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=4, task="k_sensitivity", cpu_procs=1, seeds="none")
exp = pd.read_parquet(INP["tier1a_all"])
PAR = pd.read_csv(os.path.join(REV, "tables", "S4_parity.csv")).dropna(subset=["offset"])
OFF = PAR.groupby("protein").offset.first().astype(int).to_dict()
MP = pd.read_csv(os.path.join(REV, "tables", "S4_old_column_to_frame_map.csv"))
ceil = pd.read_parquet(INP["operator_ceiling"]); CEIL = dict(zip(ceil.protein, ceil.spearman)); CEIL_OFF = dict(zip(ceil.protein, ceil.offset.astype(int)))   # 沉积结构有自己的偏移（CI2 为 +1），不能用生成器的偏移
nb = pd.read_csv(os.path.join(INP["results_tables"], "null_baselines.csv"))
prots = sorted(OFF)


def rho_of(resseq, NC, NH, p, off=None):
    x, y, _ = bv.align(list(resseq), RT * bv.aggregate(NC, NH, "count_avg"), exp[exp.protein_id == p], OFF[p] if off is None else off)
    return float(stats.spearmanr(x, y)[0])


# ---------------------------------------------------------------- K1
ens = {"old_npz": {}, "new_matched": {}, "new_allframes": {}}
for p in prots:
    old = PP.load_old(INP, "BioEmu", p); new = PP.load_new(REV, "BioEmu", p)
    fr = MP[(MP.generator == "BioEmu") & (MP.protein == p)].sort_values("old_col").frame_idx.values
    common = [r for r in old["resSeq"] if r in new["NC"].index]; io = [old["resSeq"].index(r) for r in common]
    ens["old_npz"][p] = rho_of(common, old["NC"][io, :], old["NH"][io, :], p)
    ens["new_matched"][p] = rho_of(common, new["NC"].loc[common, fr].values, new["NH"].loc[common, fr].values, p)
    ens["new_allframes"][p] = rho_of(common, new["NC"].loc[common].values, new["NH"].loc[common].values, p)
single = {}
for p in prots:
    R = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet"))
    nc = R.pivot(index="resSeq", columns="recon_k", values="NC"); nh = R.pivot(index="resSeq", columns="recon_k", values="NH")
    single[p] = [rho_of(nc.index, nc[[k]].values, nh[[k]].values, p, CEIL_OFF[p]) for k in range(10)]
rows = []


def k1_row(scn_e, scn_s, e, s):
    e = np.asarray(e); s = np.asarray(s); d = e - s
    w = revstats.wilcoxon_greater(e, s)
    return dict(ensemble=scn_e, single_structure=scn_s, n_proteins=len(e), wilcoxon_greater_p=w[1], n_improved=int((d > 0).sum()), median_delta=float(np.median(d)), mean_delta=float(np.mean(d)))


ceil_v = [CEIL[p] for p in prots]
rows.append(dict(ensemble="manuscript null_baselines.csv", single_structure="deposited_structure_manuscript", n_proteins=12, wilcoxon_greater_p=revstats.wilcoxon_greater(nb.obs_spearman.values, nb.single_struct_ceiling.values)[1],
                 n_improved=int((nb.delta_vs_ceiling.values > 0).sum()), median_delta=float(np.median(nb.delta_vs_ceiling.values)), mean_delta=float(np.mean(nb.delta_vs_ceiling.values))))
for scn in ens:
    rows.append(k1_row(scn, "deposited_structure_manuscript", [ens[scn][p] for p in prots], ceil_v))
    ps = []
    for k in range(10):
        r = k1_row(scn, f"crystal_recon_k{k}", [ens[scn][p] for p in prots], [single[p][k] for p in prots]); rows.append(r); ps.append(r["wilcoxon_greater_p"])
    rows.append(dict(ensemble=scn, single_structure="crystal_recon_k0-9 (range of p)", n_proteins=12, wilcoxon_greater_p=float(np.max(ps)), n_improved=np.nan, median_delta=np.nan, mean_delta=np.nan,
                     note=f"p range {min(ps):.5f}-{max(ps):.5f}"))
K1 = pd.DataFrame(rows); f1 = os.path.join(REV, "tables", "S4_k1_sensitivity.csv"); K1.to_csv(f1, index=False)

# ---------------------------------------------------------------- K5
sub = json.load(open(INP["tier3_subset"]))
rows5 = []


def obs_min(NC, NH):
    g = np.nanmean(0.35 * np.asarray(NC, float) + 2.0 * np.asarray(NH, float), axis=0)
    return RT * float(np.min(g))


for g in ["Str2Str_T3_d035", "Str2Str_T3", "BioEmu_T3"]:
    doms = sorted(d for d in sub if os.path.isdir(os.path.join(INP["gen_dir"], g, d)))
    vals = {k: {} for k in ["old_all_cols", "old_common_res", "new_matched", "rep1", "rep2", "rep3", "new_allframes"]}; ex = {}
    for d in doms:
        old = PP.load_old(INP, g, d)
        if old is None:
            continue
        new = PP.load_new(REV, g, d); fr = MP[(MP.generator == g) & (MP.protein == d)].sort_values("old_col").frame_idx.values
        common = [r for r in old["resSeq"] if r in new["NC"].index]; io = [old["resSeq"].index(r) for r in common]
        ex[d] = sub[d]["deltaG"]
        vals["old_all_cols"][d] = obs_min(old["NC"], old["NH"])
        vals["old_common_res"][d] = obs_min(old["NC"][io, :], old["NH"][io, :])
        vals["new_matched"][d] = obs_min(new["NC"].loc[common, fr].values, new["NH"].loc[common, fr].values)
        vals["new_allframes"][d] = obs_min(new["NC"].loc[common].values, new["NH"].loc[common].values)
        if g.startswith("Str2Str"):
            for r in (1, 2, 3):
                Rr, _ = RN.load_rep(REV, r, g, d)
                nc = Rr.pivot(index="resSeq", columns="frame_idx", values="NC"); nh = Rr.pivot(index="resSeq", columns="frame_idx", values="NH")
                vals[f"rep{r}"][d] = obs_min(nc.loc[common, fr].values, nh.loc[common, fr].values)
    for k, v in vals.items():
        if len(v) < 10:
            continue
        ds = sorted(v); a = np.array([v[d] for d in ds]); e = np.array([ex[d] for d in ds]); s = stats.spearmanr(a, e)
        ref = np.array([vals["old_common_res"][d] for d in ds])
        rows5.append(dict(generator=g, scoring=k, n_domains=len(ds), spearman_rho=float(s[0]), spearman_p_two_sided=float(s[1]),
                          pearson_r=float(stats.pearsonr(a, e)[0]), spearman_with_old_obs_min_across_domains=float(stats.spearmanr(a, ref)[0]),
                          mean_abs_diff_vs_old_kcal=float(np.mean(np.abs(a - ref))), sd_obs_min_across_domains=float(np.std(a, ddof=1))))
K5 = pd.DataFrame(rows5); f2 = os.path.join(REV, "tables", "S4_k5_sensitivity.csv"); K5.to_csv(f2, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(K1[K1.single_structure.isin(["deposited_structure_manuscript", "crystal_recon_k0-9 (range of p)"]) | K1.ensemble.str.startswith("manuscript")].round(5).to_string(index=False))
print(K5.round(4).to_string(index=False))
log.finish([f1, f2], note=f"K1 rows={len(K1)} K5 rows={len(K5)}")
