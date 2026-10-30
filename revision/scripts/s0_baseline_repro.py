"""Stage 0 — 基线复现（阳性对照 C-repro，revision_proposal.md §3 Stage 0 步骤 4）。

只读现有 npz 与结果表，按原脚本的逻辑独立复现原稿数字，逐项与稿件数值比对。
确定性条目要求 abs_diff ≤ 0.0005（p 值按相对差 ≤1%）；随机条目按原脚本种子复现，并另报
“随机种子漂移”；需要 pdbfixer 重建的条目只从已存结果表读取（S4a 才重新重建）。
输出：tables/S0_baseline_repro.csv（item, manuscript_value, reproduced_value, abs_diff, source_file, script_line, status, note）
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revstats, revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
R_GAS, T = 1.987204e-3, 298.15; RT = R_GAS * T
rows = []


def rec(item, ms, rp, src, line, kind="det", note="", tol=None):
    """kind: det（确定性，≤0.0005）| pval（相对差 ≤1%）| int（整数精确）| rand（随机，按 tol）| info（仅记录）"""
    if rp is None or (isinstance(rp, float) and np.isnan(rp)):
        st = "not_reproducible_from_disk"; ad = np.nan
    else:
        ad = abs(float(rp) - float(ms)) if ms is not None else np.nan
        if kind == "det":
            st = "PASS" if ad <= (tol if tol is not None else 0.0005) else "MISMATCH"
        elif kind == "round3":     # 稿件以 3 位小数报告：允许舍入误差 0.0005
            st = "PASS" if ad <= 0.0005 + 1e-9 else "MISMATCH"
        elif kind == "round2":
            st = "PASS" if ad <= 0.005 + 1e-9 else "MISMATCH"
        elif kind == "round1":
            st = "PASS" if ad <= 0.05 + 1e-9 else "MISMATCH"
        elif kind == "pval":
            st = "PASS" if (ms == 0 and rp == 0) or ad / max(abs(ms), 1e-300) <= 0.01 or round(rp, len(str(ms).split(".")[-1])) == ms else "MISMATCH"
        elif kind == "int":
            st = "PASS" if int(round(rp)) == int(ms) else "MISMATCH"
        elif kind == "rand":
            st = "PASS" if ad <= (tol or 0.0005) else ("within_seed_noise" if ad <= 5 * (tol or 0.0005) else "MISMATCH")
        else:
            st = "info"
    rows.append(dict(item=item, manuscript_value=ms, reproduced_value=None if rp is None else float(rp),
                     abs_diff=ad, source_file=src, script_line=line, status=st, note=note))


log = revio.RunLogger(stage=0, task="baseline_repro", cpu_procs=1, seeds="orig:0/7/42")

# ======================== K1：残基层 系综 vs 单结构 ========================
exp = pd.read_parquet(INP["tier1a_all"])
maps = json.load(open(INP["structure_maps"]))
ceil = pd.read_parquet(INP["operator_ceiling"]); ceilsp = dict(zip(ceil.protein, ceil.spearman))
nb = pd.read_csv(os.path.join(INP["results_tables"], "null_baselines.csv"))

# (a) 从 eval/pred/BioEmu/*.csv 独立重算逐蛋白系综 Spearman（原 42_null_baselines.py:19–25 的对齐规则）
obs = {}
for p in sorted(exp.protein_id.unique()):
    f = os.path.join(INP["eval_pred_dir"], "BioEmu", f"{p}.csv")
    pred = pd.read_csv(f); g = exp[exp.protein_id == p]
    off = bv.boff_null42({int(r.resSeq): bv.to1(r.resn) for _, r in pred.iterrows()}, g, maps)
    pred = pred.copy(); pred["resi"] = pred.resSeq - off
    m = g.merge(pred, on="resi", how="inner").dropna(subset=["dGopen_kcal", "dGopen_pred"])
    obs[p] = (stats.spearmanr(m.dGopen_pred.values, m.dGopen_kcal.values)[0], len(m), off)
# (b) 同一数值也可从 operator_f.npz 直接重算（count_avg，与 eval/pred 必须一致 → 检验 npz 与 csv 同源）
npz_rho = {}
for p in obs:
    d = np.load(os.path.join(INP["gen_dir"], "BioEmu", p, "operator_f.npz"))
    lnpf = bv.aggregate(d["NC"], d["NH"], "count_avg"); g = exp[exp.protein_id == p]
    x, y, _ = bv.align(d["resSeq"], RT * lnpf, g, obs[p][2])
    npz_rho[p] = stats.spearmanr(x, y)[0]

ST7 = {  # 稿件补充表 7（系综 ρ、单结构 ρ、Δ、n）
    "Im7": (0.542, 0.174, 0.368, 25), "cytochrome_c": (0.630, 0.508, 0.122, 48), "RNaseA": (0.380, 0.302, 0.078, 25),
    "RNaseH_Tt": (0.357, 0.287, 0.070, 43), "SNase": (0.722, 0.655, 0.067, 101), "Ubiquitin": (0.656, 0.590, 0.066, 41),
    "b-lactoglobulin": (0.564, 0.513, 0.051, 80), "Barnase": (0.605, 0.616, -0.011, 41), "CI2": (-0.156, -0.346, 0.190, 14),
    "UCH-L1": (0.233, 0.109, 0.124, 51), "Barstar": (-0.011, -0.107, 0.096, 15), "CheY": (0.160, 0.231, -0.071, 38)}
for p, (e_ms, s_ms, d_ms, n_ms) in ST7.items():
    rec(f"ST7 ensemble rho {p}", e_ms, obs[p][0], "eval/pred/BioEmu/*.csv", "42_null_baselines.py:19-25", "round3")
    rec(f"ST7 ensemble rho {p} (from operator_f.npz)", e_ms, npz_rho[p], "gen/BioEmu/*/operator_f.npz", "aggregate(count_avg)", "round3",
        note="npz 与 eval/pred 同源核对")
    rec(f"ST7 single-structure rho {p}", s_ms, ceilsp[p], "results/operator_ceiling.parquet", "15_operator_ceiling.py:22", "round3")
    rec(f"ST7 delta {p}", d_ms, round(obs[p][0], 3) - round(ceilsp[p], 3), "null_baselines.csv 规则（3 位小数相减）", "42:30", "round3")
    rec(f"ST7 n_res {p}", n_ms, obs[p][1], "eval/pred/BioEmu", "42:23", "int")

o = nb.obs_spearman.values; c = nb.single_struct_ceiling.values; dlt = nb.delta_vs_ceiling.values
rec("K1 one-sided paired Wilcoxon p", 0.0046, revstats.wilcoxon_greater(o, c)[1], "null_baselines.csv", "42:43 / Fig2.py:35", "pval")
rec("K1 n ensemble > single", 10, int((dlt > 0).sum()), "null_baselines.csv", "Fig2.py:39", "int")
rec("K1 median delta", 0.074, float(np.median(dlt)), "null_baselines.csv", "Fig2.py:39", "round3")
rec("K1 mean delta", 0.096, float(np.mean(dlt)), "null_baselines.csv", "manuscript l.159", "round3")
imax = int(np.argmax(dlt)); mk = np.arange(len(nb)) != imax
rec("K1 drop largest-gain (Im7) Wilcoxon p", 0.0093, revstats.wilcoxon_greater(o[mk], c[mk])[1], "null_baselines.csv", "Fig2.py:37-38", "pval",
    note=f"去掉的是 {nb.protein.iloc[imax]}")
rec("K1 drop Im7: n improved", 9, int((dlt[mk] > 0).sum()), "null_baselines.csv", "manuscript l.165", "int")
rec("K1 sign test 10/12 p (manuscript reports for 9/11 context)", 0.019, revstats.sign_test_greater(dlt)[2], "null_baselines.csv", "Fig2.py:36", "pval",
    note="稿件 l.165 把它写在“去掉最大增益后”语境下；R2-m2 指出 9/11 应为 0.033")
rec("K1 sign test 9/11 p (reviewer R2-m2 value)", 0.033, revstats.sign_test_greater(dlt[mk])[2], "null_baselines.csv", "binomtest", "pval")
# BH：用存档的 perm_p 重算 BH（置换 p 本身是随机的，按原种子才能逐位复现）
bhq = revstats.bh_like_null42(nb.perm_p.values)
rec("K1 BH q<0.05 count", 8, int((bhq < 0.05).sum()), "null_baselines.csv perm_p", "42:34-37", "int")
rec("K1 BH q recomputed vs stored (max abs diff)", 0.0, float(np.max(np.abs(np.round(bhq, 4) - nb.perm_BH.values))), "null_baselines.csv", "42:37", "det")

# (c) 置换 p 的原种子复现（42_null_baselines.py：default_rng(0)，按蛋白排序依次消耗 5000 次置换 + 2000 次 bootstrap）
rng = np.random.default_rng(0); B = 5000; perm_rep = {}
for p in sorted(exp.protein_id.unique()):
    f = os.path.join(INP["eval_pred_dir"], "BioEmu", f"{p}.csv")
    pred = pd.read_csv(f); g = exp[exp.protein_id == p]
    off = bv.boff_null42({int(r.resSeq): bv.to1(r.resn) for _, r in pred.iterrows()}, g, maps)
    pred = pred.copy(); pred["resi"] = pred.resSeq - off
    m = g.merge(pred, on="resi", how="inner").dropna(subset=["dGopen_kcal", "dGopen_pred"])
    x = m.dGopen_pred.values; y = m.dGopen_kcal.values; ob = stats.spearmanr(x, y)[0]
    nul = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(B)])
    pp = (np.sum(nul >= ob) + 1) / (B + 1)
    _ = np.array([stats.spearmanr(*((lambda i: (x[i], y[i]))(rng.integers(0, len(x), len(x)))))[0] for _ in range(2000)])
    perm_rep[p] = pp
stored = dict(zip(nb.protein, nb.perm_p))
rec("K1 perm_p exact reseeded reproduction (max abs diff over 12)", 0.0,
    float(max(abs(perm_rep[p] - stored[p]) for p in stored)), "42_null_baselines.py default_rng(0)", "42:18-27", "det", tol=1e-12)

# ======================== 绝对标定（K2） ========================
cal = pd.read_csv(os.path.join(INP["results_stats"], "stage1_calibration_crystal.csv"))
rep = pd.read_csv(os.path.join(INP["results_stats"], "stage1_reproducibility.csv"))
for p, v in {"SNase": 0.717, "Barnase": 0.624, "cytochrome_c": 0.500, "RNaseA": 0.294, "CI2": -0.483}.items():
    rec(f"ST1 per-protein Pearson {p}", v, float(cal[cal.protein == p]["pearson_defaultβ"].iloc[0]),
        "results/stats/stage1_calibration_crystal.csv", "10_stage1_calibration.py:27", "round3", note="单次重建（稿件 ST1 列）")
for p, v in {"Barnase": 0.015, "RNaseA": 0.034, "SNase": 0.007, "CI2": 0.063, "cytochrome_c": 0.026}.items():
    rec(f"ST1 recon SD {p}", v, float(rep[rep.protein == p].pearson_sd.iloc[0]), "stage1_reproducibility.csv", "16:47", "round3")
rec("Pooled Pearson (mean over 10 recon)", 0.581, float(rep[rep.protein == "POOLED"].pearson_mean.iloc[0]), "stage1_reproducibility.csv", "16:39", "round3")
rec("Pooled Pearson SD", 0.007, float(rep[rep.protein == "POOLED"].pearson_sd.iloc[0]), "stage1_reproducibility.csv", "16:48", "round3")
rec("LOPO Pearson", 0.327, float(rep[rep.protein == "LOPO"].pearson_mean.iloc[0]), "stage1_reproducibility.csv", "16:45", "round3")
rec("Mean of per-protein Pearson (R2-C2: 0.330)", 0.330,
    float(np.mean([0.717, 0.624, 0.500, 0.294, -0.483])), "ST1 数值", "R2-C2", "round3", note="审稿人算法：5 个逐蛋白 r 简单平均")
pa = json.load(open(os.path.join(INP["operator_params"], "A_crystal.json")))
rec("A_crystal.json betaC no-intercept", None, pa["betaC_noint"], "operator/params/A_crystal.json", "10:52", "info")
rec("A_crystal.json betaH no-intercept", None, pa["betaH_noint"], "operator/params/A_crystal.json", "10:52", "info")
rec("A_crystal.json betaC with-intercept", None, pa["betaC_int"], "operator/params/A_crystal.json", "10:53", "info")
rec("A_crystal.json betaH with-intercept", None, pa["betaH_int"], "operator/params/A_crystal.json", "10:53", "info")

# ======================== 敏感性网格（ST6） ========================
bs = pd.read_csv(os.path.join(INP["results_stats"], "stage1_beta_sensitivity.csv"))
ST6 = [(0.55, 2, 0.24, 0.370, 2.212, 0.568), (0.55, 2, 0.26, 0.369, 1.677, 0.554), (0.55, 3, 0.24, 0.336, 2.507, 0.545),
       (0.55, 3, 0.26, 0.338, 2.036, 0.538), (0.65, 2, 0.24, 0.230, 2.111, 0.599), (0.65, 2, 0.26, 0.233, 1.744, 0.597),
       (0.65, 3, 0.24, 0.215, 2.409, 0.580), (0.65, 3, 0.26, 0.221, 2.073, 0.584), (0.80, 2, 0.24, 0.116, 2.510, 0.600),
       (0.80, 2, 0.26, 0.119, 2.165, 0.604), (0.80, 3, 0.24, 0.113, 2.632, 0.593), (0.80, 3, 0.26, 0.118, 2.328, 0.603)]
for cnc, ij, cnh, bc, bh_, r in ST6:
    row = bs[(np.isclose(bs.cut_Nc, cnc)) & (bs.ij_Nc == ij) & (np.isclose(bs.cut_Nh, cnh))].iloc[0]
    tag = f"{cnc}/{ij}/{cnh}"
    rec(f"ST6 betaC {tag}", bc, row.betaC, "stage1_beta_sensitivity.csv", "11:34", "round3")
    rec(f"ST6 betaH {tag}", bh_, row.betaH, "stage1_beta_sensitivity.csv", "11:34", "round3")
    rec(f"ST6 pooled Pearson {tag}", r, row.pooled_pearson, "stage1_beta_sensitivity.csv", "11:34", "round3")
rec("ST6 betaH CV (manuscript '~14%')", 0.14, float(bs.betaH.std() / bs.betaH.mean()), "stage1_beta_sensitivity.csv", "11:36 (pandas std ddof=1)", "round2")
rec("ST6 betaC CV (R2-C3: 44%)", 0.44, float(bs.betaC.std() / bs.betaC.mean()), "stage1_beta_sensitivity.csv", "ddof=1", "round2")
rec("ST6 betaH min (R2-C3: 1.677)", 1.677, float(bs.betaH.min()), "stage1_beta_sensitivity.csv", "", "round3")
rec("ST6 betaH max (R2-C3: 2.632)", 2.632, float(bs.betaH.max()), "stage1_beta_sensitivity.csv", "", "round3")
rec("ST6 n cells betaH>=2.5 (R2-C3: 3 of 12)", 3, int((bs.betaH >= 2.5).sum()), "stage1_beta_sensitivity.csv", "", "int")
_mx = float(max(bs.betaH.max(), pa["betaH_noint"], pa["betaH_int"]))
rows.append(dict(item="Manuscript 'betaH 2.5-3.1' upper bound 3.1", manuscript_value=3.1, reproduced_value=_mx,
                 abs_diff=abs(_mx - 3.1), source_file="stage1_beta_sensitivity.csv + A_crystal.json", script_line="",
                 status="confirms_reviewer_R2-C3",
                 note="磁盘上任何拟合（12 格网格 + A_crystal 有/无截距）最大 βH=2.882，达不到 3.1（F9）；属审稿人指出的错误，S6 追溯并给替换值"))

# ======================== LOPO 机器学习（ST2） ========================
lo = pd.read_csv(os.path.join(INP["results_tables"], "learned_operator_lopo.csv")); lod = dict(zip(lo.model, lo.lopo_spearman))
for k, v in {"BV": 0.390, "Ridge[NC,NH]": 0.382, "Ridge[full]": 0.392, "RF[full]": 0.387, "GBM[full]": 0.403}.items():
    rec(f"ST2 LOPO {k}", v, lod[k], "results/tables/learned_operator_lopo.csv", "40:73", "round3")
# 原脚本只把 CI 打印到 stdout、未落盘 → 按 40_learned_operator_lopo.py 逐行逻辑重跑（不写原项目目录）
import mdtraj as md
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8,
      'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}


def _best_off40(pred_res, g):  # 40_learned_operator_lopo.py:11–18
    eid = {int(r.resi): bv.to1(r.resn) for _, r in g.iterrows() if str(r.resn).strip()}
    if len(eid) < 5:
        return maps.get(g.protein_id.iloc[0], {}).get('gen_offset', 0)
    best, bg = 0, -1
    for o_ in range(-8, 9):
        gg = sum(1 for er, aa in eid.items() if pred_res.get(er + o_) == aa)
        if gg > bg:
            bg, best = gg, o_
    return best


def _feats40(prot):  # 40_learned_operator_lopo.py:19–43（逐行等价）
    base = os.path.join(INP["gen_dir"], "BioEmu", prot)
    d = np.load(os.path.join(base, "operator_f.npz")); resSeq = d["resSeq"].astype(int); NC = d["NC"]; NH = d["NH"]
    pred = pd.read_csv(os.path.join(INP["eval_pred_dir"], "BioEmu", f"{prot}.csv"))
    resn = {int(r.resSeq): bv.to1(r.resn) for _, r in pred.iterrows()}
    lnPF = 0.35 * NC + 2.0 * NH
    F = dict(NC_mean=np.nanmean(NC, 1), NC_std=np.nanstd(NC, 1), NC_p10=np.nanpercentile(NC, 10, 1), NC_p90=np.nanpercentile(NC, 90, 1),
             NH_mean=np.nanmean(NH, 1), NH_std=np.nanstd(NH, 1), NH_p90=np.nanpercentile(NH, 90, 1),
             fopen=np.nanmean(lnPF < np.nanmedian(lnPF), 1), NCxNH=np.nanmean(NC * NH, 1), lnPF_std=np.nanstd(lnPF, 1))
    try:
        t = md.load(os.path.join(base, "samples.xtc"), top=os.path.join(base, "topology.pdb"))
        t = t[np.linspace(0, t.n_frames - 1, min(50, t.n_frames)).astype(int)]
        sasa = md.shrake_rupley(t, mode='residue').mean(0); dssp = md.compute_dssp(t, simplified=True)
        rs2i = {r.resSeq: i for i, r in enumerate(t.topology.residues)}
        F['SASA'] = np.array([sasa[rs2i[r]] if r in rs2i else np.nan for r in resSeq])
        F['fhelix'] = np.array([np.mean(dssp[:, rs2i[r]] == 'H') if r in rs2i else np.nan for r in resSeq])
        F['fsheet'] = np.array([np.mean(dssp[:, rs2i[r]] == 'E') if r in rs2i else np.nan for r in resSeq])
    except Exception:
        for k_ in ['SASA', 'fhelix', 'fsheet']:
            F[k_] = np.full(len(resSeq), np.nan)
    F['KD'] = np.array([KD.get(resn.get(int(r), 'X'), 0.0) for r in resSeq])
    df = pd.DataFrame(F); df['resSeq'] = resSeq; df['BV'] = 0.35 * F['NC_mean'] + 2.0 * F['NH_mean']
    g = exp[exp.protein_id == prot]
    if len(g) == 0:   # ecDHFR 无残基级 ΔG_open：原脚本此处 iloc[0] 越界；其结果本应被 len(x)>=8 过滤，跳过等价
        return None
    off = _best_off40(resn, g); df['resi'] = df.resSeq - off
    m = df.merge(g[['resi', 'dGopen_kcal']], on='resi', how='inner').dropna(subset=['dGopen_kcal'])
    m['protein'] = prot; return m


prots40 = [os.path.basename(dd) for dd in glob.glob(os.path.join(INP["gen_dir"], "BioEmu", "*")) if os.path.exists(os.path.join(dd, "operator_f.npz"))]
data40 = pd.concat([x for x in (_feats40(p) for p in sorted(prots40)) if x is not None and len(x) >= 8], ignore_index=True)
FEATS = ['NC_mean', 'NC_std', 'NC_p10', 'NC_p90', 'NH_mean', 'NH_std', 'NH_p90', 'fopen', 'NCxNH', 'lnPF_std', 'SASA', 'fhelix', 'fsheet', 'KD']
data40[FEATS] = data40[FEATS].fillna(data40[FEATS].median())


def _lopo40(model_fn, cols):
    sp = []
    for p in sorted(data40.protein.unique()):
        tr = data40[data40.protein != p]; te = data40[data40.protein == p]
        if len(te) < 5:
            continue
        mdl = model_fn(); mdl.fit(tr[cols], tr.dGopen_kcal); sp.append(stats.spearmanr(mdl.predict(te[cols]), te.dGopen_kcal)[0])
    return np.nanmean(sp), sp


bv40 = [stats.spearmanr(data40[data40.protein == p].BV, data40[data40.protein == p].dGopen_kcal)[0]
        for p in sorted(data40.protein.unique()) if len(data40[data40.protein == p]) >= 5]
res40 = {}
for name, fn, cols in [("Ridge[NC,NH]", lambda: Ridge(alpha=1.0), ['NC_mean', 'NH_mean']), ("Ridge[full]", lambda: Ridge(alpha=1.0), FEATS),
                       ("RF[full]", lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=4), FEATS),
                       ("GBM[full]", lambda: GradientBoostingRegressor(random_state=0), FEATS)]:
    res40[name] = _lopo40(fn, cols)
best40 = max(res40, key=lambda k_: res40[k_][0])
rng40 = np.random.default_rng(0); diffs40 = np.array(res40[best40][1]) - np.array(bv40)
boot40 = [np.mean(rng40.choice(diffs40, len(diffs40))) for _ in range(10000)]
rec("ST2 LOPO BV rerun", 0.390, float(np.nanmean(bv40)), "rerun of 40_learned_operator_lopo.py", "40:57", "round3")
for k_, v_ in {"Ridge[NC,NH]": 0.382, "Ridge[full]": 0.392, "RF[full]": 0.387, "GBM[full]": 0.403}.items():
    rec(f"ST2 LOPO {k_} rerun", v_, float(res40[k_][0]), "rerun of 40_learned_operator_lopo.py", "40:60-64", "round3")
# RF[full]：在 rev / 原 operator / probe 三个环境中重跑结果逐位相同（0.3860001815，n_jobs=1/4、蛋白顺序正/逆均不变），
# 与存档 0.387 差 0.001。存档 csv 写于 06-20 22:10，而磁盘上当前版本的脚本遇到当时已存在的 ecDHFR npz（21:35）会越界报错，
# 说明存档值来自一个已不在磁盘上的代码变体。该模型不是最优模型；最优模型 GBM 0.403、配对 bootstrap CI、特征重要性均精确复现，
# 对 K3 无影响 → 按 §0.3 规则 2 保留原稿数字，仅登记。
for r_ in rows:
    if r_["item"] == "ST2 LOPO RF[full] rerun" and r_["status"] == "MISMATCH":
        r_["status"] = "minor_irreproducible_no_conclusion_impact"
        r_["note"] = ("三环境重跑逐位一致=0.3860；存档 0.387 来自已不在磁盘上的代码变体（当前脚本在存档时刻会因 ecDHFR 越界报错）。"
                      "非最优模型，GBM/CI/重要性精确复现，K3 不受影响，保留原稿数字（§0.3 规则 2）")
rec("ST2 best model is GBM", 1, int(best40 == "GBM[full]"), "rerun", "40:65", "int", note=f"best={best40}")
rec("ST2 best-vs-BV delta (+0.013)", 0.013, float(res40[best40][0] - np.nanmean(bv40)), "rerun", "40:69", "round3")
rec("ST2 paired-bootstrap CI low (-0.059)", -0.059, float(np.percentile(boot40, 2.5)), "rerun default_rng(0)", "40:67-69", "round3")
rec("ST2 paired-bootstrap CI high (+0.085)", 0.085, float(np.percentile(boot40, 97.5)), "rerun default_rng(0)", "40:67-69", "round3")
rf40 = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=4).fit(data40[FEATS], data40.dGopen_kcal)
imp40 = dict(zip(FEATS, rf40.feature_importances_))
rec("ST2 RF importance NCxNH (0.283)", 0.283, float(imp40["NCxNH"]), "rerun", "40:70-72", "round3")
rec("ST2 RF importance SASA (0.10)", 0.10, float(imp40["SASA"]), "rerun", "40:70-72", "round2")
rec("ST2 LOPO data n_residues (522)", 522, int(len(data40)), "rerun", "40:45", "int")
os.makedirs(os.path.join(REV, "tables"), exist_ok=True)
pd.DataFrame([dict(model="BV", lopo_spearman=float(np.nanmean(bv40)), per_protein=json.dumps([float(x) for x in bv40]))] +
             [dict(model=k_, lopo_spearman=float(v_[0]), per_protein=json.dumps([float(x) for x in v_[1]])) for k_, v_ in res40.items()]
             ).to_csv(os.path.join(REV, "tables", "S0_lopo_rerun.csv"), index=False)

# ======================== 五类生成器（K4） ========================
lb = pd.read_parquet(INP["leaderboard"])
ORDER = ["BioEmu", "ESMFlow", "Str2Str", "ShortMD", "Boltz2"]
piv = lb.pivot(index="protein", columns="model", values="spearman")[ORDER]
for k, v in {"BioEmu": 0.39, "ESMFlow": 0.38, "Str2Str": 0.36, "ShortMD": 0.33, "Boltz2": 0.23}.items():
    rec(f"Five-paradigm mean Spearman {k}", v, float(piv[k].mean()), "eval/leaderboard.parquet", "Fig4.py:29", "round2")
chi2, pF = stats.friedmanchisquare(*[piv[cc].values for cc in ORDER])
rec("Friedman chi2", 11.6, float(chi2), "eval/leaderboard.parquet", "Fig4.py:41", "round1")
rec("Friedman p", 0.021, float(pF), "eval/leaderboard.parquet", "Fig4.py:41", "pval")
rec("Wilcoxon BioEmu vs ESMFlow p", 0.424, float(stats.wilcoxon(piv["BioEmu"], piv["ESMFlow"])[1]), "leaderboard", "Fig4.py:43", "pval")
rec("Wilcoxon BioEmu vs Str2Str p", 0.181, float(stats.wilcoxon(piv["BioEmu"], piv["Str2Str"])[1]), "leaderboard", "Fig4.py:43", "pval")
rec("TOST ESMFlow vs BioEmu (+-0.05)", 0.043, revstats.tost_paired_t((piv["ESMFlow"] - piv["BioEmu"]).values), "leaderboard", "Fig4.py:TOST", "pval")
# leaderboard 与 npz 同源核对（每生成器×蛋白的 count_avg Spearman）
mx = 0.0
for gname in ORDER:
    for p in piv.index:
        f = os.path.join(INP["gen_dir"], gname, p, "operator_f.npz")
        d = np.load(f); g = exp[exp.protein_id == p]
        pr = pd.read_csv(os.path.join(INP["eval_pred_dir"], gname, f"{p}.csv"))
        off = bv.best_offset({int(r.resSeq): bv.to1(r.resn) for _, r in pr.iterrows()}, g,
                             maps.get(p, {}).get("gen_offset", maps.get(p, {}).get("offset", 0)))
        x, y, _ = bv.align(d["resSeq"], RT * bv.aggregate(d["NC"], d["NH"], "count_avg"), g, off)
        mx = max(mx, abs(round(stats.spearmanr(x, y)[0], 3) - piv.loc[p, gname]))
rec("leaderboard vs operator_f.npz Spearman (max abs diff, 60 cells)", 0.0, mx, "eval/leaderboard + gen/*/operator_f.npz", "22_leaderboard.py:36-40", "round3")

# matched-50（Fig4.py:65–91，default_rng(42)，REPS=50，gen_offset 规则）
rng_m = np.random.default_rng(42); m50 = {}
for gname in ORDER:
    per = []
    for npz in sorted(glob.glob(os.path.join(INP["gen_dir"], gname, "*", "operator_f.npz"))):
        p = npz.split("/")[-2]; d = np.load(npz); NC, NH, rs = d["NC"], d["NH"], d["resSeq"]; F = NC.shape[1]
        e = exp[exp.protein_id == p]; off = maps.get(p, {}).get("gen_offset", maps.get(p, {}).get("offset", 0))
        emap = dict(zip(e.resi.astype(int), e.dGopen_kcal))
        if not emap:
            continue
        n = min(50, F); srs = []
        for _ in range(50):
            idx = rng_m.choice(F, n, replace=False)
            dg = R_GAS * T * (0.35 * np.nanmean(NC[:, idx], 1) + 2.0 * np.nanmean(NH[:, idx], 1))
            prd = {int(rs[i]) - off: dg[i] for i in range(len(rs))}
            pairs = [(prd[r], emap[r]) for r in emap if r in prd]
            if len(pairs) >= 5:
                a = np.array(pairs); srs.append(stats.spearmanr(a[:, 0], a[:, 1])[0])
        if srs:
            per.append(np.mean(srs))
    m50[gname] = float(np.mean(per))
    rec(f"matched-50 mean Spearman {gname}", None, m50[gname], "Fig4.py:65-91 default_rng(42)", "Fig4.py", "info",
        note="原稿只报告“排序保持”，无具体数字；登记供 S7 可比性核对")
# 偶然发现（S0 复现中）：Fig4.py matched-50 用 structure_maps 的 gen_offset 对齐，而主排行榜（22_leaderboard.py）
# 用按残基身份的 best_offset。两者只在 ShortMD×CI2 不同（gen_offset=0, best_offset=+1；ShortMD 在 2CI2 晶体上模拟，
# 编号比成熟序列多 1，即 WORKLOG 记载的 CI2 offset bug 在 Fig4 inset 中的残留）。R2-m7 正是质疑这条“排序保持”，
# 因此 S9 步骤 7 统一用与主排行榜相同的 best_offset 重算 matched 排序；此处只登记差异。
_d = np.load(os.path.join(INP["gen_dir"], "ShortMD", "CI2", "operator_f.npz")); _e = exp[exp.protein_id == "CI2"]
_ln = bv.aggregate(_d["NC"], _d["NH"], "count_avg")
_x0, _y0, _ = bv.align(_d["resSeq"], _ln, _e, 0); _x1, _y1, _ = bv.align(_d["resSeq"], _ln, _e, 1)
rows.append(dict(item="Incidental: Fig4 matched-50 ShortMD×CI2 alignment (gen_offset 0 vs best_offset +1)",
                 manuscript_value=float(piv.loc["CI2", "ShortMD"]), reproduced_value=float(stats.spearmanr(_x0, _y0)[0]),
                 abs_diff=abs(float(stats.spearmanr(_x0, _y0)[0]) - float(piv.loc["CI2", "ShortMD"])),
                 source_file="Fig4.py matched-50 vs eval/leaderboard", script_line="Fig4.py:72 gen_offset vs 22_leaderboard.py:36 best_offset",
                 status="incidental_finding_R2-m7",
                 note=f"全帧 count_avg：offset0 ρ={stats.spearmanr(_x0, _y0)[0]:.3f}，offset+1 ρ={stats.spearmanr(_x1, _y1)[0]:.3f}（=leaderboard）。"
                      "只影响原稿未给数字的 matched-50 inset；S9 用 best_offset 重算"))

# ======================== 收敛（K6） ========================
sc = pd.read_csv(os.path.join(INP["results_tables"], "scale_BioEmu.csv"))
u = sc[sc.protein == "Ubiquitin"]
rec("Convergence Ubiquitin rho at n=50", 0.65, float(u[u.n_conf == 50].spearman_mean.iloc[0]), "results/tables/scale_BioEmu.csv", "24:28", "round2")
rec("Convergence Ubiquitin rho at full n=489", 0.66, float(u[u.n_conf == u.n_conf.max()].spearman_mean.iloc[0]), "scale_BioEmu.csv", "24:32", "round2",
    note=f"full n={int(u.n_conf.max())}")

# ======================== 全局（K5） ========================
sub = json.load(open(INP["tier3_subset"]))
t3 = pd.read_parquet(INP["tier3_40_72"]); ORIG = dict(zip(t3.name, t3.origin))


def loadg(model):
    out = []
    for npz in glob.glob(os.path.join(INP["gen_dir"], model, "*", "operator_f.npz")):
        did = os.path.basename(os.path.dirname(npz))
        if did not in sub:
            continue
        d = np.load(npz); g = np.nanmean(0.35 * d["NC"] + 2.0 * d["NH"], axis=0)
        out.append(dict(domain=did, origin=ORIG.get(sub[did]["name"], "?"), exp=sub[did]["deltaG"], g=g,
                        ming=RT * np.min(g), softmin=-RT * (logsumexp(-g) - np.log(len(g))), mean=RT * np.nanmean(g)))
    return pd.DataFrame(out).sort_values("domain").reset_index(drop=True)


G = {m: loadg(m) for m in ["BioEmu_T3", "ESMFlow_T3", "Boltz2_T3", "Str2Str_T3", "Str2Str_T3_d035"]}
for m, v in {"BioEmu_T3": 0.617, "Boltz2_T3": -0.131, "Str2Str_T3": 0.261, "ESMFlow_T3": -0.032}.items():
    rec(f"e1_multicontrol soft-min Pearson {m}", v, float(stats.pearsonr(G[m].softmin, G[m].exp)[0]),
        "gen/*_T3/*/operator_f.npz", "45_e1_multicontrol.py:8-9", "round3")
d35 = G["Str2Str_T3_d035"]; d25 = G["Str2Str_T3"]
rec("Global obs_min Spearman delta0.35 (manuscript rho=0.334)", 0.334, float(stats.spearmanr(d35.ming, d35.exp)[0]), "Str2Str_T3_d035", "build_data.py:41-43", "round3")
rec("Global obs_min Pearson delta0.35 (manuscript 0.356)", 0.356, float(stats.pearsonr(d35.ming, d35.exp)[0]), "Str2Str_T3_d035", "", "round3")
rec("Global obs_min Spearman p delta0.35 (manuscript p=0.007)", 0.007, float(stats.spearmanr(d35.ming, d35.exp)[1]), "Str2Str_T3_d035", "", "pval",
    note="稿件把这个 p 写在 Pearson 0.36 后面（R2-m4）")
rec("Global obs_min Pearson p delta0.35 (R2-m4 ~0.004)", 0.004, float(stats.pearsonr(d35.ming, d35.exp)[1]), "Str2Str_T3_d035", "", "round3")
rec("Global obs_min Spearman delta0.25 (weaker control rho=0.29)", 0.286, float(stats.spearmanr(d25.ming, d25.exp)[0]), "Str2Str_T3", "", "round3")
rec("Global obs_min Spearman p delta0.25 (p=0.021)", 0.021, float(stats.spearmanr(d25.ming, d25.exp)[1]), "Str2Str_T3", "", "pval")
# jackknife（ST4 caption [0.303, 0.383]）
jk = [stats.spearmanr(np.delete(d35.ming.values, i), np.delete(d35.exp.values, i))[0] for i in range(len(d35))]
rec("Global jackknife LOO min (0.303)", 0.303, float(np.min(jk)), "Str2Str_T3_d035", "", "round3")
rec("Global jackknife LOO max (0.383)", 0.383, float(np.max(jk)), "Str2Str_T3_d035", "", "round3")
# 百分位扫描（ST4 caption 0.334→0.290→0.250→0.181→0.120→0.055）
for q, v in zip([0, 10, 25, 50, 75, 100], [0.334, 0.290, 0.250, 0.181, 0.120, 0.055]):
    rec(f"Global sweep q={q}", v, float(stats.spearmanr([RT * np.percentile(g, q) for g in d35.g], d35.exp)[0]),
        "Str2Str_T3_d035", "build_data.py:56-60 规则", "round3")
for sub_, v, pv in [("de_novo", 0.452, 0.011), ("natural", 0.225, None)]:
    dd = d35[d35.origin == sub_]
    r_, p_ = stats.spearmanr(dd.ming, dd.exp)
    rec(f"Global {sub_} Spearman (n={len(dd)})", v, float(r_), "Str2Str_T3_d035", "", "round3")
    if pv is not None:
        rec(f"Global {sub_} Spearman p uncorrected (manuscript 'p=0.011 after Bonferroni')", pv, float(p_), "Str2Str_T3_d035", "", "pval",
            note="R2-m5：0.011 实为未校正值")
        rec(f"Global {sub_} Spearman p Bonferroni x2 (R2-m5 ~0.021)", 0.021, float(min(1, 2 * p_)), "Str2Str_T3_d035", "", "det",
            tol=0.001, note="审稿人给的是近似值“≈0.021”；精确值见 reproduced_value（=2×未校正 p）")
rec("Global de_novo n", 31, int((d35.origin == "de_novo").sum()), "tier3", "", "int")
rec("Global natural n", 34, int((d35.origin == "natural").sum()), "tier3", "", "int")
be = G["BioEmu_T3"]
rec("Global BioEmu obs_min Spearman (manuscript 0.47)", 0.47, float(stats.spearmanr(be.ming, be.exp)[0]), "BioEmu_T3", "WORKLOG:536", "round2")
rec("Global BioEmu de_novo obs_min Spearman (manuscript 0.65)", 0.65, float(stats.spearmanr(be[be.origin == 'de_novo'].ming, be[be.origin == 'de_novo'].exp)[0]),
    "BioEmu_T3", "WORKLOG:541", "round2")
# 设计/天然不对称置换（SF7.py:89–96，default_rng(7)，先 boot_ci 后 perm_diff；按原调用顺序复现）
rng7 = np.random.default_rng(7)


def boot_ci(x, y, nb=10000):
    n = len(x); rr = np.empty(nb)
    for b in range(nb):
        idx = rng7.integers(0, n, n); rr[b] = stats.spearmanr(x[idx], y[idx]).statistic
    return np.nanpercentile(rr, [2.5, 97.5])


def perm_diff(dfm, nperm=10000):
    s = lambda dd: stats.spearmanr(dd.ming, dd.exp).statistic
    ob = s(dfm[dfm.origin == "de_novo"]) - s(dfm[dfm.origin == "natural"])
    mg = dfm.ming.values; ex = dfm.exp.values; ndn = int((dfm.origin == "de_novo").sum()); n = len(dfm); cnt = 0
    for _ in range(nperm):
        pr = rng7.permutation(n); dn = pr[:ndn]; na = pr[ndn:]
        if stats.spearmanr(mg[dn], ex[dn]).statistic - stats.spearmanr(mg[na], ex[na]).statistic >= ob:
            cnt += 1
    return ob, (cnt + 1) / (nperm + 1)


# SF7 读的是 figures/data 下的 csv（行序 = glob 顺序），必须按原文件行序复现
fd = INP["figures_data"]
for name, f, dms, pms in [("BioEmu", "fig5_bioemu_T3.csv", 0.45, 0.021), ("Str2Str", "fig5_str2str_d035_T3.csv", 0.23, 0.170)]:
    dfm = pd.read_csv(os.path.join(fd, f))
    for sb in ["de_novo", "natural"]:
        dd = dfm[dfm.origin == sb]; boot_ci(dd.ming.values, dd.exp.values)
    ob, pp = perm_diff(dfm)
    rec(f"Global asymmetry delta-rho {name}", dms, float(ob), f"figures/data/{f}", "SF7.py:89-96", "round2")
    rec(f"Global asymmetry permutation p {name}", pms, float(pp), f"figures/data/{f}", "SF7.py default_rng(7)", "rand", tol=0.0005)

# ======================== 输出 ========================
out = pd.DataFrame(rows)
os.makedirs(os.path.join(REV, "tables"), exist_ok=True)
fo = os.path.join(REV, "tables", "S0_baseline_repro.csv"); out.to_csv(fo, index=False)
summ = out.status.value_counts().to_dict()
print(out.to_string(index=False, max_colwidth=60))
print("\nSUMMARY", summ)
log.finish([fo], note=json.dumps(summ, ensure_ascii=False))
