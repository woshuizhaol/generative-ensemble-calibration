"""Stage 10 · 全局稳定性分析（revision_proposal.md Stage 10；R2-C4a/b/c/d、R2-m4/m5/m6）。

观测量（全部来自 S4 的逐构象计数，计数平均形式，β=0.35/2.0，RT 取 298.15 K：Tsuboyama 2023 的 ΔG 条件为 pH 7.4、298 K，见 tier3_criterion.md）：
  g(c) = 构象 c 内全部已打分残基的 0.35·N_C + 2.0·N_H 的均值；obs_min / obs_softmin / obs_q{1,5,10,25,50} / obs_mean；het_g = SD_c g(c)（ddof=0，e1 的 het_mean 定义）。
域集：D_new（S1 判据，177 个）、D_legacy65（原稿 65 个，衔接用）、D_common_AF193（有精确 AlphaFold 起点的 193 个域，五个当前生成器在同一组域上再报告一遍（偏离登记 D1-1））。
  Str2Str_AF 在 D_new 里只有 167 个域（10 个域只有亲本 AF 模型，按方案原预案不做 Str2Str_AF，见偏离登记 D1-1），n_expected 如实写在每一行。
生成器：BioEmu、ESMFlow、Boltz2、Str2Str_AF_d025、Str2Str_AF_d035（当前）；Str2Str_legacy_d025/d035（只在 D_legacy65，连续性对照）。
打分口径 scoring_basis：S4_rescored（S4a/S4b 统一重打分）与 old_npz（旧 operator_f.npz，仅 D_legacy65，用于复现原稿）。
样本量方案 sample_size_scheme：matched50（BioEmu：S4a 冻结索引，M=50、R=200）、allframes（BioEmu 全部可用帧）、native50（其余生成器本身的 50 帧）、oldcols（旧 npz 全部列）。
统计：每行同时给出 Spearman（置换 p，10⁵ 次，另附 scipy p）、Pearson（t 检验 p）、域层 bootstrap CI（B=10⁴）、留一 jackknife 区间；匹配样本量的行另给 R 次抽样的 ρ 中位数与 2.5/97.5 百分位，
  并以"ρ 最接近中位数的那次抽样"报告 p 与 CI。分层 de_novo / natural 的 p 另做 Bonferroni ×2；设计/天然不对称性 Δρ 用标签置换 10⁵ 次。
用法：python s10_global_stability.py run [--quick] [--specs a,b] [--sets D_new,…]     （--quick：B_perm=2000、B_boot=500，仅调试，输出加 _quick 后缀）
输出：tables/S10_global_all_generators.csv、S10_global_matched_M_draws.parquet、S10_bioemu_convergence.csv、S10_stratified_bonferroni.csv、S10_denovo_natural_asymmetry.csv、
      S10_heterogeneity_ordering.csv、S10_legacy_bridge.csv、S10_perm_negative_control.csv、S10_paired_delta_rho.csv、S10_domain_table.csv
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
import sys, json, argparse, time, itertools
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import logsumexp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, globalstats as GS  # noqa: E402
from lib.ensemble import np_load_verified  # noqa: E402
import s4_postprocess as PP  # noqa: E402

PROJ, REV, INP = revio.load_paths()
RT = bv.R_GAS * bv.T_REF
TAB = os.path.join(REV, "tables")
DRAW_DIR = os.path.join(REV, "raw_outputs", "s4", "draw_indices")
OBS = GS.OBS_NAMES
ALL3 = ["D_new", "D_legacy65", "D_common_AF193"]
EXTRA_SETS = ["D_new_dataset3"]   # 敏感性子集（docs/tier3_criterion.md §3）：D_new ∩ Dataset 3；只在显式 --sets D_new_dataset3 时运行，不进入默认输出
STRATA = ["all", "de_novo", "natural"]
TOL = 0.005
MS = dict(str2str_d035_rho=0.334, str2str_d035_pearson=0.356, str2str_d025_rho=0.286)   # 原稿正文数字（S0/F18 登记），仅作衔接表的参照值

SPECS = [
    dict(key="BioEmu|S4|matched50", gen="BioEmu", scoring="S4_rescored", depth="matched50", sets=ALL3),
    dict(key="BioEmu|S4|allframes", gen="BioEmu", scoring="S4_rescored", depth="allframes", sets=ALL3),
    dict(key="BioEmu|old|oldcols", gen="BioEmu", scoring="old_npz", depth="oldcols", sets=["D_legacy65"]),
    dict(key="ESMFlow|S4|native50", gen="ESMFlow", scoring="S4_rescored", depth="native50", sets=ALL3),
    dict(key="ESMFlow|old|oldcols", gen="ESMFlow", scoring="old_npz", depth="oldcols", sets=["D_legacy65"]),
    dict(key="Boltz2|S4|native50", gen="Boltz2", scoring="S4_rescored", depth="native50", sets=ALL3),
    dict(key="Boltz2|old|oldcols", gen="Boltz2", scoring="old_npz", depth="oldcols", sets=["D_legacy65"]),
    dict(key="Str2Str_AF_d025|S4|native50", gen="Str2Str_AF_d025", scoring="S4_rescored", depth="native50", sets=ALL3),
    dict(key="Str2Str_AF_d035|S4|native50", gen="Str2Str_AF_d035", scoring="S4_rescored", depth="native50", sets=ALL3),
    dict(key="Str2Str_legacy_d025|S4|native50", gen="Str2Str_legacy_d025", scoring="S4_rescored", depth="native50", sets=["D_legacy65"]),
    dict(key="Str2Str_legacy_d025|old|oldcols", gen="Str2Str_legacy_d025", scoring="old_npz", depth="oldcols", sets=["D_legacy65"]),
    dict(key="Str2Str_legacy_d035|S4|native50", gen="Str2Str_legacy_d035", scoring="S4_rescored", depth="native50", sets=["D_legacy65"]),
    dict(key="Str2Str_legacy_d035|old|oldcols", gen="Str2Str_legacy_d035", scoring="old_npz", depth="oldcols", sets=["D_legacy65"]),
]
OLD_DIR = {"BioEmu": "BioEmu_T3", "ESMFlow": "ESMFlow_T3", "Boltz2": "Boltz2_T3", "Str2Str_legacy_d025": "Str2Str_T3", "Str2Str_legacy_d035": "Str2Str_T3_d035"}
HAVE = {"BioEmu": "have_BioEmu_T3", "ESMFlow": "have_ESMFlow_T3", "Boltz2": "have_Boltz2_T3", "Str2Str_legacy_d025": "have_Str2Str_T3", "Str2Str_legacy_d035": "have_Str2Str_T3_d035"}
NEW_DIR = {"BioEmu": "BioEmu_T3new", "ESMFlow": "ESMFlow_T3new", "Boltz2": "Boltz2_T3new"}


# ------------------------------------------------------------------ 域表
def domain_table():
    t = pd.read_csv(os.path.join(TAB, "S5_domain_table.csv"))
    dn = pd.read_csv(os.path.join(TAB, "S1_tier3_Dnew.csv")).set_index("WT_name"); pv = pd.read_csv(os.path.join(TAB, "S1_tier3_legacy65_provenance.csv")).set_index("T3_id")
    dG, org = [], []
    for r in t.itertuples():
        if r.in_Dnew:
            dG.append(float(dn.loc[r.WT_name, "dG"])); org.append(str(dn.loc[r.WT_name, "origin"]))
        else:
            dG.append(float(pv.loc[r.domain_id, "dG_json"])); org.append(str(pv.loc[r.domain_id, "origin"]))
    t["dG"] = dG; t["origin"] = org
    both = t[t.in_Dnew & t.in_legacy65]
    for r in both.itertuples():   # 30 个同属两个集合的域：两个来源的 dG、origin 必须一致
        assert abs(float(pv.loc[r.domain_id, "dG_json"]) - r.dG) < 1e-9 and str(pv.loc[r.domain_id, "origin"]) == r.origin, r.domain_id
    t["in_D_common_AF193"] = t.need_Str2Str_AF.astype(bool)
    ds3 = pd.read_csv(os.path.join(TAB, "S1_tier3_Dnew.csv")).set_index("WT_name").in_dataset3.astype(bool); t["in_dataset3"] = [bool(r.in_Dnew and ds3.loc[r.WT_name]) for r in t.itertuples()]
    return t


def in_set(r, s):
    return bool({"D_new": r.in_Dnew, "D_legacy65": r.in_legacy65, "D_common_AF193": r.need_Str2Str_AF, "D_new_dataset3": r.in_Dnew and r.in_dataset3}[s])


def source_name(gen, r, scoring):
    """该生成器在该域应有的数据来源目录名（只依据域表的标志，不看文件是否存在；None = 该组合本来就不应有数据）。"""
    if scoring == "old_npz":
        return OLD_DIR[gen] if (gen in OLD_DIR and getattr(r, HAVE[gen]) and r.in_legacy65) else None
    if gen in ("BioEmu", "ESMFlow", "Boltz2"):
        return OLD_DIR[gen] if getattr(r, HAVE[gen]) else NEW_DIR[gen]
    if gen.startswith("Str2Str_AF"):
        return gen if r.need_Str2Str_AF else None
    return OLD_DIR[gen] if (getattr(r, HAVE[gen]) and r.in_legacy65) else None   # Str2Str_legacy


# ------------------------------------------------------------------ g(c)
def g_old(gdir, dom):
    d = PP.load_old(INP, gdir, dom)
    return None if d is None else np.nanmean(0.35 * d["NC"] + 2.0 * d["NH"], axis=0)   # 与 e1_multicontrol 的表达式逐字相同


def g_new(gdir, dom):
    n = PP.load_new(REV, gdir, dom)
    if n is None:
        return None
    cols = np.array(sorted(n["ok"])); x = 0.35 * n["NC"][list(cols)].values + 2.0 * n["NH"][list(cols)].values
    return pd.Series(np.nanmean(x, axis=0), index=cols)


def draw_file(gdir, dom, M):
    return os.path.join(DRAW_DIR, f"F_{gdir}_{dom}_M{M}.npz")


def draws_G(g, gdir, dom, M):
    z = np_load_verified("F", draw_file(gdir, dom, M)); fr = z["frames"]; pos = pd.Index(g.index).get_indexer(fr.ravel())
    assert (pos >= 0).all(), f"{gdir}/{dom} M{M}: 冻结索引里有不在可用帧里的帧号"
    return g.values[pos].reshape(fr.shape)


def ens_G(spec, r, gdir):
    dom = r.domain_id
    if spec["scoring"] == "old_npz":
        g = g_old(gdir, dom); return None if g is None else g[None, :]
    g = g_new(gdir, dom)
    if g is None:
        return None
    if spec["depth"] == "matched50":
        return draws_G(g, gdir, dom, 50) if os.path.exists(draw_file(gdir, dom, 50)) else None   # 缺冻结索引 = 缺数据（S4b 合并后需先运行 s4_freeze_draw_indices.py F）
    return g.values[None, :]


def obs_from_G(G):
    R, M = G.shape; out = {"obs_min": RT * G.min(1), "obs_softmin": -RT * (logsumexp(-G, axis=1) - np.log(M)), "obs_mean": RT * G.mean(1), "het_g": G.std(1)}
    for q in GS.QS:
        out[f"obs_q{q}"] = RT * np.percentile(G, q, axis=1)
    return out


def tile(a, R):
    return a if len(a) == R else np.resize(a, R)


def collect(spec, DT, sets):
    out = {}; expected = {s: 0 for s in sets}
    for r in DT.itertuples():
        ss = [s for s in sets if in_set(r, s)]
        if not ss:
            continue
        gdir = source_name(spec["gen"], r, spec["scoring"])
        if gdir is None:
            continue
        for s in ss:
            expected[s] += 1
        G = ens_G(spec, r, gdir)
        if G is not None:
            out[r.domain_id] = dict(obs=obs_from_G(G), M=G.shape[1], R=G.shape[0], gdir=gdir)
    return out, expected


# ------------------------------------------------------------------ 主分析
def run_spec(spec, DT, sets, args, rows, draw_rows, xrep, het_rows):
    t0 = time.time(); D, expected = collect(spec, DT, sets)
    Rspec = max([v["R"] for v in D.values()], default=1)
    print(f"[{spec['key']}] 域数据 {len(D)} 个 R={Rspec} ({time.time() - t0:.0f}s)", flush=True)
    for s in sets:
        doms = [r for r in DT.itertuples() if in_set(r, s) and r.domain_id in D]
        n_exp = expected[s]
        if not doms:
            rows.append(dict(generator=spec["gen"], scoring_basis=spec["scoring"], sample_size_scheme=spec["depth"], domain_set=s, stratum="all", observable="(none)", n=0, n_expected=n_exp, complete=False)); continue
        ids = [r.domain_id for r in doms]; y = np.array([r.dG for r in doms]); org = np.array([r.origin for r in doms])
        Mmin = min(D[d]["M"] for d in ids); Mmax = max(D[d]["M"] for d in ids)
        X = {o: np.stack([tile(D[d]["obs"][o], Rspec) for d in ids], axis=1) for o in OBS + ["het_g"]}   # (R, n)
        for o in OBS:
            res_all = None
            for st in STRATA:
                m = np.ones(len(ids), bool) if st == "all" else (org == st)
                base = dict(generator=spec["gen"], scoring_basis=spec["scoring"], sample_size_scheme=spec["depth"], domain_set=s, stratum=st, observable=o, n=int(m.sum()),
                            n_expected=n_exp if st == "all" else int(sum(1 for r in DT.itertuples() if in_set(r, s) and r.origin == st and source_name(spec["gen"], r, spec["scoring"]) is not None)),
                            n_frames_min=Mmin, n_frames_max=Mmax, dG_min=float(y[m].min()) if m.any() else np.nan, dG_max=float(y[m].max()) if m.any() else np.nan)
                base["complete"] = base["n"] == base["n_expected"]
                if m.sum() < 5:
                    rows.append(base); continue
                tag = f"{spec['key']}|{s}|{st}|{o}"
                st_res = GS.corr_draws(X[o][:, m], y[m], args.B_perm, args.B_boot, tag); rows.append({**base, **st_res})
                if st == "all":
                    res_all = st_res
                    k = st_res["median_draw"]; xrep[(spec["key"], s, o)] = (ids, X[o][k], y, org)
                    if Rspec > 1:
                        rr = GS.spearman_vec(X[o], y); pr = np.array([stats.pearsonr(X[o][i], y)[0] for i in range(Rspec)])
                        for i in range(Rspec):
                            draw_rows.append(dict(generator=spec["gen"], domain_set=s, observable=o, draw=i, spearman_rho=float(rr[i]), pearson_r=float(pr[i])))
        het_rows.append(dict(generator=spec["gen"], scoring_basis=spec["scoring"], sample_size_scheme=spec["depth"], domain_set=s, n=len(ids), het_g_median_over_domains=float(np.median(np.median(X["het_g"], axis=0))), het_g_mean_over_domains=float(np.mean(np.median(X["het_g"], axis=0)))))
        xrep[(spec["key"], s, "het_g")] = (ids, np.median(X["het_g"], axis=0), y, org)
    print(f"[{spec['key']}] 完成 ({time.time() - t0:.0f}s)", flush=True)


def asymmetry(xrep, args):
    rows = []
    for (key, s, o), (ids, x, y, org) in xrep.items():
        if o == "het_g" or (org == "de_novo").sum() < 5 or (org == "natural").sum() < 5:
            continue
        a = GS.asymmetry_perm(x, y, org == "de_novo", args.B_perm, tag=f"{key}|{s}|{o}")
        rd = stats.spearmanr(x[org == "de_novo"], y[org == "de_novo"])[0]; rn = stats.spearmanr(x[org == "natural"], y[org == "natural"])[0]
        rows.append(dict(spec=key, domain_set=s, observable=o, rho_de_novo=float(rd), rho_natural=float(rn), n_de_novo=a["n_first"], n_natural=a["n_second"], delta_rho_denovo_minus_natural=a["delta_rho"], asymmetry_p_perm_two_sided=a["p_perm_two_sided"], asymmetry_p_perm_one_sided_denovo_gt_natural=a["p_perm_one_sided_first_gt_second"]))
    return pd.DataFrame(rows)


def bonferroni(G):
    rows = []
    for (gen, sb, ds, s, o), d in G[G.observable.isin(OBS)].groupby(["generator", "scoring_basis", "sample_size_scheme", "domain_set", "observable"]):
        dd = d.set_index("stratum")
        if not {"de_novo", "natural"} <= set(dd.index) or dd.loc[["de_novo", "natural"], "spearman_rho"].isna().any():
            continue
        for st in ("de_novo", "natural"):
            r = dd.loc[st]
            rows.append(dict(generator=gen, scoring_basis=sb, sample_size_scheme=ds, domain_set=s, observable=o, stratum=st, n=int(r.n), spearman_rho=r.spearman_rho, spearman_p_perm_uncorrected=r.spearman_p_perm_two_sided,
                             spearman_p_perm_bonferroni_x2=min(1.0, 2 * r.spearman_p_perm_two_sided), pearson_r=r.pearson_r, pearson_p_t_uncorrected=r.pearson_p_t_two_sided, pearson_p_t_bonferroni_x2=min(1.0, 2 * r.pearson_p_t_two_sided),
                             significant_after_bonferroni_spearman=bool(2 * r.spearman_p_perm_two_sided < 0.05)))
    return pd.DataFrame(rows)


def negative_control(xrep, args):
    rows = []
    for (key, s, o), (ids, x, y, org) in xrep.items():
        if o == "het_g" or s not in ("D_new", "D_legacy65"):
            continue
        obs, p, null = GS.perm_p_spearman(x, y, 1000, tag=f"negctl|{key}|{s}|{o}")
        rows.append(dict(spec=key, domain_set=s, observable=o, n=len(x), observed_rho=obs, null_B=1000, null_median=float(np.median(null)), null_mean=float(null.mean()), null_p2_5=float(np.percentile(null, 2.5)), null_p97_5=float(np.percentile(null, 97.5)),
                         share_abs_null_ge_abs_observed=float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), pass_abs_median_lt_0p03=bool(abs(np.median(null)) < 0.03)))
    return pd.DataFrame(rows)


def paired_delta(xrep, args):
    """跨生成器相关差：同一组域上配对 bootstrap（每次重抽域，两个生成器同时算 ρ），报告 Δρ 的 95% CI，不比较两个独立的 p。"""
    cur = {"BioEmu": "BioEmu|S4|matched50", "ESMFlow": "ESMFlow|S4|native50", "Boltz2": "Boltz2|S4|native50", "Str2Str_AF_d025": "Str2Str_AF_d025|S4|native50", "Str2Str_AF_d035": "Str2Str_AF_d035|S4|native50"}
    rows = []
    for s in ("D_new", "D_common_AF193", "D_legacy65"):
        for (a, ka), (b, kb) in itertools.combinations(cur.items(), 2):
            for o in OBS:
                if (ka, s, o) not in xrep or (kb, s, o) not in xrep:
                    continue
                ia, xa, ya, _ = xrep[(ka, s, o)]; ib, xb, yb, _ = xrep[(kb, s, o)]
                common = [d for d in ia if d in set(ib)]
                if len(common) < 10:
                    continue
                pa = {d: i for i, d in enumerate(ia)}; pb = {d: i for i, d in enumerate(ib)}
                va = np.array([xa[pa[d]] for d in common]); vb = np.array([xb[pb[d]] for d in common]); yy = np.array([ya[pa[d]] for d in common]); n = len(common)
                rng = GS.seeded_rng("paired", ka, kb, s, o, n); idx = rng.integers(0, n, size=(args.B_boot, n))
                da = GS._spearman_rows(va[idx], yy[idx]) - GS._spearman_rows(vb[idx], yy[idx]); da = da[np.isfinite(da)]
                ra = stats.spearmanr(va, yy)[0]; rb = stats.spearmanr(vb, yy)[0]
                rows.append(dict(domain_set=s, generator_a=a, generator_b=b, observable=o, n_common_domains=n, rho_a=float(ra), rho_b=float(rb), delta_rho_a_minus_b=float(ra - rb), boot_lo=float(np.percentile(da, 2.5)), boot_hi=float(np.percentile(da, 97.5)), ci_excludes_0=bool(np.percentile(da, 2.5) > 0 or np.percentile(da, 97.5) < 0)))
    return pd.DataFrame(rows)


def convergence(DT, args):
    """BioEmu 收敛曲线：M ∈ {10, 25, 50, 100, 全部可用}，R=100，口径 F，8 个观测量（验收要求 obs_q 的漂移小于 obs_min）。"""
    rows = []; Ms = [10, 25, 50, 100, "all"]
    cache = {}
    for r in DT.itertuples():
        gdir = source_name("BioEmu", r, "S4_rescored")
        g = g_new(gdir, r.domain_id) if gdir else None
        if g is None:
            continue
        cache[r.domain_id] = (gdir, g)
    for s in ("D_new", "D_legacy65"):
        doms = [r for r in DT.itertuples() if in_set(r, s) and r.domain_id in cache]; ids = [r.domain_id for r in doms]; y = np.array([r.dG for r in doms])
        n_exp = sum(1 for r in DT.itertuples() if in_set(r, s))
        med = {}
        for M in Ms:
            per = {}; ok = True
            for d in ids:
                gdir, g = cache[d]; Mf = len(g) if M == "all" else M
                if not os.path.exists(draw_file(gdir, d, Mf)):
                    ok = False; break
                G = draws_G(g, gdir, d, Mf)[:100]; per[d] = obs_from_G(G)
            if not ok:
                print(f"[convergence] {s} M={M}: 缺冻结索引，跳过", flush=True); continue
            R = max(len(per[d]["obs_min"]) for d in ids)
            for o in OBS:
                X = np.stack([tile(per[d][o], R) for d in ids], axis=1); rr = GS.spearman_vec(X, y)
                rows.append(dict(domain_set=s, M=str(M), observable=o, n_domains=len(ids), n_expected=n_exp, R=R, rho_median=float(np.median(rr)), rho_p2_5=float(np.percentile(rr, 2.5)), rho_p97_5=float(np.percentile(rr, 97.5))))
                med.setdefault(o, []).append(float(np.median(rr)))
        for o, v in med.items():
            rows.append(dict(domain_set=s, M="drift(max-min of median rho over M)", observable=o, n_domains=len(ids), n_expected=n_exp, R=np.nan, rho_median=float(max(v) - min(v))))
            if len(v) == len(Ms):   # 补充的描述性指标（预先指定的是上面的 max-min）：M=50 → 全部可用帧的变化，以及 M=10 → 全部帧的单调趋势（Spearman，5 个点）
                rows.append(dict(domain_set=s, M="supplementary: rho(all) - rho(M=50)", observable=o, n_domains=len(ids), n_expected=n_exp, R=np.nan, rho_median=float(v[-1] - v[2])))
                rows.append(dict(domain_set=s, M="supplementary: monotonic trend (Spearman of rho vs M index, 5 points)", observable=o, n_domains=len(ids), n_expected=n_exp, R=np.nan, rho_median=float(stats.spearmanr(range(len(v)), v)[0])))
        if "obs_min" in med:
            dm = max(med["obs_min"]) - min(med["obs_min"])
            for q in GS.QS:
                rows.append(dict(domain_set=s, M="acceptance: drift(obs_q) < drift(obs_min)", observable=f"obs_q{q}", n_domains=len(ids), n_expected=n_exp, R=np.nan, rho_median=float(max(med[f"obs_q{q}"]) - min(med[f"obs_q{q}"])), rho_p2_5=dm, rho_p97_5=float((max(med[f"obs_q{q}"]) - min(med[f"obs_q{q}"])) < dm)))
    return pd.DataFrame(rows)


def heterogeneity_ordering(het_df, G):
    rows = []
    geo = os.path.join(TAB, "S10_het_geometry.csv"); GEO = pd.read_csv(geo) if os.path.exists(geo) else None
    for s in ("D_new", "D_legacy65", "D_common_AF193"):
        cur = het_df[(het_df.domain_set == s) & (het_df.scoring_basis == "S4_rescored") & (~het_df.generator.str.contains("legacy")) & (het_df.sample_size_scheme.isin(["matched50", "native50"]))].copy()
        rr = G[(G.domain_set == s) & (G.stratum == "all") & (G.observable == "obs_min") & (G.scoring_basis == "S4_rescored") & (G.sample_size_scheme.isin(["matched50", "native50"])) & (~G.generator.str.contains("legacy"))].set_index("generator")
        cur = cur.set_index("generator").join(rr[["n", "spearman_rho", "pearson_r"]].rename(columns={"n": "n_corr", "spearman_rho": "rho_obs_min", "pearson_r": "pearson_obs_min"}))
        if GEO is not None:
            gg = GEO[GEO.domain_set == s].set_index("generator"); cur = cur.join(gg[["median_mean_pairwise_ca_rmsd_nm", "median_rg_sd_nm"]])
        for g, r in cur.iterrows():
            rows.append(dict(kind="per_generator", domain_set=s, generator=g, **{k: r[k] for k in cur.columns if k not in ("domain_set", "generator")}))
        if len(cur) >= 4:
            for hm in [c for c in ("het_g_median_over_domains", "median_mean_pairwise_ca_rmsd_nm", "median_rg_sd_nm") if c in cur.columns and cur[c].notna().all()]:
                tau, p = stats.kendalltau(cur[hm], cur["rho_obs_min"])
                rows.append(dict(kind="kendall_tau_heterogeneity_vs_rho_obs_min", domain_set=s, generator=f"all {len(cur)} generators (descriptive, n={len(cur)})", heterogeneity_metric=hm, kendall_tau=float(tau), kendall_p=float(p)))
    return pd.DataFrame(rows)


def legacy_bridge(G, xrep):
    """原稿数字 ↔ 复现值 ↔ D_new 新值。"""
    def row(gen, sb, ds, s, o, st="all"):
        d = G[(G.generator == gen) & (G.scoring_basis == sb) & (G.sample_size_scheme == ds) & (G.domain_set == s) & (G.observable == o) & (G.stratum == st)]
        return d.iloc[0] if len(d) else None
    e1 = pd.read_csv(os.path.join(INP["results_tables"], "e1_multicontrol.csv")).set_index("generator")
    out = []

    def add(item, gen_label, old_args, stat, ms, newcmp):
        r_old = row(*old_args); r_s4 = row(old_args[0], "S4_rescored", "native50" if old_args[0] != "BioEmu" else "matched50", "D_legacy65", old_args[4])
        r_new = row(*newcmp) if newcmp else None
        col = "spearman_rho" if stat == "spearman" else "pearson_r"
        val = None if r_old is None else float(r_old[col])
        out.append(dict(item=item, generator=gen_label, observable=old_args[4], statistic=stat, manuscript_value=ms, reproduced_old_npz=val, abs_diff=None if val is None else abs(val - ms), within_tol_0p005=None if val is None else bool(abs(val - ms) <= TOL),
                        n_old=None if r_old is None else int(r_old.n), p_spearman_scipy_two_sided=None if r_old is None else r_old.spearman_p_scipy_two_sided, p_spearman_perm_two_sided=None if r_old is None else r_old.spearman_p_perm_two_sided,
                        p_pearson_t_two_sided=None if r_old is None else r_old.pearson_p_t_two_sided,
                        S4_rescored_D_legacy65=None if r_s4 is None else float(r_s4[col]), D_new_value=None if r_new is None else float(r_new[col]), D_new_generator=None if newcmp is None else newcmp[0], D_new_n=None if r_new is None else int(r_new.n), D_new_n_expected=None if r_new is None else int(r_new.n_expected)))
    add("K5: Str2Str δ0.35 Spearman (obs_min)", "Str2Str_legacy_d035", ("Str2Str_legacy_d035", "old_npz", "oldcols", "D_legacy65", "obs_min"), "spearman", MS["str2str_d035_rho"], ("Str2Str_AF_d035", "S4_rescored", "native50", "D_new", "obs_min"))
    add("K5: Str2Str δ0.35 Pearson (obs_min)", "Str2Str_legacy_d035", ("Str2Str_legacy_d035", "old_npz", "oldcols", "D_legacy65", "obs_min"), "pearson", MS["str2str_d035_pearson"], ("Str2Str_AF_d035", "S4_rescored", "native50", "D_new", "obs_min"))
    add("K5: Str2Str δ0.25 Spearman (obs_min)", "Str2Str_legacy_d025", ("Str2Str_legacy_d025", "old_npz", "oldcols", "D_legacy65", "obs_min"), "spearman", MS["str2str_d025_rho"], ("Str2Str_AF_d025", "S4_rescored", "native50", "D_new", "obs_min"))
    e1map = {"BioEmu_T3": ("BioEmu", ("BioEmu", "S4_rescored", "matched50", "D_new")), "Boltz2_T3": ("Boltz2", ("Boltz2", "S4_rescored", "native50", "D_new")),
             "Str2Str_T3": ("Str2Str_legacy_d025", ("Str2Str_AF_d025", "S4_rescored", "native50", "D_new")), "ESMFlow_T3": ("ESMFlow", ("ESMFlow", "S4_rescored", "native50", "D_new"))}
    for e1g, (gen, (ng, nsb, nds, ns)) in e1map.items():
        add(f"e1 {e1g}: ens_pearson (obs_softmin)", gen, (gen, "old_npz", "oldcols", "D_legacy65", "obs_softmin"), "pearson", float(e1.loc[e1g, "ens_pearson"]), (ng, nsb, nds, ns, "obs_softmin"))
        add(f"e1 {e1g}: mean_pearson (obs_mean)", gen, (gen, "old_npz", "oldcols", "D_legacy65", "obs_mean"), "pearson", float(e1.loc[e1g, "mean_pearson"]), (ng, nsb, nds, ns, "obs_mean"))
        key = {"BioEmu": "BioEmu|old|oldcols", "ESMFlow": "ESMFlow|old|oldcols", "Boltz2": "Boltz2|old|oldcols", "Str2Str_legacy_d025": "Str2Str_legacy_d025|old|oldcols"}[gen]
        if (key, "D_legacy65", "het_g") in xrep:
            het = float(np.mean(xrep[(key, "D_legacy65", "het_g")][1]))
            out.append(dict(item=f"e1 {e1g}: het_mean", generator=gen, observable="het_g", statistic="mean over domains of SD_c g(c) (ddof=0)", manuscript_value=float(e1.loc[e1g, "het_mean"]), reproduced_old_npz=het, abs_diff=abs(het - float(e1.loc[e1g, "het_mean"])), within_tol_0p005=bool(abs(het - float(e1.loc[e1g, "het_mean"])) <= TOL)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run"]); ap.add_argument("--quick", action="store_true"); ap.add_argument("--specs", default=""); ap.add_argument("--sets", default="")
    ap.add_argument("--skip-convergence", action="store_true"); ap.add_argument("--suffix", default=""); a = ap.parse_args()
    a.B_perm, a.B_boot = (2000, 500) if a.quick else (100000, 10000); suf = ("_quick" if a.quick else "") + a.suffix
    log = revio.RunLogger(stage=10, task="global_stability" + suf, cpu_procs=1, seeds=f"globalstats entropy {GS.ENTROPY}; draws=S4a frozen (20261024+1000+r)")
    DT = domain_table(); DT.to_csv(os.path.join(TAB, "S10_domain_table.csv"), index=False)
    print(DT.groupby(["in_legacy65", "in_Dnew"]).size().to_string(), "\n", "D_common_AF193:", int(DT.need_Str2Str_AF.sum()), flush=True)
    specs = [s for s in SPECS if (not a.specs or s["key"] in a.specs.split(","))]; rows, draw_rows, het_rows = [], [], []; xrep = {}
    for sp in specs:
        allowed = sp["sets"] + ([e for e in EXTRA_SETS if a.sets and e in a.sets.split(",")] if sp["scoring"] == "S4_rescored" and not sp["gen"].startswith("Str2Str_legacy") else [])
        sets = [s for s in allowed if (not a.sets or s in a.sets.split(","))]
        if sets:
            run_spec(sp, DT, sets, a, rows, draw_rows, xrep, het_rows)
    G = pd.DataFrame(rows); out = os.path.join(TAB, f"S10_global_all_generators{suf}.csv"); G.to_csv(out, index=False)
    pd.DataFrame(draw_rows).to_parquet(os.path.join(TAB, f"S10_global_matched_M_draws{suf}.parquet"), index=False)
    het_df = pd.DataFrame(het_rows)
    bon = bonferroni(G); bon.to_csv(os.path.join(TAB, f"S10_stratified_bonferroni{suf}.csv"), index=False)
    asy = asymmetry(xrep, a); asy.to_csv(os.path.join(TAB, f"S10_denovo_natural_asymmetry{suf}.csv"), index=False)
    neg = negative_control(xrep, a)
    if len(neg):
        neg.to_csv(os.path.join(TAB, f"S10_perm_negative_control{suf}.csv"), index=False)
    pdl = paired_delta(xrep, a)
    if len(pdl):
        pdl.to_csv(os.path.join(TAB, f"S10_paired_delta_rho{suf}.csv"), index=False)   # 空表不落盘（敏感性子集只有 D_new_dataset3 一个域集，没有跨生成器配对）
    het = heterogeneity_ordering(het_df, G)
    if len(het):
        het.to_csv(os.path.join(TAB, f"S10_heterogeneity_ordering{suf}.csv"), index=False)
    br = legacy_bridge(G, xrep); br.to_csv(os.path.join(TAB, f"S10_legacy_bridge{suf}.csv"), index=False)
    if not a.skip_convergence:
        cv = convergence(DT, a); cv.to_csv(os.path.join(TAB, f"S10_bioemu_convergence{suf}.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(br.round(4).to_string(index=False))
    print("complete 比例:", float(G.complete.mean()) if len(G) else None, " 总行数:", len(G))
    log.finish([out], note=f"rows={len(G)}")


if __name__ == "__main__":
    main()
