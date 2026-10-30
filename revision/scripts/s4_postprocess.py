"""Stage 4a · 重打分后的后处理与一致性对照（revision_proposal.md Stage 4 步骤 5–6）。

  map      旧 operator_f.npz 的列 → xtc 帧号。旧 npz 没有帧索引，且各生成器当年用过不同的抽稀（np.linspace）与超时设置，列号不一定等于帧号。
           办法：旧列在当年按帧序保存，所以对每个格子在"新打分的 N_C 向量"上做保序对齐（动态规划，最小化逐残基 L1 距离之和）。
           全重原子帧（ESMFlow、Boltz-2、ShortMD）补侧链不改重原子，N_C 应逐位相同，距离为 0 就是强证据；
           主链加 Cβ 帧（BioEmu、Str2Str）补侧链有随机性（S3 实测），只能看距离是否明显低于无关列。
  parity   C-parity：新旧逐残基系综平均 N_C、N_H 的 Pearson（验收线 ≥ 0.99），以及 count_avg 的 Spearman（对实验 ΔG_open）新旧差（验收线 ≤ 0.02）。
  leaderboard  旧 npz 复算的逐蛋白 Spearman 与原稿 leaderboard 核对（排除编号偏移/链选择问题）
  weights  权重审计：旧流程的系综平均是否等权（eval/pred 的 NC_mean 与 npz 各列的 nanmean 逐位核对），新流程用 bv.aggregate 的 nanmean。
  timing   逐生成器 × 蛋白的每帧三段耗时，与 S3 基准的外推对照。
  temps    各蛋白实验温度（T_C，来自 S1 残基表），供 dG_pred_Texp 使用。
用法：python s4_postprocess.py {map|parity|leaderboard|weights|timing|temps|all} [--allow-pending]
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402

TIER1A = ["BioEmu", "ESMFlow", "Boltz2", "Str2Str", "ShortMD"]
T3 = ["BioEmu_T3", "ESMFlow_T3", "Boltz2_T3", "Str2Str_T3", "Str2Str_T3_d035"]
RT = bv.R_GAS * bv.T_REF


def load_new(REV, g, p):
    base = os.path.join(REV, "raw_outputs", "s4", "counts", g)
    f = os.path.join(base, f"{p}.parquet")
    if not os.path.exists(f):
        return None
    R = pd.read_parquet(f); F = pd.read_parquet(os.path.join(base, f"{p}_frames.parquet"))
    ok = F[F.status.isin(["ok", "ok_retry"])].frame_idx.values
    nc = R.pivot(index="resSeq", columns="frame_idx", values="NC"); nh = R.pivot(index="resSeq", columns="frame_idx", values="NH")
    resn = R.drop_duplicates("resSeq").set_index("resSeq").resn.to_dict()
    return dict(NC=nc, NH=nh, resn=resn, ok=np.sort(ok), n_xtc=len(F), F=F)


def load_old(INP, g, p):
    f = os.path.join(INP["gen_dir"], g, p, "operator_f.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f); return dict(resSeq=[int(x) for x in z["resSeq"]], NC=np.asarray(z["NC"], float), NH=np.asarray(z["NH"], float))


def monotone_align(C):
    """C: (m, n) 代价。选 i_1 < i_2 < … < i_m 使 Σ C[j, i_j] 最小，返回 i（长度 m）。要求 m ≤ n。"""
    m, n = C.shape; D = np.full((m, n), np.inf); P = np.zeros((m, n), dtype=np.int32); D[0] = C[0]
    for j in range(1, m):
        prev = D[j - 1]; cm = np.minimum.accumulate(prev); arg = np.zeros(n, dtype=np.int32); best = 0
        for i in range(n):
            if prev[i] < prev[best]:
                best = i
            arg[i] = best
        D[j, 1:] = C[j, 1:] + cm[:-1]; P[j, 1:] = arg[:-1]; D[j, 0] = np.inf
    i = int(np.argmin(D[-1])); path = [i]
    for j in range(m - 1, 0, -1):
        i = int(P[j, i]); path.append(i)
    return np.array(path[::-1])


def cmd_map(a, INP, REV):
    log = revio.RunLogger(stage=4, task="old_column_map", cpu_procs=1, seeds="none")
    rows, summ = [], []
    for g in TIER1A + T3:
        for p in sorted(os.listdir(os.path.join(INP["gen_dir"], g))):
            old = load_old(INP, g, p); new = load_new(REV, g, p)
            if old is None or p == "ecDHFR":
                continue
            if new is None:
                if a.allow_pending:
                    continue
                raise SystemExit(f"{g}/{p}: 新打分尚未完成")
            common = [r for r in old["resSeq"] if r in new["NC"].index]
            io = [old["resSeq"].index(r) for r in common]; O = old["NC"][io, :]
            N = new["NC"].loc[common, new["ok"]].values; m, n = O.shape[1], N.shape[1]
            if m > n:
                summ.append(dict(generator=g, protein=p, n_old_cols=m, n_new_ok_frames=n, n_xtc=new["n_xtc"], status="UNMAPPABLE_old_has_more_columns_than_new_ok_frames"))
                continue
            diff = np.abs(O.T[:, None, :] - N.T[None, :, :]); mask = ~np.isnan(diff)
            C = np.where(mask, diff, 0).sum(axis=2) / np.maximum(mask.sum(axis=2), 1)   # 逐残基平均 L1
            idx = monotone_align(C); frames = new["ok"][idx]
            cost = C[np.arange(m), idx]; un = C.argmin(axis=1)
            identity = bool(m == new["n_xtc"] and np.array_equal(frames, np.arange(m)))
            lin = {mm: np.array_equal(frames, np.linspace(0, new["n_xtc"] - 1, mm).astype(int)) for mm in (50, 100, 150) if mm <= new["n_xtc"]}
            rows += [dict(generator=g, protein=p, old_col=j, frame_idx=int(frames[j]), l1_per_res=float(cost[j]), exact_NC_match=bool(cost[j] == 0)) for j in range(m)]
            summ.append(dict(generator=g, protein=p, n_old_cols=m, n_new_ok_frames=n, n_xtc=new["n_xtc"], n_common_res=len(common), identity_map=identity,
                             matches_linspace_rule=";".join(f"{k}:{v}" for k, v in lin.items() if v) or "none", median_l1_per_res=round(float(np.median(cost)), 3),
                             frac_exact_NC=round(float((cost == 0).mean()), 4), frac_dp_equals_unconstrained_argmin=round(float((idx == un).mean()), 4),
                             median_l1_unrelated=round(float(np.median(C[np.arange(m), (idx + max(n // 3, 1)) % n])), 3), status="mapped"))
    M = pd.DataFrame(rows); S = pd.DataFrame(summ)
    f1 = os.path.join(REV, "tables", "S4_old_column_to_frame_map.csv"); f2 = os.path.join(REV, "tables", "S4_old_column_map_summary.csv"); M.to_csv(f1, index=False); S.to_csv(f2, index=False)
    pd.set_option("display.width", 250); print(S.to_string(index=False)); log.finish([f1, f2], note=f"cells={len(S)}")


def cmd_parity(a, INP, REV):
    log = revio.RunLogger(stage=4, task="parity", cpu_procs=1, seeds="none")
    exp = pd.read_parquet(INP["tier1a_all"]); maps = json.load(open(INP["structure_maps"]))
    MP = pd.read_csv(os.path.join(REV, "tables", "S4_old_column_to_frame_map.csv")); rows = []
    for g in TIER1A + T3:
        for p in sorted(os.listdir(os.path.join(INP["gen_dir"], g))):
            old = load_old(INP, g, p); new = load_new(REV, g, p)
            if old is None or new is None or p == "ecDHFR":
                continue
            mp_ = MP[(MP.generator == g) & (MP.protein == p)].sort_values("old_col")
            if len(mp_) == 0:
                continue
            fr = mp_.frame_idx.values
            common = [r for r in old["resSeq"] if r in new["NC"].index]; io = [old["resSeq"].index(r) for r in common]
            oNC = np.nanmean(old["NC"][io, :], axis=1); oNH = np.nanmean(old["NH"][io, :], axis=1)
            nNCm = np.nanmean(new["NC"].loc[common, fr].values, axis=1); nNHm = np.nanmean(new["NH"].loc[common, fr].values, axis=1)
            nNCa = np.nanmean(new["NC"].loc[common, new["ok"]].values, axis=1); nNHa = np.nanmean(new["NH"].loc[common, new["ok"]].values, axis=1)
            pr = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
            row = dict(generator=g, protein=p, n_old_cols=len(fr), n_new_ok=len(new["ok"]), n_common_res=len(common),
                       pearson_NC_old_vs_new_matched=pr(oNC, nNCm), pearson_NH_old_vs_new_matched=pr(oNH, nNHm),
                       pearson_NC_old_vs_new_all=pr(oNC, nNCa), pearson_NH_old_vs_new_all=pr(oNH, nNHa),
                       max_abs_dNC_matched=float(np.abs(oNC - nNCm).max()), max_abs_dNH_matched=float(np.abs(oNH - nNHm).max()))
            if g in TIER1A:
                ge = exp[exp.protein_id == p]
                pr_csv = pd.read_csv(os.path.join(INP["eval_pred_dir"], g, f"{p}.csv"))
                off = bv.best_offset({int(r.resSeq): bv.to1(r.resn) for _, r in pr_csv.iterrows()}, ge, maps.get(p, {}).get("gen_offset", maps.get(p, {}).get("offset", 0)))
                def rho(resSeq, NC, NH):
                    x, y, _ = bv.align(resSeq, RT * bv.aggregate(NC, NH, "count_avg"), ge, off); return float(stats.spearmanr(x, y)[0]), len(x)
                ro, n_o = rho(old["resSeq"], old["NC"], old["NH"])
                rm, _ = rho(common, new["NC"].loc[common, fr].values, new["NH"].loc[common, fr].values)
                ra, _ = rho(common, new["NC"].loc[common, new["ok"]].values, new["NH"].loc[common, new["ok"]].values)
                row.update(offset=off, n_exp_residues=n_o, spearman_old=ro, spearman_new_matched=rm, spearman_new_all_frames=ra,
                           d_spearman_matched=rm - ro, d_spearman_all=ra - ro)
            rows.append(row)
    T = pd.DataFrame(rows)
    T["pass_pearson_0.99"] = (T.pearson_NC_old_vs_new_matched >= 0.99) & (T.pearson_NH_old_vs_new_matched >= 0.99)
    T["pass_spearman_0.02"] = T.d_spearman_matched.abs() <= 0.02
    f = os.path.join(REV, "tables", "S4_parity.csv"); T.to_csv(f, index=False)
    pd.set_option("display.width", 250)
    print(T.groupby("generator").agg(cells=("protein", "size"), min_pearson_NC=("pearson_NC_old_vs_new_matched", "min"), min_pearson_NH=("pearson_NH_old_vs_new_matched", "min"),
                                     max_abs_dSp=("d_spearman_matched", lambda x: x.abs().max()), fail_pearson=("pass_pearson_0.99", lambda x: int((~x).sum()))).round(4).to_string())
    log.finish([f], note=f"cells={len(T)}")


def cmd_leaderboard(a, INP, REV):
    """排除编号偏移与链选择问题：用旧 npz 复算的逐蛋白 Spearman（spearman_old，S4_parity.csv）必须与原稿 eval/leaderboard.parquet 一致。"""
    log = revio.RunLogger(stage=4, task="parity_vs_leaderboard", cpu_procs=1, seeds="none")
    L = pd.read_parquet(INP["leaderboard"]); T = pd.read_csv(os.path.join(REV, "tables", "S4_parity.csv")).dropna(subset=["spearman_old"])
    m = T.merge(L[["model", "protein", "spearman", "n"]], left_on=["generator", "protein"], right_on=["model", "protein"], how="outer", indicator=True)
    assert (m._merge == "both").all(), m[m._merge != "both"][["generator", "protein", "model"]]
    m["abs_diff"] = (m.spearman_old - m.spearman).abs(); m["n_exp_equal"] = m.n_exp_residues == m.n
    out = m[["generator", "protein", "offset", "n_exp_residues", "n", "n_exp_equal", "spearman_old", "spearman", "abs_diff"]].rename(columns=dict(n="n_leaderboard", spearman="spearman_leaderboard"))
    f = os.path.join(REV, "tables", "S4_parity_old_vs_leaderboard.csv"); out.to_csv(f, index=False)
    print(f"cells={len(out)} max|diff|={out.abs_diff.max():.4f} (leaderboard 保留 3 位小数，舍入误差上限 0.0005) n_exp 全等={bool(out.n_exp_equal.all())}")
    log.finish([f], note=f"cells={len(out)} max_abs_diff={out.abs_diff.max():.4f}")


def cmd_celllogs(a, INP, REV):
    """方案输出清单里的 logs/s4/<generator>_<protein>.log：由帧状态表生成，每个（生成器×蛋白）一份文本；为保持 logs/s4 整洁放在 logs/s4/cells/ 下。"""
    log = revio.RunLogger(stage=4, task="cell_logs", cpu_procs=1, seeds="none")
    base = os.path.join(REV, "raw_outputs", "s4", "counts"); out = os.path.join(REV, "logs", "s4", "cells"); os.makedirs(out, exist_ok=True); n = 0
    for g in sorted(os.listdir(base)):
        for fn in sorted(os.listdir(os.path.join(base, g))):
            if not fn.endswith("_frames.parquet"):
                continue
            p = fn[:-len("_frames.parquet")]; F = pd.read_parquet(os.path.join(base, g, fn)); bad = F[~F.status.isin(["ok", "ok_retry"])]
            L = [f"generator={g} protein={p}", f"frames_or_recons={len(F)}  status_counts={F.status.value_counts().to_dict()}  retried={int((F.attempts > 1).sum())}",
                 f"t_fix_s: mean={F.t_fix_s.mean():.3f} median={F.t_fix_s.median():.3f} max={F.t_fix_s.max():.3f}  t_count_s mean={F.t_count_s.mean():.4f}  t_sasa_s mean={F.t_sasa_s.mean():.4f}",
                 f"total_core_s_all_attempts={F.t_wall_all_attempts_s.sum():.1f}  n_res_scored: min={int(F.n_res_scored.min())} max={int(F.n_res_scored.max())}  seeds: det_seed(generator|protein|frame_idx|recon_k|attempt)",
                 "failed_frames: " + ("none" if len(bad) == 0 else "; ".join(f"frame {r.frame_idx}/recon {r.recon_k}: {r.status} {r.err}" for r in bad.itertuples()))]
            open(os.path.join(out, f"{g}_{p}.log"), "w").write("\n".join(L) + "\n"); n += 1
    print("cell logs:", n, "->", out); log.finish([out], note=f"cells={n}")


def cmd_weights(a, INP, REV):
    log = revio.RunLogger(stage=4, task="weights_audit", cpu_procs=1, seeds="none"); rows = []
    for g in TIER1A + T3:
        for p in sorted(os.listdir(os.path.join(INP["gen_dir"], g))):
            old = load_old(INP, g, p); f = os.path.join(INP["eval_pred_dir"], g, f"{p}.csv")
            if old is None or not os.path.exists(f) or p == "ecDHFR":
                continue
            pr = pd.read_csv(f).set_index("resSeq"); common = [r for r in old["resSeq"] if r in pr.index]; io = [old["resSeq"].index(r) for r in common]
            nc = np.nanmean(old["NC"][io, :], axis=1); nh = np.nanmean(old["NH"][io, :], axis=1); ln = bv.aggregate(old["NC"][io, :], old["NH"][io, :], "count_avg")
            rows.append(dict(generator=g, protein=p, n_cols=old["NC"].shape[1], n_res_compared=len(common),
                             max_abs_diff_NC_mean=float(np.abs(pr.loc[common, "NC_mean"].values - nc).max()), max_abs_diff_NH_mean=float(np.abs(pr.loc[common, "NH_mean"].values - nh).max()),
                             max_abs_diff_lnPF=float(np.abs(pr.loc[common, "lnPF"].values - ln).max())))
    # Boltz2_T3 在 eval/pred 里没有逐残基预测值，无法做上面的逐残基比对；改为复现 e1_multicontrol 的全局量（45_e1_multicontrol.py:8-9 对帧用 np.nanmean / logsumexp，即等权）
    sub = json.load(open(INP["tier3_subset"])); ref = pd.read_csv(os.path.join(INP["results_tables"], "e1_multicontrol.csv")).set_index("generator")
    from scipy.special import logsumexp as _lse
    from scipy.stats import pearsonr as _pr
    e1 = []
    for g in ("Boltz2_T3", "BioEmu_T3", "Str2Str_T3", "ESMFlow_T3"):
        ds = sorted(d for d in sub if os.path.exists(os.path.join(INP["gen_dir"], g, d, "operator_f.npz"))); vals = []
        for d in ds:
            z = np.load(os.path.join(INP["gen_dir"], g, d, "operator_f.npz")); gg = np.nanmean(0.35 * z["NC"] + 2.0 * z["NH"], axis=0)
            vals.append((-bv.R_GAS * bv.T_REF * (_lse(-gg) - np.log(len(gg))), bv.R_GAS * bv.T_REF * np.nanmean(gg), float(np.std(gg)), sub[d]["deltaG"]))
        v = np.array(vals); e1.append(dict(generator=g, n=len(ds), ens_pearson=_pr(v[:, 0], v[:, 3])[0], mean_pearson=_pr(v[:, 1], v[:, 3])[0], het_mean=v[:, 2].mean(),
                                          ref_ens_pearson=float(ref.loc[g, "ens_pearson"]), ref_mean_pearson=float(ref.loc[g, "mean_pearson"]), ref_het_mean=float(ref.loc[g, "het_mean"])))
    E = pd.DataFrame(e1); E["reproduces_reference_3dp"] = (E.ens_pearson.round(3) == E.ref_ens_pearson) & (E.mean_pearson.round(3) == E.ref_mean_pearson) & (E.het_mean.round(3) == E.ref_het_mean)
    f_e1 = os.path.join(REV, "tables", "S4_weights_e1_reproduction.csv"); E.round(5).to_csv(f_e1, index=False); print(E.round(4).to_string(index=False))
    T = pd.DataFrame(rows); T["weights_uniform_in_old_pipeline"] = (T[["max_abs_diff_NC_mean", "max_abs_diff_NH_mean", "max_abs_diff_lnPF"]].max(axis=1) < 1e-9)
    T["old_pipeline_weighting"] = "np.nanmean over frames (uniform); forward.compute(weights=None) → 1/F; 21_score_parallel.agg → np.nanmean"
    T["new_pipeline_weighting"] = "bv.aggregate → np.nanmean (count_avg) / mean of exp(−x) (invpf_avg), uniform; frames chosen only through frozen index files"
    f = os.path.join(REV, "tables", "S4_weights_audit.csv"); T.to_csv(f, index=False)
    print("cells", len(T), "| all uniform:", bool(T.weights_uniform_in_old_pipeline.all()), "| max diffs:", T[["max_abs_diff_NC_mean", "max_abs_diff_NH_mean", "max_abs_diff_lnPF"]].max().to_dict())
    log.finish([f], note=f"cells={len(T)}")


def cmd_timing(a, INP, REV):
    log = revio.RunLogger(stage=4, task="timing", cpu_procs=1, seeds="none"); rows = []
    base = os.path.join(REV, "raw_outputs", "s4", "counts")
    for g in sorted(os.listdir(base)):
        for f in sorted(os.listdir(os.path.join(base, g))):
            if not f.endswith("_frames.parquet"):
                continue
            F = pd.read_parquet(os.path.join(base, g, f)); ok = F[F.status.isin(["ok", "ok_retry"])]
            rows.append(dict(generator=g, protein=f[:-len("_frames.parquet")], n_frames=len(F), n_ok=len(ok), n_retry=int((F.status == "ok_retry").sum()),
                             mean_t_fix_s=ok.t_fix_s.mean(), median_t_fix_s=ok.t_fix_s.median(), p95_t_fix_s=ok.t_fix_s.quantile(.95), max_t_fix_s=ok.t_fix_s.max(),
                             mean_t_count_s=ok.t_count_s.mean(), mean_t_sasa_s=ok.t_sasa_s.mean(), total_core_h=F.t_wall_all_attempts_s.sum() / 3600))
    T = pd.DataFrame(rows).round(4); f = os.path.join(REV, "tables", "S4_timing.csv"); T.to_csv(f, index=False)
    print(T.groupby("generator").agg(cells=("protein", "size"), frames=("n_frames", "sum"), core_h=("total_core_h", "sum"), mean_fix=("mean_t_fix_s", "mean"), max_fix=("max_t_fix_s", "max")).round(3).to_string())
    log.finish([f], note=f"cells={len(T)}")


def cmd_temps(a, INP, REV):
    log = revio.RunLogger(stage=4, task="temperature_map", cpu_procs=1, seeds="none")
    r = pd.read_parquet(os.path.join(REV, "tables", "S1_tier1a_residues_final.parquet"))
    g = r.groupby("protein_id").agg(T_C=("T_C", "first"), n_T=("T_C", "nunique"), T_assumed=("T_assumed", "max"), pH=("pH", "first")).reset_index()
    assert (g.n_T == 1).all(), "同一蛋白出现多个实验温度"
    g["T_K"] = g.T_C + 273.15; g["RT_kcal_at_Texp"] = bv.R_GAS * g.T_K; g["RT_kcal_at_298.15"] = RT
    g["note"] = np.where(g.T_assumed, "实验温度为假定值（原表无记录），dG_pred_Texp 对该蛋白只是形式上的", "实验温度取自原表")
    f = os.path.join(REV, "tables", "S4_temperature_map.csv"); g.to_csv(f, index=False); print(g.round(3).to_string(index=False)); log.finish([f], note="12 proteins")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["map", "parity", "leaderboard", "weights", "timing", "temps", "celllogs", "all"]); ap.add_argument("--allow-pending", action="store_true")
    a = ap.parse_args(); PROJ, REV, INP = revio.load_paths()
    for k, fn in (("map", cmd_map), ("parity", cmd_parity), ("leaderboard", cmd_leaderboard), ("celllogs", cmd_celllogs), ("weights", cmd_weights), ("timing", cmd_timing), ("temps", cmd_temps)):
        if a.cmd in (k, "all"):
            fn(a, INP, REV)
