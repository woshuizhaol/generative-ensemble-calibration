"""Stage 6d · 生成系综的绝对指标（revision_proposal.md S6 步骤 3；R1-C3 第二点）。
对 EX2 子集 5 个蛋白 × 5 个生成器（BioEmu、ESMFlow、Boltz-2、ShortMD、Str2Str_xtal），计算与 s6a 相同的全部指标，并与沉积结构（10 次重建均值）配对比较；Barnase 单列。
系综预测：计数平均形式 ln PF = 0.35⟨N_C⟩ + 2⟨N_H⟩（等权），帧选择 = S7 主分析口径：口径 O 的 M=50（S4a 冻结的 200 次无放回抽样），M_eff = min(50, n)；
  ESMFlow、Boltz-2 本来就是 50 帧（1 个"抽样"）；BioEmu、ShortMD 报告 200 次抽样的中位数及 2.5–97.5 百分位。
  Str2Str_xtal：S4b 的打分（种子 11 的 50 帧；种子 12–15 另报种子间 SD）；需要 S4b 已合并，没有则明确跳过并在输出里标注。
编号偏移：旧生成器沿用 S4_parity 里的 offset（与原稿 leaderboard 复现一致）；Str2Str_xtal 按残基身份在 [−8, 8] 内取最佳偏移（bv.best_offset；若实验残基表的残基名为空——Barnase、RNase A、SNase——函数回退为 0，已用沉积结构残基名核对 0 为最优）。
5 个 EX2 蛋白的配对 bootstrap（B=10⁴，seed 7；共用一条 rng，Barnase 排在第一位，所以 Barnase 的区间与只做 Barnase 时逐位相同）：同一组残基重抽，ensemble 取口径 O 的全部旧列平均、structure 取 10 次重建均值，报告 Δ=ensemble−structure 的 95% CI。
输出：tables/S6_ex2_generators_absolute.csv、tables/S6_generators_vs_structure_paired.csv（ΔG_Texp 的行带 delta_ci_lo/hi）、tables/S6_barnase_generators.csv（Barnase 子集）。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402
from lib.abs_metrics import metrics  # noqa: E402
from lib.ensemble import np_load_verified  # noqa: E402
from scipy import stats  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=6, task="generators_absolute", cpu_procs=1, seeds="bootstrap seed 7; frozen draw indices")
EX2 = ["Barnase", "RNaseA", "SNase", "CI2", "cytochrome_c"]; GENS = ["BioEmu", "ESMFlow", "Boltz2", "ShortMD", "Str2Str_xtal"]; BC, BH = bv.BETA_C, bv.BETA_H; R_GAS = bv.R_GAS; B = 10000
S1 = pd.read_parquet(os.path.join(REV, "tables", "S1_tier1a_residues_final.parquet")); PAR = pd.read_csv(os.path.join(REV, "tables", "S4_parity.csv")).dropna(subset=["offset"])
ceil = pd.read_parquet(INP["operator_ceiling"]); OFF_STRUCT = dict(zip(ceil.protein, ceil.offset.astype(int)))


def exp_of(p):
    g = S1[S1.protein_id == p].dropna(subset=["dGopen_kcal"]); T = float(g.T_C.iloc[0]) + 273.15
    return g, T


def structure_pred(p):
    """沉积结构 10 次重建的 ln PF，对齐到实验残基；返回 (resi 数组, (10, n))。"""
    g, T = exp_of(p); Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet")); rows = []
    for k in range(10):
        d = Rr[Rr.recon_k == k].set_index("resSeq"); l = BC * d.NC.astype(float) + BH * d.NH.astype(float); rows.append([l.get(int(r) + OFF_STRUCT[p], np.nan) for r in g.resi.astype(int)])
    A = np.array(rows, float); keep = ~np.isnan(A).any(0); return g.resi.astype(int).values[keep], A[:, keep]


def gen_pred_old(g_name, p):
    """旧 npz（口径 O）：返回 (resSeq 数组, lnPF (res × cols))、offset。"""
    z = np.load(os.path.join(INP["gen_dir"], g_name, p, "operator_f.npz")); res = np.array([int(x) for x in z["resSeq"]]); l = BC * np.asarray(z["NC"], float) + BH * np.asarray(z["NH"], float)
    off = int(PAR[(PAR.generator == g_name) & (PAR.protein == p)].offset.iloc[0]); return res, l, off


def gen_pred_new(label, p):
    """S4b 的 parquet（Str2Str_xtal_s<k>）：返回 (resSeq 数组, lnPF (res × frames)) 或 None。"""
    f = os.path.join(REV, "raw_outputs", "s4", "counts", label, f"{p}.parquet")
    if not os.path.exists(f):
        return None
    R = pd.read_parquet(f); F = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", label, f"{p}_frames.parquet")); ok = F[F.status.isin(["ok", "ok_retry"])].frame_idx.values
    nc = R.pivot(index="resSeq", columns="frame_idx", values="NC")[sorted(ok)]; nh = R.pivot(index="resSeq", columns="frame_idx", values="NH")[sorted(ok)]
    resn = R.drop_duplicates("resSeq").set_index("resSeq").resn.to_dict()
    return nc.index.values.astype(int), BC * nc.values.astype(float) + BH * nh.values.astype(float), resn


def align(res, lnpf_cols, off, resi_struct):
    """按 resi = resSeq − off 对齐到结构的实验残基；返回 (n 个残基的行下标 in lnpf, 保留的 resi mask)。"""
    pos = {int(r) - off: i for i, r in enumerate(res)}; idx = np.array([pos.get(int(r), -1) for r in resi_struct]); return idx


def main():
    rng = np.random.default_rng(7); abs_rows, pair_rows, bar_rows = [], [], []
    for p in EX2:
        g, T = exp_of(p); resi_s, S = structure_pred(p); y_all = dict(zip(g.resi.astype(int), g.dGopen_kcal.astype(float) / (R_GAS * T)))
        for gen in GENS:
            cols = None; note = ""; n_frames = None; basis = "O"
            if gen == "Str2Str_xtal":
                got = gen_pred_new("Str2Str_xtal_s11", p)
                if got is None:
                    abs_rows.append(dict(protein=p, generator=gen, status="S4b_not_merged_yet")); continue
                res, L, resn = got; ge = g.rename(columns={"protein_id": "protein_id"}); off = bv.best_offset({int(r): bv.to1(resn[int(r)]) for r in res}, ge, 0); draws = np.arange(L.shape[1])[None, :]; n_frames = L.shape[1]; basis = "S4b (seed 11, all ok frames)"
            else:
                res, L, off = gen_pred_old(gen, p); z = np_load_verified("O", os.path.join(REV, "raw_outputs", "s4", "draw_indices", f"O_{gen}_{p}_M50.npz")); draws = z["frames"].astype(int); n_frames = L.shape[1]
            idx = align(res, L, off, resi_s); ok = idx >= 0
            if ok.sum() < 5:
                abs_rows.append(dict(protein=p, generator=gen, status="too_few_residues")); continue
            resi_c = resi_s[ok]; y = np.array([y_all[int(r)] for r in resi_c]); Lm = L[idx[ok]]          # (n, cols)
            pred_draws = np.stack([Lm[:, d].mean(1) for d in draws])                                   # (R, n)
            for space, scale, obs in (("lnPF", 1.0, y), ("dG_298", R_GAS * 298.15, y * R_GAS * T), ("dG_Texp", R_GAS * T, y * R_GAS * T)):
                m = metrics(pred_draws * scale, np.broadcast_to(obs, pred_draws.shape)); sp = np.array([stats.spearmanr(pr, obs)[0] for pr in pred_draws * scale])
                for k_, v in {**m, "spearman": sp}.items():
                    abs_rows.append(dict(protein=p, generator=gen, space=space, metric=k_, median_over_draws=float(np.median(v)), p2_5=float(np.percentile(v, 2.5)), p97_5=float(np.percentile(v, 97.5)), n_draws=len(draws), n_frames_per_draw=int(draws.shape[1]),
                                         n_frames_available=int(n_frames), n_residues=len(resi_c), offset_used=int(off), status="ok", basis=basis))
            # 与沉积结构的配对：结构取同一组残基
            sk = np.array([np.where(resi_s == r)[0][0] for r in resi_c]); Sk = S[:, sk]; ens_all = Lm.mean(1)
            for space, scale, obs in (("lnPF", 1.0, y), ("dG_Texp", R_GAS * T, y * R_GAS * T)):
                ms = [metrics(Sk[k] * scale, obs) for k in range(10)]; me = metrics(ens_all * scale, obs)
                for k_ in me:
                    sv = float(np.mean([x[k_] for x in ms])); row = dict(protein=p, generator=gen, space=space, metric=k_, ensemble_all_columns=float(me[k_]), structure_mean_10recon=sv, delta=float(me[k_]) - sv, n_residues=len(resi_c))
                    if space == "dG_Texp":
                        idxb = rng.integers(0, len(y), size=(B, len(y))); mb_e = metrics((ens_all * scale)[idxb], obs[idxb]); mb_s = np.mean([metrics((Sk[k] * scale)[idxb], obs[idxb])[k_] for k in range(10)], axis=0)
                        dd = mb_e[k_] - mb_s; row.update(delta_ci_lo=float(np.percentile(dd, 2.5)), delta_ci_hi=float(np.percentile(dd, 97.5))); bar_rows.append(row)
                    pair_rows.append(row)
    A = pd.DataFrame(abs_rows); Pp = pd.DataFrame(pair_rows); Bb = pd.DataFrame(bar_rows)
    f1 = os.path.join(REV, "tables", "S6_ex2_generators_absolute.csv"); f2 = os.path.join(REV, "tables", "S6_generators_vs_structure_paired.csv"); f3 = os.path.join(REV, "tables", "S6_barnase_generators.csv")
    A.to_csv(f1, index=False); Pp.to_csv(f2, index=False); Bb[Bb.protein == "Barnase"].to_csv(f3, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    ok = A[(A.status == "ok") & (A.space == "dG_Texp")]
    print(ok[ok.metric.isin(["pearson", "spearman", "rmse", "bias", "slope_exp_on_pred", "ccc"])].pivot_table(index=["protein", "generator"], columns="metric", values="median_over_draws").round(3).to_string())
    print("missing:", A[A.status != "ok"][["protein", "generator", "status"]].to_string(index=False))
    log.finish([f1, f2, f3], note=f"rows={len(A)} paired={len(Pp)}")


if __name__ == "__main__":
    main()
