"""Stage 6e · Str2Str_xtal 的种子间变异（同一个沉积结构起点上 5 次独立采样，种子 11–15）：EX2 子集 5 个蛋白的绝对指标。
对每个蛋白、每个种子（S4b 的 Str2Str_xtal_s11…s15，各 50 帧，全部 ok 帧作为一次“抽样”），与 s6d 相同的指标（ΔG 空间、实验温度）；
报告每个指标的种子间均值、SD、最小、最大，以及与沉积结构 10 次重建均值之差 Δ 的种子间 SD（结构基线对 5 个种子相同）。
输出 tables/S6_xtal_seed_variation.csv（逐蛋白 × 逐种子 × 指标）与 tables/S6_xtal_seed_summary.csv（种子间汇总）。"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402
from lib.abs_metrics import metrics  # noqa: E402
import s6d_generators_absolute as D  # noqa: E402

PROJ, REV, INP = revio.load_paths(); TAB = os.path.join(REV, "tables"); SEEDS = (11, 12, 13, 14, 15); R_GAS = bv.R_GAS


def main():
    log = revio.RunLogger(stage=6, task="xtal_seed_variation", cpu_procs=1, seeds="generation seeds 11-15")
    rows = []
    for p in D.EX2:
        g, T = D.exp_of(p); resi_s, S = D.structure_pred(p); y_all = dict(zip(g.resi.astype(int), g.dGopen_kcal.astype(float)))
        for k in SEEDS:
            got = D.gen_pred_new(f"Str2Str_xtal_s{k}", p)
            if got is None:
                continue
            res, L, resn = got; off = bv.best_offset({int(r): bv.to1(resn[int(r)]) for r in res}, g, 0); idx = D.align(res, L, off, resi_s); ok = idx >= 0
            resi_c = resi_s[ok]; obs = np.array([y_all[int(r)] for r in resi_c]); ens = L[idx[ok]].mean(1) * R_GAS * T; sk = np.array([np.where(resi_s == r)[0][0] for r in resi_c])
            me = metrics(ens, obs); ms = [metrics(S[j][sk] * R_GAS * T, obs) for j in range(10)]
            for m_, v in me.items():
                sv = float(np.mean([x[m_] for x in ms])); rows.append(dict(protein=p, seed=k, metric=m_, ensemble=float(v), structure_mean_10recon=sv, delta=float(v) - sv, n_residues=len(resi_c), n_frames=int(L.shape[1]), offset_used=int(off)))
    V = pd.DataFrame(rows); f1 = os.path.join(TAB, "S6_xtal_seed_variation.csv"); V.to_csv(f1, index=False)
    S_ = V.groupby(["protein", "metric"]).agg(n_seeds=("seed", "nunique"), ensemble_mean=("ensemble", "mean"), ensemble_sd=("ensemble", lambda x: float(np.std(x, ddof=1))), ensemble_min=("ensemble", "min"), ensemble_max=("ensemble", "max"),
                                           delta_mean=("delta", "mean"), delta_sd=("delta", lambda x: float(np.std(x, ddof=1))), delta_min=("delta", "min"), delta_max=("delta", "max")).reset_index()
    f2 = os.path.join(TAB, "S6_xtal_seed_summary.csv"); S_.to_csv(f2, index=False); pd.set_option("display.width", 220)
    print(S_[S_.metric.isin(["pearson", "ccc", "bias"])].round(3).to_string(index=False))
    log.finish([f1, f2], note=f"rows={len(V)}")


if __name__ == "__main__":
    main()
