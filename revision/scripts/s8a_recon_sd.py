"""Stage 8a · 沉积结构重建间 SD（revision_proposal.md S8 步骤 3；R2-m3：中位增益 0.074 对残差 SD 0.026 只差约 3 倍，不是一个数量级）。
对 12 个蛋白的沉积结构各做 10 次确定性重建（S4a），计算：
  (a) 单结构 ρ（Spearman，对实验 ΔG_open）与 Pearson 在 10 次重建间的 SD（ddof=1）
  (b) 两两重建之间的 Δρ（45 对）：SD 与 |Δρ| 的 95% 分位数——"结构对比自身另一次重建"的参照
  增益与倍数：中位增益（BioEmu 计数平均形式，口径 O，M=50 的 200 次抽样中位数 − 原稿单结构基线；以及原稿 null_baselines 的 delta_vs_ceiling）除以 12 个蛋白 SD 的中位数和最大值；
  每个蛋白的增益超过其 (b) 95% 分位数的个数。
  Barstar（1BTA）与 CheY（1CYE）的沉积结构自带氢，重建无随机性，SD 恰为 0，单独标注；SD 的汇总分"全部 12 个"与"有变异的 10 个"两种。
输出：tables/S8_recon_sd.csv、tables/S8_recon_sd_summary.csv。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, itertools
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, ensemble as E  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=8, task="recon_sd", cpu_procs=1, seeds="none")
PROTS = sorted(json.load(open(INP["tier1a_sequences"]))); ceil = pd.read_parquet(INP["operator_ceiling"]); BASE = dict(zip(ceil.protein, ceil.spearman)); OFF = dict(zip(ceil.protein, ceil.offset.astype(int)))
nb = pd.read_csv(os.path.join(INP["results_tables"], "null_baselines.csv"))


def recon_values(p):
    Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet")); e = E.exp_residues(p); y = dict(zip(e.resi.astype(int), e.dGopen_kcal.astype(float))); sp, pe = [], []
    for k in range(10):
        d = Rr[Rr.recon_k == k].set_index("resSeq"); lp = E.BC * d.NC.astype(float) + E.BH * d.NH.astype(float)
        a = np.array([(lp[int(r) + OFF[p]], y[int(r)]) for r in e.resi.astype(int) if (int(r) + OFF[p]) in lp.index]); sp.append(stats.spearmanr(a[:, 0], a[:, 1])[0]); pe.append(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    return np.array(sp), np.array(pe)


def main():
    P = pd.read_csv(os.path.join(REV, "tables", "S7_invpf_vs_countavg_per_protein.csv")); pri = P[(P.scenario == "primary") & (P.generator == "BioEmu") & (P.form == "count_avg")].set_index("protein")
    rows = []; pair_rows = []; st1 = pd.read_csv(os.path.join(INP["results_stats"], "stage1_reproducibility.csv")).set_index("protein")   # 原项目：5 个 EX2 蛋白 N=10 的 Pearson 均值/SD
    for p in PROTS:
        sp, pe = recon_values(p); pairs = np.array([sp[i] - sp[j] for i, j in itertools.combinations(range(10), 2)])
        pair_rows += [dict(protein=p, recon_i=i, recon_j=j, rho_i=float(sp[i]), rho_j=float(sp[j]), delta_rho=float(sp[i] - sp[j])) for i, j in itertools.combinations(range(10), 2)]
        gain_o = float(pri.loc[p, "rho_median"] - BASE[p]); gain_m = float(nb.set_index("protein").loc[p, "delta_vs_ceiling"]) if p in set(nb.protein) else np.nan
        rows.append(dict(protein=p, rho_orig_baseline=BASE[p], rho_recon_mean=sp.mean(), rho_recon_min=sp.min(), rho_recon_max=sp.max(), rho_sd_over_10_recon=sp.std(ddof=1), pearson_sd_over_10_recon=pe.std(ddof=1), pearson_sd_original_project_stage1=(float(st1.loc[p, 'pearson_sd']) if p in st1.index else np.nan), n_distinct_rho=len(set(np.round(sp, 9))),
                         pair_dRho_sd_45=pairs.std(ddof=1), pair_absdRho_p95_45=float(np.percentile(np.abs(pairs), 95)), pair_absdRho_max_45=float(np.abs(pairs).max()), gain_bioemu_vs_orig_baseline=gain_o, gain_manuscript_delta_vs_single_structure=gain_m,
                         gain_bioemu_vs_recon_mean=float(pri.loc[p, "rho_median"] - sp.mean()), deterministic_no_variation=bool(len(set(np.round(sp, 9))) == 1)))   # 用不同取值个数判断，不用 SD==0（相同数值的 np.std 可能是 1e-17）
    D = pd.DataFrame(rows); D["gain_exceeds_pair_p95"] = D.gain_bioemu_vs_orig_baseline > D.pair_absdRho_p95_45; D["gain_vs_recon_mean_exceeds_pair_p95"] = D.gain_bioemu_vs_recon_mean > D.pair_absdRho_p95_45
    f1 = os.path.join(REV, "tables", "S8_recon_sd.csv"); D.to_csv(f1, index=False); PR = pd.DataFrame(pair_rows); PR.to_csv(os.path.join(REV, "tables", "S8_recon_pairs.csv"), index=False); assert len(PR) == 12 * 45   # 方案验收：12 个蛋白 × 45 对齐全
    var = D[~D.deterministic_no_variation]; sd_all = D.rho_sd_over_10_recon.values; sd_var = var.rho_sd_over_10_recon.values; med_gain_m = float(np.median(D.gain_manuscript_delta_vs_single_structure.dropna())); med_gain_o = float(np.median(D.gain_bioemu_vs_orig_baseline)); med_gain_r = float(np.median(D.gain_bioemu_vs_recon_mean))
    S = pd.DataFrame([dict(gain_definition=g, median_gain=mg, n_proteins_all=len(D), sd_median_all12=float(np.median(sd_all)), sd_max_all12=float(sd_all.max()), ratio_to_median_sd_all12=mg / np.median(sd_all), ratio_to_max_sd_all12=mg / sd_all.max(),
                           n_proteins_with_variation=len(var), sd_median_varying=float(np.median(sd_var)), sd_max_varying=float(sd_var.max()), ratio_to_median_sd_varying=mg / np.median(sd_var), ratio_to_max_sd_varying=mg / sd_var.max(),
                           n_gain_exceeds_own_pair_p95=int(col.sum()), pearson_sd_cytc=float(D.set_index("protein").loc["cytochrome_c", "pearson_sd_over_10_recon"]), rho_sd_cytc=float(D.set_index("protein").loc["cytochrome_c", "rho_sd_over_10_recon"]))
                      for g, mg, col in (("manuscript delta_vs_ceiling (BioEmu all frames) vs original single baseline", med_gain_m, D.gain_bioemu_vs_orig_baseline > D.pair_absdRho_p95_45), ("BioEmu M=50 median draw vs original single baseline", med_gain_o, D.gain_exceeds_pair_p95),
                                         ("BioEmu M=50 median draw vs 10-recon mean baseline", med_gain_r, D.gain_vs_recon_mean_exceeds_pair_p95))])
    S.to_csv(os.path.join(REV, "tables", "S8_recon_sd_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(D[["protein", "rho_orig_baseline", "rho_recon_mean", "rho_recon_min", "rho_recon_max", "rho_sd_over_10_recon", "pearson_sd_over_10_recon", "pair_absdRho_p95_45", "gain_bioemu_vs_orig_baseline", "gain_bioemu_vs_recon_mean", "gain_exceeds_pair_p95"]].round(3).to_string(index=False))
    print(S.round(3).T.to_string()); log.finish([f1], note="recon sd")


if __name__ == "__main__":
    main()
