"""Stage 8c 诊断 · 沉积结构上 SASA 打分的并列程度（解释酰胺 SASA 变体的增益，R2-C6a）。
单个结构里被溶剂完全遮蔽的酰胺 N 的 SASA 恰为 0，实验残基上这些值并列；对 M 个构象取平均后并列被打破，
ρ(系综) − ρ(结构) 里因此混有“打破并列”的成分。本脚本对 12 个蛋白、10 次重建，在有实验 ΔG_open 的残基上统计：
  frac_at_mode  最大并列组（众数）占实验残基的比例；n_distinct  不同取值个数；frac_zero  SASA 恰为 0 的残基比例。
对 rSASA_res、SASA_amideNH、BV（−ln PF 的 −N 组合）三种打分各算一遍，取 10 次重建的均值。
另做对照（S8_sasa_single_vs_ensemble.csv）：同一生成器的同一批冻结抽样帧里，单个构象（M=1，逐帧）的 SASA 打分 ρ 与 50 帧系综的 ρ 对比，并给出单帧打分在实验残基上恰为 0 的比例；
回答“酰胺 SASA 变体下系综相对沉积结构的增益有多少来自把并列打散（分辨率），多少来自更好的预测”——本对照只量化前者的上限，不能把两者分开归因。
输出：tables/S8_sasa_ties.csv、tables/S8_sasa_single_vs_ensemble.csv"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, ensemble as E  # noqa: E402
from scipy import stats as _st

PROJ, REV, INP = revio.load_paths()
PROTS = sorted(json.load(open(INP["tier1a_sequences"])))
ceil = pd.read_parquet(INP["operator_ceiling"]); OFF_S = dict(zip(ceil.protein, ceil.offset.astype(int)))


def tie_stats(v):
    u, c = np.unique(np.round(v, 12), return_counts=True)
    return dict(frac_at_mode=float(c.max() / len(v)), n_distinct=int(len(u)), frac_zero=float(np.mean(v == 0.0)))


def main():
    log = revio.RunLogger(stage=8, task="sasa_ties", cpu_procs=1, seeds="none (deterministic)")
    rows = []
    for p in PROTS:
        Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet")); e = E.exp_residues(p)
        per = {"rsasa": [], "amide": [], "bv": []}
        for k in range(10):
            d = Rr[Rr.recon_k == k].set_index("resSeq"); keep = [int(r) for r in e.resi.astype(int) if (int(r) + OFF_S[p]) in d.index]; sel = d.loc[[r + OFF_S[p] for r in keep]]
            per["rsasa"].append(tie_stats(sel.rSASA_res.astype(float).values)); per["amide"].append(tie_stats(sel.SASA_amideNH.astype(float).values))
            per["bv"].append(tie_stats(E.BC * sel.NC.astype(float).values + E.BH * sel.NH.astype(float).values))
        for sc, lst in per.items():
            rows.append(dict(protein=p, score=sc, n_residues=len(keep), **{k: float(np.mean([x[k] for x in lst])) for k in ("frac_at_mode", "n_distinct", "frac_zero")}))
    T = pd.DataFrame(rows); f = os.path.join(REV, "tables", "S8_sasa_ties.csv"); T.to_csv(f, index=False)
    pd.set_option("display.width", 200)
    print(T.pivot(index="protein", columns="score", values="frac_at_mode").round(3).to_string()); print(T.pivot(index="protein", columns="score", values="n_distinct").round(1).to_string())
    print(T.groupby("score")[["frac_at_mode", "frac_zero", "n_distinct"]].median().round(3).to_string())
    # ---- 单帧 vs 50 帧系综（口径 F，冻结抽样的第 0 次；其余 49 次只用于系综 ρ 的中位数，取自 S8_sasa_baseline.csv）
    GENS = {"BioEmu": "BioEmu", "ESMFlow": "ESMFlow", "Boltz2": "Boltz2", "ShortMD": "ShortMD", "Str2Str_xtal": "Str2Str_xtal_s11", "Str2Str_legacy": "Str2Str"}; base = pd.read_csv(os.path.join(REV, "tables", "S8_sasa_baseline.csv"), float_precision="round_trip"); rows = []
    for g, dname in GENS.items():
        for p in PROTS:
            cell = E.load_cell_sasa(dname, p)
            if g == "Str2Str_xtal":   # 种子 11 的 50 帧全部使用（没有冻结抽样）；单帧 = 其中每一帧
                cols = np.arange(cell["NC"].shape[1])
            else:
                fr, meta = E.load_draws("F", dname, p, 50); cols = E.frames_to_cols(cell, fr)[0]
            idx, y, _ = E.align_idx(cell, p); c1 = cols[:, None]
            sc = {"bv": E.ln_pf(cell["NC"], cell["NH"], c1, "count_avg")[:, idx], "rsasa": -np.nanmean(cell["rSASA"][:, c1], axis=-1).T[:, idx], "amide": -np.nanmean(cell["amideSASA"][:, c1], axis=-1).T[:, idx]}
            for nm, m in sc.items():
                r = E.spearman_rows(m, y); ens = base[(base.generator == g) & (base.protein == p) & (base.score == nm)].rho_median.iloc[0]
                rows.append(dict(generator=g, protein=p, score=nm, n_single_frames=len(cols), rho_single_frame_median=float(np.nanmedian(r)), n_single_frame_rho_undefined=int(np.isnan(r).sum()), rho_ensemble_M50_median=float(ens), frac_exp_residues_exactly_zero_single_frame=float(np.mean(m == 0.0))))
    Sg = pd.DataFrame(rows); Sg.to_csv(os.path.join(REV, "tables", "S8_sasa_single_vs_ensemble.csv"), index=False)
    print(Sg.groupby(["generator", "score"])[["rho_single_frame_median", "rho_ensemble_M50_median", "frac_exp_residues_exactly_zero_single_frame"]].median().round(3).to_string())
    log.finish([f], note="sasa ties diagnostic")


if __name__ == "__main__":
    main()
