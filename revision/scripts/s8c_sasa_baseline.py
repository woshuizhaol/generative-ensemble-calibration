"""Stage 8c · SASA 基线（revision_proposal.md S8 步骤 1；R2-C6a）。
基线打分：保护度预测量取负的系综平均 rSASA（score = −⟨rSASA_res⟩）；变体 score = −⟨SASA_amideNH⟩。SASA 在全原子加氢构象上算（S4a/S4b，mdtraj shrake_rupley，探针 0.14 nm，960 点，除以 Tien 2013 理论最大 ASA）。
对象：沉积结构 10 次重建（每次重建一个 ρ，再取均值）；每个生成器——口径 F 冻结的 M=50 抽样（R=200，S4a 的 F_<生成器>_<蛋白>_M50.npz；Str2Str_xtal 为 S4b 种子 11 的 50 帧），
BV 列取同一组帧，保证可配对。编号偏移与 S7 相同（生成器：S4_parity；沉积结构：operator_ceiling.offset）。
回答三个问题（对每个生成器，12 个蛋白配对）：
  (i)  SASA 下系综是否仍优于结构（单侧配对 Wilcoxon，ρ_SASA(系综) > ρ_SASA(结构)）
  (ii) BV 是否优于 SASA 基线（同一系综：ρ_BV − ρ_SASA，配对 Wilcoxon 双侧，蛋白 bootstrap 95% CI，B=10⁴，seed 7）
  (iii) 系综在 BV 下的增益与在 SASA 下的增益之差：[ρ_BV(系综) − ρ_BV(结构)] − [ρ_SASA(系综) − ρ_SASA(结构)]
(i)(iii) 需要结构 ρ 有定义：输入为常数时 Spearman 无定义（CheY 的酰胺 SASA 在 38 个实验残基上、10 次重建里恒为 0），此时 (i)(iii) 在结构 ρ 有定义的蛋白上做（n_struct_defined、undefined_struct 列出），\n并另报“无定义记 0”的敏感性（*_undef0）；(ii) 只用系综 ρ，始终用全部蛋白。bootstrap 对每个（生成器、打分、检验）单独固定种子 [7, 生成器序号, 打分序号, 检验序号]。\n输出：tables/S8_sasa_baseline.csv（逐生成器、蛋白、打分方式）、tables/S8_sasa_tests.csv。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, revstats, ensemble as E, stats_extra as X  # noqa: E402

PROJ, REV, INP = revio.load_paths()
PROTS = sorted(json.load(open(INP["tier1a_sequences"]))); B = 10000
GENS = {"BioEmu": "BioEmu", "ESMFlow": "ESMFlow", "Boltz2": "Boltz2", "ShortMD": "ShortMD", "Str2Str_xtal": "Str2Str_xtal_s11", "Str2Str_legacy": "Str2Str"}
ceil = pd.read_parquet(INP["operator_ceiling"]); OFF_S = dict(zip(ceil.protein, ceil.offset.astype(int)))


def structure_rhos(p):
    """沉积结构 10 次重建：BV（count_avg，单构象）、−rSASA、−酰胺 SASA 各自的 ρ 均值。"""
    Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet")); e = E.exp_residues(p); y = dict(zip(e.resi.astype(int), e.dGopen_kcal.astype(float))); out = {"bv": [], "rsasa": [], "amide": []}
    for k in range(10):
        d = Rr[Rr.recon_k == k].set_index("resSeq"); keep = [int(r) for r in e.resi.astype(int) if (int(r) + OFF_S[p]) in d.index]; yy = np.array([y[r] for r in keep]); sel = d.loc[[r + OFF_S[p] for r in keep]]
        for nm, v in (("bv", E.BC * sel.NC.astype(float).values + E.BH * sel.NH.astype(float).values), ("rsasa", -sel.rSASA_res.astype(float).values), ("amide", -sel.SASA_amideNH.astype(float).values)):
            out[nm].append(stats.spearmanr(v, yy)[0])
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    log = revio.RunLogger(stage=8, task="sasa_baseline", cpu_procs=1, seeds="frozen F draws; bootstrap seed 7")
    STR = {p: structure_rhos(p) for p in PROTS}; rows = []
    MAN = pd.read_csv(os.path.join(REV, "tables", "S4_draw_indices_manifest_F.csv")); SHA = dict(zip(MAN.file.map(os.path.basename), MAN.sha256))   # 配对核对：SASA 与 BV 用同一组冻结抽样帧，记录索引文件的 sha256
    for gname, dname in GENS.items():
        for p in PROTS:
            cell = E.load_cell_sasa(dname, p)
            if cell is None:
                continue
            if gname == "Str2Str_xtal":
                cols = np.arange(cell["NC"].shape[1])[None, :]; meta = dict(M_eff=cols.shape[1], n_available=cols.shape[1], n_lt_M=cols.shape[1] < 50)
            else:
                fr, meta = E.load_draws("F", dname, p, 50); cols = E.frames_to_cols(cell, fr)
            idx, y, _ = E.align_idx(cell, p)
            sc = {"bv": E.ln_pf(cell["NC"], cell["NH"], cols, "count_avg")[:, idx], "rsasa": -np.nanmean(cell["rSASA"][:, cols], axis=-1).T[:, idx], "amide": -np.nanmean(cell["amideSASA"][:, cols], axis=-1).T[:, idx]}
            for nm, m in sc.items():
                r = E.spearman_rows(m, y); r = r[np.isfinite(r)]
                rows.append(dict(generator=gname, protein=p, score=nm, rho_median=float(np.median(r)), rho_p2_5=float(np.percentile(r, 2.5)), rho_p97_5=float(np.percentile(r, 97.5)), rho_structure_mean10recon=STR[p][nm], draw_index_sha256=(SHA.get(f"F_{dname}_{p}_M50.npz", "all_ok_frames") if gname != "Str2Str_xtal" else "all_ok_frames"), M_eff=meta["M_eff"], n_lt_M=meta["n_lt_M"], n_draws=len(r), n_residues=len(idx)))
    P = pd.DataFrame(rows); f1 = os.path.join(REV, "tables", "S8_sasa_baseline.csv"); P.to_csv(f1, index=False); rng = np.random.default_rng(7); out = []
    for gi, g in enumerate(P.generator.unique()):
        d = P[P.generator == g]; piv = d.pivot(index="protein", columns="score", values="rho_median"); st = d.drop_duplicates(["protein", "score"]).pivot(index="protein", columns="score", values="rho_structure_mean10recon"); n = len(piv)
        for si, sc in enumerate(("rsasa", "amide")):
            ib = lambda v, t: (lambda m: (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))))(np.median(v.values[np.random.default_rng([7, gi, si, t]).integers(0, len(v), size=(B, len(v)))], axis=1))
            diff_bv_sasa = piv["bv"] - piv[sc]; lo, hi = ib(diff_bv_sasa, 0)                       # (ii)：只用系综 ρ，全部蛋白
            ok = st[sc].notna() & st["bv"].notna()                                                 # (i)(iii)：结构 ρ 有定义的蛋白
            gain_sasa = (piv[sc] - st[sc])[ok]; gain_bv = (piv["bv"] - st["bv"])[ok]; gdiff = gain_bv - gain_sasa
            w1 = revstats.wilcoxon_greater(piv[sc][ok].values, st[sc][ok].values); lo2, hi2 = ib(gdiff, 1); hl2 = X.hodges_lehmann(diff_bv_sasa.values); hl3 = X.hodges_lehmann(gdiff.values)
            st0 = st[sc].fillna(0.0); g0 = piv[sc] - st0; w0 = revstats.wilcoxon_greater(piv[sc].values, st0.values); gd0 = (piv["bv"] - st["bv"]) - g0   # 敏感性：无定义记 0
            out.append(dict(generator=g, sasa_score=sc, n_proteins=n, n_struct_defined=int(ok.sum()), undefined_struct=";".join(st.index[~ok]),
                            i_wins_sasa_ens_gt_struct=int((gain_sasa > 0).sum()), i_wilcoxon_greater_p=w1[1], i_median_gain_sasa=float(np.median(gain_sasa)), median_gain_bv=float(np.median(gain_bv)),
                            ii_median_bv_minus_sasa=float(np.median(diff_bv_sasa)), ii_boot_lo=lo, ii_boot_hi=hi, ii_hl_estimate=hl2[0], ii_hl_lo=hl2[1], ii_hl_hi=hl2[2], ii_n_bv_better=int((diff_bv_sasa > 0).sum()), ii_wilcoxon_two_sided_p=float(stats.wilcoxon(diff_bv_sasa.values).pvalue),
                            iii_median_gain_bv_minus_gain_sasa=float(np.median(gdiff)), iii_boot_lo=lo2, iii_boot_hi=hi2, iii_hl_estimate=hl3[0], iii_hl_lo=hl3[1], iii_hl_hi=hl3[2], iii_n_bv_gain_larger=int((gdiff > 0).sum()),
                            median_rho_sasa_structure=float(st[sc].median()), median_rho_bv_structure=float(st["bv"].median()),
                            i_wins_undef0=int((g0 > 0).sum()), i_wilcoxon_greater_p_undef0=w0[1], i_median_gain_sasa_undef0=float(np.median(g0)), iii_median_gain_bv_minus_gain_sasa_undef0=float(np.median(gd0)), iii_n_bv_gain_larger_undef0=int((gd0 > 0).sum())))
    T = pd.DataFrame(out)
    for sc in ("rsasa", "amide"):   # 多重比较：5（或 6 个含 Str2Str_xtal）个生成器的 p 做 BH，按打分方式分别校正；(i) 单侧、(ii) 双侧各一组
        m = (T.sasa_score == sc) & (T.generator != "Str2Str_legacy")   # Str2Str_legacy 是对照，不进校正集合；有 Str2Str_xtal 后自动含入
        for col, q in (("i_wilcoxon_greater_p", "i_bh_q"), ("ii_wilcoxon_two_sided_p", "ii_bh_q")):
            ok = m & T[col].notna(); T.loc[ok, q] = X.bh(T.loc[ok, col].values)
    f2 = os.path.join(REV, "tables", "S8_sasa_tests.csv"); T.to_csv(f2, index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    cols = ["generator", "sasa_score", "n_struct_defined", "i_wins_sasa_ens_gt_struct", "i_wilcoxon_greater_p", "i_median_gain_sasa", "median_gain_bv", "ii_median_bv_minus_sasa", "ii_boot_lo", "ii_boot_hi", "ii_n_bv_better", "ii_wilcoxon_two_sided_p", "iii_median_gain_bv_minus_gain_sasa", "iii_boot_lo", "iii_boot_hi", "iii_n_bv_gain_larger"]
    print(T[cols].round(4).to_string(index=False)); print(T[["generator", "sasa_score", "undefined_struct", "i_wins_undef0", "i_wilcoxon_greater_p_undef0", "i_median_gain_sasa_undef0", "iii_median_gain_bv_minus_gain_sasa_undef0", "iii_n_bv_gain_larger_undef0"]].round(4).to_string(index=False))
    print("structure median rho: rSASA %.3f, amide %.3f (11 defined), BV %.3f" % (T[T.sasa_score == "rsasa"].median_rho_sasa_structure.iloc[0], T[T.sasa_score == "amide"].median_rho_sasa_structure.iloc[0], T.median_rho_bv_structure.iloc[0])); log.finish([f1, f2], note="sasa baseline")


if __name__ == "__main__":
    main()
