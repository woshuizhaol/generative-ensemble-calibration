"""Stage 7 · 补充分析（规则见 tables/S7_decision_rule.md §4；不改变判定）：
  w_max 支配度     S7_dominance_wmax.csv           每个（生成器, 蛋白）：1/PF 平均中贡献最大的单个构象所占比例 >0.5 的残基比例（200 次抽样的中位数与 2.5/97.5 百分位）
  系综 vs 单构象   S7_ensemble_vs_single_conformers.csv   系综 ρ 与 M_eff 个单构象 ρ 的中位数、最大值；系综 ρ 超过单构象中位数的蛋白数
  EX2 绝对指标     S7_invpf_absolute_ex2.csv       5 个 EX2 蛋白，两种形式的 Pearson、RMSE、偏差、斜率（实验对预测）、CCC（ΔG 空间, 实验温度），200 次抽样中位数
  收敛曲线         S7_convergence.csv              BioEmu，口径 F：M ∈ {10, 25, 50, 100, 200, 400}（R=50）；口径 O：只含 M ≤ n_available 的格子
全部使用 S4a 冻结的抽样索引（口径 O 的 M=50 为主）。Str2Str_xtal 在 S4b 合并后自动纳入。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, ensemble as E  # noqa: E402
from lib.abs_metrics import metrics  # noqa: E402
import s7_invpf_form as S7  # noqa: E402

PROJ, REV, INP = revio.load_paths()
EX2 = ["Barnase", "RNaseA", "SNase", "CI2", "cytochrome_c"]; R_GAS = bv.R_GAS


def primary_cols(gname, dname, p):
    cell = E.load_cell(dname, p, "F" if gname == "Str2Str_xtal" else "O")
    if cell is None:
        return None, None, None
    got = S7.get_cols(("O", 50), gname, dname, p, cell)
    return (None, None, None) if got is None else (cell, got[0], got[1])


def main():
    log = revio.RunLogger(stage=7, task="extras", cpu_procs=1, seeds="frozen draw indices")
    dom, ens, absr = [], [], []
    for gname, dname in S7.GENS.items():
        for p in S7.PROTS:
            cell, cols, meta = primary_cols(gname, dname, p)
            if cell is None:
                continue
            idx, y, _ = E.align_idx(cell, p); wm = E.w_max(cell["NC"], cell["NH"], cols)           # (R, n_res)
            frac_exp = (wm[:, idx] > 0.5).mean(1); frac_all = (np.where(np.isfinite(wm), wm, 0) > 0.5).sum(1) / np.isfinite(wm).sum(1)
            dom.append(dict(generator=gname, protein=p, M_eff=meta["M_eff"], n_lt_M=meta["n_lt_M"], n_draws=len(cols), frac_wmax_gt_0p5_expres_median=float(np.median(frac_exp)), frac_expres_p2_5=float(np.percentile(frac_exp, 2.5)), frac_expres_p97_5=float(np.percentile(frac_exp, 97.5)),
                            frac_wmax_gt_0p5_allres_median=float(np.median(frac_all)), mean_wmax_expres_median=float(np.median(wm[:, idx].mean(1))), n_expres=len(idx)))
            c0 = cols[0]; single = E.rho_draws(cell, p, c0[:, None], "count_avg"); single = single[np.isfinite(single)]
            for form in S7.FORMS:
                r = E.rho_draws(cell, p, cols, form); r = r[np.isfinite(r)]; med = float(np.median(r))
                ens.append(dict(generator=gname, protein=p, form=form, ensemble_rho_median=med, single_conformer_rho_median=float(np.median(single)), single_conformer_rho_max=float(np.max(single)), n_single=len(single), ensemble_gt_single_median=bool(med > np.median(single)), ensemble_gt_single_max=bool(med > np.max(single))))
            if p in EX2:
                e = E.exp_residues(p); T = float(e.T_C.iloc[0]) + 273.15
                for form in S7.FORMS:
                    pred = E.ln_pf(cell["NC"], cell["NH"], cols, form)[:, idx] * R_GAS * T; obs = np.broadcast_to(y, pred.shape)
                    m = metrics(pred, obs)
                    for k_, v in m.items():
                        if k_ in ("pearson", "rmse", "bias", "slope_exp_on_pred", "slope_pred_on_exp", "ccc"):
                            absr.append(dict(generator=gname, protein=p, form=form, metric=k_, median=float(np.median(v)), p2_5=float(np.percentile(v, 2.5)), p97_5=float(np.percentile(v, 97.5)), n_draws=len(cols), n_residues=len(idx)))
    D = pd.DataFrame(dom); En = pd.DataFrame(ens); A = pd.DataFrame(absr)
    summ = En.groupby(["generator", "form"]).agg(n_proteins=("protein", "size"), n_ensemble_gt_single_median=("ensemble_gt_single_median", "sum"), n_ensemble_gt_single_max=("ensemble_gt_single_max", "sum")).reset_index()
    En.to_csv(os.path.join(REV, "tables", "S7_ensemble_vs_single_conformers.csv"), index=False); summ.to_csv(os.path.join(REV, "tables", "S7_ensemble_vs_single_summary.csv"), index=False)
    D.to_csv(os.path.join(REV, "tables", "S7_dominance_wmax.csv"), index=False); A.to_csv(os.path.join(REV, "tables", "S7_invpf_absolute_ex2.csv"), index=False)
    # 收敛曲线（BioEmu）
    conv = []
    for basis in ("F", "O"):
        for p in S7.PROTS:
            cell = E.load_cell("BioEmu", p, basis); n = cell["NC"].shape[1]
            for M in (10, 25, 50, 100, 200, 400):
                f = os.path.join(REV, "raw_outputs", "s4", "draw_indices", f"{basis}_BioEmu_{p}_M{M}.npz")
                if not os.path.exists(f) or (basis == "O" and M > n):
                    continue
                fr, meta = E.load_draws(basis, "BioEmu", p, M); cols = fr if basis == "O" else E.frames_to_cols(cell, fr)
                for form in S7.FORMS:
                    r = E.rho_draws(cell, p, cols, form); r = r[np.isfinite(r)]
                    conv.append(dict(basis=basis, protein=p, M=M, M_eff=meta["M_eff"], n_available=meta["n_available"], n_draws=len(r), form=form, rho_median=float(np.median(r)), rho_p2_5=float(np.percentile(r, 2.5)), rho_p97_5=float(np.percentile(r, 97.5)), baseline_rho=S7.BASE[p]))
    C = pd.DataFrame(conv); C.to_csv(os.path.join(REV, "tables", "S7_convergence.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summ.to_string(index=False)); print(D.groupby("generator").frac_wmax_gt_0p5_expres_median.agg(["median", "min", "max"]).round(3).to_string())
    print(C[C.basis == "F"].groupby(["form", "M"]).rho_median.median().unstack().round(3).to_string())
    log.finish([os.path.join(REV, "tables", f) for f in ("S7_ensemble_vs_single_conformers.csv", "S7_dominance_wmax.csv", "S7_invpf_absolute_ex2.csv", "S7_convergence.csv")], note=f"dom={len(D)} conv={len(C)}")


if __name__ == "__main__":
    main()
