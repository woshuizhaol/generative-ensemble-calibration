"""Stage 7/6 补充 · Str2Str_xtal 的种子间变异（同一个沉积结构起点上 5 次独立采样，种子 11–15）。
对每个蛋白、每个种子的 50 帧系综，口径 F 的全部 ok 帧（一次“抽样”），计算计数平均与 1/PF 平均两种形式的残基层 Spearman；报告：
  - 每个蛋白 ρ 的种子间均值与 SD（并与 8a 的沉积结构重建间 SD 并列）；
  - 对每个种子单独做 S7 的跨蛋白检验与结论编码（E_CA、E_LL、D、code），看种子 11 的结论（gain_countavg_only）是否对种子敏感；
  - 5 个种子 ρ 的中位数系综（把 5×50 帧当作一个 250 帧系综不在此做，因为各生成器的比较口径是 50 帧）。
输出：tables/S7_xtal_seed_variation.csv（逐蛋白逐种子）、S7_xtal_seed_tests.csv（逐种子的跨蛋白检验与编码）、S7_xtal_seed_summary.csv（种子间 SD 汇总）。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, ensemble as E  # noqa: E402
import s7_invpf_form as S7  # noqa: E402

PROJ, REV, INP = revio.load_paths(); TAB = os.path.join(REV, "tables"); SEEDS = (11, 12, 13, 14, 15)


def main():
    log = revio.RunLogger(stage=7, task="xtal_seed_variation", cpu_procs=1, seeds="generation seeds 11-15")
    rows = []
    for k in SEEDS:
        for p in S7.PROTS:
            cell = E.load_cell(f"Str2Str_xtal_s{k}", p, "F")
            if cell is None:
                raise SystemExit(f"Str2Str_xtal_s{k}/{p} 尚未合并")
            cols = np.arange(cell["NC"].shape[1])[None, :]
            for form in S7.FORMS:
                r = E.rho_draws(cell, p, cols, form)
                rows.append(dict(scenario=f"seed{k}", seed=k, generator="Str2Str_xtal", group=S7.GROUP["Str2Str_xtal"], protein=p, form=form, n_frames=cols.shape[1], rho_median=float(r[0]), baseline_rho=S7.BASE[p]))
    P = pd.DataFrame(rows); P.to_csv(os.path.join(TAB, "S7_xtal_seed_variation.csv"), index=False)
    T = S7.tests_for(P); T["code"] = [S7.code_of(r) for r in T.itertuples()]; T["seed"] = [int(s[4:]) for s in T.scenario]
    T[["seed", "n_proteins", "wins_count_avg", "wilcoxon_greater_p_count_avg", "median_delta_count_avg", "E_CA", "wins_invpf_avg", "wilcoxon_greater_p_invpf_avg", "median_delta_invpf_avg", "E_LL", "median_LL_minus_CA", "LL_minus_CA_boot_lo", "LL_minus_CA_boot_hi", "D", "code"]].to_csv(os.path.join(TAB, "S7_xtal_seed_tests.csv"), index=False)
    rs = pd.read_csv(os.path.join(TAB, "S8_recon_sd.csv")).set_index("protein"); summ = []
    for form in S7.FORMS:
        w = P[P.form == form].pivot(index="protein", columns="seed", values="rho_median")
        for p in w.index:
            summ.append(dict(form=form, protein=p, rho_mean_over_seeds=w.loc[p].mean(), rho_sd_over_seeds=w.loc[p].std(ddof=1), rho_min=w.loc[p].min(), rho_max=w.loc[p].max(), deposited_recon_sd=float(rs.loc[p, "rho_sd_over_10_recon"])))
    Sm = pd.DataFrame(summ); Sm.to_csv(os.path.join(TAB, "S7_xtal_seed_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(T[["seed", "wins_count_avg", "wilcoxon_greater_p_count_avg", "E_CA", "wins_invpf_avg", "wilcoxon_greater_p_invpf_avg", "E_LL", "D", "code"]].round(4).to_string(index=False))
    for form in S7.FORMS:
        x = Sm[Sm.form == form]; print(form, "median SD over seeds %.4f ; max %.4f ; vs median recon SD %.4f" % (x.rho_sd_over_seeds.median(), x.rho_sd_over_seeds.max(), x.deposited_recon_sd.median()))
    log.finish([os.path.join(TAB, "S7_xtal_seed_tests.csv")], note="xtal seed variation")


if __name__ == "__main__":
    main()
