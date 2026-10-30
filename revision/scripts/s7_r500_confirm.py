"""Stage 7 · R=500 应急补做（S7_decision_rule.md §4 最后一条；revision_proposal.md S7 风控第 4 条）：
“若同一形式在不同抽样之间 ρ 的 95% 区间宽于 0.3，把 R 增加到 500 确认中位数稳定。”
触发情况（S7_invpf_vs_countavg_per_protein.csv）：BioEmu 的 invpf_avg，CI2 与 Im7 在 primary/s1/s2，cyt c 在 s2，共 7 个格子宽度 >0.3。
做法：对 BioEmu 的全部 12 个蛋白 × 3 个场景（primary: 口径 O M=50；s1: 口径 O M=36；s2: 口径 F M=50）用冻结规则生成 R=500 次抽样
  （第 r 次的 SeedSequence 只依赖 r、格子和 M，所以前 200 行必须与已冻结的 R=200 文件逐位相同，先校验），再用与 s7_invpf_form.py 相同的检验代码
  在 R=200（冻结）和 R=500 上各算一遍，使两者的差异只来自抽样次数；另并列 R=100（冻结文件的前 100 次抽样）作抽样次数敏感性（S7 核查发现 D 的下界对此敏感）。新索引存 raw_outputs/s4/draw_indices_r500/，冻结文件不改。
不改变判定：冻结规则以 R=200 的主分析为准；R=500 的作用是“确认中位数稳定”，结果如实并列。
输出：tables/S7_r500_per_protein.csv、S7_r500_decision.csv、S7_r500_manifest.csv"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, ensemble as E  # noqa: E402
import s4_freeze_draw_indices as FR  # noqa: E402
import s7_invpf_form as S7  # noqa: E402

PROJ, REV, INP = revio.load_paths()
SCN = (("primary", "O", 50), ("s1", "O", 36), ("s2", "F", 50)); R_NEW = 500
OUT = os.path.join(REV, "raw_outputs", "s4", "draw_indices_r500"); os.makedirs(OUT, exist_ok=True)


def main():
    log = revio.RunLogger(stage=7, task="r500_confirm", cpu_procs=1, seeds=f"frozen rule: entropy={FR.ENTROPY_BASE}+r, spawn_key=(cell,M); bootstrap seed 7")
    rows100, rows200, rows500, man, per = [], [], [], [], []
    for name, basis, M in SCN:
        for p in S7.PROTS:
            cell = E.load_cell("BioEmu", p, basis)
            z = E.np_load_verified(basis, os.path.join(REV, "raw_outputs", "s4", "draw_indices", f"{basis}_BioEmu_{p}_M{M}.npz")); ids = z["available_ids"]; fr200 = z["frames"]
            if int(z["n_available"]) <= M:   # n≤M：只有“全部帧”一次，不存在抽样次数问题
                fr500 = fr200; ndist = 1
            else:
                fr500, _, ndist, _ = FR.make_draws(ids, M, R_NEW, basis, "BioEmu", p)
                assert fr500.shape[0] == R_NEW and np.array_equal(fr500[:fr200.shape[0]], fr200), f"{name}/{p}: 前 {fr200.shape[0]} 次抽样与冻结文件不一致"
            path = os.path.join(OUT, f"{basis}_BioEmu_{p}_M{M}_R{R_NEW}.npz"); np.savez(path, frames=fr500, available_ids=ids, M=M, R=fr500.shape[0], basis=basis, protein=p)
            man.append(dict(scenario=name, protein=p, basis=basis, M=M, n_available=int(z["n_available"]), R_frozen=fr200.shape[0], R_new=fr500.shape[0], first_rows_identical_to_frozen=True, n_distinct_new=ndist, file=os.path.relpath(path, REV), sha256=revio.sha256(path)))
            for form in S7.FORMS:
                for tag, fr, rows in (("R100", fr200[:100], rows100), ("R200", fr200, rows200), ("R500", fr500, rows500)):
                    cols = fr if basis == "O" else E.frames_to_cols(cell, fr)
                    r = E.rho_draws(cell, p, cols, form); r = r[np.isfinite(r)]
                    rows.append(dict(scenario=name, basis=basis, M_requested=M, generator="BioEmu", group=S7.GROUP["BioEmu"], protein=p, form=form, n_draws=len(r), rho_median=float(np.median(r)), rho_p2_5=float(np.percentile(r, 2.5)), rho_p97_5=float(np.percentile(r, 97.5)), baseline_rho=S7.BASE[p]))
                    if tag == "R500":
                        h1 = float(np.median(r[:len(r) // 2])); h2 = float(np.median(r[len(r) // 2:])); per.append(dict(scenario=name, protein=p, form=form, width_R500=float(np.percentile(r, 97.5) - np.percentile(r, 2.5)), median_first_half=h1, median_second_half=h2, abs_diff_halves=abs(h1 - h2)))
    A = pd.DataFrame(rows200); Bn = pd.DataFrame(rows500)
    # 校验：这里重算的 R=200 中位数必须与 S7 主表一致（同一冻结索引、同一代码）
    ref = pd.read_csv(os.path.join(REV, "tables", "S7_invpf_vs_countavg_per_protein.csv"), float_precision="round_trip"); ref = ref[(ref.generator == "BioEmu") & ref.scenario.isin([s[0] for s in SCN])]
    m = A.merge(ref, on=["scenario", "protein", "form"], suffixes=("", "_ref")); assert len(m) == len(A) and np.abs(m.rho_median - m.rho_median_ref).max() < 1e-12, "R=200 重算与 S7 主表不一致"
    P = A.merge(Bn, on=["scenario", "protein", "form"], suffixes=("_R200", "_R500")); P["median_change"] = P.rho_median_R500 - P.rho_median_R200; P["width_R200"] = P.rho_p97_5_R200 - P.rho_p2_5_R200; P["width_R500"] = P.rho_p97_5_R500 - P.rho_p2_5_R500
    P = P.merge(pd.DataFrame(per), on=["scenario", "protein", "form"], suffixes=("", "_h")).drop(columns=["width_R500_h"]); P["triggered_width_gt_0p3_R200"] = P.width_R200 > 0.3
    P[["scenario", "protein", "form", "n_draws_R200", "n_draws_R500", "rho_median_R200", "rho_median_R500", "median_change", "width_R200", "width_R500", "triggered_width_gt_0p3_R200", "median_first_half", "median_second_half", "abs_diff_halves"]].to_csv(os.path.join(REV, "tables", "S7_r500_per_protein.csv"), index=False)
    # 判定：同一检验代码（每个（场景，生成器）单独以种子 7 起算 bootstrap，所以 R=100/200/500 之间的差异只来自抽样次数）
    out = []
    for tag, df in (("R100", pd.DataFrame(rows100)), ("R200", A), ("R500", Bn)):
        T = S7.tests_for(df); T["code"] = [S7.code_of(r) for r in T.itertuples()]; T["draws"] = tag; out.append(T)
    D = pd.concat(out, ignore_index=True)[["draws", "scenario", "generator", "n_proteins", "wins_count_avg", "wilcoxon_greater_p_count_avg", "wins_invpf_avg", "wilcoxon_greater_p_invpf_avg", "E_CA", "E_LL", "median_LL_minus_CA", "LL_minus_CA_boot_lo", "LL_minus_CA_boot_hi", "D", "code"]]
    D.to_csv(os.path.join(REV, "tables", "S7_r500_decision.csv"), index=False); pd.DataFrame(man).to_csv(os.path.join(REV, "tables", "S7_r500_manifest.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(D.round(4).to_string(index=False)); t = P[P.triggered_width_gt_0p3_R200]; print(t[["scenario", "protein", "form", "rho_median_R200", "rho_median_R500", "median_change", "width_R200", "width_R500", "abs_diff_halves"]].round(4).to_string(index=False))
    print("max |median change| over all BioEmu cells: %.4f ; over triggered cells: %.4f ; max |half-half| over all: %.4f" % (P.median_change.abs().max(), t.median_change.abs().max(), P.abs_diff_halves.max()))
    log.finish([os.path.join(REV, "tables", "S7_r500_decision.csv")], note="R=500 contingency")


if __name__ == "__main__":
    main()
