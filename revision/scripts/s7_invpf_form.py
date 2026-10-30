"""Stage 7 · 1/PF 平均形式与计数平均形式的对照（主脚本）。判定规则见 tables/S7_decision_rule.md（已冻结，sha256 在 S7_decision_rule.sha256）。
场景（scenario）：
  primary   口径 O，M=50：旧 npz 的列，S4a 冻结的 O_<生成器>_<蛋白>_M50.npz（R=200 次无放回抽样；n≤50 则全部列，标 n_lt_M）。Str2Str_xtal：S4b 种子 11 的全部 ok 帧（50 帧）
  s1        口径 O，M=36：共同样本量
  s2        口径 F，M=50：S4a 的全帧重打分（被旧打分丢掉的帧回来了）
  sec_a     口径 O，全部列      sec_b  口径 F，全部帧
每个（场景, 生成器, 蛋白, 形式）报告 200 次抽样 ρ 的中位数与 2.5/97.5 百分位，以及对单结构基线的 Δρ。基线：operator_ceiling.spearman（原稿，主）；沉积结构 10 次重建 ρ 的均值（敏感性）。
输出：tables/S7_invpf_vs_countavg_per_protein.csv、S7_invpf_cross_protein_tests.csv、S7_decision_table.csv、S7_decision_outcome.json。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, hashlib, argparse
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, revstats, ensemble as E  # noqa: E402

PROJ, REV, INP = revio.load_paths()
PROTS = sorted(json.load(open(INP["tier1a_sequences"]))); FORMS = ["count_avg", "invpf_avg"]
GENS = {"BioEmu": "BioEmu", "ESMFlow": "ESMFlow", "Boltz2": "Boltz2", "ShortMD": "ShortMD", "Str2Str_xtal": "Str2Str_xtal_s11", "Str2Str_legacy": "Str2Str"}   # 报告名 → 数据名
TRAINED_ON_DGFOLD = {"BioEmu": True, "ESMFlow": False, "Boltz2": False, "Str2Str_xtal": False, "ShortMD": False, "Str2Str_legacy": False}   # §0.4 标准列名 trained_on_dGfold
GROUP = {"BioEmu": "fine-tuned on experimental stability", "ESMFlow": "no stability-data training", "Boltz2": "no stability-data training", "Str2Str_xtal": "no stability-data training", "ShortMD": "physics simulation", "Str2Str_legacy": "control only"}
DECISION_GENS = ["BioEmu", "ESMFlow", "Boltz2", "Str2Str_xtal", "ShortMD"]
ceil = pd.read_parquet(INP["operator_ceiling"]); BASE = dict(zip(ceil.protein, ceil.spearman)); OFF_S = dict(zip(ceil.protein, ceil.offset.astype(int)))
B = 10000


def recon_baseline(p):
    """沉积结构 10 次重建各自的 ρ（单构象，两种形式相同），返回均值。"""
    Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet")); e = E.exp_residues(p); y = dict(zip(e.resi.astype(int), e.dGopen_kcal.astype(float))); rhos = []
    for k in range(10):
        d = Rr[Rr.recon_k == k].set_index("resSeq"); lp = E.BC * d.NC.astype(float) + E.BH * d.NH.astype(float); pairs = [(lp[int(r) + OFF_S[p]], y[int(r)]) for r in e.resi.astype(int) if (int(r) + OFF_S[p]) in lp.index]
        a = np.array(pairs); rhos.append(stats.spearmanr(a[:, 0], a[:, 1])[0])
    return float(np.mean(rhos))


RECON = {p: recon_baseline(p) for p in PROTS}


def get_cols(scn, gname, dname, p, cell):
    """返回 (cols (R,M), meta)；找不到就返回 None。"""
    basis, M = scn
    if gname == "Str2Str_xtal":   # S4b：没有冻结的口径 O 索引；全部 ok 帧
        n = cell["NC"].shape[1]
        if M == "all" or M == 50:
            return np.arange(n)[None, :], dict(M_eff=n, n_available=n, n_lt_M=bool(n < 50), n_distinct=1)
        f = os.path.join(REV, "raw_outputs", "s4", "draw_indices", f"F_{dname}_{p}_M{M}.npz")
        if not os.path.exists(f):
            return None
        E._verify_draw_file("F", f); z = np.load(f); return E.frames_to_cols(cell, z["frames"].astype(int)), dict(M_eff=int(z["M_eff"]), n_available=int(z["n_available"]), n_lt_M=bool(z["n_lt_M"]), n_distinct=int(z["n_distinct_draws"]))
    if M == "all":
        n = cell["NC"].shape[1]; return np.arange(n)[None, :], dict(M_eff=n, n_available=n, n_lt_M=False, n_distinct=1)
    fr, meta = E.load_draws(basis, dname, p, M)
    return (fr if basis == "O" else E.frames_to_cols(cell, fr)), meta


def run_scenario(name, scn, rows):
    basis = scn[0]
    for gname, dname in GENS.items():
        for p in PROTS:
            if gname == "Str2Str_xtal" and basis == "O" and name not in ("primary", "s1", "sec_a"):
                continue
            cell = E.load_cell(dname, p, "F" if (basis == "F" or gname == "Str2Str_xtal") else "O")
            if cell is None:
                continue
            got = get_cols(scn, gname, dname, p, cell)
            if got is None:
                continue
            cols, meta = got
            for form in FORMS:
                r = E.rho_draws(cell, p, cols, form); r = r[np.isfinite(r)]
                rows.append(dict(scenario=name, basis=basis, M_requested=scn[1], generator=gname, group=GROUP[gname], trained_on_dGfold=TRAINED_ON_DGFOLD[gname], protein=p, form=form, n_available=meta["n_available"], M_eff=meta["M_eff"], n_lt_M=meta["n_lt_M"], n_distinct_draws=meta["n_distinct"], n_draws=len(r),
                                 rho_median=float(np.median(r)), rho_p2_5=float(np.percentile(r, 2.5)), rho_p97_5=float(np.percentile(r, 97.5)), baseline_rho=BASE[p], delta_median=float(np.median(r)) - BASE[p],
                                 delta_p2_5=float(np.percentile(r, 2.5)) - BASE[p], delta_p97_5=float(np.percentile(r, 97.5)) - BASE[p], baseline_recon_mean=RECON[p], delta_vs_recon_mean=float(np.median(r)) - RECON[p]))


def boot_median_ci(d, rng):
    idx = rng.integers(0, len(d), size=(B, len(d))); m = np.median(d[idx], axis=1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def tests_for(df):
    out = []
    for (scn, g), d in df.groupby(["scenario", "generator"]):
        rng = np.random.default_rng(7)   # 每个（场景, 生成器）单独以 seed 7 起算，结果不依赖其他组的顺序，加入 Str2Str_xtal 也不会移动已有区间
        piv = d.pivot(index="protein", columns="form", values="rho_median"); base = d.drop_duplicates("protein").set_index("protein").baseline_rho.reindex(piv.index); n = len(piv)
        row = dict(scenario=scn, generator=g, group=GROUP[g], trained_on_dGfold=TRAINED_ON_DGFOLD[g], n_proteins=n)
        for form in FORMS:
            diff = (piv[form] - base).values; w = revstats.wilcoxon_greater(piv[form].values, base.values) if n >= 6 and np.any(diff != 0) else (np.nan, np.nan)
            row[f"wins_{form}"] = int((diff > 0).sum()); row[f"wilcoxon_greater_p_{form}"] = w[1]; row[f"median_delta_{form}"] = float(np.median(diff))
            row[f"E_{'CA' if form == 'count_avg' else 'LL'}"] = bool(w[1] < 0.05 and (diff > 0).sum() >= 9)
        dll = (piv["invpf_avg"] - piv["count_avg"]).values; lo, hi = boot_median_ci(dll, rng)
        row.update(median_LL_minus_CA=float(np.median(dll)), LL_minus_CA_boot_lo=lo, LL_minus_CA_boot_hi=hi, n_LL_gt_CA=int((dll > 0).sum()), n_LL_lt_CA=int((dll < 0).sum()),
                   wilcoxon_two_sided_p_LL_vs_CA=(float(stats.wilcoxon(dll[dll != 0]).pvalue) if (dll != 0).sum() >= 1 else np.nan),
                   sign_test_p_LL_vs_CA=(float(stats.binomtest(int((dll > 0).sum()), int((dll != 0).sum()), 0.5).pvalue) if (dll != 0).sum() else np.nan),
                   D=("LL>CA" if lo > 0 else ("CA>LL" if hi < 0 else "≈")))
        out.append(row)
    return pd.DataFrame(out)


def code_of(r):
    if r.E_CA and r.E_LL:
        return "gain_robust_to_form"
    if r.E_CA and not r.E_LL:
        return "inconsistent_flag" if r.D == "LL>CA" else "gain_countavg_only"
    if (not r.E_CA) and r.E_LL:
        return "gain_invpf_only"
    return "no_gain_either"


def main():
    log = revio.RunLogger(stage=7, task="invpf_form", cpu_procs=1, seeds="frozen draw indices; bootstrap seed 7")
    rows = []
    for name, scn in (("primary", ("O", 50)), ("s1", ("O", 36)), ("s2", ("F", 50)), ("sec_a", ("O", "all")), ("sec_b", ("F", "all"))):
        run_scenario(name, scn, rows); print(name, len(rows), flush=True)
    P = pd.DataFrame(rows); f1 = os.path.join(REV, "tables", "S7_invpf_vs_countavg_per_protein.csv"); P.to_csv(f1, index=False)
    T = tests_for(P); T["code"] = [code_of(r) if r.generator in DECISION_GENS else "control_only" for r in T.itertuples()]; f2 = os.path.join(REV, "tables", "S7_invpf_cross_protein_tests.csv"); T.to_csv(f2, index=False)
    # 判定表与机理论断（以 primary 为准；s1/s2 若改变编码则登记 changed_due_to_audit）
    pri = T[(T.scenario == "primary") & T.generator.isin(DECISION_GENS)].set_index("generator"); rows2 = []
    for g in pri.index:
        c0 = pri.loc[g, "code"]; ch = [s for s in ("s1", "s2") if ((T.scenario == s) & (T.generator == g)).any() and T[(T.scenario == s) & (T.generator == g)].code.iloc[0] != c0]
        rows2.append(dict(generator=g, group=GROUP[g], trained_on_dGfold=TRAINED_ON_DGFOLD[g], E_CA=bool(pri.loc[g, "E_CA"]), E_LL=bool(pri.loc[g, "E_LL"]), D=pri.loc[g, "D"], code=c0, wins_CA=int(pri.loc[g, "wins_count_avg"]), wins_LL=int(pri.loc[g, "wins_invpf_avg"]),
                          changed_due_to_audit=",".join(ch) if ch else ""))
    DT = pd.DataFrame(rows2); f3 = os.path.join(REV, "tables", "S7_decision_table.csv"); DT.to_csv(f3, index=False)
    be = DT[DT.generator == "BioEmu"].iloc[0] if (DT.generator == "BioEmu").any() else None
    supported = bool(be is not None and be.D in ("LL>CA", "≈") and be.code in ("gain_robust_to_form", "gain_invpf_only"))
    out = dict(decision_rule_sha256=open(os.path.join(REV, "tables", "S7_decision_rule.sha256")).read().split()[0], decision_rule_mtime=os.path.getmtime(os.path.join(REV, "tables", "S7_decision_rule.md")), outcome_mtime=None,
               mechanism_claim_supported=supported, bioemu_code=(None if be is None else be.code), bioemu_D=(None if be is None else be.D), per_generator={r.generator: dict(code=r.code, D=r.D, E_CA=r.E_CA, E_LL=r.E_LL, changed_due_to_audit=r.changed_due_to_audit) for r in DT.itertuples()},
               missing_decision_generators=[g for g in DECISION_GENS if g not in set(DT.generator)])
    f4 = os.path.join(REV, "tables", "S7_decision_outcome.json"); json.dump(out, open(f4, "w"), ensure_ascii=False, indent=1); out["outcome_mtime"] = os.path.getmtime(f4); json.dump(out, open(f4, "w"), ensure_ascii=False, indent=1)   # 第二次写入把落盘时间戳记进文件，供“规则先于结果”核对
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(DT.to_string(index=False)); print("mechanism claim supported:", supported, "| missing:", out["missing_decision_generators"])
    print(T[T.scenario == "primary"][["generator", "n_proteins", "wins_count_avg", "wilcoxon_greater_p_count_avg", "wins_invpf_avg", "wilcoxon_greater_p_invpf_avg", "median_LL_minus_CA", "LL_minus_CA_boot_lo", "LL_minus_CA_boot_hi", "D"]].round(4).to_string(index=False))
    log.finish([f1, f2, f3, f4], note=f"rows={len(P)}")


if __name__ == "__main__":
    main()
