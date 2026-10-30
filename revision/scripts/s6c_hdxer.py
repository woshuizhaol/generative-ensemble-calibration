"""Stage 6c · HDXer 实现一致性（revision_proposal.md S6 步骤 6，阳性对照 C-hdxer）。
输入：SNase 的固定结构 data/interim/hdxer/SNase_fixed.pdb（与原项目 14_hdxer_crosscheck.py 写出的文件逐字节相同，sha256 见工作文档）；
      HDXer 输出 raw_outputs/s6/hdxer/SNase_logProtection_factors.dat（HDXer commit f06191e，`calc_hdx.py -m BestVendruscolo`，本阶段重跑，与原运行逐位相同）。
(a) 本实现切到 HDXer 设置——N_H 受体 = 全部蛋白 O，N_H 截断 0.24 nm，N_H 不排除相邻残基，N_C 排除 n±2（= |i−j|≥3，本实现原本就是），跳过首残基——逐残基比较，应完全一致。
(b) 逐项切换：从生产设置（备选受体 = 主链 O，0.26 nm，|i−j|≥2 排除，首残基保留）出发，一次只改一项，再逐项累加，量化每项对 ln PF 的贡献。
(c) 复现审稿人的数字（偏差 −0.23 ln PF、配对 Wilcoxon P=9.3×10⁻⁵、18 个高于参照、65 个低于参照）：在存档的交叉核对表和一组替代口径上枚举。
(d) 两种实现分别对实验：只在 SNase 有实验值且 HDXer 输出中也有的 100 个残基上（审稿人写的是 101，HDXer 表无 K1）。
输出：tables/S6_hdxer_consistency.csv、S6_hdxer_toggles.csv、S6_reviewer_bias_reproduction.csv、S6_hdxer_vs_experiment.csv。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, itertools
import numpy as np, pandas as pd
import mdtraj as md
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402
import s4_rescore as S4  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=6, task="hdxer_consistency", cpu_procs=1, seeds="fix_frame seeds 0-19 for part (c)")
BC, BH = bv.BETA_C, bv.BETA_H
t1 = md.load(os.path.join(REV, "data", "interim", "hdxer", "SNase_fixed.pdb"))
H = np.loadtxt(os.path.join(REV, "raw_outputs", "s6", "hdxer", "SNase_logProtection_factors.dat")); hd = dict(zip(H[:, 0].astype(int), H[:, 1]))


def lnpf(nn):
    return {r: BC * v[1] + BH * v[2] for r, v in nn.items()}


PROD = dict(cut_Nh=0.26, ij_Nh=2, hbond_acceptor="backbone_O", skip_first=False)
HDX = dict(cut_Nh=0.24, ij_Nh=0, hbond_acceptor="all_O", skip_first=True)


def run(t, **kw):
    return bv.nc_nh(t, cut_Nc=0.65, ij_Nc=3, **kw)


def compare(a, b):
    k = sorted(set(a) & set(b)); x = np.array([a[i] for i in k]); y = np.array([b[i] for i in k]); d = x - y
    return dict(n=len(k), pearson=float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan, max_abs_diff=float(np.abs(d).max()), n_equal=int((np.abs(d) < 1e-9).sum()), mean_diff=float(d.mean()))


# ---------------- (a) C-hdxer
mine_prod = lnpf(run(t1, **PROD)); mine_hdx = lnpf(run(t1, **HDX))
rows = [dict(setting="本实现 · HDXer 设置（受体全部 O, 0.24 nm, 不排除相邻残基, 跳过首残基）", **compare(mine_hdx, hd)), dict(setting="本实现 · 生产设置", **compare(mine_prod, hd))]
for ij in (2,):   # 唯一不确定的一项：HDXer 是否排除相邻残基的 O
    rows.append(dict(setting="本实现 · HDXer 设置但 N_H 仍排除 |i−j|<2", **compare(lnpf(run(t1, **{**HDX, "ij_Nh": ij})), hd)))
C = pd.DataFrame(rows); f1 = os.path.join(REV, "tables", "S6_hdxer_consistency.csv"); C.to_csv(f1, index=False)
# ---------------- (b) 逐项切换
steps = [("受体: 主链 O → 全部蛋白 O", dict(hbond_acceptor="all_O")), ("N_H 截断: 0.26 → 0.24 nm", dict(cut_Nh=0.24)), ("N_H 排除相邻残基 |i−j|<2 → 不排除", dict(ij_Nh=0)), ("首残基: 保留 → 跳过", dict(skip_first=True))]
base_vals = mine_prod; trows = []
for nm, kw in steps:
    v = lnpf(run(t1, **{**PROD, **kw})); c = compare(v, mine_prod); ks = sorted(set(v) & set(mine_prod)); d = np.array([v[i] - mine_prod[i] for i in ks])
    trows.append(dict(mode="single (from production)", change=nm, n_residues_compared=len(ks), n_changed=int((np.abs(d) > 1e-9).sum()), n_up=int((d > 1e-9).sum()), n_down=int((d < -1e-9).sum()), mean_delta_lnPF=float(d.mean()), residues_dropped=130 - len(v)))
cum = {}
for i, (nm, kw) in enumerate(steps):
    cum.update(kw); v = lnpf(run(t1, **{**PROD, **cum})); ks = sorted(set(v) & set(mine_prod)); d = np.array([v[j] - mine_prod[j] for j in ks]); c = compare(v, hd)
    trows.append(dict(mode="cumulative", change=" + ".join(s[0] for s in steps[:i + 1]), n_residues_compared=len(ks), n_changed=int((np.abs(d) > 1e-9).sum()), n_up=int((d > 1e-9).sum()), n_down=int((d < -1e-9).sum()), mean_delta_lnPF=float(d.mean()), residues_dropped=130 - len(v),
                     max_abs_diff_vs_hdxer=c["max_abs_diff"], n_equal_to_hdxer=c["n_equal"], n_vs_hdxer=c["n"]))
T = pd.DataFrame(trows); f2 = os.path.join(REV, "tables", "S6_hdxer_toggles.csv"); T.to_csv(f2, index=False)
# ---------------- (c) 审稿人数字
xc = pd.read_csv(os.path.join(INP["results_stats"], "hdxer_crosscheck_SNase.csv"), float_precision="round_trip"); rr = []   # round_trip：精确解析，计数不随 pandas 版本变化（默认快速解析会把相邻双精度数解析成同一个值）


def signflip_p(nz, B=1000000, seed=0):
    """符号翻转置换检验（双侧，+1 校正），统计量为符号秩和；并列取平均秩。对并列（这里 |d| 几乎全是 2）是精确的参照。"""
    r = stats.rankdata(np.abs(nz)); T = r[nz > 0].sum(); E = r.sum() / 2; rng = np.random.default_rng(seed); out = 0; done = 0
    while done < B:
        m = min(200000, B - done); S = rng.integers(0, 2, size=(m, len(nz))).astype(bool); Tm = (S * r).sum(1); out += int(np.sum(np.abs(Tm - E) >= abs(T - E) - 1e-9)); done += m
    return (out + 1) / (B + 1)


def tests(tag, ref, other, note=""):
    """差值先舍入到 1e-9 再检验：本实现与 HDXer 的差几乎全是 N_H 计数差 1 × 系数 2.0，原始浮点里这些本应并列的 |d| 相差 ~4e-15，会被当作不同的秩（浮点并列伪影，见 S7 的同类问题）。"""
    ks = sorted(set(ref) & set(other)); d = np.array([other[k] - ref[k] for k in ks]); nz_raw = d[np.abs(d) > 1e-9]; nz = np.round(nz_raw, 9)
    w = stats.wilcoxon(nz) if len(nz) > 0 else None; w_raw = stats.wilcoxon(nz_raw) if len(nz_raw) > 0 else None
    sp = float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) > 0 else np.nan
    return dict(candidate=tag, n_residues=len(ks), n_nonzero=len(nz), n_other_higher=int((nz > 0).sum()), n_other_lower=int((nz < 0).sum()), mean_diff_other_minus_ref=float(d.mean()), wilcoxon_p=float(w.pvalue) if w else np.nan,
                wilcoxon_p_unrounded_float=float(w_raw.pvalue) if w_raw else np.nan, sign_test_p=sp, note=note)


rr.append(dict(candidate="审稿人报告值", n_residues=np.nan, n_nonzero=83, n_other_higher=18, n_other_lower=65, mean_diff_other_minus_ref=-0.23, wilcoxon_p=9.3e-5, note="审稿人原文：偏差 −0.23 ln PF，配对 Wilcoxon P=9.3×10⁻⁵，18 个高于参照、65 个低于参照（n=83 非零差）"))
ref = dict(zip(xc.resSeq, xc.lnPF_hdxer)); mine = dict(zip(xc.resSeq, xc.lnPF_mine))
rr.append(tests("存档交叉核对表（Zenodo/SF1b 数据），去掉 1e-9 以内的差：本实现 − HDXer", ref, mine, "SF1b 的来源数据；n_nonzero 只含真实差异"))
rr.append(tests("同上，HDXer − 本实现", mine, ref, "符号相反"))
mp = dict(zip(xc.resSeq, np.exp(xc.lnPF_mine))); hp = dict(zip(xc.resSeq, np.exp(xc.lnPF_hdxer)))
rr.append(tests("PF（不取对数）：本实现 − HDXer", hp, mp, "换算口径不同"))
exp_s = pd.read_parquet(os.path.join(REV, "tables", "S1_tier1a_residues_final.parquet")); ex = exp_s[(exp_s.protein_id == "SNase")].dropna(subset=["dGopen_kcal"]); ex_res = set(ex.resi.astype(int))
rr.append(tests("存档表限于 SNase 有实验值且 HDXer 输出中也有的 100 个残基", {k: v for k, v in ref.items() if k in ex_res}, {k: v for k, v in mine.items() if k in ex_res}, "审稿人写的 n=101；本表实际 n=100（HDXer 输出无 K1）"))
# 替代结构：同一份 1SNO，用 20 个不同的确定性种子重建，每次氢放置不同；HDXer 设置的本实现 = HDXer（(a) 已验证）
pdb = os.path.join(INP["pdb_dir"], "1SNO.pdb"); fr = md.load(pdb); fr = fr[0] if fr.n_frames > 1 else fr
cnt = []
for sd in range(20):
    t = bv.fix_frame(fr, True, seed=1000 + sd); a = lnpf(run(t, **PROD)); b = lnpf(run(t, **HDX)); r_ = tests(f"重建种子 {1000 + sd}：生产设置 − HDXer 设置", b, a, "每个种子的氢放置不同"); rr.append(r_); cnt.append((r_["n_nonzero"], r_["n_other_higher"], r_["n_other_lower"], r_["mean_diff_other_minus_ref"], r_["wilcoxon_p"]))
# 审稿人数字的逐项复现（存档交叉核对表，float_precision="round_trip" 精确解析，不设容差）。
# 发现：表里 33 个残基的两种实现只差 1e-15 量级——本实现把 ln PF 算成 0.35·N_C+2·N_H（如 4.8999999999999995），HDXer 输出写成 4.9——且 33 个全部落在"本实现更低"一侧。
dr = (xc.lnPF_mine - xc.lnPF_hdxer).values; noise = (dr != 0) & (np.abs(dr) < 1e-9); real = np.abs(dr) >= 1e-9
rep_ok = lambda got, want, rel=0.02: abs(got - want) <= rel * abs(want)
rep_rows = []


def add_row(item, reviewer, got, ok, note=""):
    rep_rows.append(dict(item=item, reviewer=reviewer, reproduced=got, status=("reproduced" if ok else "not_reproduced"), note=note))


add_row("偏差（本实现 − HDXer 的均值）", -0.23, float(dr.mean()), round(float(dr.mean()), 2) == -0.23, "含 33 个浮点噪声残基，对均值的影响 <1e-13")
add_row("本实现高于参照的残基数", 18, int((dr > 0).sum()), int((dr > 0).sum()) == 18)
add_row("本实现低于参照的残基数", 65, int((dr < 0).sum()), int((dr < 0).sum()) == 65, "其中 %d 个是 1e-15 量级的浮点噪声，不是真实差异" % int((dr[noise] < 0).sum()))
add_row("非零差个数（按原始浮点）", 83, int((dr != 0).sum()), int((dr != 0).sum()) == 83, "= 真实差异 %d + 浮点噪声 %d" % (int(real.sum()), int(noise.sum())))
w_main = stats.wilcoxon(xc.lnPF_mine, xc.lnPF_hdxer, zero_method="wilcox", alternative="two-sided")
add_row("配对 Wilcoxon p（scipy 默认：zero_method=wilcox，双侧，原始浮点）", 9.3e-5, float(w_main.pvalue), rep_ok(float(w_main.pvalue), 9.3e-5), "审稿人 P=9.3×10⁻⁵；复现值 %.2e" % w_main.pvalue)
for zm in ("pratt", "zsplit"):
    add_row(f"配对 Wilcoxon p（zero_method={zm}，双侧，原始浮点）", 9.3e-5, float(stats.wilcoxon(xc.lnPF_mine, xc.lnPF_hdxer, zero_method=zm, alternative="two-sided").pvalue), False, "备选实现，作对照")
add_row("符号检验 p（原始浮点，18 对 65）", 9.3e-5, float(stats.binomtest(int((dr > 0).sum()), int((dr != 0).sum()), 0.5).pvalue), False, "备选检验，作对照")
rr_ = dr[real]; rr9 = np.round(rr_, 9)
add_row("【去掉浮点噪声后】真实差异个数", np.nan, int(real.sum()), True, "18 个更高、%d 个更低；其中 %d 个的 |差| 恰为 2.0（N_H 计数差 1 × 系数 2.0），1 个为 4.0" % (int((rr_ < 0).sum()), int((np.abs(np.round(rr_, 6)) == 2.0).sum())))
add_row("【去掉浮点噪声后】这 %d 个真实差异在原始浮点里 |d|≈2 的最大离散" % int(real.sum()), np.nan, float(np.ptp(np.abs(rr_)[np.abs(np.abs(rr_) - 2) < 1e-6])), True, "应并列的差值被浮点噪声打散；不舍入的 Wilcoxon p=%.2e 是并列被打破的伪影，不可用" % float(stats.wilcoxon(rr_).pvalue))
add_row("【去掉浮点噪声后，差值舍入到 1e-9】配对 Wilcoxon p（%d 个真实差异，scipy 默认，并列校正）" % int(real.sum()), np.nan, float(stats.wilcoxon(rr9).pvalue), True, "边缘显著；方案预期 p≈0.04")
add_row("【去掉浮点噪声后，舍入】符号翻转置换 p（双侧，10⁶ 次，seed 0）", np.nan, signflip_p(rr9), True, "并列下的精确参照；0.05 附近")
add_row("【去掉浮点噪声后，舍入】符号检验 p（18 对 %d）" % int((rr9 < 0).sum()), np.nan, float(stats.binomtest(int((rr9 > 0).sum()), len(rr9), 0.5).pvalue), True, "")
add_row("审稿人 P 值被浮点噪声夸大的倍数（9.28e-5 对舍入后 Wilcoxon p）", np.nan, float(stats.wilcoxon(rr9).pvalue) / float(w_main.pvalue), True, "约 400 倍")
RR = pd.DataFrame(rep_rows); RR.to_csv(os.path.join(REV, "tables", "S6_reviewer_bias_numbers.csv"), index=False)
cnt = np.array(cnt); R = pd.DataFrame(rr); f3 = os.path.join(REV, "tables", "S6_reviewer_bias_reproduction.csv"); R.to_csv(f3, index=False)
# ---------------- (d) 对实验
exv = dict(zip(ex.resi.astype(int), ex.dGopen_kcal.astype(float) / (bv.R_GAS * (float(ex.T_C.iloc[0]) + 273.15)))); vrows = []
for nm, d in (("本实现·生产设置", mine_prod), ("本实现·HDXer 设置", mine_hdx), ("HDXer", hd)):
    ks = sorted(set(d) & set(exv)); x = np.array([d[k] for k in ks]); y = np.array([exv[k] for k in ks])
    vrows.append(dict(implementation=nm, n_residues=len(ks), pearson=float(np.corrcoef(x, y)[0, 1]), spearman=float(stats.spearmanr(x, y)[0]), rmse=float(np.sqrt(np.mean((x - y) ** 2))), mean_bias=float(np.mean(x - y))))
ks100 = sorted(set(mine_prod) & set(hd) & set(exv)); x = np.array([mine_prod[k] for k in ks100]); y = np.array([exv[k] for k in ks100])   # 与 HDXer 相同的 100 个残基上比较生产设置（首残基被 HDXer 跳过）
vrows.insert(1, dict(implementation="本实现·生产设置（限于与 HDXer 相同的残基）", n_residues=len(ks100), pearson=float(np.corrcoef(x, y)[0, 1]), spearman=float(stats.spearmanr(x, y)[0]), rmse=float(np.sqrt(np.mean((x - y) ** 2))), mean_bias=float(np.mean(x - y))))
V = pd.DataFrame(vrows); f4 = os.path.join(REV, "tables", "S6_hdxer_vs_experiment.csv"); V.to_csv(f4, index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)
print(C.round(4).to_string(index=False)); print(T.round(3).to_string(index=False)); print(R.iloc[:6].round(5).to_string(index=False))
print("20 个替代重建：非零差个数 %d–%d，其中 本实现(生产)更高 %d–%d、更低 %d–%d；均值差 %.3f 到 %.3f；Wilcoxon p 范围 %.1e–%.1e" % (cnt[:, 0].min(), cnt[:, 0].max(), cnt[:, 1].min(), cnt[:, 1].max(), cnt[:, 2].min(), cnt[:, 2].max(), cnt[:, 3].min(), cnt[:, 3].max(), np.nanmin(cnt[:, 4]), np.nanmax(cnt[:, 4])))
print(V.round(3).to_string(index=False)); log.finish([f1, f2, f3, f4, os.path.join(REV, "tables", "S6_reviewer_bias_numbers.csv")], note="C-hdxer")
