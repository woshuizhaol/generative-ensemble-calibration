"""Stage 7 · 阳性对照（先于一切结果，全部通过才继续；判定规则文件 tables/S7_decision_rule.md §5）。
  C-linear   count_avg 的 ln PF = 逐构象 ln PF 的平均，最大绝对差 < 1e-9（在真实数据的 60 个 Tier-1a 格子上）
  C-F1       单构象时 count_avg 与 invpf_avg 完全相等，最大绝对差 < 1e-9
  C-twostate 合成两态系综（99 帧 N_C=30、N_H=1；1 帧 N_C=N_H=0）：count_avg = 12.375，invpf_avg = −ln(0.99·e^−12.5 + 0.01) ≈ 4.605
  Jensen     ln PF^LL ≤ ln PF^CA（所有残基、所有格子）
  C-lib      库自检：lib/ensemble.py 用旧 npz 全部列算的 count_avg ρ 与原稿 eval/leaderboard.parquet 一致（最大差 ≤ 0.0006，leaderboard 保留 3 位小数）
             以及与 bv.aggregate 逐元素一致
输出 tables/S7_controls.csv。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, ensemble as E  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=7, task="controls", cpu_procs=1, seeds="none")
GENS = ["BioEmu", "ESMFlow", "Boltz2", "Str2Str", "ShortMD", "Str2Str_xtal"]; checks = []   # Str2Str_xtal 为 S4b 新增（种子 11），与旧 5 个生成器同样做线性/Jensen/库一致性检查


def add(name, ok, detail):
    checks.append(dict(control=name, passed=bool(ok), detail=detail)); print(("PASS " if ok else "FAIL ") + name + " | " + detail, flush=True)


seqs = list(__import__("json").load(open(INP["tier1a_sequences"])))
lin, jen, agg, rho_new = [], [], [], {}
for g in GENS:
    for p in seqs:
        c = E.load_cell("Str2Str_xtal_s11", p, "F") if g == "Str2Str_xtal" else E.load_cell(g, p, "O"); M = c["NC"].shape[1]; cols = np.arange(M)[None, :]   # Str2Str_xtal：S4b 的 parquet（口径 F）
        ca = E.ln_pf(c["NC"], c["NH"], cols, "count_avg")[0]; ll = E.ln_pf(c["NC"], c["NH"], cols, "invpf_avg")[0]
        per_frame = (E.BC * c["NC"] + E.BH * c["NH"]); mean_pf = np.nanmean(per_frame, axis=1)
        lin.append(np.nanmax(np.abs(ca - mean_pf))); jen.append(np.nanmax(ll - ca)); agg.append(np.nanmax(np.abs(ca - bv.aggregate(c["NC"], c["NH"], "count_avg"))))
        rho_new[(g, p)] = float(E.rho_draws(c, p, cols, "count_avg")[0])
add(f"C-linear: count_avg ln PF == mean of per-frame ln PF ({len(lin)} Tier-1a cells)", max(lin) < 1e-9, f"max abs diff {max(lin):.2e}")
add("lib == bv.aggregate (count_avg)", max(agg) < 1e-9, f"max abs diff {max(agg):.2e}")
add(f"Jensen: ln PF^LL <= ln PF^CA (all residues, {len(jen)} cells)", max(jen) <= 1e-9, f"max (LL - CA) = {max(jen):.2e}")
# C-F1
rng = np.random.default_rng(0); NC1 = rng.integers(0, 60, size=(200, 1)).astype(float); NH1 = rng.integers(0, 4, size=(200, 1)).astype(float); c1 = np.array([[0]])
d = np.nanmax(np.abs(E.ln_pf(NC1, NH1, c1, "count_avg") - E.ln_pf(NC1, NH1, c1, "invpf_avg")))
add("C-F1: single conformer, count_avg == invpf_avg", d < 1e-9, f"max abs diff {d:.2e}")
# C-twostate
NC2 = np.array([[30.0] * 99 + [0.0]]); NH2 = np.array([[1.0] * 99 + [0.0]]); cols2 = np.arange(100)[None, :]
a = float(E.ln_pf(NC2, NH2, cols2, "count_avg")[0, 0]); b = float(E.ln_pf(NC2, NH2, cols2, "invpf_avg")[0, 0]); a0 = 0.99 * (0.35 * 30 + 2.0); b0 = -np.log(0.99 * np.exp(-12.5) + 0.01)
add("C-twostate", abs(a - 12.375) < 1e-9 and abs(a - a0) < 1e-9 and abs(b - b0) < 1e-9 and abs(b - 4.605) < 5e-4, f"count_avg {a:.6f} (expect 12.375), invpf_avg {b:.6f} (expect {b0:.6f} ≈ 4.605)")
# C-lib：与原稿 leaderboard
L = pd.read_parquet(INP["leaderboard"]); diffs = []
for r in L.itertuples():
    diffs.append(abs(rho_new[(r.model, r.protein)] - r.spearman))
add("C-lib: lib count_avg rho (all old columns) == leaderboard spearman (60 cells)", max(diffs) <= 0.0006, f"max abs diff {max(diffs):.5f}; n={len(diffs)}")
T = pd.DataFrame(checks); f = os.path.join(REV, "tables", "S7_controls.csv"); T.to_csv(f, index=False); log.finish([f], status="ok" if T.passed.all() else "controls_failed", note=f"passed={int(T.passed.sum())}/{len(T)}")
sys.exit(0 if T.passed.all() else 1)
