"""Stage 4b · rSASA 取值范围诊断（偏离 D4-5 的依据）：对 raw_outputs/s4/counts 下全部类别，统计非末端残基帧里 rSASA>1.2 的数量，
区分非噪声类别与噪声类别（noise_anm / noise_cart），并给出超线残基离链端的距离（≤3 个残基）。输出 tables/S4b_rsasa_range_diagnostic.csv。"""
import os, sys, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths(); base = os.path.join(REV, "raw_outputs", "s4", "counts")
log = revio.RunLogger(stage=4, task="rsasa_diagnostic", cpu_procs=1, seeds="none")
out = []
for g in sorted(os.listdir(base)):
    fs = [f for f in glob.glob(os.path.join(base, g, "*.parquet")) if not f.endswith("_frames.parquet")]
    tot = over = near = tt = tover = 0; mx = 0.0
    for f in fs:
        R = pd.read_parquet(f, columns=["resSeq", "rSASA_res"]); lo = R.resSeq.min(); n = R.resSeq.max(); term = (R.resSeq == lo) | (R.resSeq == n); X = R[~term]
        tot += len(X); o = X[X.rSASA_res > 1.2]; over += len(o); mx = max(mx, float(X.rSASA_res.max())); d = np.minimum(o.resSeq - lo, n - o.resSeq) + 1; near += int((d <= 3).sum()); tt += int(term.sum()); tover += int((R[term].rSASA_res > 1.2).sum())
    out.append(dict(group=g, is_noise=g.startswith("noise_"), n_files=len(fs), n_nonterminal_rows=tot, n_over_1p2=over, frac_over_1p2=over / max(tot, 1), max_nonterminal_rSASA=mx, n_over_within_3_of_chain_end=near, n_terminal_rows=tt, n_terminal_over_1p2=tover))
T = pd.DataFrame(out); f = os.path.join(REV, "tables", "S4b_rsasa_range_diagnostic.csv"); T.to_csv(f, index=False)
nn = T[~T.is_noise]; print("non-noise groups:", len(nn), "rows", int(nn.n_nonterminal_rows.sum()), "over 1.2:", int(nn.n_over_1p2.sum()), "max", round(nn.max_nonterminal_rSASA.max(), 3))
for k in ("noise_anm", "noise_cart"):
    x = T[T.group.str.startswith(k)]; print(k, "rows", int(x.n_nonterminal_rows.sum()), "over", int(x.n_over_1p2.sum()), "near end", int(x.n_over_within_3_of_chain_end.sum()), "max", round(x.max_nonterminal_rSASA.max(), 3))
log.finish([f], note="rsasa diagnostic")
