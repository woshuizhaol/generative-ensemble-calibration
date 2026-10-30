"""Stage 14 · 为从 BioEmu 构象出发的 ShortMD 轨迹（r4、r5）生成并冻结口径 F 的抽样索引（规则同 s4_freeze_draw_indices.py：M_eff = min(50, n_available)，无放回，R=200，
entropy = 20261024 + 1000 + r，spawn_key = (cell_id("F", generator, protein), M)），在 r4、r5 打分合并之后、任何 ρ 计算之前运行。
输出：raw_outputs/s4/draw_indices/F_ShortMD_r<k>_<蛋白>_M50.npz（格式同 S4）与 tables/S14_draw_indices_manifest.csv（含 sha256）。
重跑时已存在的文件必须与按规则重新生成的内容逐位相同，否则中止（冻结后不得改写）。
用法：python s14_freeze_draws.py [--reps 4,5]"""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402
import s4_freeze_draw_indices as FZ  # noqa: E402

PROJ, REV, INP = revio.load_paths()
ap = argparse.ArgumentParser(); ap.add_argument("--reps", default="4,5"); a = ap.parse_args()
log = revio.RunLogger(stage=14, task="freeze_draws", cpu_procs=1, seeds=f"entropy={FZ.ENTROPY_BASE}+r, spawn_key=(cell_id('F', generator, protein), 50)")
out_dir = os.path.join(REV, "raw_outputs", "s4", "draw_indices"); seqs = json.load(open(INP["tier1a_sequences"])); rows = []
for k in [int(x) for x in a.reps.split(",")]:
    g = f"ShortMD_r{k}"
    for p in sorted(seqs):
        F = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", g, f"{p}_frames.parquet")); ids = F[F.status.isin(["ok", "ok_retry"])].frame_idx.values
        fr, m_eff, ndist, lt = FZ.make_draws(ids, 50, 200, "F", g, p); fn = f"F_{g}_{p}_M50.npz"; path = os.path.join(out_dir, fn)
        if os.path.exists(path):
            z = np.load(path)
            if not (np.array_equal(z["frames"], fr) and int(z["M_eff"]) == m_eff and np.array_equal(z["available_ids"], np.asarray(sorted(ids), dtype=np.int32))):
                raise SystemExit(f"{fn} 已存在且与按规则重新生成的内容不同：冻结后的索引不得改写")
        else:
            np.savez(path, frames=fr, available_ids=np.asarray(sorted(ids), dtype=np.int32), n_available=len(ids), M_eff=m_eff, M_requested=50, n_lt_M=lt, n_distinct_draws=ndist, R=fr.shape[0], R_requested=200,
                     basis="F", generator=g, protein=p, entropy_base=FZ.ENTROPY_BASE, spawn_key=np.array([FZ.cell_id("F", g, p), 50]))
        rows.append(dict(basis="F", generator=g, protein=p, M_requested=50, M_eff=m_eff, n_available=len(ids), n_lt_M=lt, R_requested=200, R_stored=fr.shape[0], n_distinct_draws=ndist,
                         file=os.path.join("raw_outputs", "s4", "draw_indices", fn), sha256=revio.sha256(path)))
T = pd.DataFrame(rows); f = os.path.join(REV, "tables", "S14_draw_indices_manifest.csv"); T.to_csv(f, index=False)
print(T.groupby("generator").agg(files=("file", "size"), min_avail=("n_available", "min"), min_distinct=("n_distinct_draws", "min"), n_lt_M=("n_lt_M", "sum")).to_string())
log.finish([f], note=f"files={len(T)}")
