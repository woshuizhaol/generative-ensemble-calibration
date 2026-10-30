"""Stage 4a · 抽样索引冻结（revision_proposal.md Stage 4，"抽样索引冻结"一段）。

S7、S8、S9、S10 用到的全部子抽样在这里一次性生成并冻结，之后各 Stage 只读取、校验 sha256，不再自己抽样。
文件：raw_outputs/s4/draw_indices/<口径>_<generator>_<protein>_M<M>.npz；清单：tables/S4_draw_indices_manifest.csv（含每个文件的 sha256）。

口径：
  O  原稿口径：作用于旧 operator_f.npz 的列（列号 0 基）。只依赖旧 npz 的列数，可以在重打分之前生成。
  F  全帧口径：作用于新打分 parquet 的 frame_idx（原 xtc 的 0 基帧号）；可用帧 = 状态为 ok 或 ok_retry 的帧。必须在重打分合并之后生成。
规则：
  M_eff = min(M, n_available)。n_available > M：无放回抽 R 次；n_available ≤ M：只保留 1 个"全部帧"索引，n_lt_M = (n_available < M)。不做有放回抽样。
  每个文件同时存 n_available、M_eff、M_requested、n_lt_M、n_distinct_draws（实际不重复的子集数）、available_ids（可用帧的编号，frames 的值都取自它）。
  frames 的每一行升序排列（稳定、便于哈希）。
种子（偏离 D4-1，见工作文档）：方案对子抽样种子有两处互相矛盾的写法（"20261024 + stage×100 + k"配例子"S7 第 r 次为 20261124 + r"；而 S10 又明写
  "20261024 + 1000 + r"），且 R=200 时按"阶段 ×100"取种子会让相邻阶段的区间重叠。因为所有阶段读取同一批冻结文件，种子只在冻结时用一次，
  这里统一：第 r 次抽样的 entropy = 20261024 + 1000 + r（即 S10 那条明确公式），再把（格子、M）混入 SeedSequence 的 spawn_key，
  使不同格子、不同 M 的随机流相互独立，不会重叠。
格子与 M 的清单：
  Tier-1a（5 生成器 × 12 蛋白）：M=50 R=200（S7 主分析、S8、S9）；O 口径另有 M=36 R=200（S7 (s1) 共同样本量）。
  BioEmu 收敛曲线（S7 (c)）：M ∈ {10, 25, 50, 100, 200, 400} R=50，F 口径全部可行（每个蛋白 433–499 帧）；O 口径只报告 M ≤ n_available 的格子。
  ShortMD（F 口径）：M=50 R=200；另补 M=100 与全部帧（S7 应急分支，R=50）。
  T3 的 BioEmu_T3（S10）：F 口径 M=50 R=200；收敛曲线 M ∈ {10, 25, 50, 100, 全部可用} R=100。其余 T3 生成器每域 50 帧，不需要抽样。
用法：python s4_freeze_draw_indices.py O | F | S5 ...   （S5/S4b 新系综在打分后再调用，规则相同）
"""
import os, sys, json, hashlib, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

ENTROPY_BASE = 20261024 + 1000
TIER1A_GENS = ["BioEmu", "ESMFlow", "Boltz2", "Str2Str", "ShortMD"]
CONV_M = [10, 25, 50, 100, 200, 400]


def cell_id(basis, generator, protein):
    return int(hashlib.sha256(f"{basis}|{generator}|{protein}".encode()).hexdigest()[:8], 16)


def make_draws(available_ids, M, R, basis, generator, protein):
    ids = np.asarray(sorted(available_ids), dtype=np.int32); n = len(ids)
    if n <= M:
        return ids[None, :].copy(), n, 1, n < M
    cid = cell_id(basis, generator, protein); out = np.empty((R, M), dtype=np.int32)
    for r in range(R):
        rng = np.random.default_rng(np.random.SeedSequence(ENTROPY_BASE + r, spawn_key=(cid, M)))
        out[r] = np.sort(ids[rng.choice(n, size=M, replace=False)])
    return out, M, len({tuple(x) for x in out}), False


def plan_cells(basis, REV, INP):
    """返回 [(generator, protein, available_ids, [(M, R), …])]"""
    cells = []
    seqs = json.load(open(INP["tier1a_sequences"]))
    if basis == "O":
        for g in TIER1A_GENS:
            for p in seqs:
                n = int(np.load(os.path.join(INP["gen_dir"], g, p, "operator_f.npz"))["NC"].shape[1])
                MR = [(50, 200), (36, 200)]
                if g == "BioEmu":
                    MR += [(m, 50) for m in CONV_M if m != 50 and m <= n]
                cells.append((g, p, np.arange(n), MR))
    elif basis == "F":
        for g in TIER1A_GENS:
            for p in seqs:
                F = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", g, f"{p}_frames.parquet"))
                ids = F[F.status.isin(["ok", "ok_retry"])].frame_idx.values
                MR = [(50, 200)]
                if g == "BioEmu":
                    MR += [(m, 50) for m in CONV_M if m != 50]
                if g == "ShortMD":   # S7 应急分支（1/PF 平均形式下 w_max>0.5）要求口径 F 补报 M=100/200，只有 BioEmu 和 ShortMD 帧数够；M=200 即全部帧
                    MR += [(100, 50), (len(ids), 50)]
                cells.append((g, p, ids, MR))
        for gT3 in ("BioEmu_T3", "BioEmu_T3new"):   # BioEmu_T3new 是 S4b（S5 新生成的 143 个域），要在 S4b 合并之后才有目录；缺目录时跳过并提示，由 S10 的完整性检查兜底
            base = os.path.join(REV, "raw_outputs", "s4", "counts", gT3)
            if not os.path.isdir(base):
                print(f"[freeze] 跳过 {gT3}：{base} 不存在（S4b 尚未合并）", flush=True); continue
            for f in sorted(os.listdir(base)):
                if f.endswith("_frames.parquet"):
                    p = f[:-len("_frames.parquet")]; F = pd.read_parquet(os.path.join(base, f))
                    ids = F[F.status.isin(["ok", "ok_retry"])].frame_idx.values
                    MR = [(50, 200)] + [(m, 100) for m in (10, 25, 100)] + [(len(ids), 100)]   # 最后一档 = 全部可用
                    cells.append((gT3, p, ids, MR))
        # S7 对 Str2Str_xtal 的共同样本量敏感性（M=36）：Str2Str_xtal 没有旧 npz，所以没有口径 O 的 M36 文件，在口径 F 补；每个种子 50 帧，M=50 即全部帧
        for p in seqs:
            for k in (11,):
                base = os.path.join(REV, "raw_outputs", "s4", "counts", f"Str2Str_xtal_s{k}")
                if os.path.exists(os.path.join(base, f"{p}_frames.parquet")):
                    F = pd.read_parquet(os.path.join(base, f"{p}_frames.parquet")); ids = F[F.status.isin(["ok", "ok_retry"])].frame_idx.values
                    cells.append((f"Str2Str_xtal_s{k}", p, ids, [(50, 200), (36, 200)]))
                else:
                    print(f"[freeze] 跳过 Str2Str_xtal_s{k}/{p}：尚未打分合并", flush=True)
    else:
        raise SystemExit("basis 只能是 O 或 F")
    return cells


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("basis", choices=["O", "F"]); a = ap.parse_args()
    PROJ, REV, INP = revio.load_paths()
    log = revio.RunLogger(stage=4, task=f"freeze_draw_indices_{a.basis}", cpu_procs=1, seeds=f"entropy={ENTROPY_BASE}+r, spawn_key=(cell,M)")
    out_dir = os.path.join(REV, "raw_outputs", "s4", "draw_indices"); os.makedirs(out_dir, exist_ok=True)
    rows = []
    for g, p, ids, MR in plan_cells(a.basis, REV, INP):
        seen = set()
        for M, R in MR:
            if (M) in seen:
                continue
            seen.add(M)
            fr, m_eff, ndist, lt = make_draws(ids, M, R, a.basis, g, p)
            fn = f"{a.basis}_{g}_{p}_M{M}.npz"; path = os.path.join(out_dir, fn)
            if os.path.exists(path):
                # 幂等重跑：抽样规则是确定性的，已存在的文件必须与重新生成的内容逐位相同，才跳过（保留原文件和原 sha256）；不同则中止，冻结后的索引不得改写
                z = np.load(path)
                if not (np.array_equal(z["frames"], fr) and int(z["M_eff"]) == m_eff and np.array_equal(z["available_ids"], np.asarray(sorted(ids), dtype=np.int32))):
                    raise SystemExit(f"{fn} 已存在且与按规则重新生成的内容不同：冻结后的索引不得改写；如需重做，先人工归档旧目录并在工作文档里记录原因")
                rows.append(dict(basis=a.basis, generator=g, protein=p, M_requested=M, M_eff=m_eff, n_available=len(ids), n_lt_M=lt, R_requested=R, R_stored=fr.shape[0],
                                 n_distinct_draws=ndist, file=os.path.join("raw_outputs", "s4", "draw_indices", fn), sha256=revio.sha256(path)))
                continue
            np.savez(path, frames=fr, available_ids=np.asarray(sorted(ids), dtype=np.int32), n_available=len(ids), M_eff=m_eff, M_requested=M, n_lt_M=lt,
                     n_distinct_draws=ndist, R=fr.shape[0], R_requested=R, basis=a.basis, generator=g, protein=p, entropy_base=ENTROPY_BASE, spawn_key=np.array([cell_id(a.basis, g, p), M]))
            rows.append(dict(basis=a.basis, generator=g, protein=p, M_requested=M, M_eff=m_eff, n_available=len(ids), n_lt_M=lt, R_requested=R, R_stored=fr.shape[0],
                             n_distinct_draws=ndist, file=os.path.join("raw_outputs", "s4", "draw_indices", fn), sha256=revio.sha256(path)))
    T = pd.DataFrame(rows); f = os.path.join(REV, "tables", f"S4_draw_indices_manifest_{a.basis}.csv"); T.to_csv(f, index=False)
    pd.set_option("display.width", 220)
    print(T.groupby(["basis", "M_requested"]).agg(files=("file", "size"), n_lt_M=("n_lt_M", "sum"), min_avail=("n_available", "min"), min_distinct=("n_distinct_draws", "min")).to_string())
    print("n_lt_M cells (M=50):", T[(T.M_requested == 50) & T.n_lt_M][["generator", "protein", "n_available"]].values.tolist())
    print("n_distinct_draws < R cells (M<n):", T[(T.n_available > T.M_eff) & (T.n_distinct_draws < T.R_stored)][["generator", "protein", "M_requested", "n_distinct_draws"]].values.tolist())
    log.finish([f], note=f"files={len(T)}")


if __name__ == "__main__":
    main()
