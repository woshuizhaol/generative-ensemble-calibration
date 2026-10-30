"""Stage 10b · 系综的几何异质性（R2-C4b；s10_global_stability.py 的异质性排序表要用）。
对每个生成器 × 域，在匹配样本量下计算：平均成对 CA-RMSD（叠合后，nm）、Rg（CA 回转半径）在构象间的 SD（ddof=0，与 het_g 一致，nm）。
  BioEmu：S4a/S4b 冻结的 M=50、R=200 抽样（每个域取 R 次抽样的中位数）；其余生成器本身 50 帧，直接用。
  帧集合：优先用 S4 打分状态为 ok/ok_retry 的帧（与相关性分析同一批帧）；S4b 尚未合并时退回 xtc 全部帧，并在 frames_source 列标出。
  BioEmu 的 BioEmu_T3new 要等 S4b 合并并冻结索引后才有抽样，缺的域在 n 与 n_expected 里如实体现。
输出：tables/S10_het_geometry_per_domain.csv、tables/S10_het_geometry.csv（生成器 × 域集的域间中位数）
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
import sys
import numpy as np, pandas as pd
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, geom  # noqa: E402
import s10_global_stability as S  # noqa: E402

PROJ, REV, INP = S.PROJ, S.REV, S.INP
GEN_SETS = [("BioEmu", "S4_rescored", S.ALL3), ("ESMFlow", "S4_rescored", S.ALL3), ("Boltz2", "S4_rescored", S.ALL3), ("Str2Str_AF_d025", "S4_rescored", S.ALL3), ("Str2Str_AF_d035", "S4_rescored", S.ALL3),
            ("Str2Str_legacy_d025", "S4_rescored", ["D_legacy65"]), ("Str2Str_legacy_d035", "S4_rescored", ["D_legacy65"])]


def traj_paths(gdir, dom):
    if gdir in ("BioEmu_T3", "ESMFlow_T3", "Boltz2_T3", "Str2Str_T3", "Str2Str_T3_d035"):
        d = os.path.join(INP["gen_dir"], gdir, dom)
    elif gdir.startswith("Str2Str_AF"):
        d = os.path.join(REV, "raw_outputs", "s5", "gen", gdir, dom, "seed11")
    else:
        d = os.path.join(REV, "raw_outputs", "s5", "gen", gdir, dom)
    return os.path.join(d, "samples.xtc"), os.path.join(d, "topology.pdb")


def ok_frames(gdir, dom, n_xtc):
    f = os.path.join(REV, "raw_outputs", "s4", "counts", gdir, f"{dom}_frames.parquet")
    if os.path.exists(f):
        F = pd.read_parquet(f); return np.sort(F[F.status.isin(["ok", "ok_retry"])].frame_idx.values), "S4_ok_frames"
    return np.arange(n_xtc), "xtc_all_frames"


def one(gen, r, gdir):
    xtc, top = traj_paths(gdir, r.domain_id)
    if not (os.path.exists(xtc) and os.path.exists(top)):
        return None
    t = md.load(xtc, top=top); ca = t.topology.select("name CA"); X = t.xyz[:, ca, :].astype(float)
    ids, src = ok_frames(gdir, r.domain_id, len(X)); X = X[ids]
    cen = X - X.mean(1, keepdims=True); rg = np.sqrt((cen ** 2).sum(2).mean(1))
    D = geom.pairwise_rmsd(X); n = len(ids)
    if gen == "BioEmu":
        f50 = S.draw_file(gdir, r.domain_id, 50)
        if not os.path.exists(f50):
            return None
        fr = S.np_load_verified("F", f50)["frames"]; pos = np.searchsorted(ids, fr)
        assert np.array_equal(ids[pos], fr), f"{gdir}/{r.domain_id}: 冻结索引与帧集合不一致"
        iu = np.triu_indices(fr.shape[1], 1); sub = D[pos[:, :, None], pos[:, None, :]][:, iu[0], iu[1]]
        return dict(mean_pairwise_ca_rmsd_nm=float(np.median(sub.mean(1))), rg_sd_nm=float(np.median(rg[pos].std(1))), n_frames=int(fr.shape[1]), n_draws=int(fr.shape[0]), n_available=n, frames_source=src, n_ca=int(len(ca)))
    iu = np.triu_indices(n, 1)
    return dict(mean_pairwise_ca_rmsd_nm=float(D[iu].mean()), rg_sd_nm=float(rg.std()), n_frames=n, n_draws=1, n_available=n, frames_source=src, n_ca=int(len(ca)))


def main():
    log = revio.RunLogger(stage=10, task="het_geometry", cpu_procs=1, seeds="none (frozen draws)")
    DT = S.domain_table(); rows = []; summ = []
    for gen, sb, sets in GEN_SETS:
        spec = dict(gen=gen, scoring=sb)
        for r in DT.itertuples():
            ss = [s for s in sets if S.in_set(r, s)]
            gdir = S.source_name(gen, r, sb) if ss else None
            if not ss or gdir is None:
                continue
            res = one(gen, r, gdir)
            rows.append(dict(generator=gen, domain_id=r.domain_id, source=gdir, domain_sets=";".join(ss), expected=True, **(res or {})))
        print(f"[{gen}] 完成", flush=True)
    P = pd.DataFrame(rows); P.to_csv(os.path.join(S.TAB, "S10_het_geometry_per_domain.csv"), index=False)
    for gen, sb, sets in GEN_SETS:
        for s in sets:
            d = P[(P.generator == gen) & P.domain_sets.str.split(";").apply(lambda x: s in x)]
            got = d.dropna(subset=["mean_pairwise_ca_rmsd_nm"])
            summ.append(dict(generator=gen, domain_set=s, n=len(got), n_expected=len(d), median_mean_pairwise_ca_rmsd_nm=float(got.mean_pairwise_ca_rmsd_nm.median()) if len(got) else np.nan,
                             median_rg_sd_nm=float(got.rg_sd_nm.median()) if len(got) else np.nan, frames_source=";".join(sorted(got.frames_source.unique())) if len(got) else ""))
    T = pd.DataFrame(summ); f = os.path.join(S.TAB, "S10_het_geometry.csv"); T.to_csv(f, index=False)
    pd.set_option("display.width", 220); print(T.round(4).to_string(index=False)); log.finish([f], note=f"rows={len(P)}")


if __name__ == "__main__":
    main()
