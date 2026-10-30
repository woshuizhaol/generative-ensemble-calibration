"""Stage 9 追加（U-S3-2）· 步骤 7：把新的 cyt c 轨迹（带血红素 / 无血红素对照）按与其他生成器完全相同的流程打分。
流程：full.xtc（含水）→ 只留蛋白原子（去掉水、离子、血红素 HEC 和铁 FE；血红素原子不进入接触数，与所有蛋白、所有生成器一致）→ ensemble.xtc / ensemble.pdb
  （残基名改回原 ensemble.pdb 的惯例：HID → HIS，CYX → CYS）→ `s4_rescore.run_task`（逐帧 PDBFixer 重新加氢 pH 7.0、N_C / N_H、rSASA；确定性种子 det_seed）→ 合并成
  raw_outputs/s4/counts/<generator>/cytochrome_c.parquet 与 _frames.parquet（生成器名 ShortMD_cytc_apo_s<seed> / ShortMD_cytc_heme_s<seed>，与已有的 ShortMD 分开，不覆盖任何旧文件）。
所有 200 帧都打分，失败帧记入帧状态表（不丢弃）。CPU 单线程（OPENMM_CPU_THREADS=1，须在 OpenMM 加载前设置）。
用法：python s9d_cytc_score.py --runs cytc_apo_s101,cytc_apo_s102,… [--workers 4]"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, argparse, glob
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np, pandas as pd
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402
import s4_rescore as S4  # noqa: E402

PROJ, REV, INP = revio.load_paths(); MD = os.path.join(REV, "md_cytc_heme", "md"); PARTS = os.path.join(REV, "md_cytc_heme", "analysis", "parts")
OUT = os.path.join(REV, "raw_outputs", "s4", "counts"); CHUNK = 25
ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
log = revio.RunLogger(stage=9, task="cytc_score", cpu_procs=a.workers, seeds="det_seed(generator, cytochrome_c, frame, 0, attempt)")
DROP = {"HOH", "WAT", "Na+", "Cl-", "NA", "CL", "HEC", "FE"}


def make_ensemble(run):
    d = os.path.join(MD, run); t = md.load(os.path.join(d, "full.xtc"), top=os.path.join(d, "final.pdb"))
    keep = [at.index for at in t.topology.atoms if at.residue.name not in DROP]
    prot = t.atom_slice(keep)
    for r in prot.topology.residues:
        if r.name == "HID":
            r.name = "HIS"
        if r.name == "CYX":
            r.name = "CYS"
    prot.save_xtc(os.path.join(d, "ensemble.xtc")); prot[0].save_pdb(os.path.join(d, "ensemble.pdb"))
    return d, prot.n_frames, prot.n_atoms


def main():
    max_asa = os.path.join(REV, "data", "external", "tien2013_maxasa", "tien2013_theoretical_maxASA.csv"); tasks = []; info = {}
    for run in a.runs.split(","):
        d, nfr, nat = make_ensemble(run); g = "ShortMD_" + run; info[run] = (g, nfr, nat)
        for f0 in range(0, nfr, CHUNK):
            tasks.append(dict(generator=g, protein="cytochrome_c", recon_k=0, kind="xtc", xtc=os.path.join(d, "ensemble.xtc"), top=os.path.join(d, "ensemble.pdb"), f0=f0, f1=min(nfr, f0 + CHUNK), group="cytc_md", parts_root=PARTS))
        print(run, "frames", nfr, "protein atoms", nat, flush=True)
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        futs = [ex.submit(S4.run_task, (t, REV, max_asa)) for t in tasks]
        for k, f in enumerate(cf.as_completed(futs), 1):
            f.result()
            if k % 8 == 0 or k == len(futs):
                print(f"{k}/{len(futs)} chunks done", flush=True)
    outs = []
    for run, (g, nfr, nat) in info.items():
        base = os.path.join(PARTS, g, "cytochrome_c"); rs = sorted(glob.glob(os.path.join(base, "*.parquet")))
        res_files = [f for f in rs if not f.endswith("_frames.parquet")]; frm_files = [f for f in rs if f.endswith("_frames.parquet")]
        R = pd.concat([pd.read_parquet(f) for f in res_files], ignore_index=True); F = pd.concat([pd.read_parquet(f) for f in frm_files], ignore_index=True)
        assert set(F.frame_idx) == set(range(nfr)) and not F.duplicated(["frame_idx", "recon_k"]).any(), (g, "帧不完整或重复")
        ok = F.status.isin(["ok", "ok_retry"]); assert all((int(r.frame_idx), 0) in set(zip(R.frame_idx.astype(int), R.recon_k.astype(int))) for r in F[ok].itertuples())
        d = os.path.join(OUT, g); os.makedirs(d, exist_ok=True)
        R.sort_values(["frame_idx", "recon_k", "resSeq"]).to_parquet(os.path.join(d, "cytochrome_c.parquet"), index=False); F.sort_values(["frame_idx", "recon_k"]).to_parquet(os.path.join(d, "cytochrome_c_frames.parquet"), index=False)
        print(g, "frames", len(F), "ok", int((F.status == "ok").sum()), "retry", int((F.status == "ok_retry").sum()), "failed", int((~ok).sum()), "residues scored", R.resSeq.nunique(), "core-s", round(float(F.t_wall_all_attempts_s.sum()), 1))
        outs.append(os.path.join(d, "cytochrome_c.parquet"))
    log.finish(outs)


if __name__ == "__main__":
    main()
