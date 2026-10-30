"""Stage 4a · 偏离 D4-2 的证据：链端残基 rSASA>1.2 的原因（去端基原子的消融）。

对一批 rSASA>1.2 的链首/末残基，复现该帧（确定性种子 → 同一个加氢结构），分别计算：
  orig       ：完整结构上该残基的 rSASA（应与 parquet 一致）
  no_Cterm   ：去掉 C 端羧基的额外氧（OXT/OT1/OT2 中非主链 O 的那个）后
  no_Nterm   ：去掉 N 端 NH3⁺ 的多余氢（H2、H3，保留 H1/H）后
  no_both    ：两者都去掉
  interior_median：同一帧所有非末端残基的 rSASA 中位数（量级参照）
目的：检验"超过 1.2 是端基原子造成的"这一假设（结论：不是主因；真正的原因没有单独查明）。结果写入 tables/S4_terminal_ablation.csv。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 逐位复现的前提：数值库加载之前就把线程数固定为 1（S4a 返工发现：先导入 numpy/mdtraj 再设置则不起作用，同种子的主链+Cβ 帧会与存档不一致）
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
import numpy as np, pandas as pd
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402
import s4_rescore as S4  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=4, task="terminal_ablation", cpu_procs=1, seeds="det_seed of the scored frame")
max_asa = bv.load_max_asa(os.path.join(REV, "data", "external", "tien2013_maxasa", "tien2013_theoretical_maxASA.csv"))
base = os.path.join(REV, "raw_outputs", "s4", "counts")
nb = pd.read_csv(os.path.join(REV, "tables", "S1_deposited_structure_numbering.csv"))
cases = [("ESMFlow", "Im7"), ("ESMFlow", "CI2"), ("Boltz2", "CI2"), ("ShortMD", "CI2"), ("ShortMD", "Ubiquitin"), ("Crystal", "Ubiquitin"), ("Crystal", "CI2"), ("BioEmu", "CI2"), ("Str2Str", "CI2"), ("ESMFlow_T3", "T3_002")]
rows = []
for g, p in cases:
    R = pd.read_parquet(os.path.join(base, g, f"{p}.parquet"), columns=["frame_idx", "recon_k", "resSeq", "resn", "rSASA_res"])
    F = pd.read_parquet(os.path.join(base, g, f"{p}_frames.parquet"))
    top = md.load(os.path.join(INP["pdb_dir"], f"{nb[nb.protein == p].pdb.iloc[0]}.pdb") if g == "Crystal" else os.path.join(INP["gen_dir"], g, p, "topology.pdb")).topology
    ch = max(top.chains, key=lambda c: sum(1 for r in c.residues if r.name in bv.T2O)); lo, hi = 1, ch.n_residues
    for end, rs in (("N", lo), ("C", hi)):
        hit = R[(R.resSeq == rs) & (R.rSASA_res > 1.2)]
        if hit.empty:
            continue
        r0 = hit.iloc[0]; k, rk = int(r0.frame_idx), int(r0.recon_k); st = F[(F.frame_idx == k) & (F.recon_k == rk)].iloc[0]
        if g == "Crystal":
            fr = md.load(os.path.join(INP["pdb_dir"], f"{nb[nb.protein == p].pdb.iloc[0]}.pdb")); fr = fr[0] if fr.n_frames > 1 else fr
        else:
            fr = md.load_frame(os.path.join(INP["gen_dir"], g, p, "samples.xtc"), k, top=os.path.join(INP["gen_dir"], g, p, "topology.pdb"))
        assert S4.det_seed(g, p, k, rk, int(st.attempts) - 1) == int(st.seed)
        t1 = bv.fix_frame(fr, True, seed=int(st.seed)); T = t1.topology
        res = {r.resSeq: r for r in T.residues}; tr = res[rs]
        def rs_of(drop):
            keep = [a.index for a in T.atoms if a.index not in drop]
            return bv.sasa_features(t1.atom_slice(keep), max_asa)[rs][0]
        names = [a.name for a in tr.atoms]
        oxt = [a.index for a in tr.atoms if a.name in ("OXT", "OT1", "OT2")] if end == "C" else []
        extra_h = [a.index for a in tr.atoms if a.name in ("H2", "H3")] if end == "N" else []
        orig = bv.sasa_features(t1, max_asa)[rs][0]; interior = [v[0] for q, v in bv.sasa_features(t1, max_asa).items() if q not in (lo, hi)]
        rows.append(dict(generator=g, protein=p, frame_idx=k, recon_k=rk, end=end, resSeq=rs, resn=tr.name, rSASA_parquet=float(r0.rSASA_res), rSASA_recomputed=float(orig),
                         rSASA_without_end_group_atoms=float(rs_of(set(oxt + extra_h))) if (oxt or extra_h) else np.nan, n_end_group_atoms_removed=len(oxt + extra_h),
                         atoms=" ".join(names), interior_median_rSASA=float(np.median(interior))))
D = pd.DataFrame(rows); D["drop_by_removing_end_group"] = D.rSASA_recomputed - D.rSASA_without_end_group_atoms
D["still_above_1p2_after_removal"] = D.rSASA_without_end_group_atoms > 1.2
f = os.path.join(REV, "tables", "S4_terminal_ablation.csv"); D.to_csv(f, index=False)
pd.set_option("display.width", 250); print(D.drop(columns=["atoms"]).round(3).to_string(index=False))
log.finish([f], note=f"cases={len(D)}")
