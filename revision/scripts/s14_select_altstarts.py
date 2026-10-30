"""Stage 14 · 选择短 MD 的替代起点（Editorial Requests Table 1d：结果是否独立于初始构型）。

对 12 个蛋白，各从 BioEmu 系综（原项目 gen/BioEmu/<蛋白>/samples.xtc，S4 帧状态为 ok 的帧）取两个构象作为 MD 起点：
  与沉积结构（链 A）的 Cα 均方根偏差（核心残基，去掉每端 3 个残基，Kabsch 叠合）处于该蛋白系综分布第 25 与第 75 百分位的构象。
  取各目标百分位处偏差最近的帧；若该帧的 Rg 偏离沉积结构 >15% 则顺延到次近的帧（记录在表中）。
序列对应：BioEmu 序列与沉积结构链 A 序列做全局比对（Biopython PairwiseAligner），按比对位点配对 Cα。
输出：md_altstart/starts/<蛋白>_a<1|2>.pdb（该帧的 BioEmu 原子，链 A，残基从 1 编号）与 tables/S14_altstart_selection.csv。
不使用随机数。用法（rev 环境）：python s14_select_altstarts.py [--proteins A,B]"""
import os, sys, json, argparse
import numpy as np, pandas as pd
import mdtraj as md
from Bio import Align
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
ap = argparse.ArgumentParser(); ap.add_argument("--proteins", default=""); ap.add_argument("--out-dir", default=os.path.join(REV, "md_altstart", "starts")); a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
         "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "MSE": "M"}
S3 = pd.read_csv(os.path.join(REV, "tables", "S3_md_protocol_per_protein.csv"))
PERC = {1: 25.0, 2: 75.0}
END = 3


def kabsch_rmsd(P, Q):
    """P, Q: (n, 3)。返回叠合后的 RMSD（nm）。"""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc; U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T)); D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean()))


def chain_seq_ca(traj, chain_index=0):
    top = traj.topology; res = [r for r in top.chain(chain_index).residues if r.is_protein and any(at.name == "CA" for at in r.atoms)]
    seq = "".join(THREE.get(r.name, "X") for r in res); ca = [r.atom("CA").index for r in res]
    return seq, np.array(ca)


def main():
    rows = []
    prots = [r for r in S3.itertuples() if (not a.proteins or r.protein in a.proteins.split(","))]
    aligner = Align.PairwiseAligner(); aligner.mode = "global"; aligner.match_score = 2.0; aligner.mismatch_score = -1.0; aligner.open_gap_score = -3.0; aligner.extend_gap_score = -0.5
    for r in prots:
        p, pdb = r.protein, r.pdb_id
        gd = os.path.join(INP["gen_dir"], "BioEmu", p)
        ens = md.load(os.path.join(gd, "samples.xtc"), top=os.path.join(gd, "topology.pdb"))
        F = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "BioEmu", f"{p}_frames.parquet")); okf = np.array(sorted(F[F.status.isin(["ok", "ok_retry"])].frame_idx.unique()), int)
        dep = md.load(os.path.join(PROJ, "data", "external", "pdb", f"{pdb}.pdb"))
        dep = dep.atom_slice(dep.topology.select("protein and chainid 0")); dep = dep[0]
        s_dep, ca_dep = chain_seq_ca(dep); s_ens, ca_ens = chain_seq_ca(ens)
        aln = aligner.align(s_ens, s_dep)[0]; pairs = []
        for (a0, a1), (b0, b1) in zip(*aln.aligned):
            pairs += list(zip(range(a0, a1), range(b0, b1)))
        pairs = [(i, j) for i, j in pairs if s_ens[i] == s_dep[j]]
        core = pairs[END:-END]; ie = np.array([ca_ens[i] for i, _ in core]); jd = np.array([ca_dep[j] for _, j in core])
        X = ens.xyz[:, ie, :]; Y = dep.xyz[0, jd, :]
        rm = np.array([kabsch_rmsd(X[f], Y) for f in range(ens.n_frames)]) * 10.0   # Å
        # Rg（Cα）比较
        rg = np.array([md.compute_rg(ens[f:f + 1], masses=None)[0] for f in range(ens.n_frames)]) * 10.0 if False else None
        ca_all = ens.xyz[:, ca_ens, :]; rg_f = np.sqrt(((ca_all - ca_all.mean(1, keepdims=True)) ** 2).sum(2).mean(1)) * 10.0
        cad = dep.xyz[0, ca_dep, :]; rg_d = float(np.sqrt(((cad - cad.mean(0)) ** 2).sum(1).mean()) * 10.0)
        pool = rm[okf]
        for k, pc in PERC.items():
            target = np.percentile(pool, pc); order = okf[np.argsort(np.abs(pool - target))]
            chosen, skipped = None, []
            for f in order:
                if abs(rg_f[f] - rg_d) / rg_d <= 0.15:
                    chosen = int(f); break
                skipped.append(int(f))
            assert chosen is not None, (p, k)
            sub = ens[chosen]; sub.topology = ens.topology
            fn = os.path.join(a.out_dir, f"{p}_a{k}.pdb"); sub.save_pdb(fn)
            rows.append(dict(protein=p, start=f"a{k}", replicate=3 + k, frame_idx=chosen, target_percentile=pc, calpha_rmsd_to_deposited_A=round(float(rm[chosen]), 3),
                             pool_rmsd_percentile_25_A=round(float(np.percentile(pool, 25)), 3), pool_rmsd_median_A=round(float(np.median(pool)), 3), pool_rmsd_percentile_75_A=round(float(np.percentile(pool, 75)), 3),
                             rg_start_A=round(float(rg_f[chosen]), 3), rg_deposited_A=round(rg_d, 3), skipped_frames_rg=";".join(map(str, skipped)), n_pool=int(len(pool)),
                             n_aligned_core=len(core), n_residues_ensemble=len(s_ens), n_residues_deposited=len(s_dep), pdb_id=pdb, file=os.path.relpath(fn, REV)))
            print(p, k, chosen, f"{rm[chosen]:.2f} Å (pool 25/50/75: {np.percentile(pool, 25):.2f}/{np.median(pool):.2f}/{np.percentile(pool, 75):.2f})", flush=True)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(REV, "tables", "S14_altstart_selection.csv"), index=False); print(T.to_string())


if __name__ == "__main__":
    main()
