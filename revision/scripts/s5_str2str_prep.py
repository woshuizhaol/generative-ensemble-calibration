"""Stage 5A · Str2Str 输入准备：与单结构基线完全相同的沉积结构（revision_proposal.md S5 5A 步骤 1）。

规则与 bv.fix_frame 的取链逻辑一致：md.load 沉积结构（多模型取第 0 个），取"标准残基数最多的链"，只保留标准氨基酸残基的重原子
（去掉水、配体、血红素、乙酰帽等），不做 pdbfixer 补全，原样保留 PDB 编号。输出 R/data/interim/str2str_xtal_in/<protein>.pdb，
并登记每个输入的 PDB 号、原文件 sha256、链、残基数、原子数、是否缺主链原子，以及与 BioEmu 拓扑的 CA-RMSD（验收：必须 >0.05 nm，防止重犯 F4）。
用法：python s5_str2str_prep.py
输出：data/interim/str2str_xtal_in/*.pdb；tables/S5_str2str_xtal_inputs.csv
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
import numpy as np, pandas as pd
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=5, task="str2str_prep", cpu_procs=1, seeds="none")
out_dir = os.path.join(REV, "data", "interim", "str2str_xtal_in"); os.makedirs(out_dir, exist_ok=True)
nb = pd.read_csv(os.path.join(REV, "tables", "S1_deposited_structure_numbering.csv")); seqs = list(__import__("json").load(open(INP["tier1a_sequences"])))
rows = []
for p in sorted(seqs):
    pdb = nb[nb.protein == p].pdb.iloc[0]; src = os.path.join(INP["pdb_dir"], f"{pdb}.pdb")
    t = md.load(src); t = t[0] if t.n_frames > 1 else t
    ch = max(t.topology.chains, key=lambda c: sum(1 for r in c.residues if r.name in bv.T2O))
    keep = [a.index for a in ch.atoms if a.residue.name in bv.T2O and a.element is not None and a.element.symbol != "H"]
    s = t.atom_slice(keep)
    # Str2Str 的 PDB 解析按 (链, 残基号, 插入码) 识别残基；mdtraj 读入时丢掉插入码，1RIL 的 80/80A 会变成两个同号残基，Str2Str 把它们并成一个（输出少 1 个残基，
    # S5A 第一次运行时 QC 发现）。所以保证残基号严格递增：n_i = max(原号, n_{i-1}+1)，只改动重号及其后被挤占的号，原有的空缺（缺失残基）保留。
    prev = None; n_renum = 0
    for r_ in s.topology.residues:
        new = r_.resSeq if prev is None else max(r_.resSeq, prev + 1)
        if new != r_.resSeq:
            n_renum += 1; r_.resSeq = new
        prev = new
    dst = os.path.join(out_dir, f"{p}.pdb"); s.save_pdb(dst)
    res = list(s.topology.residues); have = {r.index: {a.name for a in r.atoms} for r in res}
    miss_bb = sum(1 for r in res if not {"N", "CA", "C"} <= have[r.index])
    gaps = sum(1 for a, b in zip(res[:-1], res[1:]) if b.resSeq - a.resSeq != 1)
    # 与 BioEmu 拓扑的 CA-RMSD（序列相同、残基数相同时）：必须 >0.05 nm
    bt = md.load(os.path.join(INP["gen_dir"], "BioEmu", p, "topology.pdb")); ca_s = s.topology.select("name CA"); ca_b = bt.topology.select("name CA")
    # 残基数可能不同（如 CI2：2CI2 为 65，BioEmu 为 64），所以按单字母序列比对取公共残基，再叠合算 CA-RMSD
    import difflib
    seq_s = "".join(bv.T2O[s.topology.atom(i).residue.name] for i in ca_s); seq_b = "".join(bv.T2O[bt.topology.atom(i).residue.name] for i in ca_b)
    idx_s, idx_b = [], []
    for blk in difflib.SequenceMatcher(None, seq_s, seq_b, autojunk=False).get_matching_blocks():
        idx_s += list(range(blk.a, blk.a + blk.size)); idx_b += list(range(blk.b, blk.b + blk.size))
    rmsd = float(md.rmsd(s.atom_slice(ca_s[idx_s]), bt.atom_slice(ca_b[idx_b]))[0]) if len(idx_s) >= 30 else float("nan")
    n_common = len(idx_s)
    rows.append(dict(protein=p, pdb=pdb, src_sha256=revio.sha256(src), chain_index=ch.index, n_residues=len(res), n_atoms=s.n_atoms, n_res_missing_NCAC=miss_bb, n_chain_gaps=gaps, n_resSeq_renumbered=n_renum,
                     first_resSeq=res[0].resSeq, last_resSeq=res[-1].resSeq, n_common_CA=n_common, ca_rmsd_to_bioemu_nm=rmsd, out=os.path.relpath(dst, REV), out_sha256=revio.sha256(dst)))
D = pd.DataFrame(rows); f = os.path.join(REV, "tables", "S5_str2str_xtal_inputs.csv"); D.to_csv(f, index=False)
pd.set_option("display.width", 250); print(D.drop(columns=["src_sha256", "out_sha256", "out"]).round(3).to_string(index=False))
log.finish([f, out_dir], note=f"proteins={len(D)}")
