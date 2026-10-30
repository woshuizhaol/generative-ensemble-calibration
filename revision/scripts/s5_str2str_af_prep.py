"""Stage 5B · Str2Str 从 AlphaFold 模型起始：193 个有精确 AF 模型的域的输入（方案 5B 步骤 1-2；19 个只有亲本模型的域不做，补充起点口径已撤回；S10 在 193 个共有域上补报）。
起点 = Tsuboyama 公布的 AlphaFold 模型（R/data/external/tsuboyama_af/unz/AlphaFold_model_PDBs/<af_file>），只保留重原子（与 5A 沉积结构输入一致），残基号保证严格递增。
文件名用 safe_id（域编号里的"."换成"_"），避免 Str2Str 按 ".pdb" 切文件名出错；对应关系写入表。
验收：起点不得是 BioEmu 构象——与 P/gen/BioEmu_T3 拓扑（有则比）的 CA-RMSD 必须 >0.05 nm（防止重犯 F4）。
输出：data/interim/str2str_af_in/<safe_id>.pdb；tables/S5_str2str_af_inputs.csv"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np, pandas as pd
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio, geom  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=5, task="str2str_af_prep", cpu_procs=1, seeds="none")
out_dir = os.path.join(REV, "data", "interim", "str2str_af_in"); os.makedirs(out_dir, exist_ok=True)
src_dir = os.path.join(REV, "data", "external", "tsuboyama_af", "unz", "AlphaFold_model_PDBs")
d = pd.read_csv(os.path.join(REV, "tables", "S5_domain_table.csv")); d = d[d.need_Str2Str_AF]; rows = []
for r in d.itertuples():
    src = os.path.join(src_dir, r.af_file); t = md.load(src)
    s = t.atom_slice([a.index for a in t.topology.atoms if a.element is not None and a.element.symbol != "H" and a.residue.name in bv.T2O])
    prev = None; nren = 0
    for q in s.topology.residues:
        new = q.resSeq if prev is None else max(q.resSeq, prev + 1)
        nren += int(new != q.resSeq); q.resSeq = new; prev = new
    sid = r.domain_id.replace(".", "_"); dst = os.path.join(out_dir, f"{sid}.pdb"); s.save_pdb(dst)
    ca = s.topology.select("name CA"); rmsd = np.nan; bpath = os.path.join(INP["gen_dir"], "BioEmu_T3", r.domain_id, "topology.pdb")
    if os.path.exists(bpath):
        b = md.load(bpath); cb = b.topology.select("name CA")
        if len(cb) == len(ca):
            rmsd = float(geom.rmsd_to_ref(s.xyz[:, ca], b.xyz[0, cb])[0])
    rows.append(dict(domain_id=r.domain_id, safe_id=sid, af_file=r.af_file, src_sha256=revio.sha256(src), n_res=s.n_residues, n_atoms=s.n_atoms, n_resSeq_renumbered=nren,
                     seq_equals_domain_table="".join(bv.T2O[q.name] for q in s.topology.residues) == r.seq, ca_rmsd_to_bioemu_nm=rmsd, out_sha256=revio.sha256(dst)))
D = pd.DataFrame(rows); f = os.path.join(REV, "tables", "S5_str2str_af_inputs.csv"); D.to_csv(f, index=False)
chk = dict(n=len(D), all_seq_equal=bool(D.seq_equals_domain_table.all()), renumbered=int(D.n_resSeq_renumbered.sum()), n_compared_with_bioemu=int(D.ca_rmsd_to_bioemu_nm.notna().sum()),
           min_rmsd_to_bioemu_nm=float(D.ca_rmsd_to_bioemu_nm.min()), all_gt_0p05=bool((D.ca_rmsd_to_bioemu_nm.dropna() > 0.05).all()), unique_safe_ids=int(D.safe_id.nunique()))
print(chk); log.finish([f, out_dir], note=str(chk))
