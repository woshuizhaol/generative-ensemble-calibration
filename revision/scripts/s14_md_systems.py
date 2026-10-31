"""Stage 14 · 体系设置表（编辑表 4a）：从 BioEmu 构象出发的 24 条短 MD（12 蛋白 × r4、r5）的体系统计。
列与 S13_md_systems.csv 相同，但链数和残基数取自各轨迹自己的 md_meta.json（r4、r5 的起点是 BioEmu 构象的单链，与 r1 的多链体系不同）（链数、残基数、初始盒子与生产段均值盒子、总原子数、水分子数、Na+/Cl- 数、盐浓度、种子、墙钟、版本）。
数据来源：md_replicates/md/<蛋白>_r4|r5/ 下的 md_meta.json、ensemble.xtc、ensemble.pdb、md.log（s14_altstart_md.py 的输出）。
输出：tables/S14_md_systems.csv。不改动任何已有表。"""
import os, sys, json, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, s13util as U  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=14, task="md_systems", cpu_procs=1, seeds="none (deterministic table)")
S3 = pd.read_csv(os.path.join(REV, "tables", "S3_md_protocol_per_protein.csv")).set_index("protein")
NA_AVO = 0.602214076
WATER_NM3 = 0.0300
PROTS = sorted(json.load(open(INP["tier1a_sequences"])))


def row(p, rep):
    P = U.md_paths(rep, p); s3 = S3.loc[p]; tr = md.load(P["ens_xtc"], top=P["ens_pdb"]); L = U.read_md_log(P["md_log"]); m = json.load(open(P["meta"]))
    r = dict(protein=p, replicate=rep, pdb_id=s3.pdb_id, n_chains_in_box=int(m["n_protein_chains"]), protein_residues_in_box=int(m["n_protein_residues_all_chains"]),
             n_protein_atoms=int(tr.n_atoms), box_prod_mean_a_nm=float(tr.unitcell_lengths[:, 0].mean()), box_prod_mean_b_nm=float(tr.unitcell_lengths[:, 1].mean()), box_prod_mean_c_nm=float(tr.unitcell_lengths[:, 2].mean()),
             box_prod_mean_volume_nm3=float(tr.unitcell_volumes.mean()), mean_T_K=float(L.temperature_K.mean()), sd_T_K=float(L.temperature_K.std(ddof=1)), n_log_rows=len(L), n_frames=tr.n_frames,
             n_atoms_total=m["n_atoms_total"], n_waters=m["n_waters"], n_Na=m["n_Na"], n_Cl=m["n_Cl"], box_initial_a_nm=m["box_initial_nm"][0], box_initial_b_nm=m["box_initial_nm"][1], box_initial_c_nm=m["box_initial_nm"][2],
             seed_md=m["seed_md"], seed_build=m["seed_build"], wall_clock_min=round(m["wall_s_total"] / 60.0, 2), started=m["started"], net_charge_total_e=m["net_charge_total_e"])
    v0 = r["box_initial_a_nm"] * r["box_initial_b_nm"] * r["box_initial_c_nm"]
    pairs = min(r["n_Na"], r["n_Cl"])
    r.update(box_initial_volume_nm3=float(v0), n_NaCl_pairs_beyond_neutralization=int(pairs), salt_conc_M_per_box_volume=pairs / (NA_AVO * v0), salt_conc_M_per_water_volume=pairs / (NA_AVO * r["n_waters"] * WATER_NM3), waters_per_nm3=r["n_waters"] / v0)
    return r


rows = [row(p, rep) for p in PROTS for rep in ("r4", "r5")]
df = pd.DataFrame(rows); out = os.path.join(REV, "tables", "S14_md_systems.csv"); df.to_csv(out, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df.shape); print(df[["protein", "replicate", "n_atoms_total", "n_waters", "n_Na", "n_Cl", "box_initial_a_nm", "box_prod_mean_a_nm", "mean_T_K", "n_frames", "seed_md", "wall_clock_min"]].round(3).to_string(index=False))
log.finish([out])
