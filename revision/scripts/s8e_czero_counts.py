"""Stage 8 · C-zero 验收未全部通过后的诊断：C-zero（ANM，幅度 0.01 Å，种子 21）与沉积结构重建在计数层面差多少。
对每个蛋白的有实验值的残基（编号 resSeq = resi + offset，offset 取原稿 operator_ceiling.offset）：
  frac_NC_changed / frac_NH_changed：C-zero 50 个构象的平均 N_C（N_H）与沉积结构 recon_0 相差 ≥0.5 的残基比例；mean_abs_dNC / mean_abs_dNH：平均绝对差；
  recon_NC_range_max / recon_NH_range_max：沉积结构 10 次重建之间同一残基计数的最大极差（Barstar、CheY 自带氢，N_C 与 N_H 皆为 0）。
另给 C-zero 帧（xtc，约束最小化之后的重原子坐标）相对参考结构（未最小化的 recon_0 重原子，topology_heavy.pdb）的位移：czero_heavy_rmsd_to_ref_A（全部重原子，不叠合，帧均值）、czero_ca_rmsd_to_ref_A（CA，不叠合）、czero_frame_spread_A（各帧到平均帧的 RMSD 的均值，即噪声本身的散布）。
这是描述性诊断，不是新的检验。输出 tables/S8_czero_count_diff.csv。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdtraj as md  # noqa: E402
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths(); TAB = os.path.join(REV, "tables"); CNT = os.path.join(REV, "raw_outputs", "s4", "counts")
log = revio.RunLogger(stage=8, task="czero_counts", cpu_procs=1, seeds="none")
S1 = pd.read_parquet(os.path.join(TAB, "S1_tier1a_residues_final.parquet")); OC = pd.read_parquet(os.path.join(PROJ, "results", "operator_ceiling.parquet")).set_index("protein").offset.to_dict()
rows = []
for p in sorted(S1.protein_id.unique()):
    res = (S1[S1.protein_id == p].dropna(subset=["dGopen_kcal"]).resi.astype(int) + int(OC[p])).values
    C = pd.read_parquet(os.path.join(CNT, "Crystal", f"{p}.parquet")); C = C[C.resSeq.isin(res)]
    z = pd.read_parquet(os.path.join(CNT, "noise_anm_czero_s21", f"{p}.parquet")); z = z[z.resSeq.isin(res)]
    c0 = C[C.recon_k == 0].set_index("resSeq").sort_index(); zm = z.groupby("resSeq")[["NC", "NH"]].mean().sort_index(); common = c0.index.intersection(zm.index)
    dNC = (zm.loc[common, "NC"] - c0.loc[common, "NC"]).abs(); dNH = (zm.loc[common, "NH"] - c0.loc[common, "NH"]).abs()
    rng_ = C.groupby("resSeq")[["NC", "NH"]].agg(lambda x: x.max() - x.min())
    top = md.load(os.path.join(REV, "raw_outputs", "s5", "noise_work", "xtc", p, "topology_heavy.pdb")); ref = top.xyz[0]; tr = md.load(os.path.join(REV, "raw_outputs", "s5", "noise_work", "xtc", p, "anm", "czero", "seed21.xtc"), top=top.topology)
    ca = np.array([a_.index for a_ in top.topology.atoms if a_.name == "CA"]); dd = np.linalg.norm(tr.xyz - ref[None], axis=2) * 10; mean_frame = tr.xyz.mean(0)
    spread = float((np.sqrt(((tr.xyz - mean_frame[None]) ** 2).sum(2).mean(1)) * 10).mean())
    rows.append(dict(protein=p, n_exp_residues=len(common), frac_NC_changed=float((dNC >= 0.5).mean()), mean_abs_dNC=float(dNC.mean()), frac_NH_changed=float((dNH >= 0.5).mean()), mean_abs_dNH=float(dNH.mean()),
                     recon_NC_range_max=float(rng_.NC.max()), recon_NH_range_max=float(rng_.NH.max()),
                     czero_heavy_rmsd_to_ref_A=float(np.sqrt((dd ** 2).mean(1)).mean()), czero_ca_rmsd_to_ref_A=float(np.sqrt((dd[:, ca] ** 2).mean(1)).mean()), czero_frame_spread_A=spread))
T = pd.DataFrame(rows); f = os.path.join(TAB, "S8_czero_count_diff.csv"); T.to_csv(f, index=False); pd.set_option("display.width", 220); print(T.round(3).to_string(index=False))
log.finish([f], note="czero count diff")
