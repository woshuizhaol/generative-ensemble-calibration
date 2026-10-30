"""Stage 13 (d)(e) · 体系设置表（清单 4a）与“结果与初始构型的关系”的数据（清单 1d；只给数据，不作“独立于初始构型”的论断）。
输出（tables/）：
  S13_md_systems.csv                      每条运行（12 蛋白 × r1、r2、r3）一行：链数、残基数、盒子（初始三边与生产段均值三边）、总原子数、水分子数、Na+/Cl- 数、盐浓度、平均温度、墙钟、种子、软件版本
  S13_shortmd_initial_structure_data.csv  每蛋白 × 副本：ρ、Cα RMSD（对沉积结构）、副本 ln PF 谱与沉积结构（10 次重建均值）ln PF 谱的 Spearman
  S13_shortmd_replicate_pair_data.csv     每蛋白 × 副本对（r1-r2、r1-r3、r2-r3）：平均结构间 Cα RMSD、跨副本 vs 同副本的帧—帧 Cα RMSD、ln PF 谱 Spearman、ρ 差
r1 的体系统计取自其 full_top.pdb / ensemble.xtc / md.log 与 S3 表（原脚本没有保存 initial_system、种子和日志细节，见备注列）。
用法：python s13_systems_and_1d.py [--reps r1,r2,r3] [--out-dir DIR] [--proteins A,B]"""
import os, sys, json, argparse, warnings, itertools
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import mdtraj as md
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, ensemble as E, s13util as U, bv  # noqa: E402

PROJ, REV, INP = revio.load_paths()
ap = argparse.ArgumentParser(); ap.add_argument("--reps", default="r1,r2,r3"); ap.add_argument("--out-dir", default=os.path.join(REV, "tables")); ap.add_argument("--proteins", default=""); a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True); REPS = a.reps.split(",")
log = revio.RunLogger(stage=13, task="systems_and_1d", cpu_procs=1, seeds="none (deterministic frame subsets)")
PROTS = [p for p in sorted(json.load(open(INP["tier1a_sequences"]))) if not a.proteins or p in a.proteins.split(",")]
S3 = pd.read_csv(os.path.join(REV, "tables", "S3_md_protocol_per_protein.csv")).set_index("protein")
NA_AVO = 0.602214076   # n / (0.602214 × V[nm³]) = mol/L
WATER_NM3 = 0.0300     # 1 个水分子在 0.997 g/mL 下占的体积（nm³），用来由水分子数估水体积
CEIL = pd.read_parquet(INP["operator_ceiling"]); OFF_CRYST = dict(zip(CEIL.protein, CEIL.offset.astype(int)))
SEL = [9 + 20 * k for k in range(10)]   # 第 10、30、…、190 帧（1 基）


def count_res(top):
    names = [r.name for r in top.residues]
    return dict(n_waters=sum(x in ("HOH", "WAT") for x in names), n_Na=sum(x in ("NA", "Na+") for x in names), n_Cl=sum(x in ("CL", "Cl-") for x in names))


def system_row(p, rep):
    P = U.md_paths(rep, p); s3 = S3.loc[p]; tr = md.load(P["ens_xtc"], top=P["ens_pdb"]); L = U.read_md_log(P["md_log"])
    row = dict(protein=p, replicate=rep, pdb_id=s3.pdb_id, n_chains_in_box=int(s3.n_chains_in_box), protein_residues_in_box=int(s3.protein_residues_in_box), scored_chain_residues=int(s3.scored_chain_residues),
               n_protein_atoms=int(tr.n_atoms), box_prod_mean_a_nm=float(tr.unitcell_lengths[:, 0].mean()), box_prod_mean_b_nm=float(tr.unitcell_lengths[:, 1].mean()), box_prod_mean_c_nm=float(tr.unitcell_lengths[:, 2].mean()),
               box_prod_mean_volume_nm3=float(tr.unitcell_volumes.mean()), mean_T_K=float(L.temperature_K.mean()), sd_T_K=float(L.temperature_K.std(ddof=1)), n_log_rows=len(L), n_frames=tr.n_frames, nominal_salt="0.1 M NaCl (Modeller ionicStrength) + neutralizing ions",
               force_field="amber14-all.xml + amber14/tip3pfb.xml (ff14SB, TIP3P-FB, Joung-Cheatham ions)", ph=7.0, padding_nm=1.0, production_ns=2.0, equilibration_ps_npt=100.0)
    if rep == "r1":
        ft = md.load(P["full_top"]); c = count_res(ft.topology); lens = ft.unitcell_lengths[0] if ft.unitcell_lengths is not None else [np.nan] * 3
        row.update(n_atoms_total=int(ft.n_atoms), **c, box_initial_a_nm=float(lens[0]), box_initial_b_nm=float(lens[1]), box_initial_c_nm=float(lens[2]), seed_md="unset (OpenMM default random seed)", seed_build="unset",
                   wall_clock_min=float(s3.wall_clock_min) if str(s3.wall_clock_min).replace(".", "", 1).isdigit() else np.nan, wall_clock_note=str(s3.wall_clock_note), openmm_version=str(s3.openmm_version_env_md), platform=str(s3.platform), gpu=str(s3.gpu_model),
                   started="2026-06-20 (原 ShortMD；启动时刻未保留)", note="r1 = 原 ShortMD：full_top.pdb 的 CRYST1 是溶剂化后的初始盒子；原脚本未保存 initial_system.pdb、种子和平衡段日志；墙钟是上限（5 个蛋白并行或排队）", net_charge_total_e=np.nan, net_charge_protein_e=np.nan)
    else:
        m = json.load(open(P["meta"]))
        row.update(n_atoms_total=m["n_atoms_total"], n_waters=m["n_waters"], n_Na=m["n_Na"], n_Cl=m["n_Cl"], box_initial_a_nm=m["box_initial_nm"][0], box_initial_b_nm=m["box_initial_nm"][1], box_initial_c_nm=m["box_initial_nm"][2],
                   seed_md=m["seed_md"], seed_build=m["seed_build"], wall_clock_min=round(m["wall_s_total"] / 60.0, 2), wall_clock_note="单任务独占 1 张 GPU（其间另一张卡同时跑另一条），含体系构建与 ensemble 切片", openmm_version=m["openmm"], platform=m["platform"],
                   gpu=f'{m["gpu_name"]} (CUDA_VISIBLE_DEVICES={m["cuda_visible_devices"]})', started=m["started"], note="", net_charge_total_e=m["net_charge_total_e"], net_charge_protein_e=m["net_charge_protein_e"])
    v0 = row["box_initial_a_nm"] * row["box_initial_b_nm"] * row["box_initial_c_nm"]
    pairs = min(row["n_Na"], row["n_Cl"])   # 中和蛋白净电荷之外的 NaCl 对数
    row.update(box_initial_volume_nm3=float(v0), n_NaCl_pairs_beyond_neutralization=int(pairs), salt_conc_M_per_box_volume=pairs / (NA_AVO * v0), salt_conc_M_per_water_volume=pairs / (NA_AVO * row["n_waters"] * WATER_NM3),
               ionic_strength_M_from_ions=0.5 * (row["n_Na"] + row["n_Cl"]) / (NA_AVO * v0), waters_per_nm3=row["n_waters"] / v0)
    return row


def crystal_cell(p):
    Rr = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "Crystal", f"{p}.parquet")); nc = Rr.pivot(index="resSeq", columns="recon_k", values="NC"); nh = Rr.pivot(index="resSeq", columns="recon_k", values="NH")
    return dict(res=nc.index.values.astype(int), NC=nc.values.astype(float), NH=nh.values.astype(float), offset=OFF_CRYST[p])


def profile(cell, p):
    resi, lp, y = U.lnpf_profile(cell, p); return pd.Series(lp, index=resi)


def main():
    SYS, INI, PAIR = [], [], []
    Rt = pd.read_csv(os.path.join(a.out_dir, "S13_shortmd_replicate_rho.csv")) if os.path.exists(os.path.join(a.out_dir, "S13_shortmd_replicate_rho.csv")) else None
    for p in PROTS:
        pdb_id = S3.loc[p, "pdb_id"]; prof = {}; mean_xyz = {}; sel_xyz = {}; ca_rmsd = {}
        cc = crystal_cell(p); prof["crystal"] = profile(cc, p)
        ref_pos = None
        for rep in REPS:
            SYS.append(system_row(p, rep)); P = U.md_paths(rep, p); t = md.load(P["ens_xtc"], top=P["ens_pdb"]); ca = U.chain0_ca(t.topology); tca = t.atom_slice(ca)
            pos, dxyz, nmatch, ndep = U.deposited_ca(pdb_id, [at.residue.name for at in tca.topology.atoms]); sub = tca.atom_slice(pos); ref = md.Trajectory(dxyz[None], sub.topology)
            if ref_pos is None:
                ref_pos = pos
            assert np.array_equal(pos, ref_pos), (p, rep, "副本间的沉积结构对应残基不一致")
            sub.superpose(ref, 0); mean_xyz[rep] = sub.xyz.mean(0); sel_xyz[rep] = sub.slice(SEL); ca_rmsd[rep] = md.rmsd(sub, ref, 0) * 10.0
            cell, _ = U.load_rep_cell(rep, p); prof[rep] = profile(cell, p)
        for rep in REPS:
            r = ca_rmsd[rep]; j = prof[rep].index.intersection(prof["crystal"].index)
            INI.append(dict(protein=p, replicate=rep, rho_median=float(Rt[(Rt.row_type == "replicate") & (Rt.protein == p) & (Rt.replicate == rep)].rho_median.iloc[0]) if Rt is not None else np.nan,
                            baseline_rho_deposited=float(Rt[(Rt.row_type == "replicate") & (Rt.protein == p)].baseline_rho_deposited.iloc[0]) if Rt is not None else np.nan,
                            ca_rmsd_to_deposited_mean_A=float(r.mean()), ca_rmsd_to_deposited_min_A=float(r.min()), ca_rmsd_to_deposited_max_A=float(r.max()), ca_rmsd_first_half_mean_A=float(r[:100].mean()), ca_rmsd_second_half_mean_A=float(r[100:].mean()),
                            lnpf_profile_spearman_vs_deposited_recon_mean=float(stats.spearmanr(prof[rep][j], prof["crystal"][j])[0]), n_residues_in_profile=len(j), n_ca_matched=int(len(ref_pos))))
        for r1_, r2_ in itertools.combinations(REPS, 2):
            m1 = md.Trajectory(mean_xyz[r1_][None], sel_xyz[r1_].topology); m2 = md.Trajectory(mean_xyz[r2_][None], sel_xyz[r2_].topology); mm = float(md.rmsd(m1, m2, 0)[0] * 10.0)
            A, B = sel_xyz[r1_], sel_xyz[r2_]; cross = np.array([md.rmsd(B, A, i) for i in range(len(SEL))]) * 10.0   # [i, j] = RMSD(A_i, B_j)
            wa = np.array([md.rmsd(A, A, i) for i in range(len(SEL))]) * 10.0; wb = np.array([md.rmsd(B, B, i) for i in range(len(SEL))]) * 10.0; tri = np.triu_indices(len(SEL), 1)
            j = prof[r1_].index.intersection(prof[r2_].index)
            rr1 = float(Rt[(Rt.row_type == "replicate") & (Rt.protein == p) & (Rt.replicate == r1_)].rho_median.iloc[0]) if Rt is not None else np.nan; rr2 = float(Rt[(Rt.row_type == "replicate") & (Rt.protein == p) & (Rt.replicate == r2_)].rho_median.iloc[0]) if Rt is not None else np.nan
            PAIR.append(dict(protein=p, replicate_pair=f"{r1_}-{r2_}", mean_structure_ca_rmsd_A=mm, cross_replicate_frame_ca_rmsd_mean_A=float(cross.mean()), cross_replicate_frame_ca_rmsd_min_A=float(cross.min()), cross_replicate_frame_ca_rmsd_max_A=float(cross.max()),
                             within_first_replicate_frame_ca_rmsd_mean_A=float(wa[tri].mean()), within_second_replicate_frame_ca_rmsd_mean_A=float(wb[tri].mean()), n_cross_pairs=int(cross.size), n_within_pairs=int(len(tri[0])),
                             lnpf_profile_spearman_between_replicates=float(stats.spearmanr(prof[r1_][j], prof[r2_][j])[0]), n_residues_in_profile=len(j), rho_first=rr1, rho_second=rr2, rho_diff_second_minus_first=rr2 - rr1))
        print(p, "ok", flush=True)
    SYS = pd.DataFrame(SYS); INI = pd.DataFrame(INI); PAIR = pd.DataFrame(PAIR)
    for name, df in (("md_systems", SYS), ("shortmd_initial_structure_data", INI), ("shortmd_replicate_pair_data", PAIR)):
        df.to_csv(os.path.join(a.out_dir, f"S13_{name}.csv"), index=False); print(name, df.shape)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(SYS[["protein", "replicate", "n_atoms_total", "n_waters", "n_Na", "n_Cl", "box_initial_a_nm", "box_prod_mean_a_nm", "mean_T_K", "n_NaCl_pairs_beyond_neutralization", "salt_conc_M_per_water_volume", "ionic_strength_M_from_ions", "wall_clock_min"]].round(3).to_string(index=False))
    log.finish([os.path.join(a.out_dir, f"S13_{n}.csv") for n in ("md_systems", "shortmd_initial_structure_data", "shortmd_replicate_pair_data")])


if __name__ == "__main__":
    main()
