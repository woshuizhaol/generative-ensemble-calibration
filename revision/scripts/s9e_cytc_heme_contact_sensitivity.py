"""Stage 9 追加（U-S3-2）· 敏感性：血红素对 N_C 的直接贡献。
所有生成器的打分都只用蛋白原子（血红素与铁在打分前去掉），所以主分析只检验血红素对构象采样的间接作用。本脚本补一个敏感性：把血红素重原子也计入 N_C
（每个残基的酰胺氮 6.5 Å 以内的血红素 / 铁重原子数，加到该残基的 N_C 上；血红素不在序列里，不受 |i−j| 排除），N_H 不变，其余同主分析（冻结抽样、计数平均、M=50）。
思路来自总自检复核员对本项工作的独立敏感性检验（其脚本 tmp/verify_heme/v6_hemecontact.py），这里按本项目的数据路径重写并脚本化。
输出：tables/S9e_cytc_heme_contact_sensitivity.csv（每条带血红素轨迹：不含 / 含血红素原子的 ρ、含血红素接触的残基数、实验残基里接触数的均值）。
用法：python s9e_cytc_heme_contact_sensitivity.py --heme cytc_heme_s201,cytc_heme_s202,cytc_heme_s203"""
import os, sys, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd, mdtraj as md
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, bv  # noqa: E402

PROJ, REV, INP = revio.load_paths(); TAB = os.path.join(REV, "tables")
ap = argparse.ArgumentParser(); ap.add_argument("--heme", required=True); a = ap.parse_args()
log = revio.RunLogger(stage=9, task="cytc_heme_contact_sensitivity", cpu_procs=1, seeds="frozen draws F_ShortMD_cytochrome_c_M50")
BC, BH = bv.BETA_C, bv.BETA_H
s1 = pd.read_parquet(os.path.join(TAB, "S1_tier1a_residues_final.parquet")); e = s1[(s1.protein_id == "cytochrome_c") & s1.dGopen_kcal.notna()]; resi = e.resi.astype(int).values; y = e.dGopen_kcal.astype(float).values
frames = np.load(os.path.join(REV, "raw_outputs", "s4", "draw_indices", "F_ShortMD_cytochrome_c_M50.npz"))["frames"].astype(int)


def rho_med(NC, NH, rs):
    pos = [list(rs).index(r) for r in resi]; out = []
    for dr in frames:
        pred = BC * NC[pos][:, dr].mean(1) + BH * NH[pos][:, dr].mean(1); out.append(stats.spearmanr(pred, y).statistic)
    return float(np.median(out))


rows = []
for name in a.heme.split(","):
    d = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", "ShortMD_" + name, "cytochrome_c.parquet"))
    NC = d.pivot(index="resSeq", columns="frame_idx", values="NC"); NH = d.pivot(index="resSeq", columns="frame_idx", values="NH"); rs = NC.index.values
    t = md.load(os.path.join(REV, "md_cytc_heme", "md", name, "full.xtc"), top=os.path.join(REV, "md_cytc_heme", "md", name, "final.pdb")); top = t.topology
    heme_heavy = [at.index for at in top.atoms if at.residue.name in ("HEC", "FE") and at.element.symbol != "H"]
    nidx = []
    for r in rs:
        ns = [at.index for at in top.atoms if at.residue.resSeq == r and at.name == "N" and at.residue.name not in ("HEC", "FE", "HOH") and at.residue.index < 104]; assert len(ns) == 1, (r, ns); nidx.append(ns[0])
    pairs = np.array([[n, h] for n in nidx for h in heme_heavy])
    dist = md.compute_distances(t, pairs, periodic=True).reshape(t.n_frames, len(nidx), len(heme_heavy)); extra = (dist < bv.GEOM["cut_Nc"]).sum(2).T
    assert extra.shape == NC.shape
    me = extra.mean(1); ep = [list(rs).index(r) for r in resi]
    rows.append(dict(run=name, rho_without_heme_atoms=rho_med(NC.values, NH.values, rs), rho_with_heme_heavy_atoms_in_NC=rho_med(NC.values + extra, NH.values, rs), n_heme_heavy_atoms=len(heme_heavy), n_residues_with_heme_contacts=int((me > 0).sum()),
                     n_experimental_residues_with_heme_contacts=int((me[ep] > 0).sum()), mean_extra_NC_over_experimental_residues=float(me[ep].mean()), max_mean_extra_NC=float(me.max())))
R_ = pd.DataFrame(rows); R_["delta"] = R_.rho_with_heme_heavy_atoms_in_NC - R_.rho_without_heme_atoms
apo = pd.read_csv(os.path.join(TAB, "S9e_cytc_heme_rho.csv")); apo = apo[apo.group == "apo"].rho_median.mean()
R_["no_heme_control_mean_rho"] = apo; R_["with_heme_atoms_minus_control_mean"] = R_.rho_with_heme_heavy_atoms_in_NC - apo
R_.to_csv(os.path.join(TAB, "S9e_cytc_heme_contact_sensitivity.csv"), index=False); pd.set_option("display.width", 250); print(R_.round(4).to_string(index=False))
print("mean over heme runs: without %.4f, with %.4f (shift %+.4f); control %.4f" % (R_.rho_without_heme_atoms.mean(), R_.rho_with_heme_heavy_atoms_in_NC.mean(), R_.delta.mean(), apo))
log.finish([os.path.join(TAB, "S9e_cytc_heme_contact_sensitivity.csv")])
