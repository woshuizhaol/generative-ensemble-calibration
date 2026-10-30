"""Stage 9 追加（U-S3-2）· 步骤 3b：血红素 c 与铁的 RESP 电荷（由大模型的 MK 静电势拟合）。
输入：qm/large_esp.log（g16，UB3LYP/6-31G*，pop=MK iop(6/33=2,6/42=6)）→ espgen 转成 .esp；qm/large_esp_atoms.tsv（原子序号 → 标签）。
拟合（`lib/resp_fit.py`，两阶段 RESP，已在甲醇上与 antechamber 的 RESP 逐位核对）：
  - 约束：HEC + FE 总电荷 = −1（卟啉二价阴离子 −2、两个丙酸根 −2、Fe(III) +3）；四个配体片段（Cys14 的 SG-CH3 帽、Cys17 的 SG-CH3 帽、4-甲基咪唑、二甲硫醚）各自电中性；
    两个丙酸根的两个氧各自等价（O1A=O2A、O1D=O2D）；
  - 阶段 1：所有重原子（Fe 除外）受弱双曲约束 0.0005；阶段 2：甲基 / 亚甲基的氢拉成等价组后重拟合（约束 0.001），其余冻结在阶段 1 的值；
  - Fe 不受约束（金属中心的电荷由静电势决定）。
  - 对称性约束（默认开启，`--no-symmetry` 关闭）：不加约束的 RESP 在大共轭环上过拟合（四个吡咯氮的电荷在 −0.54 到 +0.03 之间，没有物理依据：Fe 卟啉核心近似 D4h 对称），
    所以在阶段 1 把按化学对称性相同的原子取为等电荷——四个吡咯氮；四个 meso 碳及其氢；八个 α 碳；β 碳按取代基分三类（带甲基 C3A、C2B、C2C、C2D；带丙酸 C2A、C3D；带硫醚 C3B、C3C）；
    四个丙酸氧；两个羧基碳；两个硫醚 CH 的氢（HAB、HAC；碳 CAB、CAC 不并组）；阶段 2 再把各类甲基 / 亚甲基的氢拉成等价并与其碳一起重拟合。
    两种拟合（有 / 无对称约束）的静电势残差都写入报告，输出 resp_charges.tsv 用对称约束的结果，resp_charges_unsymmetrized.tsv 保存无约束的结果。
输出（`param/`）：resp_charges.tsv（residue、atom、charge，只含 HEC 与 FE 的原子）、resp_all_atoms.tsv（含配体片段，仅供检查）、resp_report.json（拟合残差、电荷和、迭代次数）。
用法：python s9d_cytc_resp.py [--esp qm/large_esp.esp]"""
import os, sys, json, argparse, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, resp_fit as RF  # noqa: E402

PROJ, REV, INP = revio.load_paths(); D = os.path.join(REV, "md_cytc_heme"); QM = os.path.join(D, "qm"); OUT = os.path.join(D, "param"); os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--esp", default=os.path.join(QM, "large_esp.esp")); ap.add_argument("--no-symmetry", action="store_true"); a = ap.parse_args()
log = revio.RunLogger(stage=9, task="cytc_resp", cpu_procs=1, seeds="none")
if not os.path.exists(a.esp) or os.path.getsize(a.esp) == 0:
    env = dict(os.environ, AMBERHOME="/opt/amber/amber22", PATH="/opt/amber/amber22/bin:" + os.environ["PATH"])
    subprocess.run(["espgen", "-i", os.path.join(QM, "large_esp.log"), "-o", a.esp], check=True, env=env, capture_output=True)
num = lambda x: float(x.replace("D", "E"))
L = open(a.esp).read().split("\n"); nat, npt = int(L[0][:5]), int(L[0][5:10])   # espgen 头行是 (2I5) 固定宽度，原子数与点数可能连在一起（如 “10621393”）
assert len([l for l in L[1:] if l.strip()]) == nat + npt, (len([l for l in L[1:] if l.strip()]), nat, npt)
coords = np.array([[num(v) for v in L[1 + i].split()] for i in range(nat)]); pts = np.array([[num(v) for v in L[1 + nat + k].split()] for k in range(npt)]); esp = pts[:, 0]; xyz = pts[:, 1:4]
lab = [l.split("\t") for l in open(os.path.join(QM, "large_esp_atoms.tsv")).read().strip().split("\n")[1:]]; assert len(lab) == nat, (len(lab), nat)
labels = [x[1] for x in lab]; elem = [x[2] for x in lab]
idx = {l: i for i, l in enumerate(labels)}
heme = [i for i, l in enumerate(labels) if l.startswith("HEC.") or l == "FE"]
frag = {"CYS14": [i for i, l in enumerate(labels) if l.startswith("CYS14.")], "CYS17": [i for i, l in enumerate(labels) if l.startswith("CYS17.")],
        "HID18": [i for i, l in enumerate(labels) if l.startswith("HID18.")], "MET80": [i for i, l in enumerate(labels) if l.startswith("MET80.")]}
assert len(heme) + sum(len(v) for v in frag.values()) == nat
group_sums = [(heme, -1.0)] + [(v, 0.0) for v in frag.values()]
heavy = np.array([e != "H" for e in elem]); free = np.ones(nat, bool)
is_fe = np.array([l == "FE" for l in labels]); heavy_restr = heavy & ~is_fe
H_of = lambda pref, ks: [idx[f"HEC.{pref}{k}"] for k in ks if f"HEC.{pref}{k}" in idx]
C_ = lambda names: [idx["HEC." + n] for n in names]
# 阶段 2（标准 RESP）：甲基 / 亚甲基的氢相等，且与其碳一起重新拟合
equiv2 = [H_of(pr, (1, 2, 3)) for pr in ("HMA", "HMB", "HMC", "HMD")]
equiv2 = [sum(equiv2, [])] + [C_(["CMA", "CMB", "CMC", "CMD"])] + [sum([H_of(pr, (1, 2, 3)) for pr in ("HBB", "HBC")], [])] + [C_(["CBB", "CBC"])]
equiv2 += [sum([H_of(pr, (1, 2)) for pr in ("HAA", "HAD")], [])] + [C_(["CAA", "CAD"])] + [sum([H_of(pr, (1, 2)) for pr in ("HBA", "HBD")], [])] + [C_(["CBA", "CBD"])]
for fr_ in ("CYS14", "CYS17", "HID18", "MET80"):
    g = [i for i in frag[fr_] if labels[i].endswith("(cap)")]
    if len(g) > 1:
        equiv2.append(g)
equiv2 = [g for g in equiv2 if len(g) > 1]
sym1 = [C_(["O1A", "O2A", "O1D", "O2D"]), C_(["NA", "NB", "NC", "ND"]), C_(["CHA", "CHB", "CHC", "CHD"]), C_(["HHA", "HHB", "HHC", "HHD"]), C_(["C1A", "C4A", "C1B", "C4B", "C1C", "C4C", "C1D", "C4D"]),
        C_(["C3A", "C2B", "C2C", "C2D"]), C_(["C2A", "C3D"]), C_(["C3B", "C3C"]), C_(["CGA", "CGD"]), C_(["HAB", "HAC"])]
asym1 = [C_(["O1A", "O2A"]), C_(["O1D", "O2D"])]   # 无对称约束时，羧基的两个氧仍然等价
def go(equiv1):
    return RF.two_stage(coords, xyz, esp, heavy_restr, free, {}, equiv2, group_sums, equiv_stage1=equiv1)
q1u, q2u, infou = go(asym1); q1s, q2s, infos = go(sym1 + [g for g in equiv2 if len(g) > 1 and labels[g[0]].startswith("HEC.C") and False])
q1, q = (q1u, q2u) if a.no_symmetry else (q1s, q2s); info = infou if a.no_symmetry else infos
rep = dict(natoms=nat, npts=npt, symmetry_constraints=not a.no_symmetry, charge_sum_heme_plus_fe=float(q[heme].sum()), fragment_sums={k: float(q[v].sum()) for k, v in frag.items()}, rms_esp_residual_used=info, rms_esp_residual_unsymmetrized=infou, rms_esp_residual_symmetrized=infos,
           fe_charge=float(q[idx["FE"]]), fe_charge_unsymmetrized=float(q2u[idx["FE"]]), n_equivalence_groups_stage1_sym=len(sym1), n_equivalence_groups_stage2=len(equiv2), esp_points_per_atom=round(npt / nat, 1),
           pyrrole_N_charges_symmetrized=[round(float(q[idx["HEC." + n]]), 4) for n in ("NA", "NB", "NC", "ND")], pyrrole_N_charges_unsymmetrized=[round(float(q2u[idx["HEC." + n]]), 4) for n in ("NA", "NB", "NC", "ND")])
with open(os.path.join(OUT, "resp_all_atoms.tsv"), "w") as fh:
    fh.write("index\tlabel\telement\tq_stage1\tq_final\n")
    for i in range(nat):
        fh.write(f"{i + 1}\t{labels[i]}\t{elem[i]}\t{q1[i]:.6f}\t{q[i]:.6f}\n")
with open(os.path.join(OUT, "resp_charges_unsymmetrized.tsv"), "w") as fh:
    fh.write("residue\tatom\tcharge\n")
    for i in heme:
        l = labels[i]; res, at = ("FE", "FE") if l == "FE" else ("HEC", l.split(".", 1)[1]); fh.write(f"{res}\t{at}\t{q2u[i]:.6f}\n")
with open(os.path.join(OUT, "resp_charges.tsv"), "w") as fh:
    fh.write("residue\tatom\tcharge\n")
    for i in heme:
        l = labels[i]; res, at = ("FE", "FE") if l == "FE" else ("HEC", l.split(".", 1)[1])
        fh.write(f"{res}\t{at}\t{q[i]:.6f}\n")
json.dump(rep, open(os.path.join(OUT, "resp_report.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(rep, ensure_ascii=False))
log.finish([os.path.join(OUT, "resp_charges.tsv"), os.path.join(OUT, "resp_all_atoms.tsv"), os.path.join(OUT, "resp_report.json")])
