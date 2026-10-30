"""Stage 9 追加（U-S3-2）· 步骤 2：Gaussian 量子化学模型（血红素 c 的 Fe 位点力常数与 RESP 电荷）。
起始坐标全部取自 `md_cytc_heme/prep/cytc_heme_start.pdb`（1HRC 晶体结构 + 氢）。
小模型（`small.com`）：Fe(III) 卟吩（8 个 β 取代基换成 H）+ 4-甲基咪唑（His18，NE2 配位）+ 二甲硫醚（Met80，SD 配位）；电荷 +1、自旋二重态（低自旋 d5），
    UB3LYP/6-31G* 优化 + 频率，用于 Seminario 方法求 Fe–N、Fe–S 键与 N–Fe–N/L 角的力常数（MCPB 的标准做法，6-31G*）。
大模型（`large_esp.com`）：Fe + 完整血红素 c（含两个去质子化丙酸根）+ 两个 CH3–S 硫醚帽（Cys14、Cys17 的 CB 以甲基代替）+ 4-甲基咪唑 + 二甲硫醚；电荷 −1、二重态，
    晶体重原子坐标上做 UB3LYP/6-31G* 的 Merz–Kollman 静电势（IOp(6/33=2,6/42=6)，与 Amber antechamber 的 RESP 流程相同：每 Å² 6 个点，势值写进 log，由 espgen 读取），供 RESP 拟合血红素与 Fe 的电荷。
  （曾用 iop(6/50=1) 在甲醇上试验，本机 g16 Rev A.03 的 l602 会崩溃，所以不用它。）
截断处的甲基氢按四面体几何补上（C–H 1.09 Å，H–C–X 109.5°）。
输出：qm/small.com、qm/large_esp.com、qm/small_atoms.tsv、qm/large_atoms.tsv（原子序号 → PDB 原子映射）。
用法：env/rev/bin/python s9d_cytc_heme_qm_models.py [--nproc 16] [--mem 48GB]"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
D = os.path.join(REV, "md_cytc_heme"); QM = os.path.join(D, "qm"); os.makedirs(QM, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=16); ap.add_argument("--mem", default="48GB"); ap.add_argument("--which", default="both", choices=["both", "small", "large"]); a = ap.parse_args()
log = revio.RunLogger(stage=9, task="cytc_heme_qm_models", cpu_procs=1, seeds="none")

atoms = []   # (resname, resi, name, element, xyz)
for l in open(os.path.join(D, "prep", "cytc_heme_start.pdb")):
    if l.startswith(("ATOM", "HETATM")):
        atoms.append((l[17:20].strip(), int(l[22:26]), l[12:16].strip(), (l[76:78].strip() or l[12:16].strip()[0]).capitalize(), np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
X = {(r, i, n): (el, xyz) for r, i, n, el, xyz in atoms}
HEC = 106; FE = ("FE", 105, "FE")


def hec(n):
    return X[("HEC", HEC, n)]


def unit(v):
    return v / np.linalg.norm(v)


def methyl_hs(c, x, existing=()):
    """补全以 c 为中心、与重原子 x 相连的甲基氢（共 3 个，existing 为已有氢的坐标），返回新增的氢坐标。"""
    u = unit(x - c); t = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0]); p = unit(np.cross(u, t)); q = np.cross(u, p)
    th = np.deg2rad(109.5); phi0 = 0.0
    if existing:
        v = unit(existing[0] - c); phi0 = np.arctan2(np.dot(v, q), np.dot(v, p))
    hs = [c + 1.09 * (np.cos(th) * u + np.sin(th) * (np.cos(phi0 + k * 2 * np.pi / 3) * p + np.sin(phi0 + k * 2 * np.pi / 3) * q)) for k in range(3)]
    keep = []
    for h in hs:
        if all(np.linalg.norm(h - e) > 0.5 for e in existing):
            keep.append(h)
    return keep[:3 - len(existing)]


def replace_with_h(beta, sub):
    return hec(beta)[1] + 1.08 * unit(hec(sub)[1] - hec(beta)[1])


def build(model):
    """返回 (原子列表[(label, element, xyz)], 总电荷, 自旋多重度)。"""
    out = []
    add = lambda lab, el, xyz: out.append((lab, el, np.array(xyz, float)))
    add("FE", "Fe", hec("FE")[1] if ("HEC", HEC, "FE") in X else X[FE][1])
    ring = ["NA", "NB", "NC", "ND", "C1A", "C2A", "C3A", "C4A", "C1B", "C2B", "C3B", "C4B", "C1C", "C2C", "C3C", "C4C", "C1D", "C2D", "C3D", "C4D", "CHA", "CHB", "CHC", "CHD", "HHA", "HHB", "HHC", "HHD"]
    for n in ring:
        add("HEC." + n, *hec(n))
    subs = {"C2A": "CAA", "C3A": "CMA", "C2B": "CMB", "C3B": "CAB", "C2C": "CMC", "C3C": "CAC", "C2D": "CMD", "C3D": "CAD"}
    if model == "small":
        for beta, sub in subs.items():
            add(f"HEC.H({sub})", "H", replace_with_h(beta, sub))
    else:   # 大模型：完整取代基
        for beta, sub in subs.items():
            add("HEC." + sub, *hec(sub))
        for n in ["HMA1", "HMA2", "HMA3", "HMB1", "HMB2", "HMB3", "HMC1", "HMC2", "HMC3", "HMD1", "HMD2", "HMD3", "HAB", "HAC", "CBB", "CBC", "HBB1", "HBB2", "HBB3", "HBC1", "HBC2", "HBC3",
                  "HAA1", "HAA2", "CBA", "CGA", "O1A", "O2A", "HBA1", "HBA2", "HAD1", "HAD2", "CBD", "CGD", "O1D", "O2D", "HBD1", "HBD2"]:
            if ("HEC", HEC, n) not in X:
                continue
            add("HEC." + n, *hec(n))
        for r, cap in ((14, "CYS14"), (17, "CYS17")):
            sg = X[("CYX", r, "SG")]; cb = X[("CYX", r, "CB")]
            add(f"{cap}.SG", *sg); add(f"{cap}.CB", *cb)
            for h in methyl_hs(cb[1], sg[1]):
                add(f"{cap}.HB(cap)", "H", h)
    # His18 → 4-甲基咪唑（CB 作为甲基）
    for n in ["CG", "ND1", "HD1", "CD2", "HD2", "CE1", "HE1", "NE2"]:
        add("HID18." + n, *X[("HID", 18, n)])
    cb = X[("HID", 18, "CB")]; add("HID18.CB", *cb)
    ex = [X[("HID", 18, "HB2")][1], X[("HID", 18, "HB3")][1]]
    for lab, h in (("HID18.HB2", ex[0]), ("HID18.HB3", ex[1])):
        add(lab, "H", h)
    for h in methyl_hs(cb[1], X[("HID", 18, "CG")][1], ex):
        add("HID18.HB(cap)", "H", h)
    # Met80 → 二甲硫醚（CG 作为甲基）
    sd = X[("MET", 80, "SD")]; add("MET80.SD", *sd)
    cg = X[("MET", 80, "CG")]; add("MET80.CG", *cg)
    for lab, h in (("MET80.HG2", X[("MET", 80, "HG2")][1]), ("MET80.HG3", X[("MET", 80, "HG3")][1])):
        add(lab, "H", h)
    for h in methyl_hs(cg[1], sd[1], [X[("MET", 80, "HG2")][1], X[("MET", 80, "HG3")][1]]):
        add("MET80.HG(cap)", "H", h)
    ce = X[("MET", 80, "CE")]; add("MET80.CE", *ce)
    for n in ("HE1", "HE2", "HE3"):
        add("MET80." + n, "H", X[("MET", 80, n)][1])
    charge = 1 if model == "small" else -1
    return out, charge, 2


def write(model, name, route, extra=""):
    at, q, mult = build(model)
    with open(os.path.join(QM, name + ".com"), "w") as fh:
        fh.write(f"%nprocshared={a.nproc}\n%mem={a.mem}\n%chk={name}.chk\n{route}\n\n{name}: cytochrome c heme c model ({model})\n\n{q} {mult}\n")
        for lab, el, xyz in at:
            fh.write(f" {el:<2}  {xyz[0]:>12.6f} {xyz[1]:>12.6f} {xyz[2]:>12.6f}\n")
        fh.write("\n" + extra)
    with open(os.path.join(QM, name + "_atoms.tsv"), "w") as fh:
        fh.write("index\tlabel\telement\n")
        for i, (lab, el, xyz) in enumerate(at, 1):
            fh.write(f"{i}\t{lab}\t{el}\n")
    return len(at), q, mult


n1 = write("small", "small", "#p UB3LYP/6-31G* opt=(maxcycles=300) freq scf=(xqc,maxcycle=512) nosymm") if a.which in ("both", "small") else (0, 0, 0)
n2 = write("large", "large_esp", "#p UB3LYP/6-31G* pop=MK iop(6/33=2,6/42=6) scf=(xqc,maxcycle=512) nosymm", extra="") if a.which in ("both", "large") else (0, 0, 0)
print("small: atoms=%d charge=%d mult=%d | large_esp: atoms=%d charge=%d mult=%d" % (n1 + n2))
log.finish([os.path.join(QM, f) for f in ("small.com", "large_esp.com", "small_atoms.tsv", "large_esp_atoms.tsv")])
