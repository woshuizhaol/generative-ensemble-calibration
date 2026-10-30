"""Stage 9 追加（U-S3-2，用户 2026-10-28 裁决：只重跑 cyt c 一个体系的 MD，带血红素 c，作为 R2-C7a 的敏感性对照）· 步骤 1：起始结构。
目标：与原 ShortMD（`P/code/analysis/12_short_md.py`）的起始结构只差血红素。
  - 蛋白：1HRC 链 A，去掉结晶水和 N 端乙酰基 ACE（与原协议一致：removeHeterogens(False) 删掉了 HOH、ACE、HEC），PDBFixer 同样的步骤补缺失原子、加氢 pH 7.0。
    原 MD 里三个 His 都是 delta 质子化（ND1–H，没有 HE2），这正是 His18 的 NE2 与 Fe 配位所需的互变异构体，所以不改。
  - 血红素 c（HEC 105）：坐标取晶体结构，氢取自 AmberTools `reduce`（wwPDB 化学成分字典里的 HEC，pH 7 下两个丙酸根为羧酸根，不带 H2A/H2D），
    CAB、CAC 上的氢需要硫的位置，由本脚本按几何补上（C–H 1.09 Å，沿三个邻居单位向量之和的反方向）。
  - 铁：从 HEC 里拆成独立残基 FE（MCPB / tleap 的惯例）；Cys14、Cys17 命名为 CYX 并去掉 HG（SG 与 CAB/CAC 成键）；His18 命名 HID。
  - 对照：同一路径的无血红素体系（蛋白 PDBFixer 输出，Cys 带 HG，与原 MD 相同），用于区分血红素效应与单条轨迹的随机波动。
  - 两个体系都改成 Amber 的命名习惯：N 端 Gly1 的氢 H、H2、H3 → H1、H2、H3；三个 His 都是 delta 质子化（HD1，没有 HE2，与原 MD 相同）→ HID。
输出（`md_cytc_heme/prep/`）：cytc_apo_start.pdb、cytc_heme_start.pdb（残基名为 Amber 命名）、prep_report.json。
用法：env/md/bin/python s9d_cytc_heme_prep.py"""
import os, sys, json
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
from openmm.app import PDBFile, Modeller
from pdbfixer import PDBFixer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
D = os.path.join(REV, "md_cytc_heme", "prep"); os.makedirs(D, exist_ok=True)
log = revio.RunLogger(stage=9, task="cytc_heme_prep", cpu_procs=1, seeds="none")


def read_pdb(path):
    out = []
    for l in open(path):
        if l.startswith(("ATOM", "HETATM")):
            out.append(dict(rec=l[:6].strip(), name=l[12:16].strip(), res=l[17:20].strip(), chain=l[21], resi=int(l[22:26]), xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]), el=(l[76:78].strip() or l[12:16].strip()[0])))
    return out


def pdb_line(serial, name, res, chain, resi, xyz, el, rec="ATOM"):
    nm = f" {name:<3}" if len(name) < 4 else name
    return f"{rec:<6}{serial:>5} {nm:<4} {res:>3} {chain}{resi:>4}    {xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}  1.00  0.00          {el:>2}\n"


# ---- 1. 蛋白：与原协议相同的 PDBFixer 步骤（HEC、ACE、HOH 先去掉）
src = os.path.join(PROJ, "data", "external", "pdb", "1HRC.pdb")
prot_lines = [l for l in open(src) if l.startswith("ATOM")]
tmp = os.path.join(D, "1HRC_protein_only.pdb"); open(tmp, "w").writelines(prot_lines + ["END\n"])
fx = PDBFixer(filename=tmp); fx.findMissingResidues(); fx.missingResidues = {}; fx.findNonstandardResidues(); fx.replaceNonstandardResidues(); fx.removeHeterogens(False)
fx.findMissingAtoms(); n_missing = sum(len(v) for v in fx.missingAtoms.values()); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
apo_raw = os.path.join(D, "cytc_apo_pdbfixer.pdb"); PDBFile.writeFile(fx.topology, fx.positions, open(apo_raw, "w"), keepIds=True)
atoms = read_pdb(apo_raw)
his_atoms = {}
for a_ in atoms:
    if a_["res"] == "HIS":
        his_atoms.setdefault(a_["resi"], set()).add(a_["name"])
assert sorted(his_atoms) == [18, 26, 33] and all("HD1" in v and "HE2" not in v for v in his_atoms.values()), his_atoms   # 三个 His 都是 HD1 型（与原 MD 的 full_top.pdb 一致）
for a_ in atoms:
    if a_["res"] == "HIS":
        a_["res"] = "HID"
    if a_["resi"] == 1 and a_["name"] == "H":
        a_["name"] = "H1"
apo = os.path.join(D, "cytc_apo_start.pdb"); apo_lines = []
for k, a_ in enumerate(atoms, 1):
    apo_lines.append(pdb_line(k, a_["name"], a_["res"], "A", a_["resi"], a_["xyz"], a_["el"]))
open(apo, "w").writelines(apo_lines + ["TER\nEND\n"])

# ---- 2. 血红素 c：reduce 加氢后的 HEC 原子（晶体坐标）
red = [a for a in read_pdb(os.path.join(D, "1HRC_reduce.pdb")) if a["res"] == "HEC"]
hec = {a["name"]: a for a in red}
assert "FE" in hec and len(red) == 73, (len(red), "reduce 应给出 43 个重原子 + 30 个氢")
sg = {r: next(a for a in atoms if a["resi"] == r and a["name"] == "SG") for r in (14, 17)}


def methine_h(c, nbrs):
    v = sum((n - c) / np.linalg.norm(n - c) for n in nbrs)
    return c - 1.09 * v / np.linalg.norm(v)


hab = methine_h(hec["CAB"]["xyz"], [hec["C3B"]["xyz"], hec["CBB"]["xyz"], sg[14]["xyz"]]); hac = methine_h(hec["CAC"]["xyz"], [hec["C3C"]["xyz"], hec["CBC"]["xyz"], sg[17]["xyz"]])

# ---- 3. 组装（Amber 命名）：HID18、CYX14/17（去 HG）、MET80 不变
lines = []; serial = 1; rep = dict(his18_atoms=[], cys_hg_removed=[], n_protein_atoms=0)
for a in atoms:
    res = a["res"]
    if a["resi"] == 18 and res == "HID":
        rep["his18_atoms"].append(a["name"])
    if a["resi"] in (14, 17) and res == "CYS":
        res = "CYX"
        if a["name"] == "HG":
            rep["cys_hg_removed"].append(a["resi"]); continue
    lines.append(pdb_line(serial, a["name"], res, "A", a["resi"], a["xyz"], a["el"])); serial += 1
rep["n_protein_atoms"] = serial - 1
assert "HD1" in rep["his18_atoms"] and "HE2" not in rep["his18_atoms"], rep["his18_atoms"]
lines.append("TER\n")
lines.append(pdb_line(serial, "FE", "FE", "A", 105, hec["FE"]["xyz"], "FE", "HETATM")); serial += 1
lines.append("TER\n")
hec_names = [n for n in hec if n != "FE"]
for n in hec_names:
    lines.append(pdb_line(serial, n, "HEC", "A", 106, hec[n]["xyz"], hec[n]["el"], "HETATM")); serial += 1
lines.append(pdb_line(serial, "HAB", "HEC", "A", 106, hab, "H", "HETATM")); serial += 1; lines.append(pdb_line(serial, "HAC", "HEC", "A", 106, hac, "H", "HETATM")); serial += 1
lines.append("TER\nEND\n")
out = os.path.join(D, "cytc_heme_start.pdb"); open(out, "w").writelines(lines)
rep.update(n_hec_atoms=len(hec_names) + 2, n_heme_hydrogens=sum(1 for n in hec_names if hec[n]["el"] == "H") + 2, n_missing_heavy_atoms_added_by_pdbfixer=n_missing, apo_pdb=apo, heme_pdb=out)
json.dump(rep, open(os.path.join(D, "prep_report.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(rep, ensure_ascii=False)); log.finish([apo, out, os.path.join(D, "prep_report.json")])
