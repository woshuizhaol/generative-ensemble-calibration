"""Stage 9 追加（U-S3-2）· 步骤 4：血红素 c 的 Amber 残基（mol2）与 tleap 体系脚本。
原子类型沿用 Amber 自带的 Giammona/Bayly 血红素参数（`$AMBERHOME/dat/contrib/heme/heme_all.in`、`frcmod.hemall`，原为血红素 b）：
  卟啉大环 NA/NC → NP，NB/ND → NO，C1x/C4x → CC，C2x/C3x → CB，meso CHx → CD，环上甲基 CMx 与丙酸 CAA/CBA/CAD/CBD → CT，羧酸 CGA/CGD → C，O1x/O2x → O2，氢 → HC。
  血红素 c 与 b 的差别：C3B、C3C 上不是乙烯基（CY/CX）而是 CH(S–)CH3，故 CAB/CAC → CT，CBB/CBC → CT，HAB/HAC → H1（ff14SB 中接 S 的 sp3 碳上的氢），HBB1–3、HBC1–3 → HC。
  铁是独立残基 FE（类型 FE）；Cys14/Cys17 为 CYX（SG 与 CAB/CAC 成键），His18 为 HID（NE2 与 Fe 成键），Met80 的 SD 与 Fe 成键。
电荷：`--charges` 给出 RESP 结果（tsv：residue、atom、charge）时写入，否则写 0（仅用于检查参数是否齐全）。
输出（`md_cytc_heme/param/`）：HEC.mol2、FE.mol2、bonds_report.json。
用法：env/rev/bin/python s9d_cytc_heme_residues.py [--charges param/resp_charges.tsv]"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
D = os.path.join(REV, "md_cytc_heme"); OUT = os.path.join(D, "param"); os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--charges", default=""); a = ap.parse_args()
log = revio.RunLogger(stage=9, task="cytc_heme_residues", cpu_procs=1, seeds="none")

TYPE = {}
for n in ("NA", "NC"): TYPE[n] = "NP"
for n in ("NB", "ND"): TYPE[n] = "NO"
for r in "ABCD":
    TYPE[f"C1{r}"] = "CC"; TYPE[f"C4{r}"] = "CC"; TYPE[f"C2{r}"] = "CB"; TYPE[f"C3{r}"] = "CB"; TYPE[f"CH{r}"] = "CD"; TYPE[f"CM{r}"] = "CT"
for n in ("CAA", "CBA", "CAD", "CBD", "CAB", "CBB", "CAC", "CBC"): TYPE[n] = "CT"
for n in ("CGA", "CGD"): TYPE[n] = "C"
for n in ("O1A", "O2A", "O1D", "O2D"): TYPE[n] = "O2"

ELEM = lambda n: n[0] if n[0] in "CNOH" else n[:1]
atoms = []
for l in open(os.path.join(D, "prep", "cytc_heme_start.pdb")):
    if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() in ("HEC", "FE"):
        atoms.append((l[17:20].strip(), l[12:16].strip(), np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
hec = [(n, x) for r, n, x in atoms if r == "HEC"]; fe = [(n, x) for r, n, x in atoms if r == "FE"]
assert len(hec) == 74 and len(fe) == 1, (len(hec), len(fe))
for n, _ in hec:
    if n not in TYPE:
        TYPE[n] = "H1" if n in ("HAB", "HAC") else ("HC" if n.startswith("H") else None)
assert all(TYPE[n] for n, _ in hec), [n for n, _ in hec if not TYPE.get(n)]
q = {}
if a.charges:
    for l in open(a.charges):
        if l.startswith("#") or not l.strip() or l.startswith("residue"):
            continue
        r, n, c = l.split()[:3]; q[(r, n)] = float(c)
    # 六位小数的舍入使 HEC + FE 的总电荷偏离 −1 约 1e-5；把残差加到 Fe 上（Fe 本身的电荷不确定度远大于此），使体系总电荷严格为整数
    resid = -1.0 - sum(q.values()); q[("FE", "FE")] += resid

xyz = np.array([x for _, x in hec]); names = [n for n, _ in hec]
bonds = []
for i in range(len(hec)):
    for j in range(i + 1, len(hec)):
        d = float(np.linalg.norm(xyz[i] - xyz[j])); hi = names[i][0] == "H" or names[j][0] == "H"
        if (hi and d < 1.25 and not (names[i][0] == "H" and names[j][0] == "H")) or ((not hi) and d < 1.75):
            bonds.append((i, j, d))
deg = {n: 0 for n in names}
for i, j, d in bonds:
    deg[names[i]] += 1; deg[names[j]] += 1
rep = dict(n_atoms=len(hec), n_bonds=len(bonds), degrees={n: deg[n] for n in names if names[0] and (deg[n] > 4 or deg[n] == 0)}, bond_distance_range=[round(min(b[2] for b in bonds), 3), round(max(b[2] for b in bonds), 3)])


def mol2(path, resname, mem, types, bonds, charges):
    with open(path, "w") as fh:
        fh.write(f"@<TRIPOS>MOLECULE\n{resname}\n{len(mem)} {len(bonds)} 1 0 0\nSMALL\nUSER_CHARGES\n\n@<TRIPOS>ATOM\n")
        for k, (n, x) in enumerate(mem, 1):
            fh.write(f"{k:>6} {n:<5} {x[0]:>10.4f} {x[1]:>10.4f} {x[2]:>10.4f} {types[n]:<4} 1 {resname} {charges.get((resname, n), 0.0):>10.6f}\n")
        fh.write("@<TRIPOS>BOND\n")
        for k, (i, j, d) in enumerate(bonds, 1):
            fh.write(f"{k:>6} {i + 1:>5} {j + 1:>5} 1\n")
        fh.write("@<TRIPOS>SUBSTRUCTURE\n     1 " + f"{resname}      1 TEMP              0 ****  ****    0 ROOT\n")


mol2(os.path.join(OUT, "HEC.mol2"), "HEC", hec, TYPE, bonds, q)
mol2(os.path.join(OUT, "FE.mol2"), "FE", fe, {"FE": "FE"}, [], q)
rep["charges_from"] = a.charges or "none (all zero; parameter-completeness check only)"
rep["net_charge_hec"] = round(sum(q.get(("HEC", n), 0.0) for n, _ in hec), 6)
json.dump(rep, open(os.path.join(OUT, "bonds_report.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(rep, ensure_ascii=False))
log.finish([os.path.join(OUT, "HEC.mol2"), os.path.join(OUT, "FE.mol2"), os.path.join(OUT, "bonds_report.json")])
