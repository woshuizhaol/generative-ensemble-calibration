"""Stage 9 追加（U-S3-2）· 体系构建路径的等价性检验（QC）。
问题：原 ShortMD 用 OpenMM ForceField('amber14-all.xml','amber14/tip3pfb.xml') 构建体系；cyt c 带血红素必须走 tleap（OpenMM 的力场里没有血红素）。
两条路径对蛋白、水、离子应当给出同一个力场。检验：取原 cyt c（无血红素）MD 的第 0 帧（14,297 个原子，含水和离子，坐标逐原子相同），
  路径 A：OpenMM XML 力场（与原脚本相同）→ createSystem(PME, 1.0 nm, HBonds) → 势能与力；
  路径 B：同一坐标由 tleap（ff14SB + FB3 水 + Joung–Cheatham 离子）建 prmtop → OpenMM AmberPrmtopFile.createSystem（同样的 PME / 截断 / 约束）→ 势能与力。
两者都在 OpenMM 的 CPU 平台上算（避免 CUDA 单精度带来的差异），分项（键、角、二面角、非键）也各自比较。
输出：tables/S9d_cytc_path_equivalence.csv（总势能差、力的最大 / RMS 差、分项能量）。
用法：env/md/bin/python s9d_cytc_path_equivalence.py"""
import os, sys, json, subprocess, textwrap
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
import mdtraj as md
from openmm.app import ForceField, PDBFile, AmberPrmtopFile, AmberInpcrdFile, PME, HBonds
from scipy.spatial import cKDTree
from openmm import Platform, VerletIntegrator, Context, unit

HERE = os.path.dirname(os.path.abspath(__file__)); REV = os.path.abspath(os.path.join(HERE, "..")); PROJ = os.path.dirname(REV)
W = os.path.join(REV, "md_cytc_heme", "param", "equiv_test"); os.makedirs(W, exist_ok=True)
AMB = "/opt/amber/amber22"
t = md.load(os.path.join(PROJ, "gen", "MD_short", "cytochrome_c", "full.xtc"), top=os.path.join(PROJ, "gen", "MD_short", "cytochrome_c", "full_top.pdb"))
frame = os.path.join(W, "frame0.pdb"); t[0].save_pdb(frame)

# ---- 路径 A：OpenMM XML
pdb = PDBFile(frame); ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
sysA = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1.0 * unit.nanometer, constraints=HBonds)

# ---- 路径 B：同一坐标 → tleap → prmtop
lines = []
for l in open(frame):
    if not l.startswith(("ATOM", "HETATM")):
        continue
    res = l[17:20].strip(); nm = l[12:16].strip()
    if res == "HIS":
        l = l[:17] + "HID" + l[20:]
    if res == "GLY" and l[22:26].strip() == "1" and nm == "H":
        l = l[:12] + " H1 " + l[16:]
    if res == "NA":
        l = l[:12] + "Na+ " + l[16:17] + "Na+" + l[20:]
    if res == "CL":
        l = l[:12] + "Cl- " + l[16:17] + "Cl-" + l[20:]
    lines.append(l)
open(os.path.join(W, "frame0_amber.pdb"), "w").writelines(lines + ["END\n"])
open(os.path.join(W, "tleap_equiv.in"), "w").write(textwrap.dedent("""\
    source leaprc.protein.ff14SB
    source leaprc.water.fb3
    loadamberparams frcmod.ionsjc_tip3p
    x = loadpdb frame0_amber.pdb
    set x box {%s %s %s}
    saveamberparm x equiv.prmtop equiv.inpcrd
    quit
    """) % tuple(f"{v * 10:.4f}" for v in t.unitcell_lengths[0]))
env = dict(os.environ, AMBERHOME=AMB, PATH=f"{AMB}/bin:" + os.environ["PATH"])
r = subprocess.run(["tleap", "-s", "-f", "tleap_equiv.in"], cwd=W, env=env, capture_output=True, text=True); open(os.path.join(W, "tleap_equiv.out"), "w").write(r.stdout + r.stderr)
prm = AmberPrmtopFile(os.path.join(W, "equiv.prmtop")); sysB = prm.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.0 * unit.nanometer, constraints=HBonds)
assert pdb.topology.getNumAtoms() == prm.topology.getNumAtoms() == sysA.getNumParticles() == sysB.getNumParticles()
crdB = AmberInpcrdFile(os.path.join(W, "equiv.inpcrd")); posA = np.array(pdb.positions.value_in_unit(unit.nanometer)); posB = np.array(crdB.positions.value_in_unit(unit.nanometer))
# tleap 按自己的库顺序重排原子（C、O 放到侧链之后，离子排在水之前），所以按坐标配对原子（两边坐标来自同一个 PDB，逐原子相同）
shift = posB.mean(0) - posA.mean(0)   # tleap 的 set box 把体系整体平移到新盒子中心；平移不改变能量，按质心差对齐再配对
dist, idxB_for_A = cKDTree(posB - shift).query(posA); assert dist.max() < 1e-3 and len(set(idxB_for_A.tolist())) == len(posA), (dist.max(), len(set(idxB_for_A.tolist())))
namesA = [a.name for a in pdb.topology.atoms()]; namesB = [a.name for a in prm.topology.atoms()]
n_name_diff = sum(1 for i, j in enumerate(idxB_for_A) if namesA[i].replace("Cl", "Cl-").replace("Na", "Na+") not in (namesB[j], namesA[i]) and namesA[i] != namesB[j])


def evaluate(system, positions, box):
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)
    ctx = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("CPU")); ctx.setPeriodicBoxVectors(*box); ctx.setPositions(positions)
    out = {}
    for i, f in enumerate(system.getForces()):
        s = ctx.getState(getEnergy=True, groups={i}); out[f.__class__.__name__] = s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    s = ctx.getState(getEnergy=True, getForces=True)
    return s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer), out


box = pdb.topology.getPeriodicBoxVectors(); pos = pdb.positions
eA, fA, partsA = evaluate(sysA, pos, box); eB, fB, partsB = evaluate(sysB, crdB.positions, crdB.boxVectors)
dF = np.linalg.norm(fA - fB[idxB_for_A], axis=1)
top = np.argsort(-dF)[:5]; atomsA = list(pdb.topology.atoms())
for k in top:
    print("top |ΔF| atom:", atomsA[k].residue.name, atomsA[k].residue.index, atomsA[k].name, "|ΔF|=%.1f" % dF[k], "|F|=%.1f" % np.linalg.norm(fA[k]))
rel_big = float(np.mean(dF / np.maximum(np.linalg.norm(fA, axis=1), 1.0) > 0.05))
rows = [dict(quantity="total potential energy (kJ/mol)", pathA_openmm_xml=eA, pathB_tleap_prmtop=eB, difference=eB - eA, relative=(eB - eA) / abs(eA))]
for k in sorted(set(partsA) | set(partsB)):
    rows.append(dict(quantity=f"{k} (kJ/mol)", pathA_openmm_xml=partsA.get(k, np.nan), pathB_tleap_prmtop=partsB.get(k, np.nan), difference=partsB.get(k, np.nan) - partsA.get(k, np.nan)))
rows += [dict(quantity="max |ΔF| per atom (kJ/mol/nm)", difference=float(dF.max())), dict(quantity="fraction of atoms with |ΔF|/|F| > 5%", difference=rel_big), dict(quantity="RMS ΔF (kJ/mol/nm)", difference=float(np.sqrt((dF ** 2).mean()))), dict(quantity="RMS |F| path A (kJ/mol/nm)", pathA_openmm_xml=float(np.sqrt((np.linalg.norm(fA, axis=1) ** 2).mean()))),
         dict(quantity="atoms / atom-name mismatches", difference=float(n_name_diff), pathA_openmm_xml=float(pdb.topology.getNumAtoms()))]
out = os.path.join(REV, "tables", "S9d_cytc_path_equivalence.csv"); pd.DataFrame(rows).to_csv(out, index=False)
pd.set_option("display.width", 200); print(pd.DataFrame(rows).to_string(index=False))
