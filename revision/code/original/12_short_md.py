"""Stage1 基底b:单蛋白显式溶剂短MD(amber14+TIP3P,PME),产MD系综供算子标定对照。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, os
from openmm.app import *; from openmm import *; from openmm.unit import *
from pdbfixer import PDBFixer
import mdtraj as md
prot, pdb, gpu, ns = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
PROJ=_GEC_ROOT
os.environ["CUDA_VISIBLE_DEVICES"]=gpu
outd=f"{PROJ}/gen/MD_short/{prot}"; os.makedirs(outd,exist_ok=True)
fx=PDBFixer(filename=f"{PROJ}/data/external/pdb/{pdb}.pdb")
fx.findMissingResidues(); fx.missingResidues={}; fx.findNonstandardResidues(); fx.replaceNonstandardResidues(); fx.removeHeterogens(False)
fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
mod=Modeller(fx.topology, fx.positions)
ff=ForceField('amber14-all.xml','amber14/tip3pfb.xml')
mod.addSolvent(ff, padding=1.0*nanometer, model='tip3p', ionicStrength=0.1*molar)
system=ff.createSystem(mod.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds)
system.addForce(MonteCarloBarostat(1*bar,300*kelvin))
integ=LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 2*femtosecond)
sim=Simulation(mod.topology, system, integ, Platform.getPlatformByName('CUDA'))
sim.context.setPositions(mod.positions); sim.minimizeEnergy(maxIterations=2000)
sim.context.setVelocitiesToTemperature(300*kelvin)
sim.step(50000)  # 100ps 平衡
nframes=200; interval=int(ns*1e6/2/nframes)  # 2fs步;ns->步数/帧数
sim.reporters.append(XTCReporter(f"{outd}/full.xtc", interval))
sim.reporters.append(StateDataReporter(f"{outd}/md.log", interval*10, step=True, temperature=True, potentialEnergy=True))
sim.step(int(ns*1e6/2))  # 生产
# 存拓扑(含水) + 抽蛋白
with open(f"{outd}/full_top.pdb","w") as fh: PDBFile.writeFile(mod.topology, sim.context.getState(getPositions=True).getPositions(), fh)
t=md.load(f"{outd}/full.xtc", top=f"{outd}/full_top.pdb")
prot_sel=t.topology.select("protein")
t.atom_slice(prot_sel).save_xtc(f"{outd}/ensemble.xtc")
t[0].atom_slice(prot_sel).save_pdb(f"{outd}/ensemble.pdb")
print(f"{prot}: MD完成 {t.n_frames}帧 (蛋白{len(prot_sel)}原子)")
