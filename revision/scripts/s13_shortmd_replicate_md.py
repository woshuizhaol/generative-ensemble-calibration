"""Stage 13 · ShortMD 副本（编辑 MD 可靠性清单）：单个蛋白的一条短 MD 副本，协议逐项同原脚本 `P/code/analysis/12_short_md.py`。
  OpenMM 8.2.0（env/md）、CUDA 单精度（平台默认）、amber14-all.xml + amber14/tip3pfb.xml、PDBFixer（替换非标准残基、去杂原子与结晶水、补缺失重原子、pH 7.0 加氢，不补缺失残基）、
  Modeller.addSolvent(padding=1.0 nm, model="tip3p", ionicStrength=0.1 M)、PME 1.0 nm、HBonds、MonteCarloBarostat(1 bar, 300 K)、LangevinMiddleIntegrator(300 K, 1/ps, 2 fs)、
  LocalEnergyMinimizer(maxIterations=2000)、setVelocitiesToTemperature(300 K)、100 ps NPT 平衡（50000 步，无 NVT、无位置约束）、2 ns 生产、每 5000 步（10 ps）一帧共 200 帧。
与原脚本的差别（预登记 PREREG_replicates.md 第 2 节）：
  ① 显式随机种子：同一整数 --seed 用于 setVelocitiesToTemperature、LangevinMiddleIntegrator、MonteCarloBarostat，也用于 random.seed()（PDBFixer.addMissingHydrogens 与 Modeller.addSolvent
     的氢、离子放置用 Python 全局 random；两处之前各设一次）与 PDBFixer.addMissingAtoms(seed=…)；
  ② 记录：平衡阶段每 500 步（1 ps）、生产阶段每 5000 步（10 ps）写 StateDataReporter（势能、动能、温度、盒子体积、密度）；保存溶剂化后最小化前的 initial_system.pdb、final.pdb、
     本脚本副本 input_script.py 与 md_meta.json（版本、平台、原子数、水与离子数、盒子、种子、墙钟）。
输出：<out-root>/<protein>_r<replicate>/{initial_system.pdb, equil.log, md.log, full.xtc, final.pdb, ensemble.xtc, ensemble.pdb, input_script.py, md_meta.json}。
用法：CUDA_VISIBLE_DEVICES=7 python s13_shortmd_replicate_md.py --protein CI2 --pdb 2CI2 --replicate 2 --seed 13042 [--ns 2.0] [--equil-steps 50000] [--out-root DIR]"""
import os, sys, json, time, argparse, random, shutil, hashlib, csv, subprocess, builtins, platform as _plat
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
from openmm import __version__ as OMM_VERSION
from openmm.app import *; from openmm import *; from openmm.unit import *
from pdbfixer import PDBFixer
import mdtraj as md
import numpy as np
import importlib.metadata as im

HERE = os.path.dirname(os.path.abspath(__file__)); REV_DEFAULT = os.path.abspath(os.path.join(HERE, ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--protein", required=True); ap.add_argument("--pdb", required=True); ap.add_argument("--replicate", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--ns", type=float, default=2.0); ap.add_argument("--equil-steps", type=int, default=50000)
ap.add_argument("--out-root", default=os.path.join(REV_DEFAULT, "md_replicates", "md"))
ap.add_argument("--pdb-dir", default=os.path.join(os.environ.get("GEC_PROJECT_ROOT", os.path.join(REV_DEFAULT, "..")), "data", "external", "pdb"))
a = ap.parse_args()
prot, pdb, rep, seed, ns = a.protein, a.pdb, a.replicate, a.seed, a.ns
name = f"{prot}_r{rep}"; outd = os.path.abspath(os.path.join(a.out_root, name))
if os.path.exists(os.path.join(outd, "md_meta.json")):
    print("已完成，跳过:", outd); sys.exit(0)
if os.path.exists(outd):   # 上次未完成的目录不覆盖、不删除：改名留存
    os.rename(outd, outd + ".partial_" + time.strftime("%Y%m%d%H%M%S"))
os.makedirs(outd)
shutil.copy(os.path.abspath(__file__), os.path.join(outd, "input_script.py"))
t_start = time.time(); iso = lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z")
started = iso(); print(name, "start", started, "seed", seed, "CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

# ---------------- 体系构建（与原脚本逐行对应）
fx = PDBFixer(filename=os.path.join(a.pdb_dir, f"{pdb}.pdb"))
fx.findMissingResidues(); fx.missingResidues = {}; fx.findNonstandardResidues(); fx.replaceNonstandardResidues(); fx.removeHeterogens(False)
fx.findMissingAtoms(); fx.addMissingAtoms(seed=seed)
random.seed(seed); fx.addMissingHydrogens(7.0)
mod = Modeller(fx.topology, fx.positions)
ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
random.seed(seed); mod.addSolvent(ff, padding=1.0 * nanometer, model='tip3p', ionicStrength=0.1 * molar)
with open(os.path.join(outd, "initial_system.pdb"), "w") as fh:
    PDBFile.writeFile(mod.topology, mod.positions, fh)
system = ff.createSystem(mod.topology, nonbondedMethod=PME, nonbondedCutoff=1.0 * nanometer, constraints=HBonds)
barostat = MonteCarloBarostat(1 * bar, 300 * kelvin); barostat.setRandomNumberSeed(seed); system.addForce(barostat)
integ = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 2 * femtosecond); integ.setRandomNumberSeed(seed)
sim = Simulation(mod.topology, system, integ, Platform.getPlatformByName('CUDA'))

# ---------------- 体系统计（initial_system 的组成）
WATER, NA, CL = {"HOH", "WAT"}, {"NA", "Na+"}, {"CL", "Cl-"}
res = list(mod.topology.residues()); nw = builtins.sum(r.name in WATER for r in res); nna = builtins.sum(r.name in NA for r in res); ncl = builtins.sum(r.name in CL for r in res)   # openmm.unit 的 * 导入会覆盖内置 sum
prot_res = [r for r in res if r.name not in WATER | NA | CL]; chains_prot = sorted({r.chain.index for r in prot_res})
nbf = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
q = np.array([nbf.getParticleParameters(i)[0].value_in_unit(elementary_charge) for i in range(system.getNumParticles())])
q_prot = float(builtins.sum(q[at.index] for r in prot_res for at in r.atoms())); q_total = float(q.sum())
bv0 = mod.topology.getPeriodicBoxVectors().value_in_unit(nanometer); box0 = [float(np.linalg.norm(np.array(v))) for v in bv0]
n_dof_removed = system.getNumConstraints()

sim.context.setPositions(mod.positions)
e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
t_md0 = time.time(); sim.minimizeEnergy(maxIterations=2000)
e1 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
sim.context.setVelocitiesToTemperature(300 * kelvin, seed)
sim.reporters.append(StateDataReporter(os.path.join(outd, "equil.log"), 500, step=True, time=True, potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, density=True, separator=","))
sim.step(a.equil_steps)  # 100 ps 平衡
sim.reporters.clear()
nframes = 200; interval = int(ns * 1e6 / 2 / nframes)  # 2 fs 步；ns -> 步数/帧数
sim.reporters.append(XTCReporter(os.path.join(outd, "full.xtc"), interval))
sim.reporters.append(StateDataReporter(os.path.join(outd, "md.log"), interval, step=True, time=True, potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, density=True, separator=","))
sim.step(int(ns * 1e6 / 2))  # 生产
t_md1 = time.time()
st = sim.context.getState(getPositions=True, getEnergy=True)
sim.topology.setPeriodicBoxVectors(st.getPeriodicBoxVectors())
with open(os.path.join(outd, "final.pdb"), "w") as fh:
    PDBFile.writeFile(sim.topology, st.getPositions(), fh)
box1 = [float(np.linalg.norm(np.array(v))) for v in st.getPeriodicBoxVectors().value_in_unit(nanometer)]
del sim

# ---------------- 蛋白原子切片（与原脚本相同：ensemble.xtc / ensemble.pdb 只含 mdtraj 的 "protein" 选择，所有链）
t = md.load(os.path.join(outd, "full.xtc"), top=os.path.join(outd, "final.pdb"))
prot_sel = t.topology.select("protein")
t.atom_slice(prot_sel).save_xtc(os.path.join(outd, "ensemble.xtc"))
t[0].atom_slice(prot_sel).save_pdb(os.path.join(outd, "ensemble.pdb"))

# ---------------- 元数据
with open(os.path.join(outd, "md.log")) as fh:
    rows = list(csv.reader(fh)); hdr = [h.strip('#" ') for h in rows[0]]; body = np.array([[float(x) for x in r] for r in rows[1:]])
Tcol = [i for i, h in enumerate(hdr) if h.startswith("Temperature")][0]
def smi(q_):
    try:
        return subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0"), f"--query-gpu={q_}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"unavailable:{e}"
meta = dict(name=name, protein=prot, pdb=pdb, replicate=rep, seed_md=seed, seed_build=seed, seed_note="同一整数用于 random.seed（氢与离子放置）、addMissingAtoms、速度初始化、积分器、气压计",
            ns=ns, equil_steps=a.equil_steps, frames=int(t.n_frames), interval_steps=interval, production_rows_in_md_log=int(body.shape[0]),
            openmm=OMM_VERSION, openmm_full=Platform.getOpenMMVersion() if hasattr(Platform, "getOpenMMVersion") else "", pdbfixer=im.version("pdbfixer"), mdtraj=im.version("mdtraj"), python=_plat.python_version(),
            platform="CUDA", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_name=smi("name"), host=_plat.node(),
            n_atoms_total=int(system.getNumParticles()), n_protein_atoms_mdtraj=int(len(prot_sel)), n_protein_residues_all_chains=len(prot_res), n_protein_chains=len(chains_prot),
            n_waters=nw, n_Na=nna, n_Cl=ncl, net_charge_total_e=round(q_total, 6), net_charge_protein_e=round(q_prot, 6), n_constraints=int(n_dof_removed),
            box_initial_nm=[round(x, 4) for x in box0], box_final_nm=[round(x, 4) for x in box1], box_initial_volume_nm3=round(float(np.prod(box0)), 3),
            potential_before_min_kJmol=e0, potential_after_min_kJmol=e1, production_mean_T_K=float(body[:, Tcol].mean()), production_sd_T_K=float(body[:, Tcol].std(ddof=1)),
            started=started, finished=iso(), wall_s_total=round(time.time() - t_start, 1), wall_s_minimize_equil_production=round(t_md1 - t_md0, 1),
            script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(), cmdline=" ".join(sys.argv))
json.dump(meta, open(os.path.join(outd, "md_meta.json"), "w"), indent=1)
print(name, "done", outd, "frames", t.n_frames, "protein atoms", len(prot_sel), "wall_s", meta["wall_s_total"], flush=True)
