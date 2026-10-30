"""Stage 14 · 短 MD 的替代起点轨迹（Editorial Requests Table 1d）：协议逐项同 s13_shortmd_replicate_md.py，唯一差别是起始构象来自 BioEmu 系综中选出的一帧
  （`--start-pdb`，由 s14_select_altstarts.py 生成，只含 BioEmu 的骨架与 Cβ 原子），由 PDBFixer 补全侧链重原子（seed 同下）并加氢（pH 7.0），其余步骤（Modeller.addSolvent、amber14-all + tip3pfb、
  PME 1.0 nm、HBonds、MonteCarloBarostat、LangevinMiddleIntegrator、局部能量最小化、100 ps NPT 平衡无约束、2 ns 生产、200 帧）完全相同。最小化迭代数由 `--min-iter` 给出（默认 2000，与主分析相同；
  补全的侧链可能有空间冲突，必要时提高并记录在 md_meta.json）。副本号：a1 → 4，a2 → 5（输出目录 <蛋白>_r4、<蛋白>_r5，便于沿用 S13 的打分脚本）。
用法：CUDA_VISIBLE_DEVICES=7 python s14_altstart_md.py --protein CI2 --start-pdb DIR/CI2_a1.pdb --replicate 4 --seed 14041 [--ns 2.0] [--min-iter 2000] [--out-root DIR]"""
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
ap.add_argument("--protein", required=True); ap.add_argument("--start-pdb", required=True); ap.add_argument("--replicate", type=int, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--min-iter", type=int, default=2000)
ap.add_argument("--ns", type=float, default=2.0); ap.add_argument("--equil-steps", type=int, default=50000)
ap.add_argument("--out-root", default=os.path.join(REV_DEFAULT, "md_replicates", "md"))
a = ap.parse_args()
prot, pdb, rep, seed, ns = a.protein, os.path.basename(a.start_pdb), a.replicate, a.seed, a.ns
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
fx = PDBFixer(filename=os.path.abspath(a.start_pdb))
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
t_md0 = time.time(); sim.minimizeEnergy(maxIterations=a.min_iter)
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
meta = dict(name=name, protein=prot, start_pdb=pdb, min_iter=a.min_iter, replicate=rep, seed_md=seed, seed_build=seed, seed_note="同一整数用于 random.seed（氢与离子放置）、addMissingAtoms、速度初始化、积分器、气压计",
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
