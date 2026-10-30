"""Stage 9 追加（U-S3-2）· 步骤 6：cyt c 短 MD（带血红素 / 无血红素对照），协议逐项同原 ShortMD（`P/code/analysis/12_short_md.py`）：
  OpenMM 8.2.0（env/md）、CUDA 单精度（平台默认）、PME 1.0 nm、HBonds 约束、水刚性、2 fs、LangevinMiddle 300 K 摩擦 1/ps、MonteCarloBarostat 1 bar 300 K（频率 25）、
  LocalEnergyMinimizer maxIterations=2000、setVelocitiesToTemperature(300 K)、100 ps NPT 平衡（50000 步，无 NVT 段）、2 ns 生产、每 10 ps 一帧（200 帧，XTC）。
与原脚本的差别（各自有记录）：①体系由 tleap 构建（ff14SB + TIP3P-FB + Joung–Cheatham 离子），而不是 OpenMM Modeller + amber14-all.xml；两者的蛋白、水、离子参数相同
  （amber14/protein.ff14SB.xml 与 amber14/tip3pfb.xml 均由同名 Amber 文件转换），带与不带血红素的体系走同一条构建路径，所以两者之间的比较不受路径影响；②原脚本没有设随机种子，这里显式设置
  速度初始化、Langevin 积分器和气压计的种子（--seed），以便复现并估计轨迹间波动。
输入：param/<name>_solv.prmtop、<name>_solv.inpcrd。输出：md/<name>_s<seed>/{full.xtc, md.log, md_meta.json, final.pdb}。
用法：CUDA_VISIBLE_DEVICES=7 python s9d_cytc_md.py --name cytc_heme --seed 1 [--ns 2.0]"""
import os, sys, json, time, argparse
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, PME, HBonds, Simulation, StateDataReporter, XTCReporter, PDBFile
from openmm import MonteCarloBarostat, LangevinMiddleIntegrator, Platform, __version__ as OMM_VERSION
from openmm.unit import nanometer, kelvin, bar, picosecond, femtosecond

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--ns", type=float, default=2.0)
ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "md_cytc_heme")); a = ap.parse_args()
root = os.path.abspath(a.root); out = os.path.join(root, "md", f"{a.name}_s{a.seed}"); os.makedirs(out, exist_ok=True)
prmtop = AmberPrmtopFile(os.path.join(root, "param", f"{a.name}_solv.prmtop")); inpcrd = AmberInpcrdFile(os.path.join(root, "param", f"{a.name}_solv.inpcrd"))
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.0 * nanometer, constraints=HBonds)
barostat = MonteCarloBarostat(1 * bar, 300 * kelvin); barostat.setRandomNumberSeed(a.seed); system.addForce(barostat)
integ = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 2 * femtosecond); integ.setRandomNumberSeed(a.seed)
sim = Simulation(prmtop.topology, system, integ, Platform.getPlatformByName("CUDA"))
if inpcrd.boxVectors is not None:
    sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
sim.context.setPositions(inpcrd.positions)
t0 = time.time(); sim.minimizeEnergy(maxIterations=2000)
sim.context.setVelocitiesToTemperature(300 * kelvin, a.seed)
sim.step(50000)   # 100 ps 平衡
nframes = 200; interval = int(a.ns * 1e6 / 2 / nframes)
sim.reporters.append(XTCReporter(os.path.join(out, "full.xtc"), interval))
sim.reporters.append(StateDataReporter(os.path.join(out, "md.log"), interval * 10, step=True, temperature=True, potentialEnergy=True, density=True))
sim.step(int(a.ns * 1e6 / 2))
st = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
with open(os.path.join(out, "final.pdb"), "w") as fh:
    PDBFile.writeFile(sim.topology, st.getPositions(), fh)
json.dump(dict(name=a.name, seed=a.seed, ns=a.ns, openmm=OMM_VERSION, platform=sim.context.getPlatform().getName(), n_atoms=system.getNumParticles(), frames=nframes, interval_steps=interval, wall_s=round(time.time() - t0, 1),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")), open(os.path.join(out, "md_meta.json"), "w"), indent=1)
print("done", out)
