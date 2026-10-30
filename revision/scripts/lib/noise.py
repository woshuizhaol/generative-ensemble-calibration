"""S5C 噪声系综的核心函数（纯 numpy + OpenMM）。坐标单位 nm。
  anm_ca_samples      ProDy ANM 采样 CA 位移（传入参考 AtomGroup，样本对参考的叠合 RMSD = rmsd 参数，原型已核对）
  displace_allatom    把 CA 位移带到全部原子：每个残基用 CA_{i-1},CA_i,CA_{i+1} 三点的 Kabsch 刚体变换（链端/断点用两点最小旋转加平移）
  cart_noise          全部重原子各向同性高斯位移 N(0, σ²)，氢随所连重原子刚性平移
  Minimizer           OpenMM 真空 amber14 + 重原子谐振约束(1000 kJ/mol/nm²) + LocalEnergyMinimizer(maxIterations=200)，CPU 单线程
"""
import numpy as np


def kabsch_rt(P, Q):
    """使 |R P + t − Q| 最小的刚体变换。P、Q: (k,3)。返回 (R, t)。"""
    pc, qc = P.mean(0), Q.mean(0); H = (P - pc).T @ (Q - qc); U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def min_rotation(u, v):
    """把单位向量 u 转到 v 的最小旋转矩阵（Rodrigues）。"""
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v); c = float(u @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        a = np.cross(u, [1.0, 0, 0]);  a = a if np.linalg.norm(a) > 1e-6 else np.cross(u, [0, 1.0, 0]); a /= np.linalg.norm(a)
        return 2 * np.outer(a, a) - np.eye(3)
    k = np.cross(u, v); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); return np.eye(3) + K + K @ K / (1 + c)


def anm_ca_samples(ca_ref_nm, n_confs, rmsd_A, seed, n_modes=20, cutoff_A=15.0, gamma=1.0):
    """返回 (n_confs, nres, 3) 的 CA 坐标（nm）。seed 设全局 numpy RNG（ProDy 用它）。"""
    import prody
    prody.confProDy(verbosity="none")
    X = np.asarray(ca_ref_nm, float) * 10.0
    ag = prody.AtomGroup("ref"); ag.setCoords(X)
    anm = prody.ANM("ref"); anm.buildHessian(X, cutoff=cutoff_A, gamma=gamma); anm.calcModes(n_modes=n_modes, zeros=False)
    np.random.seed(int(seed))
    return prody.sampleModes(anm, atoms=ag, n_confs=n_confs, rmsd=float(rmsd_A)).getCoordsets() / 10.0


def displace_allatom(xyz_ref, res_atoms, ca_idx, ca_new, bond_max_nm=0.45):
    """xyz_ref: (N,3) 参考全原子坐标；res_atoms: 每个残基的原子下标数组；ca_idx: 每个残基 CA 的原子下标；ca_new: (nres,3) 位移后的 CA。返回 (N,3)。"""
    ca_ref = xyz_ref[ca_idx]; n = len(ca_idx); out = xyz_ref.copy()
    bonded = np.linalg.norm(np.diff(ca_ref, axis=0), axis=1) < bond_max_nm   # 相邻 CA 是否成键（断点处不用邻居）
    for i in range(n):
        nb = [j for j in (i - 1, i, i + 1) if 0 <= j < n and (j == i or (j < i and bonded[j]) or (j > i and bonded[i]))]
        P = ca_ref[nb]; Q = ca_new[nb]; a = res_atoms[i]
        if len(nb) >= 3:
            R, t = kabsch_rt(P, Q)
        elif len(nb) == 2:
            j = nb[0] if nb[1] == i else nb[1]; R = min_rotation(ca_ref[j] - ca_ref[i], ca_new[j] - ca_new[i]); t = ca_new[i] - R @ ca_ref[i]
        else:
            R = np.eye(3); t = ca_new[i] - ca_ref[i]
        out[a] = xyz_ref[a] @ R.T + t
    return out


def cart_noise(xyz_ref, heavy_idx, h_parent, sigma_nm, rng):
    """heavy_idx: 重原子下标；h_parent: dict {氢下标: 所连重原子下标}。返回加噪后的坐标。"""
    out = xyz_ref.copy(); disp = np.zeros_like(xyz_ref); disp[heavy_idx] = rng.normal(0.0, sigma_nm, size=(len(heavy_idx), 3))
    for h, p in h_parent.items():
        disp[h] = disp[p]
    return out + disp


class Minimizer:
    def __init__(self, topology, k_kj_nm2=1000.0, max_iter=200):
        import openmm as mm, openmm.app as app
        self.mm = mm; self.max_iter = max_iter
        ff = app.ForceField("amber14-all.xml"); system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None)
        f = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"); f.addGlobalParameter("k", k_kj_nm2)
        for p in ("x0", "y0", "z0"):
            f.addPerParticleParameter(p)
        self.heavy = [a.index for a in topology.atoms() if a.element is not None and a.element.symbol != "H"]
        for i in self.heavy:
            f.addParticle(i, [0.0, 0.0, 0.0])
        system.addForce(f); self.force = f
        plat = mm.Platform.getPlatformByName("CPU"); self.ctx = mm.Context(system, mm.VerletIntegrator(0.001), plat, {"Threads": "1"})
        assert self.ctx.getPlatform().getName() == "CPU"

    def minimize(self, xyz_nm):
        mm = self.mm; self.ctx.setPositions(xyz_nm)
        for j, i in enumerate(self.heavy):
            self.force.setParticleParameters(j, i, [float(xyz_nm[i, 0]), float(xyz_nm[i, 1]), float(xyz_nm[i, 2])])
        self.force.updateParametersInContext(self.ctx)
        mm.LocalEnergyMinimizer.minimize(self.ctx, 10.0, self.max_iter)
        return np.array(self.ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(mm.unit.nanometer))
