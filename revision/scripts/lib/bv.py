"""Best–Vendruscolo (BV) 保护因子模型：返修统一实现。

* `fix_frame` / `nc_nh` 逐字复制自原项目 operator/forward.py:15–45（原函数名 _fix_frame/_nc_nh），
  只把临时目录参数化，算法与默认阈值完全不变，以保证与原稿可比（revision_proposal.md S4 步骤 2）。
* `aggregate` 实现两种系综平均形式（等权）：
    count_avg : ln PF_i = βC·⟨N_C,i⟩ + βH·⟨N_H,i⟩                       （原稿 Eq. 1）
    invpf_avg : ln PF_i = −ln⟨exp(−βC·N_C,i(c) − βH·N_H,i(c))⟩_c          （R2-C1 要求的 Linderstrøm-Lang 形式）
  NaN（某帧缺该残基）按“仅在出现该残基的帧上平均”处理，与原 21_score_parallel.py 的 np.nanmean 一致。
"""
import os
# 打分必须在 CPU 上做：原项目所有打分脚本（21_score_parallel.py、43_boltz_t3_control.py 等）都在 CUDA_VISIBLE_DEVICES="" 下运行，
# 注释写明“算子 CPU，防 openmm 泄漏”。Modeller.addHydrogens 在未指定平台时让 OpenMM 自行选平台（该环境里实测落到 OpenCL，不是 CUDA；见 tables/S3_gpu_cpu_check.csv），
# GPU 可见时 PDBFixer 加氢会在每个进程里开 GPU 计算上下文、占用他人的卡。S3 基准的首次运行因漏了这一条占用了 GPU 0–3（约 17 分钟），已更正。
# 必须在任何 openmm/pdbfixer 导入之前赋值（fix_frame 内部才导入它们）。
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tempfile, warnings
import numpy as np
from scipy.special import logsumexp

warnings.filterwarnings("ignore")
R_KCAL = 0.0019872041          # 原 operator/forward.py:11
R_GAS = 1.987204e-3            # 原 21_score_parallel.py:8（两者在第 7 位小数一致）
T_REF = 298.15
BETA_C, BETA_H = 0.35, 2.0
GEOM = dict(cut_Nc=0.65, cut_Nh=0.26, ij_Nc=3, ij_Nh=2)
T2O = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
_SHM = "/dev/shm" if os.path.isdir("/dev/shm") else None


_THREAD_CHECKED = False


def _require_single_thread_cpu():
    """给定种子时的前提检查（S4a 返工）：同种子逐位复现要求 OpenMM 的 CPU 平台只用 1 个线程，且 OPENMM_CPU_THREADS=1 必须在 OpenMM 被加载**之前**
    进入进程环境（`import mdtraj` 就会带入 openmm，所以要设在导入 mdtraj/numpy 之前）。在 OpenMM 加载之后再设置无效，会静默得到不可复现的结果，
    所以这里直接读平台的实际线程数，不是 1 就报错。每个进程只检查一次。"""
    global _THREAD_CHECKED
    if _THREAD_CHECKED:
        return
    import openmm as mm
    sy = mm.System(); sy.addParticle(1.0); ctx = mm.Context(sy, mm.VerletIntegrator(0.001)); pl = ctx.getPlatform()
    name, n = pl.getName(), pl.getPropertyValue(ctx, "Threads") if pl.getName() == "CPU" else "n/a"
    del ctx
    if name != "CPU" or n != "1":
        raise RuntimeError(f"逐位复现的前提不满足：OpenMM 平台={name}，线程数={n}（需要 CPU 且 1 个线程）。"
                           "请在导入 numpy/mdtraj/openmm 之前设置 OPENMM_CPU_THREADS=1 与 CUDA_VISIBLE_DEVICES=''")
    _THREAD_CHECKED = True


def fix_frame(traj1, add_missing_atoms=True, tmpdir=_SHM, seed=None):
    """单帧：取最大蛋白链 → PDBFixer（补缺失重原子 + pH 7.0 加氢）→ 带 H 的 mdtraj 单帧。
    逐字复制自 operator/forward.py:_fix_frame，只多了一个可选的 seed：
      seed=None（默认）→ 与原函数完全相同，不设任何种子；
      seed=整数 → random.seed(seed)（PDBFixer/Modeller 放氢用 Python 全局 random）并把同一个整数传给 addMissingAtoms(seed=…)
      （补出的侧链原子碰撞时 PDBFixer 会用 Langevin 动力学拆开，其积分器种子默认随机）。S4 起逐帧传确定性种子，使重跑逐位一致。
    逐位一致的前提：OpenMM CPU 平台单线程（结果依赖线程数，同一线程数下可复现），见 _require_single_thread_cpu；给定 seed 时会检查。"""
    import mdtraj as md, random
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    if seed is not None:
        _require_single_thread_cpu()
        random.seed(seed)
    top = traj1.topology
    ch = max(top.chains, key=lambda c: sum(1 for r in c.residues if r.name in T2O))
    t = traj1.atom_slice([a.index for a in ch.atoms])
    tmp = tempfile.mktemp(suffix='.pdb', dir=tmpdir); out = None
    try:   # S4 起：异常或超时（SIGALRM 抛出 FrameTimeout）时也清理临时文件；计算本身与原函数逐字相同
        t.save_pdb(tmp)
        fx = PDBFixer(filename=tmp); fx.findMissingResidues()
        fx.missingResidues = {}
        fx.findNonstandardResidues(); fx.findMissingAtoms()
        if not add_missing_atoms:
            fx.missingAtoms = {}; fx.missingTerminals = {}
        fx.addMissingAtoms(seed=seed); fx.addMissingHydrogens(7.0)
        out = tempfile.mktemp(suffix='.pdb', dir=tmpdir)
        with open(out, 'w') as fh:
            PDBFile.writeFile(fx.topology, fx.positions, fh)
        r = md.load(out)
    finally:
        for p_ in (tmp, out):
            if p_ and os.path.exists(p_):
                os.remove(p_)
    return r


def fix_frame_openmm(traj1, add_missing_atoms=True, tmpdir=_SHM, seed=None):
    """与 fix_frame 的每一步相同（取最大链、PDBFixer 补原子与加氢、同样的种子处理），但直接返回 OpenMM 的 (topology, positions[nm, ndarray])，不经 PDB 往返与 mdtraj 的原子改名。
    S5 的噪声系综需要把参考结构交给 OpenMM 做约束最小化，所以用这个函数；与 fix_frame 的坐标在同一种子下逐位相同（s5_noise.py 的自检会核对）。"""
    import random
    from pdbfixer import PDBFixer
    if seed is not None:
        _require_single_thread_cpu()
        random.seed(seed)
    top = traj1.topology
    ch = max(top.chains, key=lambda c: sum(1 for r in c.residues if r.name in T2O))
    t = traj1.atom_slice([a.index for a in ch.atoms])
    tmp = tempfile.mktemp(suffix='.pdb', dir=tmpdir)
    try:
        t.save_pdb(tmp)
        fx = PDBFixer(filename=tmp); fx.findMissingResidues()
        fx.missingResidues = {}
        fx.findNonstandardResidues(); fx.findMissingAtoms()
        if not add_missing_atoms:
            fx.missingAtoms = {}; fx.missingTerminals = {}
        fx.addMissingAtoms(seed=seed); fx.addMissingHydrogens(7.0)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return fx.topology, np.array(fx.positions.value_in_unit(fx.positions.unit)) if hasattr(fx.positions, "value_in_unit") else np.asarray(fx.positions)


def nc_nh(t1, cut_Nc=0.65, cut_Nh=0.26, ij_Nc=3, ij_Nh=2, hbond_acceptor="backbone_O", skip_first=False):
    """逐残基 (resn1, N_C, N_H)。默认参数与原 operator/forward.py:_nc_nh 完全一致（单位 nm）。
    额外开关只用于 S6 的 HDXer 一致性核对：
      hbond_acceptor="all_O" → N_H 受体为全部蛋白 O（HDXer 的 'protein and symbol O'）；
      skip_first=True       → 跳过链首残基（HDXer skip_first）。默认值下输出与原函数逐字节相同。"""
    top = t1.topology; xyz = t1.xyz[0]
    heavy = np.array([a.index for a in top.atoms if a.element and a.element.symbol != 'H'])
    hv_res = np.array([top.atom(i).residue.index for i in heavy])
    if hbond_acceptor == "backbone_O":
        Oc = np.array([a.index for a in top.atoms if a.name == 'O'])
    elif hbond_acceptor == "all_O":
        Oc = np.array([a.index for a in top.atoms if a.element and a.element.symbol == 'O'
                       and a.residue.name in T2O])
    else:
        raise ValueError(hbond_acceptor)
    o_res = np.array([top.atom(i).residue.index for i in Oc])
    out = {}
    first_index = min(r.index for r in top.residues)
    for res in top.residues:
        if res.name == 'PRO' or res.name not in T2O:
            continue
        if skip_first and res.index == first_index:
            continue
        Ns = [a for a in res.atoms if a.name == 'N']; Hs = [a for a in res.atoms if a.name in ('H', 'HN', 'H1')]
        if not Ns:
            continue
        ri = res.index; npos = xyz[Ns[0].index]
        NC = int(((np.linalg.norm(xyz[heavy] - npos, axis=1) < cut_Nc) & (np.abs(hv_res - ri) >= ij_Nc)).sum())
        NH = 0
        if Hs and len(Oc):
            NH = int(((np.linalg.norm(xyz[Oc] - xyz[Hs[0].index], axis=1) < cut_Nh) & (np.abs(o_res - ri) >= ij_Nh)).sum())
        out[res.resSeq] = (T2O[res.name], NC, NH)
    return out


def load_max_asa(path):
    """读 Tien et al. 2013 的理论最大可及表面积（Å²），转成 nm²：{三字母名: nm²}。"""
    import csv
    return {r["resname"]: float(r["theoretical_max_ASA_A2"]) * 0.01 for r in csv.DictReader(open(path))}


def sasa_features(t1, max_asa):
    """逐残基 (rSASA_res, SASA_amideNH)，残基筛选规则与 nc_nh 完全相同（跳过 Pro、非标准残基、无酰胺 N 的残基），键为 resSeq。
      rSASA_res   = mdtraj shrake_rupley(mode='residue', probe_radius=0.14 nm, n_sphere_points=960) / 理论最大 ASA；
      SASA_amideNH = 同参数下原子级 SASA 中酰胺 N 与酰胺 H 之和（H 的选取与 nc_nh 一致：原子名在 H/HN/H1 中的第一个；没有则只算 N）。
    t1 必须是 fix_frame 的输出（全原子、已加氢）。单位 nm²（rSASA 无量纲）。"""
    import mdtraj as md
    sr = md.shrake_rupley(t1, probe_radius=0.14, n_sphere_points=960, mode="residue")[0]
    sa = md.shrake_rupley(t1, probe_radius=0.14, n_sphere_points=960, mode="atom")[0]
    out = {}
    for res in t1.topology.residues:
        if res.name == 'PRO' or res.name not in T2O:
            continue
        Ns = [a for a in res.atoms if a.name == 'N']; Hs = [a for a in res.atoms if a.name in ('H', 'HN', 'H1')]
        if not Ns:
            continue
        out[res.resSeq] = (float(sr[res.index] / max_asa[res.name]), float(sa[Ns[0].index] + (sa[Hs[0].index] if Hs else 0.0)))
    return out


def x_matrix(NC, NH, betaC=BETA_C, betaH=BETA_H):
    """逐构象 x_i(c) = βC·N_C + βH·N_H（ln PF 单位），形状 [残基, 构象]。"""
    return betaC * np.asarray(NC, float) + betaH * np.asarray(NH, float)


def aggregate(NC, NH, form="count_avg", frames=None, betaC=BETA_C, betaH=BETA_H, weights="uniform"):
    """返回逐残基 ln PF（等权系综平均）。frames: 使用的列索引（None=全部）。
    weights：只接受 "uniform"（默认）或一个常数向量；非常数向量直接报错（方案 R2-C1b：代码断言权重向量是常数向量）。
    本函数不支持非等权平均，所以这个参数只是显式的断言入口，不改变任何数值。"""
    if not (isinstance(weights, str) and weights == "uniform"):
        w = np.asarray(weights, float)
        if w.size == 0 or not np.allclose(w, w.flat[0]):
            raise ValueError("aggregate 只支持等权系综平均：传入的权重向量不是常数向量")
    NC = np.asarray(NC, float); NH = np.asarray(NH, float)
    if frames is not None:
        NC = NC[:, frames]; NH = NH[:, frames]
    if form == "count_avg":
        return betaC * np.nanmean(NC, axis=1) + betaH * np.nanmean(NH, axis=1)
    if form == "invpf_avg":
        x = x_matrix(NC, NH, betaC, betaH)
        out = np.full(x.shape[0], np.nan)
        for i in range(x.shape[0]):
            v = x[i][~np.isnan(x[i])]
            if len(v):
                out[i] = -(logsumexp(-v) - np.log(len(v)))
        return out
    raise ValueError(form)


def wmax_dominance(NC, NH, frames=None, betaC=BETA_C, betaH=BETA_H):
    """1/PF 平均中单个构象的最大贡献比例 w_max（逐残基）。"""
    NC = np.asarray(NC, float); NH = np.asarray(NH, float)
    if frames is not None:
        NC = NC[:, frames]; NH = NH[:, frames]
    x = x_matrix(NC, NH, betaC, betaH)
    out = np.full(x.shape[0], np.nan)
    for i in range(x.shape[0]):
        v = x[i][~np.isnan(x[i])]
        if len(v):
            w = np.exp(-(v - v.min()))
            out[i] = w.max() / w.sum()
    return out


# ---------------- 残基对齐（逐字复制原 42_null_baselines.py / 22_leaderboard.py 的规则） ----------------
A1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H',
      'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
      'TYR': 'Y', 'VAL': 'V'}


def to1(x):
    x = str(x)
    return A1.get(x.upper(), x[:1].upper()) if len(x) == 3 else (x[:1].upper() if x.strip() else '')


def best_offset(pred_resn_by_resseq, exp_group, default):
    """按残基身份在 [−8, 8] 内找最佳编号偏移（原 42_null_baselines.boff / 22_leaderboard.best_offset）。"""
    eid = {int(r.resi): to1(r.resn) for _, r in exp_group.iterrows() if str(r.resn).strip()}
    if len(eid) < 5 or not pred_resn_by_resseq:
        return default
    best, bg = default, -1
    for o in range(-8, 9):
        gg = sum(1 for er, aa in eid.items() if pred_resn_by_resseq.get(er + o) == aa)
        if gg > bg:
            bg, best = gg, o
    return best


def boff_null42(pred_resn_by_resseq, exp_group, maps):
    """原 42_null_baselines.py:boff 的精确复制（初值 0，而非 default；<5 时回落 gen_offset）。"""
    eid = {int(r.resi): to1(r.resn) for _, r in exp_group.iterrows() if str(r.resn).strip()}
    if len(eid) < 5:
        return maps.get(exp_group.protein_id.iloc[0], {}).get('gen_offset', 0)
    b, bg = 0, -1
    for o in range(-8, 9):
        gg = sum(1 for er, aa in eid.items() if pred_resn_by_resseq.get(er + o) == aa)
        if gg > bg:
            bg, b = gg, o
    return b


def align(resSeq, values, exp_group, offset):
    """把逐残基预测按 resi = resSeq − offset 对齐到实验表，返回 (pred, exp, resi) 数组。"""
    m = dict(zip([int(r) - offset for r in resSeq], values))
    e = exp_group.dropna(subset=['dGopen_kcal'])
    rr = [int(r) for r in e.resi if int(r) in m and np.isfinite(m[int(r)])]
    ev = dict(zip(e.resi.astype(int), e.dGopen_kcal.astype(float)))
    return np.array([m[r] for r in rr]), np.array([ev[r] for r in rr]), np.array(rr)
