"""两阶段 RESP 电荷拟合（Bayly et al. 1993, J. Phys. Chem. 97, 10269），带冻结原子与任意线性等式约束。
为什么自己写：Amber 的 resp 程序不能直接表达“只拟合血红素与铁的电荷、其余原子冻结、血红素 + Fe 总电荷为 −1”这类约束；本实现与 resp 程序在水分子上互相核对（见 scripts/s9d_cytc_resp_selftest.py）。
目标函数（Amber resp 的约定）：正规方程 (AᵀA + diag(a / sqrt(q_j² + b²)))·q = Aᵀ·V，b = 0.1 e；r 以 Bohr 为单位，V 以原子单位。也就是每个受约束的重原子的约束项在目标函数里是 (a/2)·(sqrt(q_j² + b²) − b)（相对 ½Σ(V − Σq/r)² 则是 a·(sqrt(q_j² + b²) − b)），所以 a=0.0005 / 0.001 是 Bayly 论文的权重，而不是把这两个值直接乘在 Σ(V − …)² 的形式上。已在甲醇上与 antechamber 的 RESP 核对（差 <1e-6）。
阶段 1：a = 0.0005，所有可变的重原子（非氢）受约束；阶段 2：a = 0.001，仅甲基 / 亚甲基等“等价组”重新拟合，其余原子冻结在阶段 1 的值。
约束：总电荷（或任意分组电荷和）、等价原子（共享同一个 q）、冻结原子（固定 q）。用拉格朗日乘子在每次迭代解线性 KKT 系统，迭代到电荷变化 <1e-6 e。
"""
import numpy as np

B_HYP = 0.1


def _design(coords, esp_xyz):
    d = np.linalg.norm(esp_xyz[:, None, :] - coords[None, :, :], axis=2)
    return 1.0 / d   # (npts, natoms)，单位 1/Bohr


def fit(coords, esp_xyz, esp_val, restrained, frozen_q, equiv=None, group_sums=None, a=0.0005, q0=None, maxit=200):
    """coords: (n,3) Bohr；esp_xyz: (m,3) Bohr；esp_val: (m,) a.u.；restrained: bool(n)；frozen_q: {idx: charge}；equiv: [[idx...], ...]；group_sums: [([idx...], total), ...]。"""
    n = len(coords); A = _design(coords, esp_xyz); equiv = equiv or []; group_sums = group_sums or []
    # 变量缩减：冻结的原子不是变量；等价组共享一个变量
    var_of = {}; free_vars = []
    for g in equiv:
        g = [i for i in g if i not in frozen_q]
        if not g:
            continue
        v = len(free_vars); free_vars.append(g)
        for i in g:
            var_of[i] = v
    for i in range(n):
        if i not in frozen_q and i not in var_of:
            var_of[i] = len(free_vars); free_vars.append([i])
    nv = len(free_vars)
    P = np.zeros((n, nv))
    for v, g in enumerate(free_vars):
        for i in g:
            P[i, v] = 1.0
    qfix = np.zeros(n)
    for i, qq in frozen_q.items():
        qfix[i] = qq
    M = A @ P; y = esp_val - A @ qfix
    C = []; d = []
    for idx, tot in group_sums:
        C.append(P[idx].sum(0)); d.append(tot - qfix[idx].sum())
    C = np.array(C); d = np.array(d)
    rest_w = np.array([restrained[g[0]] for g in free_vars], float)   # 等价组按其首个原子判断
    count = np.array([len(g) for g in free_vars], float)
    q = np.zeros(nv) if q0 is None else np.array([q0[g[0]] for g in free_vars], float)
    MtM = M.T @ M; Mty = M.T @ y
    for it in range(maxit):
        w = a * rest_w * count / np.sqrt(q ** 2 + B_HYP ** 2)   # 约束的导数 a·q/sqrt(q²+b²) 对每个变量的权重（等价组乘原子数）
        K = MtM + np.diag(w)
        nc = len(d)
        if nc:
            KKT = np.block([[K, C.T], [C, np.zeros((nc, nc))]]); rhs = np.concatenate([Mty, d])
        else:
            KKT = K; rhs = Mty
        sol = np.linalg.solve(KKT, rhs); qn = sol[:nv]
        if np.max(np.abs(qn - q)) < 1e-7:
            q = qn; break
        q = qn
    out = qfix + P @ q
    resid = y - M @ q
    return out, float(np.sqrt(np.mean(resid ** 2))), it + 1


def two_stage(coords, esp_xyz, esp_val, is_heavy, free_mask, frozen_q, equiv_groups, group_sums, qwt1=0.0005, qwt2=0.001, equiv_stage1=None):
    """free_mask(n)：True 表示可变；冻结原子由 frozen_q 给出。equiv_groups：阶段 2 中重新拟合的原子组（组内电荷相等：如甲基 / 亚甲基的氢，以及与它们相连的碳按各自的组）；
    equiv_stage1：阶段 1 就要等价的原子组（如按化学对称性相等的原子、羧酸根的两个氧）。"""
    n = len(coords); restrained1 = np.array([bool(is_heavy[i]) and bool(free_mask[i]) for i in range(n)])
    q1, rms1, it1 = fit(coords, esp_xyz, esp_val, restrained1, frozen_q, equiv=equiv_stage1, group_sums=group_sums, a=qwt1)
    # 阶段 2：等价组的原子（及其所连的重原子）重新拟合，其余冻结在 q1；组内氢共享同一电荷
    in_groups = sorted({i for g in equiv_groups for i in g})
    frozen2 = dict(frozen_q)
    for i in range(n):
        if i not in in_groups and i not in frozen2:
            frozen2[i] = float(q1[i])
    restrained2 = np.array([bool(is_heavy[i]) and (i in in_groups) and bool(free_mask[i]) for i in range(n)])
    # 阶段 2 的分组电荷约束只保留仍然涉及可变原子的组，且目标值按“阶段 1 冻结值”扣除后自动调整
    q2, rms2, it2 = fit(coords, esp_xyz, esp_val, restrained2, frozen2, equiv=equiv_groups, group_sums=[(idx, tot) for idx, tot in group_sums if any(i not in frozen2 for i in idx)], a=qwt2, q0=q1)
    return q1, q2, dict(rms_stage1=rms1, rms_stage2=rms2, iterations=(it1, it2))
