"""Stage 9 追加（U-S3-2）· 步骤 3：由 Gaussian 的小模型 Hessian 用 Seminario 方法求 Fe 位点的键与角力常数。
输入：qm/small.fchk（`formchk small.chk` 的输出，含优化后的坐标与笛卡尔 Hessian）和 qm/small_atoms.tsv（原子序号 → 标签）。
方法（Seminario 1996, Int. J. Quantum Chem. 60, 1271）：
  键 i–j：取 Hessian 中 (i,j) 的 3×3 子块（取负），其本征值 λ_k 与本征向量 v_k，沿键方向 u 的力常数 k_ij = Σ_k λ_k |u·v_k|；i→j 与 j→i 两个方向取平均。
  角 A–B–C：k_θ^-1 = 1/(R_AB² k_PA) + 1/(R_BC² k_PC)，k_PA = Σ_k λ_k^{AB} |u_PA·v_k^{AB}|，u_PA 为垂直于 AB 且位于 ABC 平面内的方向（对 BC 同理）；
    近似线性的角（A–B–C 夹角 >175°，如 N–Fe–N 反式、N–Fe–S）没有唯一的平面，取两个互相垂直的平面方向的平均。
  单位：Hessian 为 Hartree/Bohr²，1 Hartree/Bohr² = 2240.87 kcal/mol/Å²；Amber 的谐振子形式是 K·(x−x0)²，所以 K_Amber = k/2（键：kcal/mol/Å²；角：kcal/mol/rad²）。
输出：param/seminario_fe_site.tsv（逐键、逐角的 Seminario 值与优化几何），param/seminario_summary.json（按 Amber 原子类型归并后的平均值与标准差）。
用法：python s9d_cytc_heme_seminario.py [--fchk qm/small.fchk]"""
import os, sys, json, argparse, itertools
import numpy as np

BOHR = 0.529177210903; HB2_TO_KCAL_A2 = 627.5094740631 / BOHR ** 2


def read_fchk(path):
    """标准 Gaussian formatted checkpoint：标签占前 40 列，第 44 列是类型字符（I / R / C），带 N= 的是数组。"""
    L = open(path).read().split("\n"); data = {}; i = 1
    while i < len(L):
        l = L[i]
        if "N=" in l[40:]:
            key = l[:40].strip(); n = int(l.split("N=")[1]); vals = []; i += 1
            while len(vals) < n and i < len(L):
                vals += L[i].split(); i += 1
            data[key] = vals; continue
        if len(l) > 49 and l[:40].strip():
            data[l[:40].strip()] = l[49:].strip()
        i += 1
    z = np.array(data["Atomic numbers"], int); xyz = np.array(data["Current cartesian coordinates"], float).reshape(-1, 3)
    nat = len(z); lt = np.array(data["Cartesian Force Constants"], float); H = np.zeros((3 * nat, 3 * nat)); k = 0
    for r in range(3 * nat):
        for c in range(r + 1):
            H[r, c] = H[c, r] = lt[k]; k += 1
    return z, xyz * BOHR, H * HB2_TO_KCAL_A2   # Å、kcal/mol/Å²


def block(H, i, j):
    return -H[3 * i:3 * i + 3, 3 * j:3 * j + 3]


def k_along(Hb, u):
    w, v = np.linalg.eigh(Hb)   # 3×3 子块不对称时用对称化后的本征分解
    return float(np.sum(np.abs(w) * np.abs(v.T @ u)))


def k_bond(H, xyz, i, j):
    u = xyz[j] - xyz[i]; u /= np.linalg.norm(u)
    kij = k_along(0.5 * (block(H, i, j) + block(H, i, j).T), u); kji = k_along(0.5 * (block(H, j, i) + block(H, j, i).T), -u)
    return 0.5 * (kij + kji)


def k_angle(H, xyz, a, b, c):
    uab = xyz[a] - xyz[b]; rab = np.linalg.norm(uab); uab /= rab; ucb = xyz[c] - xyz[b]; rcb = np.linalg.norm(ucb); ucb /= rcb
    theta = np.degrees(np.arccos(np.clip(np.dot(uab, ucb), -1, 1)))
    Hab = block(H, a, b); Hcb = block(H, c, b); Hab = 0.5 * (Hab + Hab.T); Hcb = 0.5 * (Hcb + Hcb.T)

    def one(un):
        upa = np.cross(un, uab); upa /= np.linalg.norm(upa); upc = np.cross(ucb, un); upc /= np.linalg.norm(upc)
        kpa = k_along(Hab, upa); kpc = k_along(Hcb, upc)
        return 1.0 / (1.0 / (rab ** 2 * kpa) + 1.0 / (rcb ** 2 * kpc))
    if theta > 175 or theta < 5:   # 近线性：两个互相垂直的弯曲平面取平均
        t = np.array([1.0, 0, 0]) if abs(uab[0]) < 0.9 else np.array([0, 1.0, 0]); p1 = np.cross(uab, t); p1 /= np.linalg.norm(p1); p2 = np.cross(uab, p1)
        # 平面法向 un 垂直于所取的弯曲方向：弯曲方向 = p1 时 un = p2，反之亦然
        def one_dir(bend):
            upa = bend; upc = -bend
            kpa = k_along(Hab, upa); kpc = k_along(Hcb, upc)
            return 1.0 / (1.0 / (rab ** 2 * kpa) + 1.0 / (rcb ** 2 * kpc))
        return theta, 0.5 * (one_dir(p1) + one_dir(p2))
    un = np.cross(uab, ucb); un /= np.linalg.norm(un)
    return theta, one(un)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fchk", default="qm/small.fchk"); ap.add_argument("--atoms", default="qm/small_atoms.tsv"); ap.add_argument("--out", default="param"); a = ap.parse_args()
    base = os.path.dirname(os.path.abspath(a.fchk)); os.makedirs(a.out, exist_ok=True)
    z, xyz, H = read_fchk(a.fchk); lab = [l.split("\t")[1] for l in open(a.atoms).read().strip().split("\n")[1:]]; assert len(lab) == len(z), (len(lab), len(z))
    fe = 0; assert z[fe] == 26
    lig = [i for i in range(len(z)) if i != fe and np.linalg.norm(xyz[i] - xyz[fe]) < 2.6 and z[i] in (7, 16)]
    rows = []
    for i in lig:
        r = float(np.linalg.norm(xyz[i] - xyz[fe])); k = k_bond(H, xyz, fe, i)
        rows.append(dict(kind="bond", atoms=f"{lab[fe]}-{lab[i]}", idx=f"{fe+1}-{i+1}", geom=r, K_amber=k / 2))
    for i, j in itertools.combinations(lig, 2):
        th, k = k_angle(H, xyz, i, fe, j)
        rows.append(dict(kind="angle", atoms=f"{lab[i]}-{lab[fe]}-{lab[j]}", idx=f"{i+1}-{fe+1}-{j+1}", geom=th, K_amber=k / 2))
    # 配体内：Fe–N–C、Fe–S–C 角
    for i in lig:
        nb = [j for j in range(len(z)) if j not in (fe, i) and z[j] in (6,) and np.linalg.norm(xyz[j] - xyz[i]) < 1.9]
        for j in nb:
            th, k = k_angle(H, xyz, fe, i, j); rows.append(dict(kind="angle", atoms=f"{lab[fe]}-{lab[i]}-{lab[j]}", idx=f"{fe+1}-{i+1}-{j+1}", geom=th, K_amber=k / 2))
    import csv
    with open(os.path.join(a.out, "seminario_fe_site.tsv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "atoms", "idx", "geom", "K_amber"], delimiter="\t"); w.writeheader(); [w.writerow({**r, "geom": round(r["geom"], 3), "K_amber": round(r["K_amber"], 2)}) for r in rows]
    for r in rows:
        print(f"{r['kind']:<5} {r['atoms']:<34} geom={r['geom']:8.3f}  K={r['K_amber']:9.2f}")
