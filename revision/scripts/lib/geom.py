"""几何小工具（纯 numpy）：叠合后 RMSD 与成对 RMSD。S5 的 Str2Str 几何 QC 与噪声系综的散布匹配共用。坐标单位随输入（nm 进 nm 出）。"""
import numpy as np


def _center(X):
    return X - X.mean(axis=-2, keepdims=True)


def rmsd_to_ref(X, ref):
    """X: (n, N, 3)，ref: (N, 3)。每帧对 ref 做最优刚体叠合（Kabsch，含反射校正）后的 RMSD，返回 (n,)。"""
    X = _center(np.asarray(X, float)); R = _center(np.asarray(ref, float))
    H = np.einsum("nai,aj->nij", X, R); U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt)); S2 = S.copy(); S2[:, -1] *= d
    e = (X ** 2).sum((1, 2)) + (R ** 2).sum() - 2 * S2.sum(1)
    return np.sqrt(np.clip(e, 0, None) / X.shape[1])


def pairwise_rmsd(X):
    """X: (n, N, 3)。返回 (n, n) 的叠合后成对 RMSD 矩阵。"""
    n = len(X); M = np.zeros((n, n))
    for i in range(n):
        M[i] = rmsd_to_ref(X, X[i])
    return (M + M.T) / 2


def mean_pairwise_rmsd(X):
    M = pairwise_rmsd(X); iu = np.triu_indices(len(X), 1)
    return float(M[iu].mean())
