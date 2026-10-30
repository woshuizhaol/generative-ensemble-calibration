"""Stage 10 用的全局稳定性统计工具（观测量、Spearman/Pearson 全套统计、分层置换）。
所有随机性来自 seeded_rng(*tags)：种子 = [20261034, sha256(tags)[:8]]，与计算顺序无关，可单独复现任何一个格子。
  OBS_NAMES / observables(g, RT)  由每帧 g(c) 得到 8 个观测量
  corr_stats(x, y, ...)            Spearman（置换 p、scipy p）、Pearson（t 检验 p）、域层 bootstrap CI、留一 jackknife 区间
  corr_draws(X, y, ...)            X 为 (R, n) 的多次抽样：逐抽样 ρ 与中位数的汇总
  asymmetry_perm(x, y, label, ...) Δρ = ρ(第一类) − ρ(第二类) 的标签置换检验
"""
import hashlib
import numpy as np
from scipy import stats
from scipy.special import logsumexp

ENTROPY = 20261024 + 10
QS = (1, 5, 10, 25, 50)
OBS_NAMES = ["obs_min", "obs_softmin"] + [f"obs_q{q}" for q in QS] + ["obs_mean"]


def seeded_rng(*tags):
    h = int(hashlib.sha256("|".join(str(t) for t in tags).encode()).hexdigest()[:8], 16)
    return np.random.default_rng([ENTROPY, h])


def observables(g, RT):
    """g: (M,) 每个构象内全部已打分残基的 x_i = 0.35·N_C + 2.0·N_H 的均值。返回 dict（单位 kcal/mol）。"""
    g = np.asarray(g, float); M = len(g)
    out = {"obs_min": RT * float(np.min(g)), "obs_softmin": -RT * float(logsumexp(-g) - np.log(M)), "obs_mean": RT * float(np.mean(g))}
    for q in QS:
        out[f"obs_q{q}"] = RT * float(np.percentile(g, q))
    return out


def centered_ranks(v):
    r = stats.rankdata(v, axis=-1); return r - r.mean(axis=-1, keepdims=True)


def spearman_vec(X, y):
    """X: (R, n) 或 (n,)；y: (n,)。返回每行的 Spearman（向量化）。"""
    X = np.atleast_2d(X); rx = centered_ranks(X); ry = centered_ranks(y)
    return (rx @ ry) / (np.sqrt((rx ** 2).sum(1)) * np.sqrt((ry ** 2).sum()))


def _spearman_rows(RX, RY):
    """RX, RY: (B, n) 已是原始值（非秩）；逐行 Spearman。"""
    rx = centered_ranks(RX); ry = centered_ranks(RY)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


_PERM_CACHE = {}


def perm_matrix(n, B, tag):
    key = (n, B, tag)
    if key not in _PERM_CACHE:
        rng = seeded_rng("perm", n, B, tag)
        if len(_PERM_CACHE) >= 5:   # 每个 n 的矩阵最多 B×n×2 字节（n=177、B=1e5 时 35 MB），只保留最近 5 个
            _PERM_CACHE.pop(next(iter(_PERM_CACHE)))
        _PERM_CACHE[key] = np.argsort(rng.random((B, n)), axis=1).astype(np.int16 if n < 32000 else np.int32)
    return _PERM_CACHE[key]


def perm_p_spearman(x, y, B=100000, tag=""):
    """双侧置换 p：|ρ_perm| ≥ |ρ_obs| 的比例（+1 校正）。置换 y 的秩，等价于打乱标签。"""
    rx = centered_ranks(x); ry = centered_ranks(y); den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum()); obs = float((rx * ry).sum() / den)
    P = perm_matrix(len(x), B, tag); null = (ry[P] @ rx) / den
    return obs, float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (B + 1)), null


def corr_stats(x, y, B_perm=100000, B_boot=10000, tag=""):
    """x: 观测量（每个域一个值）；y: ΔG_fold。全部统计量配对返回，列名写明统计量。"""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    rho, p_perm, _ = perm_p_spearman(x, y, B_perm, "shared"); sp = stats.spearmanr(x, y)   # 置换矩阵按 n 共享（tag 只影响 bootstrap 的随机流）
    pr = stats.pearsonr(x, y)
    rng = seeded_rng("boot", tag, n); idx = rng.integers(0, n, size=(B_boot, n)); xb = x[idx]; yb = y[idx]
    rho_b = _spearman_rows(xb, yb)
    xc = xb - xb.mean(1, keepdims=True); yc = yb - yb.mean(1, keepdims=True); r_b = (xc * yc).sum(1) / np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))
    rho_b = rho_b[np.isfinite(rho_b)]; r_b = r_b[np.isfinite(r_b)]
    jk_rho = np.array([stats.spearmanr(np.delete(x, i), np.delete(y, i))[0] for i in range(n)]); jk_r = np.array([stats.pearsonr(np.delete(x, i), np.delete(y, i))[0] for i in range(n)])
    return dict(n=n, spearman_rho=float(rho), spearman_p_perm_two_sided=p_perm, spearman_p_scipy_two_sided=float(sp[1]), spearman_boot_lo=float(np.percentile(rho_b, 2.5)), spearman_boot_hi=float(np.percentile(rho_b, 97.5)),
                spearman_jackknife_min=float(jk_rho.min()), spearman_jackknife_max=float(jk_rho.max()),
                pearson_r=float(pr[0]), pearson_p_t_two_sided=float(pr[1]), pearson_boot_lo=float(np.percentile(r_b, 2.5)), pearson_boot_hi=float(np.percentile(r_b, 97.5)),
                pearson_jackknife_min=float(jk_r.min()), pearson_jackknife_max=float(jk_r.max()))


def corr_draws(X, y, B_perm=100000, B_boot=10000, tag=""):
    """X: (R, n)，每行是一次抽样得到的各域观测量。R=1 时退化为 corr_stats。
    返回 corr_stats（取 ρ 最接近中位数的那次抽样，即“中位那次抽样”）+ 200 次的 ρ 中位数与 2.5/97.5 百分位 + 该次抽样的 draw 编号。"""
    X = np.atleast_2d(np.asarray(X, float)); R = X.shape[0]
    if R == 1:
        s = corr_stats(X[0], y, B_perm, B_boot, tag); s.update(n_draws=1, rho_median=s["spearman_rho"], rho_p2_5=s["spearman_rho"], rho_p97_5=s["spearman_rho"], median_draw=0); return s
    rhos = spearman_vec(X, y); med = float(np.median(rhos)); k = int(np.argmin(np.abs(rhos - med)))
    s = corr_stats(X[k], y, B_perm, B_boot, tag); r_all = np.array([stats.pearsonr(X[i], y)[0] for i in range(R)])
    s.update(n_draws=R, rho_median=med, rho_p2_5=float(np.percentile(rhos, 2.5)), rho_p97_5=float(np.percentile(rhos, 97.5)), median_draw=k,
             pearson_median_over_draws=float(np.median(r_all)), pearson_p2_5_over_draws=float(np.percentile(r_all, 2.5)), pearson_p97_5_over_draws=float(np.percentile(r_all, 97.5)))
    return s


def asymmetry_perm(x, y, is_first, B=100000, tag="", chunk=10000):
    """Δρ = ρ(is_first==True 的子集) − ρ(其余)；打乱 is_first 标签（保持两类样本数）B 次，双侧 p（+1 校正）。"""
    x = np.asarray(x, float); y = np.asarray(y, float); is_first = np.asarray(is_first, bool); n = len(x); n1 = int(is_first.sum())
    obs = float(stats.spearmanr(x[is_first], y[is_first])[0] - stats.spearmanr(x[~is_first], y[~is_first])[0])
    rng = seeded_rng("asym", tag, n, n1); null = np.empty(B); done = 0
    while done < B:
        m = min(chunk, B - done); P = np.argsort(rng.random((m, n)), axis=1); a = P[:, :n1]; b = P[:, n1:]
        null[done:done + m] = _spearman_rows(x[a], y[a]) - _spearman_rows(x[b], y[b]); done += m
    return dict(delta_rho=obs, p_perm_two_sided=float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (B + 1)), p_perm_one_sided_first_gt_second=float((np.sum(null >= obs - 1e-12) + 1) / (B + 1)), null_sd=float(null.std()), n_first=n1, n_second=n - n1)   # 原稿报告的 0.021 / 0.170 是单侧（设计 > 天然）
