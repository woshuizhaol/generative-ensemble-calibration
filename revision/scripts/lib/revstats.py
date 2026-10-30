"""返修统计工具。凡与原稿统计相关的函数，均逐行对照原脚本实现，并在 docstring 注明出处。"""
import numpy as np
from scipy import stats


def bh_like_null42(pvals):
    """BH 校正，逐字复制原 code/analysis/42_null_baselines.py:34–37 的实现。"""
    pp = np.asarray(pvals, float); o = np.argsort(pp); bh = np.empty_like(pp); n = len(pp)
    for rank, i in enumerate(o):
        bh[i] = min(1, pp[i] * n / (rank + 1))
    for rank in range(n - 2, -1, -1):
        bh[o[rank]] = min(bh[o[rank]], bh[o[rank + 1]])
    return bh


def bh(pvals):
    """标准 Benjamini–Hochberg q 值（与 statsmodels fdr_bh 等价）；NaN 保留为 NaN。"""
    p = np.asarray(pvals, float); out = np.full_like(p, np.nan); m = ~np.isnan(p)
    if m.sum() == 0:
        return out
    from statsmodels.stats.multitest import multipletests
    out[m] = multipletests(p[m], method="fdr_bh")[1]
    return out


def sign_test_greater(d):
    """单侧精确符号检验：正差数 vs 非零差数（与原 Fig2.py:36 一致）。"""
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    npos = int((d > 0).sum()); nnz = int((d != 0).sum())
    return npos, nnz, float(stats.binomtest(npos, nnz, 0.5, alternative="greater").pvalue) if nnz else np.nan


def wilcoxon_greater(a, b):
    """单侧配对 Wilcoxon（a > b），scipy 默认 exact/auto，与原 42/Fig2 调用一致。"""
    r = stats.wilcoxon(a, b, alternative="greater")
    return float(r.statistic), float(r.pvalue)


def tost_paired_t(d, eps=0.05):
    """配对 t-TOST，逐字复制原 figures/scripts/Fig4.py 的实现（±eps 等价界）。"""
    d = np.asarray(d, float); nn = len(d); md = d.mean(); se = d.std(ddof=1) / np.sqrt(nn)
    return float(max(stats.t.sf((md + eps) / se, nn - 1), stats.t.cdf((md - eps) / se, nn - 1)))


def _wplus_exact_cdf(n):
    """W+（Wilcoxon 符号秩统计量）在 H0 下的精确 CDF，s=0..n(n+1)/2（动态规划枚举 2^n 符号组合）。"""
    maxs = n * (n + 1) // 2; cnt = np.zeros(maxs + 1); cnt[0] = 1
    for r in range(1, n + 1):
        new = cnt.copy(); new[r:] += cnt[:-r]; cnt = new
    return np.cumsum(cnt) / cnt.sum()


def hodges_lehmann_ci(d, alpha=0.05):
    """一样本 Hodges–Lehmann 位置估计（Walsh 平均的中位数）及精确 (1−α) CI。
    CI 端点 = 第 C 小与第 C 大的 Walsh 平均（1 基），C = 1 + max{c: P(W+ ≤ c) ≤ α/2}
    （Hollander & Wolfe 经典构造；n=12, α=0.05 时 C=14，m=78 → [W_(14), W_(65)]）。
    单元测试用“CI 内的 θ 恰好不被精确双侧 Wilcoxon 拒绝”做反演核对。"""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    w = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)]); m = len(w)
    hl = float(np.median(w))
    if n <= 25:
        cdf = _wplus_exact_cdf(n)
        C = int(np.sum(cdf <= alpha / 2))          # = 1 + 最大满足条件的 c（c 从 0 计）
        if C < 1:
            return hl, -np.inf, np.inf
        return hl, float(w[C - 1]), float(w[m - C])
    z = stats.norm.ppf(1 - alpha / 2)
    C = int(np.floor(m / 2 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    return hl, float(w[max(C - 1, 0)]), float(w[min(m - C, m - 1)])


def boot_mean_ci(v, B=10000, seed=7, ci=95):
    rng = np.random.default_rng(seed); v = np.asarray(v, float); v = v[~np.isnan(v)]
    bs = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(B)])
    return float(v.mean()), float(np.percentile(bs, (100 - ci) / 2)), float(np.percentile(bs, 100 - (100 - ci) / 2))


def fisher_z(r):
    return np.arctanh(np.clip(np.asarray(r, float), -0.999999, 0.999999))


def dersimonian_laird(effects, variances):
    """随机效应合并（DerSimonian–Laird）。返回 dict(est, se, lo, hi, tau2, I2, Q, p)。"""
    y = np.asarray(effects, float); v = np.asarray(variances, float)
    m = np.isfinite(y) & np.isfinite(v) & (v > 0); y, v = y[m], v[m]
    w = 1 / v; ybar = np.sum(w * y) / np.sum(w); Q = float(np.sum(w * (y - ybar) ** 2)); k = len(y)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w); tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    ws = 1 / (v + tau2); est = float(np.sum(ws * y) / np.sum(ws)); se = float(np.sqrt(1 / np.sum(ws)))
    I2 = float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0
    z = est / se; p = float(2 * stats.norm.sf(abs(z)))
    return dict(est=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se, tau2=float(tau2), I2=I2, Q=Q, k=k, p=p)


def spearman_perm_p(x, y, B=100000, seed=0, alternative="two-sided"):
    """Spearman 置换 p 值（打乱 y）。"""
    rng = np.random.default_rng(seed); x = np.asarray(x, float); y = np.asarray(y, float)
    rx = stats.rankdata(x); ry = stats.rankdata(y)
    obs = np.corrcoef(rx, ry)[0, 1]
    rxc = rx - rx.mean(); ryc = ry - ry.mean(); den = np.sqrt((rxc ** 2).sum() * (ryc ** 2).sum())
    cnt = 0; done = 0; chunk = 10000
    while done < B:
        nb = min(chunk, B - done)
        perm = np.argsort(rng.random((nb, len(y))), axis=1)
        r = (ryc[perm] @ rxc) / den
        if alternative == "two-sided":
            cnt += int((np.abs(r) >= abs(obs) - 1e-12).sum())
        elif alternative == "greater":
            cnt += int((r >= obs - 1e-12).sum())
        else:
            cnt += int((r <= obs + 1e-12).sum())
        done += nb
    return float(obs), float((cnt + 1) / (B + 1))
