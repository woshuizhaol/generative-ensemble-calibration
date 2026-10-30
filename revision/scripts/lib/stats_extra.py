"""S9 用的统计工具。每个函数在 s9_selftest.py 里有自检（对照 scipy 或已知值）。
  signed_rank_cdf(n)           Wilcoxon 符号秩统计量 W+ 的精确分布（动态规划）
  hodges_lehmann(d, conf=0.95) HL 估计（Walsh 平均的中位数）与精确（保守）置信区间
  bh(p)                        Benjamini–Hochberg q 值
  sign_test_greater(d)         精确二项符号检验（单侧，零差舍去）
  dersimonian_laird(y, v)      随机效应荟萃：合并效应、SE、τ²、I²、Q
  zou_diff_ci(r1, r2, r12, n)  Zou (2007) 两个依赖且共享一个变量的相关系数之差的 CI（Fisher-z 尺度的重叠法）
"""
import numpy as np
from scipy import stats


def signed_rank_cdf(n):
    """返回数组 cdf[k] = P(W+ <= k)，k = 0..n(n+1)/2。"""
    top = n * (n + 1) // 2; cnt = np.zeros(top + 1); cnt[0] = 1.0
    for i in range(1, n + 1):
        new = cnt.copy(); new[i:] += cnt[:top + 1 - i]; cnt = new
    return np.cumsum(cnt) / 2.0 ** n


def hodges_lehmann(d, conf=0.95):
    d = np.asarray(d, float); n = len(d); walsh = np.sort(np.array([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])); est = float(np.median(walsh))
    cdf = signed_rank_cdf(n); alpha = (1 - conf) / 2; k = int(np.max(np.where(cdf <= alpha)[0])) if np.any(cdf <= alpha) else -1   # P(W<=k) <= alpha 的最大 k
    N = len(walsh)
    if k < 0:
        return est, float(walsh[0]), float(walsh[-1]), float(2 * cdf[0])
    return est, float(walsh[k]), float(walsh[N - 1 - k]), float(2 * cdf[k])   # 最后一项：实际覆盖的尾概率之和 (<= 1-conf)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = o[rank]; prev = min(prev, p[i] * n / (rank + 1)); q[i] = prev
    return q


def sign_test_greater(d):
    d = np.asarray(d, float); pos = int((d > 0).sum()); n = int((d != 0).sum()); return pos, n, float(stats.binomtest(pos, n, 0.5, alternative="greater").pvalue) if n else (pos, n, np.nan)


def dersimonian_laird(y, v):
    y = np.asarray(y, float); v = np.asarray(v, float); w = 1 / v; mu_fe = (w * y).sum() / w.sum(); Q = float((w * (y - mu_fe) ** 2).sum()); k = len(y)
    c = w.sum() - (w ** 2).sum() / w.sum(); tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0; ws = 1 / (v + tau2); mu = (ws * y).sum() / ws.sum(); se = float(np.sqrt(1 / ws.sum()))
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(mu=float(mu), se=se, lo=float(mu - 1.96 * se), hi=float(mu + 1.96 * se), tau2=float(tau2), I2=float(I2), Q=Q, mu_fixed=float(mu_fe), k=k)


def _z_ci(r, n, bw=False):
    """bw=True：Spearman 的 Fisher-z 方差用 Bonett–Wright (2000) 的 (1 + r²/2)/(n−3)，比 Pearson 的 1/(n−3) 约大 3%–10%。默认 False 以保持自检覆盖率检验针对 Pearson。"""
    z = np.arctanh(r); se = np.sqrt((1 + r ** 2 / 2) / (n - 3)) if bw else 1 / np.sqrt(n - 3); return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def zou_diff_ci(r1, r2, r12, n, bw=False):
    """r1 = corr(X, Y)，r2 = corr(Z, Y)，r12 = corr(X, Z)；返回 r1−r2 的 95% CI（Zou 2007）。bw=True 时对 Spearman 用 Bonett–Wright 方差。"""
    l1, u1 = _z_ci(r1, n, bw); l2, u2 = _z_ci(r2, n, bw)
    lo = (r1 - r2) - np.sqrt((r1 - l1) ** 2 + (u2 - r2) ** 2 - 2 * r12 * (r1 - l1) * (u2 - r2))
    hi = (r1 - r2) + np.sqrt((u1 - r1) ** 2 + (r2 - l2) ** 2 - 2 * r12 * (u1 - r1) * (r2 - l2))
    return float(lo), float(hi)
