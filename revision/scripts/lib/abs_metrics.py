"""绝对标度指标（S6）：逐蛋白 Pearson、RMSE、MAE、平均偏差、OLS 斜率与截距（两个方向）、Lin 一致性相关系数 CCC。向量化：前导维度是重复（bootstrap 或抽样）。"""
import numpy as np


def metrics(pred, obs):
    """pred、obs: (...,n)。返回 dict，每项形状 (...)。"""
    pm = pred.mean(-1, keepdims=True); om = obs.mean(-1, keepdims=True); dp = pred - pm; do = obs - om
    cov = (dp * do).mean(-1); vp = (dp ** 2).mean(-1); vo = (do ** 2).mean(-1); err = pred - obs
    return dict(pearson=cov / np.sqrt(vp * vo), rmse=np.sqrt((err ** 2).mean(-1)), mae=np.abs(err).mean(-1), bias=err.mean(-1), slope_pred_on_exp=cov / vo, intercept_pred_on_exp=pm[..., 0] - cov / vo * om[..., 0],
                slope_exp_on_pred=cov / vp, intercept_exp_on_pred=om[..., 0] - cov / vp * pm[..., 0], ccc=2 * cov / (vp + vo + (pm[..., 0] - om[..., 0]) ** 2))
