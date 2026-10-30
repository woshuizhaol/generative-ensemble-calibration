"""系综读取、与实验残基对齐、两种平均形式、逐抽样 Spearman（S7–S10 共用）。

load_cell(generator, protein, basis)  → dict(res, NC, NH, offset, resn)
    basis "O"：旧 operator_f.npz 的列（P/gen/<generator>/<protein>/）
    basis "F"：S4a 的逐构象 parquet（只含状态 ok / ok_retry 的帧；列 = frame_idx）
    新系综（Str2Str_xtal_s<k>、Str2Str_AF_d035/d025、ESMFlow_T3new 等）：S4b 的 parquet，basis 固定为 "F"
exp_residues(protein)                 → 实验残基表（S1 终表，dGopen_kcal 非空）
ln_pf(NC, NH, cols, form)             → (R, n_res) 的逐残基 ln PF，cols 是 (R, M) 的列索引；form ∈ {"count_avg", "invpf_avg"}
rho_draws(cell, protein, draws, form) → (R,) Spearman（与 leaderboard 同一套对齐：resi = resSeq − offset）
所有函数不修改输入；NaN（个别帧上缺定义的残基）只对有定义的帧平均，与 bv.aggregate 一致。
"""
import os, json
import numpy as np, pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata
from . import bv, revio

PROJ, REV, INP = revio.load_paths()
BC, BH = bv.BETA_C, bv.BETA_H
_S1 = None; _PAR = None


def s1():
    global _S1
    if _S1 is None:
        _S1 = pd.read_parquet(os.path.join(REV, "tables", "S1_tier1a_residues_final.parquet"))
    return _S1


def parity():
    global _PAR
    if _PAR is None:
        _PAR = pd.read_csv(os.path.join(REV, "tables", "S4_parity.csv")).dropna(subset=["offset"])
    return _PAR


def exp_residues(p):
    return s1()[s1().protein_id == p].dropna(subset=["dGopen_kcal"])


def load_cell(g, p, basis="O"):
    """返回 dict(res, NC, NH, offset, resn)。res 为 resSeq 数组，NC/NH 形状 (n_res, n_cols)。"""
    if basis == "O":
        z = np.load(os.path.join(INP["gen_dir"], g, p, "operator_f.npz"))
        res = np.array([int(x) for x in z["resSeq"]]); NC = np.asarray(z["NC"], float); NH = np.asarray(z["NH"], float); resn = None
    else:
        f = os.path.join(REV, "raw_outputs", "s4", "counts", g, f"{p}.parquet")
        if not os.path.exists(f):
            return None
        R = pd.read_parquet(f); F = pd.read_parquet(os.path.join(REV, "raw_outputs", "s4", "counts", g, f"{p}_frames.parquet"))
        ok = np.sort(F[F.status.isin(["ok", "ok_retry"])].frame_idx.values)
        nc = R.pivot(index="resSeq", columns="frame_idx", values="NC")[ok]; nh = R.pivot(index="resSeq", columns="frame_idx", values="NH")[ok]
        res = nc.index.values.astype(int); NC = nc.values.astype(float); NH = nh.values.astype(float); resn = R.drop_duplicates("resSeq").set_index("resSeq").resn.to_dict()
        cols_map = {int(c): i for i, c in enumerate(ok)}
    P = parity(); q = P[(P.generator == g) & (P.protein == p)]
    if len(q):
        off = int(q.offset.iloc[0])
    elif resn is not None:   # 新系综：按残基身份在 [−8, 8] 内取最佳偏移
        off = bv.best_offset({int(r): bv.to1(resn[int(r)]) for r in res}, exp_residues(p), 0)
    else:
        raise KeyError((g, p))
    out = dict(res=res, NC=NC, NH=NH, offset=off, resn=resn)
    if basis != "O":
        out["frame_to_col"] = cols_map
    return out


def ln_pf(NC, NH, cols, form):
    """cols: (R, M) 列索引（可含重复列无）。返回 (R, n_res)。"""
    if form == "count_avg":
        # 与原稿 Eq. 1 和 bv.aggregate 完全相同的运算顺序：先平均计数，再乘系数相加。整数计数的和是精确的，所以计数相同的残基得到精确相等（并列）的 ln PF；
        # 若改成"逐帧 ln PF 再平均"，数学上相等，但 1e-14 量级的浮点噪声会打散并列，使 Spearman 的平均秩改变（实测最大差 0.002，对不上 leaderboard）。
        a = NC[:, cols]; b = NH[:, cols]
        return (BC * np.nanmean(a, axis=-1) + BH * np.nanmean(b, axis=-1)).T
    x = BC * NC[:, cols] + BH * NH[:, cols]            # (n_res, R, M)
    valid = np.isfinite(x); cnt = valid.sum(-1)
    if form == "invpf_avg":
        out = -(logsumexp(np.where(valid, -x, -np.inf), axis=-1) - np.log(np.maximum(cnt, 1)))
    else:
        raise ValueError(form)
    out = np.where(cnt > 0, out, np.nan)
    return out.T


def w_max(NC, NH, cols):
    """1/PF 平均中贡献最大的单个构象所占比例，(R, n_res)。"""
    x = BC * NC[:, cols] + BH * NH[:, cols]; a = np.where(np.isfinite(x), -x, -np.inf); a = a - a.max(-1, keepdims=True)
    e = np.exp(a); return (e.max(-1) / e.sum(-1)).T


def align_idx(cell, p):
    """实验残基在 cell.res 里的位置与实验值：resi = resSeq − offset。返回 (idx, y, resi)。"""
    e = exp_residues(p); pos = {int(r) - cell["offset"]: i for i, r in enumerate(cell["res"])}
    resi = [int(r) for r in e.resi if int(r) in pos]; ev = dict(zip(e.resi.astype(int), e.dGopen_kcal.astype(float)))
    return np.array([pos[r] for r in resi]), np.array([ev[r] for r in resi]), np.array(resi)


def spearman_rows(P, y):
    """P: (R, n)，y: (n,)。逐行 Spearman（含并列的平均秩）；有非有限值的行返回 NaN。"""
    ok = np.isfinite(P).all(1); out = np.full(P.shape[0], np.nan)
    if ok.any():
        rp = rankdata(P[ok], axis=1); ry = rankdata(y); rp = rp - rp.mean(1, keepdims=True); ry = ry - ry.mean()
        out[ok] = (rp @ ry) / np.sqrt((rp ** 2).sum(1) * (ry ** 2).sum())
    return out


def rho_draws(cell, p, cols, form):
    idx, y, _ = align_idx(cell, p)
    return spearman_rows(ln_pf(cell["NC"], cell["NH"], cols, form)[:, idx], y)


_DRAW_SHA = {}


def _verify_draw_file(basis, path):
    """读取冻结索引前，核对文件 sha256 与 S4a 清单一致（冻结后的索引不得被改动）。"""
    if basis not in _DRAW_SHA:
        m = pd.read_csv(os.path.join(REV, "tables", f"S4_draw_indices_manifest_{basis}.csv")); _DRAW_SHA[basis] = dict(zip(m.file.map(os.path.basename), m.sha256))
    name = os.path.basename(path)
    if name not in _DRAW_SHA[basis]:
        raise RuntimeError(f"冻结的抽样索引 {name} 不在 S4a 清单 (basis {basis}) 里")
    if _DRAW_SHA[basis][name] != revio.sha256(path):
        raise RuntimeError(f"冻结的抽样索引 {name} 的 sha256 与 S4a 清单不一致")


def np_load_verified(basis, path):
    """np.load 之前核对冻结索引的 sha256（供不经 load_draws 直接读索引文件的脚本使用）。"""
    _verify_draw_file(basis, path); return np.load(path)


def load_draws(basis, g, p, M):
    path = os.path.join(REV, "raw_outputs", "s4", "draw_indices", f"{basis}_{g}_{p}_M{M}.npz"); _verify_draw_file(basis, path); z = np.load(path)
    return z["frames"].astype(int), dict(M_eff=int(z["M_eff"]), n_available=int(z["n_available"]), n_lt_M=bool(z["n_lt_M"]), n_distinct=int(z["n_distinct_draws"]))


def frames_to_cols(cell, frames):
    """口径 F：把 frame_idx 的抽样索引转成 parquet 列号。"""
    m = cell["frame_to_col"]; return np.vectorize(m.__getitem__)(frames)


def load_cell_sasa(g, p):
    """口径 F：与 load_cell(g, p, "F") 同一套残基、同一套列（ok / ok_retry 帧），额外返回 rSASA_res 与 SASA_amideNH（形状同 NC）。"""
    cell = load_cell(g, p, "F")
    if cell is None:
        return None
    f = os.path.join(REV, "raw_outputs", "s4", "counts", g, f"{p}.parquet"); R = pd.read_parquet(f, columns=["frame_idx", "resSeq", "rSASA_res", "SASA_amideNH"])
    cols = sorted(cell["frame_to_col"], key=cell["frame_to_col"].get)
    cell["rSASA"] = R.pivot(index="resSeq", columns="frame_idx", values="rSASA_res")[cols].reindex(cell["res"]).values.astype(float)
    cell["amideSASA"] = R.pivot(index="resSeq", columns="frame_idx", values="SASA_amideNH")[cols].reindex(cell["res"]).values.astype(float)
    return cell
