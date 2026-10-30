"""Stage 13 共用：ShortMD 副本（r1 = 原 ShortMD，r2、r3 = 本阶段新跑）的路径、日志读取、冻结抽样与窗口抽样、ρ 计算。
约定：
  副本 r4、r5（BioEmu 构象起点，Stage 14）→ "ShortMD_r4"、"ShortMD_r5"，冻结抽样清单 tables/S14_draw_indices_manifest.csv。
  副本名 r1 → 生成器名 "ShortMD"（现有打分 raw_outputs/s4/counts/ShortMD/，冻结抽样 F_ShortMD_<蛋白>_M50.npz，已在 S4 清单）；
        r2、r3 → "ShortMD_r2"、"ShortMD_r3"（S13 打分，冻结抽样 F_ShortMD_r<k>_<蛋白>_M50.npz，清单 tables/S13_draw_indices_manifest.csv）。
  编号偏移一律取 r1 的偏移（S4_parity.csv 的 ShortMD 行）。ρ = 计数平均形式、口径 F、残基层 Spearman（对实验 ΔG_open）。
  帧 i（0 基）对应生产段时间 (i+1)×10 ps。
窗口抽样（预登记 PREREG_replicates.md 第 5 节）：entropy = 20261024 + 5000 + r，spawn_key = (cell_id_T(generator, protein, window), M)，M=50，无放回，R=200；窗口帧数 ≤50 则取全部。"""
import os, hashlib, difflib, warnings
import numpy as np, pandas as pd
import mdtraj as md
from . import ensemble as E, revio

PROJ, REV, INP = revio.load_paths()
REPS = ["r1", "r2", "r3"]
WIN_ENTROPY = 20261024 + 5000
FRAME_DT_PS = 10.0


def gen_of(rep):
    return "ShortMD" if rep == "r1" else f"ShortMD_{rep}"


def md_paths(rep, p):
    """返回该副本轨迹相关文件。r1 的 xtc 是原脚本输出；r2、r3 是 S13 输出。"""
    if rep == "r1":
        d = os.path.join(INP["gen_dir"], "MD_short", p)
        return dict(dir=d, ens_xtc=os.path.join(d, "ensemble.xtc"), ens_pdb=os.path.join(d, "ensemble.pdb"), full_xtc=os.path.join(d, "full.xtc"), full_top=os.path.join(d, "full_top.pdb"),
                    md_log=os.path.join(d, "md.log"), equil_log=None, initial=None, meta=None)
    d = os.path.join(REV, "md_replicates", "md", f"{p}_{rep}")
    return dict(dir=d, ens_xtc=os.path.join(d, "ensemble.xtc"), ens_pdb=os.path.join(d, "ensemble.pdb"), full_xtc=os.path.join(d, "full.xtc"), full_top=os.path.join(d, "final.pdb"),
                md_log=os.path.join(d, "md.log"), equil_log=os.path.join(d, "equil.log"), initial=os.path.join(d, "initial_system.pdb"), meta=os.path.join(d, "md_meta.json"))


def read_md_log(path):
    """StateDataReporter 日志 → DataFrame(step, time_ps, potential_kJ, kinetic_kJ, temperature_K, volume_nm3, density)；缺列为 NaN。
    r1 的 md.log 只有 step / 势能 / 温度，每 50000 步（100 ps）一行，step 从平衡段末（50000）起算。"""
    df = pd.read_csv(path); df.columns = [c.strip('#" ') for c in df.columns]
    ren = {"Step": "step", "Time (ps)": "time_ps", "Potential Energy (kJ/mole)": "potential_kJ", "Kinetic Energy (kJ/mole)": "kinetic_kJ", "Temperature (K)": "temperature_K",
           "Box Volume (nm^3)": "volume_nm3", "Density (g/mL)": "density"}
    df = df.rename(columns=ren)
    for c in ren.values():
        if c not in df:
            df[c] = np.nan
    return df[list(ren.values())]


def prod_time_ps_from_step(step, equil_steps=50000, dt_fs=2.0):
    return (np.asarray(step, float) - equil_steps) * dt_fs / 1000.0


def offset_r1(p):
    return int(E.load_cell("ShortMD", p, "F")["offset"])


def load_rep_cell(rep, p):
    """返回 (cell, offset_by_identity)。cell["offset"]：r1–r3（沉积结构起点）强制为 r1 的偏移，并同时给出按残基身份推得的偏移作核对；
    r4、r5（BioEmu 构象起点；MD 输出的残基号由 1 重新编号，与 BioEmu 原编号不同）直接取按残基身份推得的偏移（与其他新系综相同的做法）。"""
    g = gen_of(rep); cell = E.load_cell(g, p, "F")
    if cell is None:
        return None, None
    off_id = int(cell["offset"])
    if rep not in ("r4", "r5"):
        cell["offset"] = offset_r1(p)
    return cell, off_id


# ------------------------------------------------------------------ 冻结抽样（r2、r3）
def s13_manifest():
    """r2、r3 的清单（S13）与 r4、r5 的清单（S14）合并读取。"""
    fs = [os.path.join(REV, "tables", n) for n in ("S13_draw_indices_manifest.csv", "S14_draw_indices_manifest.csv")]
    ms = [pd.read_csv(f) for f in fs if os.path.exists(f)]
    return pd.concat(ms, ignore_index=True) if ms else None


def load_frozen_draws(rep, p, M=50):
    """r1：E.load_draws（核对 S4 清单 sha）；r2、r3：核对 S13 清单 sha 后读取。返回 (frames[R, M_eff], meta)。"""
    if rep == "r1":
        return E.load_draws("F", "ShortMD", p, M)
    g = gen_of(rep); name = f"F_{g}_{p}_M{M}.npz"; path = os.path.join(REV, "raw_outputs", "s4", "draw_indices", name); m = s13_manifest()
    if m is None or name not in set(m.file.map(os.path.basename)):
        raise RuntimeError(f"{name} 不在 S13 清单里")
    if dict(zip(m.file.map(os.path.basename), m.sha256))[name] != revio.sha256(path):
        raise RuntimeError(f"{name} 的 sha256 与 S13 清单不一致")
    z = np.load(path)
    return z["frames"].astype(int), dict(M_eff=int(z["M_eff"]), n_available=int(z["n_available"]), n_lt_M=bool(z["n_lt_M"]), n_distinct=int(z["n_distinct_draws"]))


def rho_frozen(cell, p, frames):
    """frames: (R, M) 的 frame_idx → R 个 ρ（计数平均）。"""
    return E.rho_draws(cell, p, E.frames_to_cols(cell, frames), "count_avg")


def rho_all_frames(cell, p):
    """不抽样：cell 中全部 ok 帧一次计算。"""
    ncol = cell["NC"].shape[1]
    return float(E.rho_draws(cell, p, np.arange(ncol)[None, :], "count_avg")[0])


def rho_subset_frames(cell, p, frame_ids):
    """给定 frame_idx 集合（全部用），一次计算。"""
    ids = [f for f in frame_ids if f in cell["frame_to_col"]]
    cols = np.array([cell["frame_to_col"][f] for f in ids])[None, :]
    return float(E.rho_draws(cell, p, cols, "count_avg")[0]), len(ids)


def window_cell_id(gen, p, win):
    return int(hashlib.sha256(f"T|{gen}|{p}|{win}".encode()).hexdigest()[:8], 16)


def window_draws(ids, M, R, gen, p, win):
    ids = np.asarray(sorted(ids), dtype=np.int32); n = len(ids)
    if n <= M:
        return ids[None, :].copy()
    cid = window_cell_id(gen, p, win); out = np.empty((R, M), dtype=np.int32)
    for r in range(R):
        rng = np.random.default_rng(np.random.SeedSequence(WIN_ENTROPY + r, spawn_key=(cid, M)))
        out[r] = np.sort(ids[rng.choice(n, size=M, replace=False)])
    return out


def rho_window_matched(cell, p, frame_ids, gen, win, M=50, R=200):
    """窗口内等量 M 帧（无放回）抽样的 ρ 中位数；窗口帧数 ≤M 时等于全部帧的值。返回 (median, n_frames_in_window, n_draws)。"""
    ids = [f for f in frame_ids if f in cell["frame_to_col"]]
    fr = window_draws(ids, M, R, gen, p, win); r = rho_frozen(cell, p, fr)
    return float(np.nanmedian(r)), len(ids), fr.shape[0]


def lnpf_profile(cell, p, frame_ids=None):
    """实验残基集上的 ln PF（计数平均；frame_ids=None 用全部 ok 帧）。返回 (resi, lnpf, y)。"""
    idx, y, resi = E.align_idx(cell, p)
    cols = np.arange(cell["NC"].shape[1]) if frame_ids is None else np.array([cell["frame_to_col"][f] for f in frame_ids if f in cell["frame_to_col"]])
    lp = E.ln_pf(cell["NC"], cell["NH"], cols[None, :], "count_avg")[0][idx]
    return resi, lp, y


def windows():
    """(标签, 起始帧, 结束帧（不含）, 类型)：4 个独立 0.5 ns 窗口与 4 个累积窗口。帧 i 的时间 = (i+1)×10 ps。"""
    ind = [(f"{0.5 * k:.1f}-{0.5 * (k + 1):.1f}ns", 50 * k, 50 * (k + 1), "independent") for k in range(4)]
    cum = [(f"0-{0.5 * (k + 1):.1f}ns", 0, 50 * (k + 1), "cumulative") for k in range(4)]
    return ind + cum


# ------------------------------------------------------------------ 沉积结构对应（Cα）
NORM_RESNAME = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "CYX": "CYS", "CYM": "CYS", "MSE": "MET"}


def chain0_ca(top):
    """第一条含蛋白残基的链上的 Cα 原子序号（A 链 = 打分链）。"""
    for ch in top.chains:
        idx = [at.index for at in ch.atoms if at.name == "CA" and at.residue.is_protein]
        if len(idx) >= 20:
            return idx
    raise RuntimeError("没有找到蛋白链")


def deposited_ca(pdb_id, md_names):
    """沉积结构（第一个模型）第一条蛋白链的 Cα 与 MD A 链的 Cα 按残基名序列比对（difflib）逐一对应。
    返回 (MD 链内 Cα 序号数组, 沉积 Cα xyz[nm]（与前者一一对应）, 匹配数, 沉积链 Cα 数)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); u = md.load(os.path.join(INP["pdb_dir"], f"{pdb_id}.pdb"))[0]
    idx = chain0_ca(u.topology)
    dep_names = [NORM_RESNAME.get(u.topology.atom(i).residue.name, u.topology.atom(i).residue.name) for i in idx]; mdn = [NORM_RESNAME.get(x, x) for x in md_names]
    sm = difflib.SequenceMatcher(None, dep_names, mdn, autojunk=False); pairs = [(i + k, j + k) for i, j, n in sm.get_matching_blocks() for k in range(n)]
    return np.array([j for _, j in pairs]), u.xyz[0][[idx[i] for i, _ in pairs]], len(pairs), len(idx)
