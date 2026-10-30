"""Stage 4a · 统一重打分（revision_proposal.md Stage 4）：逐构象的 N_C、N_H、rSASA、酰胺 SASA，带帧索引，失败帧不静默丢弃。

范围（S4a：只处理已经在盘上的系综与沉积结构）：
  s5       S4b：S5 新生成的系综与噪声系综（Str2Str_xtal_s<k>、Str2Str_AF_d035/d025、ESMFlow/Boltz2/BioEmu_T3new、noise_<类型>_<条件>_s<k>）；--only 支持以 * 结尾的前缀
  tier1a   BioEmu、ESMFlow、Boltz2、Str2Str（旧版，起点为 BioEmu 首帧）、ShortMD × 12 个蛋白，全部 xtc 帧（口径 F）
  crystal  12 个蛋白的沉积结构 × 10 次独立的 PDBFixer 重建（recon_k = 0…9）
  t3       D_legacy65 ∪ D_new 中已有的系综（BioEmu_T3、ESMFlow_T3、Boltz2_T3、Str2Str_T3、Str2Str_T3_d035），全部帧
S4b（S5 新生成的系综、噪声系综）用同一脚本、同一参数，届时在 GROUPS 里加组即可。

方法（与原稿可比的部分一字不改）：
  fix_frame / nc_nh 复用 lib/bv.py（逐字复制自原 operator/forward.py），几何阈值 0.65 / 0.26 nm、|i-j|≥3 / ≥2。
  新增：rSASA_res（mdtraj shrake_rupley residue 模式，探针 0.14 nm、960 个点，除以 Tien et al. 2013 的理论最大 ASA）与 SASA_amideNH（酰胺 N 加酰胺 H 的原子级 SASA）。
与原打分的差别（都是方案规定的）：
  · 单帧超时 40 s → 900 s；超时或报错的帧换种子重试 1 次；仍失败的写进帧状态表，不丢弃；
  · 每帧用确定性种子：seed = sha256(生成器|蛋白|frame_idx|recon_k|attempt) 的前 8 位十六进制 mod 2^31，同时传给 random.seed（PDBFixer/Modeller 放氢用
    Python 全局 random）和 PDBFixer.addMissingAtoms(seed=…)（补出的侧链原子碰撞时用 Langevin 动力学拆开，积分器种子默认随机），所以重跑逐位一致；
  · 不再做 MAXFRAMES 抽稀（原脚本对 >MAXFRAMES 的系综做 np.linspace 抽稀），全部帧都打分。
并行：每个任务（≤数百秒的帧块）在全新进程里跑（max_tasks_per_child=1），避免 OpenMM CPU 平台的线程在长寿命进程里累积；CUDA_VISIBLE_DEVICES 置空，
并断言 OpenMM 实际平台为 CPU。分片落盘，已完成的分片自动跳过（可重入）。

用法：
  python s4_rescore.py plan   [--groups tier1a,crystal,t3]                 打印任务与成本估计，不执行
  python s4_rescore.py run    [--groups …] [--workers W] [--only gen1,gen2] 执行
  python s4_rescore.py merge  [--groups …]                                  把分片合并成 counts/<generator>/<protein>.parquet 与 _frames.parquet，并做完整性检查
输出：raw_outputs/s4/parts/…（分片）、raw_outputs/s4/counts/<generator>/<protein>.parquet、…/<protein>_frames.parquet
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 打分是纯 CPU 流程；须在 import 任何 openmm/pdbfixer 之前设置，spawn 子进程会继承
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, time, json, glob, hashlib, argparse, signal
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

TIMEOUT_S = 900
CHUNK_TARGET_S = 480            # 每个任务的估计工作量（秒）；只影响负载均衡与断点粒度
TIER1A_GENS = ["BioEmu", "ESMFlow", "Boltz2", "Str2Str", "ShortMD"]
T3_GENS = ["BioEmu_T3", "ESMFlow_T3", "Boltz2_T3", "Str2Str_T3", "Str2Str_T3_d035"]
BACKBONE_TYPE = {"BioEmu", "Str2Str", "BioEmu_T3", "Str2Str_T3", "Str2Str_T3_d035"}   # 主链加 Cβ 帧：补全全部侧链，慢（S3 实测）
DETERMINISTIC_COLS = ["generator", "protein", "frame_idx", "recon_k", "resSeq", "resn", "NC", "NH", "rSASA_res", "SASA_amideNH"]


def det_seed(generator, protein, frame_idx, recon_k, attempt, rep=0):
    """rep=0 是正式重打分（与此前逐位相同）；rep≥1 只用于 s4_repeat_noise.py 的独立重复，量化 PDBFixer 随机性。"""
    h = hashlib.sha256((f"{generator}|{protein}|{frame_idx}|{recon_k}|{attempt}" + (f"|rep{rep}" if rep else "")).encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 2) + 1


def is_backbone(generator):
    return generator in BACKBONE_TYPE or generator.startswith(("Str2Str_xtal", "Str2Str_AF", "BioEmu_T3new"))


def est_cost_s(generator, n_res):
    """每帧估计耗时（秒），只用于分块与排序。依据 S3 基准：主链加 Cβ 帧 ≈ 9.56·(L/86)^2.58，全重原子帧 ≈ 0.82·(L/86)^0.97；ShortMD 含氢帧按全重原子的 1.5 倍。"""
    if is_backbone(generator):
        return 9.56 * (n_res / 86.0) ** 2.58
    c = 0.82 * (n_res / 86.0) ** 0.97
    return c * 1.5 if generator == "ShortMD" else c


# ------------------------------------------------------------------ 任务清单
def build_items(groups, INP, REV):
    items = []   # dict(group, generator, protein, kind, path_xtc, path_top, n_frames, n_res, recon_ks)
    import mdtraj as md
    seqs = json.load(open(INP["tier1a_sequences"]))
    if "tier1a" in groups:
        for g in TIER1A_GENS:
            for p in seqs:
                d = os.path.join(INP["gen_dir"], g, p)
                with md.open(os.path.join(d, "samples.xtc")) as f:
                    n = len(f)
                items.append(dict(group="tier1a", generator=g, protein=p, kind="xtc", xtc=os.path.join(d, "samples.xtc"), top=os.path.join(d, "topology.pdb"),
                                  n_frames=n, n_res=seqs[p]["length"]))
    if "crystal" in groups:
        nb = pd.read_csv(os.path.join(REV, "tables", "S1_deposited_structure_numbering.csv"))
        for r in nb.itertuples():
            items.append(dict(group="crystal", generator="Crystal", protein=r.protein, kind="pdb", pdb=os.path.join(INP["pdb_dir"], f"{r.pdb}.pdb"),
                              n_frames=10, n_res=seqs[r.protein]["length"], pdb_id=r.pdb))
    if "t3" in groups:
        L = pd.read_csv(os.path.join(REV, "tables", "S1_tier3_legacy65_provenance.csv"))
        U = L[L.in_legacy65 | L.in_Dnew]
        for g in T3_GENS:
            for r in U.itertuples():
                d = os.path.join(INP["gen_dir"], g, r.T3_id)
                if not os.path.exists(os.path.join(d, "samples.xtc")):
                    continue
                with md.open(os.path.join(d, "samples.xtc")) as f:
                    n = len(f)
                items.append(dict(group="t3", generator=g, protein=r.T3_id, kind="xtc", xtc=os.path.join(d, "samples.xtc"), top=os.path.join(d, "topology.pdb"),
                                  n_frames=n, n_res=int(r.length)))
    if "s5" in groups:
        items += s5_items(INP, REV)
    return items


S5_MISSING_NOISE = []   # 尚未完成的噪声条件（plan/run 时打印，不静默）


def s5_items(INP, REV):
    """S4b：S5 新生成的系综与噪声系综。生成器名：Str2Str_xtal_s<种子>、Str2Str_AF_d035/d025、ESMFlow_T3new、Boltz2_T3new、BioEmu_T3new、noise_<anm|cart>_<条件>_s<种子>（含 czero）。
    protein = 真实蛋白名（噪声、Str2Str_xtal）或域编号（T3 类）。"""
    import mdtraj as md
    G = os.path.join(REV, "raw_outputs", "s5", "gen"); N = os.path.join(REV, "raw_outputs", "s5", "noise_work", "xtc")
    dom = pd.read_csv(os.path.join(REV, "tables", "S5_domain_table.csv")).set_index("domain_id"); seqs = json.load(open(INP["tier1a_sequences"])); out = []

    def add(gen, prot, xtc, top, n_res):
        with md.open(xtc) as fh:
            n = len(fh)
        out.append(dict(group="s5", generator=gen, protein=prot, kind="xtc", xtc=xtc, top=top, n_frames=n, n_res=int(n_res)))
    for p in sorted(seqs):
        for k in (11, 12, 13, 14, 15):
            d = os.path.join(G, "Str2Str_xtal", p, f"seed{k}"); add(f"Str2Str_xtal_s{k}", p, os.path.join(d, "samples.xtc"), os.path.join(d, "topology.pdb"), seqs[p]["length"])
        top = os.path.join(N, p, "topology_heavy.pdb"); tdir = os.path.join(REV, "raw_outputs", "s5", "noise_work", "tables")
        for typ in ("anm", "cart"):
            for cond in ("BioEmu", "ESMFlow", "Boltz2", "ShortMD", "Str2Str_xtal", "czero"):
                files = [os.path.join(N, p, typ, cond, f"seed{k}.xtc") for k in (21, 22, 23, 24, 25)]
                # 只收录噪声任务已完成（有 __done 标记）且 5 个种子文件齐全的条件，避免把生成到一半的系综当成完整的去打分
                if not (os.path.exists(os.path.join(tdir, f"{p}__{typ}__{cond}__done")) and all(os.path.exists(x) for x in files)):
                    S5_MISSING_NOISE.append((p, typ, cond)); continue
                for k, x in zip((21, 22, 23, 24, 25), files):
                    add(f"noise_{typ}_{cond}_s{k}", p, x, top, seqs[p]["length"])
    for gen, flag in (("ESMFlow_T3new", "need_ESMFlow_T3"), ("Boltz2_T3new", "need_Boltz2_T3"), ("BioEmu_T3new", "need_BioEmu_T3")):
        for did in dom[dom[flag]].index:
            d = os.path.join(G, gen, did); add(gen, did, os.path.join(d, "samples.xtc"), os.path.join(d, "topology.pdb"), dom.loc[did, "length"])
    for tag in ("035", "025"):
        for did in dom[dom.need_Str2Str_AF].index:
            d = os.path.join(G, f"Str2Str_AF_d{tag}", did, "seed11"); add(f"Str2Str_AF_d{tag}", did, os.path.join(d, "samples.xtc"), os.path.join(d, "topology.pdb"), dom.loc[did, "length"])
    return out


def match_gen(g, keep):
    return any(g == k or (k.endswith("*") and g.startswith(k[:-1])) for k in keep)


def build_tasks(items):
    tasks = []
    for it in items:
        c = est_cost_s(it["generator"], it["n_res"])
        per = int(np.clip(CHUNK_TARGET_S // max(c, 0.05), 1, 100))
        if it["kind"] == "pdb":
            for k in range(10):
                tasks.append(dict(it, recon_k=k, f0=0, f1=1, est_s=c))
        else:
            for f0 in range(0, it["n_frames"], per):
                tasks.append(dict(it, recon_k=-1, f0=f0, f1=min(f0 + per, it["n_frames"]), est_s=c * (min(f0 + per, it["n_frames"]) - f0)))
    return sorted(tasks, key=lambda t: -t["est_s"])


def part_paths(REV, t):
    root = t.get("parts_root") or os.path.join(REV, "raw_outputs", "s4", "parts")
    base = os.path.join(root, t["generator"], t["protein"])
    if "frame_list" in t:   # s4_repeat_noise.py：只重打分指定帧
        tag = f"c{t['chunk']:04d}"
    else:
        tag = f"r{t['recon_k']:02d}" if t["kind"] == "pdb" else f"f{t['f0']:05d}-{t['f1']:05d}"
    return os.path.join(base, tag + ".parquet"), os.path.join(base, tag + "_frames.parquet")


# ------------------------------------------------------------------ 执行
class FrameTimeout(Exception):
    pass


def _alarm(sig, frm):
    raise FrameTimeout()


def run_task(args):
    t, REV, max_asa_path = args
    p_res, p_frm = part_paths(REV, t)
    if os.path.exists(p_res) and os.path.exists(p_frm):
        return p_res, "skipped"
    import mdtraj as md
    import openmm as mm
    sys.path.insert(0, os.path.join(REV, "scripts"))
    from lib import bv
    _s = mm.System(); _s.addParticle(1.0)
    plat = mm.Context(_s, mm.VerletIntegrator(0.001)).getPlatform().getName()
    if plat != "CPU":
        raise SystemExit(f"OpenMM 平台为 {plat}，不是 CPU；中止以免占用他人的卡")
    max_asa = bv.load_max_asa(max_asa_path)
    if t["kind"] == "pdb":
        tr = md.load(t["pdb"]); tr = tr[0] if tr.n_frames > 1 else tr
        frames = [(0, tr)]
    else:
        tr = md.load(t["xtc"], top=t["top"])
        frames = [(k, tr[k]) for k in (t["frame_list"] if "frame_list" in t else range(t["f0"], t["f1"]))]
    rep = t.get("rep", 0)
    signal.signal(signal.SIGALRM, _alarm)
    res_rows, frm_rows = [], []
    for k, fr in frames:
        t_all0 = time.perf_counter(); status = "error"; err = ""; seed = 0; used = 0; tf = tc = ts = np.nan; rows = None
        for attempt in (0, 1):
            seed = det_seed(t["generator"], t["protein"], k, t["recon_k"], attempt, rep); used = attempt + 1
            try:
                signal.alarm(TIMEOUT_S)
                a = time.perf_counter(); t1 = bv.fix_frame(fr, True, seed=seed); tf = time.perf_counter() - a
                a = time.perf_counter(); nn = bv.nc_nh(t1); tc = time.perf_counter() - a
                a = time.perf_counter(); sf = bv.sasa_features(t1, max_asa); ts = time.perf_counter() - a
                signal.alarm(0)
                if set(nn) != set(sf):
                    raise RuntimeError(f"nc_nh 与 sasa_features 的残基集合不一致: {len(nn)} vs {len(sf)}")
                rows = [(t["generator"], t["protein"], k, t["recon_k"], rs, nn[rs][0], nn[rs][1], nn[rs][2], sf[rs][0], sf[rs][1]) for rs in sorted(nn)]
                status = "ok" if attempt == 0 else "ok_retry"; err = ""
                break
            except FrameTimeout:
                signal.alarm(0); status = "timeout"; err = f"> {TIMEOUT_S} s (attempt {attempt + 1})"
            except Exception as e:   # 失败帧不丢弃：记入帧状态表
                signal.alarm(0); status = "error"; err = f"{type(e).__name__}: {str(e)[:150]} (attempt {attempt + 1})"
        if rows is not None:
            res_rows += rows
        frm_rows.append(dict(generator=t["generator"], protein=t["protein"], frame_idx=k, recon_k=t["recon_k"], status=status, attempts=used, seed=seed,
                             n_res_scored=len(rows) if rows is not None else 0, t_fix_s=tf, t_count_s=tc, t_sasa_s=ts, t_wall_all_attempts_s=time.perf_counter() - t_all0,
                             err=err, pid=os.getpid()))
    R = pd.DataFrame(res_rows, columns=["generator", "protein", "frame_idx", "recon_k", "resSeq", "resn", "NC", "NH", "rSASA_res", "SASA_amideNH"])
    R = R.astype(dict(frame_idx="int32", recon_k="int8", resSeq="int16", NC="int16", NH="int16", rSASA_res="float32", SASA_amideNH="float32"))
    F = pd.DataFrame(frm_rows).astype(dict(frame_idx="int32", recon_k="int8", attempts="int8", seed="int64", n_res_scored="int16"))
    os.makedirs(os.path.dirname(p_res), exist_ok=True)
    for df, p in ((R, p_res), (F, p_frm)):
        df.to_parquet(p + ".tmp", index=False); os.replace(p + ".tmp", p)   # 先写临时文件再改名：存在即完成
    return p_res, "done"


def cmd_plan(a, INP, REV):
    items = build_items(a.groups.split(","), INP, REV); tasks = build_tasks(items)
    if S5_MISSING_NOISE:
        print(f"注意：噪声系综还有 {len(S5_MISSING_NOISE)}/{12 * 2 * 6} 个（蛋白×类型×条件）没有完成，不在本次任务里")
    df = pd.DataFrame([dict(group=t["group"], generator=t["generator"], est_s=t["est_s"], frames=(t["f1"] - t["f0"])) for t in tasks])
    g = df.groupby(["group", "generator"]).agg(tasks=("est_s", "size"), frames=("frames", "sum"), est_core_h=("est_s", lambda x: x.sum() / 3600)).round(2)
    print(g.to_string()); print("总计: tasks", len(tasks), "frames", int(df.frames.sum()), "estimated core-h", round(df.est_s.sum() / 3600, 1))
    done = sum(os.path.exists(part_paths(REV, t)[0]) and os.path.exists(part_paths(REV, t)[1]) for t in tasks)
    print("已完成分片:", done, "/", len(tasks))


def cmd_run(a, INP, REV):
    items = build_items(a.groups.split(","), INP, REV)
    if a.only:
        keep = a.only.split(","); items = [i for i in items if match_gen(i["generator"], keep)]
    tasks = [t for t in build_tasks(items) if not (os.path.exists(part_paths(REV, t)[0]) and os.path.exists(part_paths(REV, t)[1]))]
    if a.reverse:   # 第二个任务池：从便宜的任务倒着做，与第一个池在中间会合（run_task 开头检查分片是否存在，已完成的会跳过）
        tasks = tasks[::-1]
    log = revio.RunLogger(stage=4, task=f"rescore_{a.groups.replace(',', '_')}", cpu_procs=a.workers, seeds="per-frame sha256 deterministic seeds")
    print(f"待执行任务 {len(tasks)}，估计 {sum(t['est_s'] for t in tasks) / 3600:.1f} core-h，workers={a.workers}，load={os.getloadavg()}", flush=True)
    max_asa_path = os.path.join(REV, "data", "external", "tien2013_maxasa", "tien2013_theoretical_maxASA.csv")
    t0 = time.time(); n_done = 0; fails = []
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_task, (t, REV, max_asa_path)): t for t in tasks}
        for fu in cf.as_completed(futs):
            t = futs[fu]
            try:
                fu.result(); n_done += 1
            except Exception as e:
                fails.append((t["generator"], t["protein"], t["f0"], repr(e)[:200])); print("任务失败:", fails[-1], flush=True)
            if n_done % 25 == 0 or n_done == len(tasks):
                print(f"[{time.time() - t0:.0f}s] {n_done}/{len(tasks)} 任务完成，load={os.getloadavg()[0]:.1f}", flush=True)
    log.finish([os.path.join(REV, "raw_outputs", "s4", "parts")], exit_code=0 if not fails else 1, status="ok" if not fails else "partial",
               note=f"tasks={len(tasks)} failed_tasks={len(fails)} wall={time.time() - t0:.0f}s")
    if fails:
        raise SystemExit(f"{len(fails)} 个任务失败")


def cmd_merge(a, INP, REV):
    req = a.groups.split(","); all_items = build_items(sorted(set(["tier1a", "crystal", "t3"]) | (set(req) & {"s5"})), INP, REV)
    items = [i for i in all_items if i["group"] in req]
    if a.only:
        keep = a.only.split(","); items = [i for i in items if match_gen(i["generator"], keep)]
    log = revio.RunLogger(stage=4, task="rescore_merge", cpu_procs=1, seeds="none")
    out_dir = os.path.join(REV, "raw_outputs", "s4", "counts")
    for it in items:
        base = os.path.join(REV, "raw_outputs", "s4", "parts", it["generator"], it["protein"])
        rs = sorted(glob.glob(os.path.join(base, "*.parquet")))
        res_files = [f for f in rs if not f.endswith("_frames.parquet")]; frm_files = [f for f in rs if f.endswith("_frames.parquet")]
        if not res_files and not frm_files:
            raise SystemExit(f"{it['generator']}/{it['protein']}: 没有分片，请先 run")
        R = pd.concat([pd.read_parquet(f) for f in res_files], ignore_index=True) if res_files else pd.DataFrame(columns=DETERMINISTIC_COLS)
        F = pd.concat([pd.read_parquet(f) for f in frm_files], ignore_index=True)
        exp = set(range(it["n_frames"])) if it["kind"] == "xtc" else set(range(10))
        got = set(F["frame_idx"]) if it["kind"] == "xtc" else set(F["recon_k"])
        dup = int(F.duplicated(["frame_idx", "recon_k"]).sum())
        if got != exp or dup:
            raise SystemExit(f"{it['generator']}/{it['protein']}: 帧不完整或重复（缺 {sorted(exp - got)[:5]}，重复 {dup}）")
        ok = F.status.isin(["ok", "ok_retry"])
        okset = set(zip(R.frame_idx.astype(int), R.recon_k.astype(int)))
        assert all((int(r.frame_idx), int(r.recon_k)) in okset for r in F[ok].itertuples()), "状态为 ok 的帧在残基表里没有行"
        d = os.path.join(out_dir, it["generator"]); os.makedirs(d, exist_ok=True)
        R.sort_values(["frame_idx", "recon_k", "resSeq"]).to_parquet(os.path.join(d, f"{it['protein']}.parquet"), index=False)
        F.sort_values(["frame_idx", "recon_k"]).to_parquet(os.path.join(d, f"{it['protein']}_frames.parquet"), index=False)
    # 汇总表：每次扫描全部已合并的结果重新生成（增量合并也能得到完整表）
    grp = {(i["generator"], i["protein"]): i["group"] for i in all_items}; summ = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*", "*_frames.parquet"))):
        g = os.path.basename(os.path.dirname(f)); p = os.path.basename(f)[:-len("_frames.parquet")]; F = pd.read_parquet(f)
        bad = int((~F.status.isin(["ok", "ok_retry"])).sum())
        summ.append(dict(group=grp.get((g, p), "?"), generator=g, protein=p, n_xtc_frames_or_recons=len(F), n_ok=int((F.status == "ok").sum()),
                         n_ok_retry=int((F.status == "ok_retry").sum()), n_timeout=int((F.status == "timeout").sum()), n_error=int((F.status == "error").sum()),
                         fail_rate=round(bad / len(F), 5), mean_t_fix_s=round(float(F.t_fix_s.mean()), 3), median_t_fix_s=round(float(F.t_fix_s.median()), 3),
                         mean_t_count_s=round(float(F.t_count_s.mean()), 4), mean_t_sasa_s=round(float(F.t_sasa_s.mean()), 4), total_core_s=round(float(F.t_wall_all_attempts_s.sum()), 1)))
    S = pd.DataFrame(summ); S4A = {"tier1a", "crystal", "t3"}; outs = []
    if S4A & set(req):   # S4a 的汇总表只含 S4a 的三个组，S5 不进入（保持 S4a 通过自检时的内容）
        f1 = os.path.join(REV, "tables", "S4_frame_status_summary.csv"); S[S.group.isin(S4A)].to_csv(f1, index=False); outs.append(f1)
    if "s5" in req:
        f2 = os.path.join(REV, "tables", "S4b_frame_status_summary.csv"); S[S.group == "s5"].to_csv(f2, index=False); outs.append(f2)
    S = S[S.group.isin(set(req))]
    pd.set_option("display.width", 220)
    print(S.groupby(["group", "generator"]).agg(proteins=("protein", "size"), frames=("n_xtc_frames_or_recons", "sum"), ok=("n_ok", "sum"), retry=("n_ok_retry", "sum"),
                                                timeout=("n_timeout", "sum"), error=("n_error", "sum"), core_h=("total_core_s", lambda x: round(x.sum() / 3600, 2))).to_string())
    log.finish(outs, note=f"merged_items={len(items)} summary_cells={len(S)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["plan", "run", "merge"]); ap.add_argument("--groups", default="tier1a,crystal,t3")
    ap.add_argument("--workers", type=int, default=20); ap.add_argument("--only", default=""); ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args(); PROJ, REV, INP = revio.load_paths()
    {"plan": cmd_plan, "run": cmd_run, "merge": cmd_merge}[a.cmd](a, INP, REV)
