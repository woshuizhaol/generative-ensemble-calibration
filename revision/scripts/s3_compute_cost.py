"""Stage 3 · 算力与耗时汇总表（revision_proposal.md Stage 3 输出 S3_compute_cost.csv；对应 R2-m8）。

只读：把每个生成任务、ShortMD、BV 打分的墙钟与资源折算成 GPU-h / core-h。所有数字从原日志或本阶段基准解析，不手填。
墙钟都是上限：原运行期间服务器被他人共享使用，且若干任务多进程并发，日志时间含排队与争用。
来源优先级：tqdm 推理循环耗时（仅采样）> 脚本起止时间戳（含预处理、转换，Boltz-2 的 T3 脚本还含 BV 打分）。
输入：tables/S3_generation_runs.csv、tables/S3_md_protocol_per_protein.csv、raw_outputs/s3/benchmark_summary.csv、P/logs/gen/*。
输出：tables/S3_compute_cost.csv（任务级）、tables/S3_bv_scoring_cost_estimate.csv（按基准外推到主文 12 蛋白五类生成器的打分总成本，标注为估计）。
"""
import os, re, sys, glob, datetime, subprocess
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
TAB = os.path.join(REV, "tables"); GL = os.path.join(PROJ, "logs", "gen")
NR = "not_recorded"
log = revio.RunLogger(stage=3, task="compute_cost", cpu_procs=1, seeds="none")
GPU = "NVIDIA GeForce RTX 3090 (24 GB)"
lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
CPU = [l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")][0] + " ×2 sockets, 56 cores"


def rd(p):
    return open(p, errors="ignore").read()


def hms(s):
    a = [int(x) for x in s.split(":")]
    return a[0] * 3600 + a[1] * 60 + a[2] if len(a) == 3 else a[0] * 60 + a[1]


def ts(s):
    return datetime.datetime.strptime(s.replace(" CST", "").strip(), "%a %b %d %I:%M:%S %p %Y")


rows = []


def add(**kw):
    base = dict(generator="", task="", scope="", n_targets=NR, samples_per_target=NR, hardware="", n_units=NR, wall_clock_s=NR, unit_hours=NR,
                unit="", per_target_s_median=NR, per_target_s_min=NR, per_target_s_max=NR, wall_basis="", is_upper_bound=True, source="", note="")
    base.update(kw); rows.append(base)


def per_target_stats(v):
    v = [x for x in v if x is not None]
    return (float(np.median(v)), float(np.min(v)), float(np.max(v))) if v else (NR, NR, NR)


# ---------------------------------------------------------------- BioEmu
R = pd.read_csv(os.path.join(TAB, "S3_generation_runs.csv"))
for kind, scope, nsamp in (("BioEmu", "Tier-1a 12 蛋白", 500), ("BioEmu_T3", "T3 锚定集（有日志的 80 域）", 200)):
    sub = R[R.generator == kind]
    w = pd.to_numeric(sub.sampling_loop_wall_s, errors="coerce")
    ok = w.notna()
    med, mn, mx = per_target_stats(w[ok].tolist())
    add(generator="BioEmu v1.1", task="采样（含首批 MSA 查询与 AF2 嵌入）", scope=scope, n_targets=len(sub), samples_per_target=nsamp,
        hardware=GPU, n_units=1, wall_clock_s=float(w[ok].sum()), unit_hours=round(float(w[ok].sum()) / 3600, 3), unit="GPU-h（每个目标 1 卡，求和）",
        per_target_s_median=med, per_target_s_min=mn, per_target_s_max=mx, wall_basis="日志 tqdm 'Sampling batches' 总耗时",
        source="tables/S3_generation_runs.csv ← logs/gen/BioEmu_*.log, genonly_*.log, t3_T3_*.log",
        note=f"{int(ok.sum())}/{len(sub)} 个目标有该耗时；不含模型载入与 xtc 转换；多卡并发、服务器共享，墙钟为上限。"
             + ("首批含 ColabFold MSA 服务器排队与 AF2 嵌入（冷启动约 4 分钟，其后用缓存嵌入）。" if kind == "BioEmu" else ""))

# ---------------------------------------------------------------- ESMFlow（7 块并行）
ch_times = []; ch_n = []
for f in sorted(glob.glob(os.path.join(GL, "esmflow_chunk_*.log"))):
    t = rd(f).replace("\r", "\n")
    m = re.findall(r"\| 50/50 \[(\d+(?::\d+)+)<", t)
    if len(m) % 2 == 0 and m[0::2] == m[1::2]:
        m = m[0::2]
    ch_times.append([hms(x) for x in m]); ch_n.append(len(m))
allt = [x for c in ch_times for x in c]
mlog = rd(os.path.join(GL, "esmflow_multi.log"))
t0 = ts(re.search(r"\[multi start (.+?) weights=", mlog).group(1)); t1 = ts(re.search(r"\[全部生成完成 (.+?)\]", mlog).group(1))
wall = (t1 - t0).total_seconds()
med, mn, mx = per_target_stats(allt)
add(generator="ESMFlow（MD base）", task="采样（50 个构象/目标；7 块各占 1 卡并行）", scope="Tier-1a 12 + T3 65 = 77 个目标", n_targets=len(allt), samples_per_target=50,
    hardware=GPU, n_units=7, wall_clock_s=wall, unit_hours=round(7 * wall / 3600, 3), unit="GPU-h（7 卡 × 作业墙钟，上限）",
    per_target_s_median=med, per_target_s_min=mn, per_target_s_max=mx, wall_basis="作业起止时间戳（含模型载入）；每目标取 tqdm 50/50 耗时",
    source="logs/gen/esmflow_multi.log, esmflow_chunk_*.log",
    note=f"各块解析到的目标数 {ch_n}（合计 {len(allt)}，应为 77；差额 {77 - len(allt)} 个目标的 tqdm 行缺失或重复，不影响作业墙钟）。"
         "作业墙钟含各块的模型载入：7 块同时载入，日志里 Loading 到 Model has been loaded 约 7.2–7.4 分钟（单进程的 esmflow_base_smoke.log 为约 2 分 50 秒）。ESM-2 3B 骨干与 ESMFlow 权重在显存内。")

# ---------------------------------------------------------------- Boltz-2
def boltz_stamps(path, tag_re):
    out = []
    cur = {}
    for ln in rd(path).splitlines():
        m = re.match(rf"\[{tag_re} (\S+) (start|done OK|OK|FAIL)\] (.+)", ln.strip())
        if m:
            d = m.group(1)
            if m.group(2) == "start":
                cur[d] = ts(m.group(3))
            elif d in cur:
                out.append((d, (ts(m.group(3)) - cur[d]).total_seconds(), m.group(2)))
    return out


ext = boltz_stamps(os.path.join(GL, "extend_boltz.log"), r"Boltz2")
pil = boltz_stamps(os.path.join(GL, "boltz_t3_pilot_nohup.log"), r"Boltz2_T3")
ful = boltz_stamps(os.path.join(GL, "boltz_t3_full_nohup.log"), r"Boltz2_T3")
t3 = [x for x in pil + ful if x[2] != "FAIL"]
med, mn, mx = per_target_stats([x[1] for x in t3])
add(generator="Boltz-2（单序列，--no_kernels）", task="采样 50 个扩散样本 + BV 打分（43_boltz_t3_control.py 一并完成）", scope="T3 65 域", n_targets=len(t3), samples_per_target=50,
    hardware=GPU + " + CPU 打分", n_units=1, wall_clock_s=float(sum(x[1] for x in t3)), unit_hours=round(sum(x[1] for x in t3) / 3600, 3), unit="GPU-h（串行，1 卡；含 CPU 打分时间）",
    per_target_s_median=med, per_target_s_min=mn, per_target_s_max=mx, wall_basis="脚本起止时间戳（含 Boltz 预处理、推理、xtc 转换、BV 打分）",
    source="logs/gen/boltz_t3_pilot_nohup.log, boltz_t3_full_nohup.log",
    note="含 BV 打分，故高估纯生成时间；各目标串行执行。")
med, mn, mx = per_target_stats([x[1] for x in ext if x[2] != "FAIL"])
add(generator="Boltz-2（单序列，--no_kernels）", task="采样 50 个扩散样本（GEN_ONLY，不含打分）", scope="Tier-1a 中后补的 7 个蛋白", n_targets=len(ext), samples_per_target=50,
    hardware=GPU, n_units=1, wall_clock_s=float(sum(x[1] for x in ext)), unit_hours=round(sum(x[1] for x in ext) / 3600, 3), unit="GPU-h（串行，1 卡）",
    per_target_s_median=med, per_target_s_min=mn, per_target_s_max=mx, wall_basis="脚本起止时间戳（含预处理、推理、xtc 转换）",
    source="logs/gen/extend_boltz.log", note="")
pt = []
for p in ("Barnase", "cytochrome_c", "RNaseA", "SNase"):
    m = re.findall(r"1/1 \[(\d+(?::\d+)+)<", rd(os.path.join(GL, f"boltz_{p}.log")).replace("\r", "\n"))
    if m:
        pt.append(hms(m[-1]))
med, mn, mx = per_target_stats(pt)
add(generator="Boltz-2（单序列，--no_kernels）", task="仅推理耗时（tqdm 'Predicting'）", scope="Tier-1a 最先生成的 4 个蛋白", n_targets=len(pt), samples_per_target=50,
    hardware=GPU, n_units=1, wall_clock_s=float(sum(pt)), unit_hours=round(sum(pt) / 3600, 3), unit="GPU-h", per_target_s_median=med, per_target_s_min=mn, per_target_s_max=mx,
    wall_basis="tqdm 推理循环", source="logs/gen/boltz_{Barnase,cytochrome_c,RNaseA,SNase}.log", note="CI2 的推理耗时在 derisk 日志里，未计入。")

# ---------------------------------------------------------------- Str2Str
S2 = [("hdx12", "str2str_hdx12.log", "Tier-1a 12 蛋白，δ=0.25"), ("t3", "str2str_t3.log", "T3 65 域，δ=0.25"),
      ("hi_d035", "str2str_hi_d035.log", "T3 65 域，δ=0.35"), ("hi_d040", "str2str_hi_d040.log", "T3 65 域，δ=0.40")]
for tag, fn, scope in S2:
    t = rd(os.path.join(GL, fn)).replace("\r", "\n")
    m = re.findall(r"Predicting DataLoader 0: 100%\|[^\|]*\| (\d+)/(\d+) \[(\d+(?::\d+)+)<", t)
    n, tot, el = int(m[-1][0]), int(m[-1][1]), hms(m[-1][2])
    add(generator="Str2Str", task="前向-后向采样（n_replica=50 每目标）", scope=scope, n_targets=n, samples_per_target=50, hardware=GPU, n_units=1,
        wall_clock_s=float(el), unit_hours=round(el / 3600, 3), unit="GPU-h（1 卡）", per_target_s_median=round(el / n, 1), per_target_s_min=NR, per_target_s_max=NR,
        wall_basis="tqdm 'Predicting' 总耗时 ÷ 目标数（均值，不是中位数）", source=f"logs/gen/{fn}", note="两个任务同时占用 2 张卡并行（hdx12+t3，或 δ0.35+δ0.40）。")

# ---------------------------------------------------------------- ShortMD
M = pd.read_csv(os.path.join(TAB, "S3_md_protocol_per_protein.csv"))
w = pd.to_numeric(M.wall_clock_min, errors="coerce")
nat = M.n_waters + M.protein_residues_in_box
add(generator="ShortMD", task="显式溶剂 MD：最小化 + 100 ps 平衡 + 2 ns 产出（单副本）", scope="Tier-1a 12 蛋白", n_targets=len(M), samples_per_target="200 帧（10 ps 间隔）", hardware=GPU + "（OpenMM CUDA）",
    n_units=1, wall_clock_s=float((w * 60).sum()), unit_hours=round(float((w * 60).sum()) / 3600, 3), unit="GPU-h（11 个蛋白求和；CI2 墙钟未记录，未计入）",
    per_target_s_median=float(np.median(w.dropna() * 60)), per_target_s_min=float(w.min() * 60), per_target_s_max=float(w.max() * 60),
    wall_basis="extend_shortmd.log 起止时间戳（后 7 个蛋白）；前 5 个蛋白并行启动，取 stage1_waiter 记录的启动时刻到 md.log 末次写入", source="tables/S3_md_protocol_per_protein.csv",
    note=f"{int(w.notna().sum())}/12 个蛋白有墙钟；前 4 个（Barnase、RNaseA、SNase、cyt c）并行跑，墙钟含排队，偏高；UCH-L1 与 Barnase 体系最大（约 3.5 万个水分子）。")

# ---------------------------------------------------------------- BV 打分（基准）
BS = os.path.join(REV, "raw_outputs", "s3", "benchmark_summary.csv")
est_rows = []
if os.path.exists(BS):
    B = pd.read_csv(BS)
    meta = __import__("json").load(open(os.path.join(REV, "raw_outputs", "s3", "benchmark_meta.json")))
    RAWB_ALL = pd.read_csv(os.path.join(REV, "raw_outputs", "s3", "benchmark_raw.csv"))
    RAWB = RAWB_ALL[RAWB_ALL.status == "ok"].assign(tot=lambda d: d.t_pdbfixer + d.t_count + d.t_sasa)
    REPR = RAWB.groupby(["protein", "generator", "scenario", "repeat"]).tot.mean().groupby(["protein", "generator", "scenario"]).agg(["min", "max"])
    probe = B[B.wall_s_per_frame.isna()]
    for r in probe.itertuples():
        errs = RAWB_ALL[(RAWB_ALL.protein == r.protein) & (RAWB_ALL.generator == r.generator) & (RAWB_ALL.scenario == r.scenario)].status.str.slice(0, 70).unique()
        add(generator=f"BV 打分（{r.generator} 帧）", task=f"失败探针：{r.scenario}", scope=f"{r.protein}（{r.n_res} aa），前 5 帧", n_targets=1, samples_per_target=5,
            hardware=meta["cpu_model"] + "（单核）", n_units=1, wall_clock_s=NR, unit_hours=NR, unit="不适用", wall_basis="未产生耗时：5/5 帧在 PDBFixer 处失败",
            source="raw_outputs/s3/benchmark_raw.csv", note="BioEmu 输出只有主链加 Cβ。对这 3 个含 His 的蛋白，不补侧链就加氢时 PDBFixer 直接报错：" + "; ".join(errs) + "。S3_bioemu_h_only_probe.csv 把这个测试扩到 12 个蛋白：10 个含 His 的全部失败，只有无 His 的 CI2 与 CheY 能跑通（但结构里仍缺侧链重原子）。", is_upper_bound=False)
    B = B[B.wall_s_per_frame.notna()]
    for r in B.itertuples():
        add(generator=f"BV 打分（{r.generator} 帧）", task=f"逐帧打分：{r.scenario}", scope=f"{r.protein}（{r.n_res} aa），前 50 帧 × {r.n_repeats} 次重复", n_targets=1, samples_per_target=50,
            hardware=meta["cpu_model"] + "（单核）", n_units=1, wall_clock_s=round(float(r.wall_s_per_frame) * 50, 2), unit_hours=round(float(r.cpu_core_s_per_frame) * 50 / 3600, 5),
            unit="core-h（CPU 时间 × 50 帧）", per_target_s_median=round(float(r.wall_s_per_frame), 3), per_target_s_min=NR, per_target_s_max=NR,
            wall_basis="s3_benchmark.py 分段计时：每帧墙钟均值（pdbfixer + 计数 + SASA）", source="raw_outputs/s3/benchmark_summary.csv",
            note=(f"每帧分段（s）: pdbfixer {r.pdbfixer_s:.3f} / 计数 {r.count_s:.3f} / SASA {r.sasa_s:.3f}；CV 总 {r.cv_total:.2f}、pdbfixer {r.cv_pdbfixer:.2f}；"
                  f"失败帧 {int(r.n_failed_frames)}；每帧 CPU 时间 {r.cpu_core_s_per_frame:.3f} core-s；各次重复的每帧均值范围 "
                  f"{REPR.loc[(r.protein, r.generator, r.scenario), 'min']:.2f}–{REPR.loc[(r.protein, r.generator, r.scenario), 'max']:.2f} s；CV 判据"
                  f"{'通过' if r.cv_criterion_pass else '未通过（已加测到 5 次；各次重复的逐帧耗时相关性弱，原因未单独验证：可能是补侧链步骤本身的随机性，也可能是共享服务器的内存/缓存争用，见工作文档）'}"), is_upper_bound=True)
    # 外推：按“帧类型”分开。基准测了两种帧类型（BioEmu 帧：主链+Cβ，要补全全部侧链；ESMFlow 帧：全重原子，不含氢），
    # 其余生成器按实际输出的帧类型归入（用每残基原子数与是否含氢核对过：Str2Str 与 BioEmu 同为主链+Cβ，Boltz-2 与 ESMFlow 同为全重原子）。
    # ShortMD 的帧已含氢，没有测过这种类型，不外推。每帧 CPU 时间对链长做对数-对数线性插值（3 个基准蛋白，两端沿最近一段的斜率延伸，并标注）。
    import json
    L = {k: v["length"] for k, v in json.load(open(INP["tier1a_sequences"])).items()}
    TYPE = {"BioEmu": ("BioEmu", "sidechain_plus_H", "主链+Cβ 帧"), "Str2Str": ("BioEmu", "sidechain_plus_H", "主链+Cβ 帧（借用 BioEmu 帧的基准）"),
            "ESMFlow": ("ESMFlow", "sidechain_plus_H", "全重原子帧"), "Boltz2": ("ESMFlow", "sidechain_plus_H", "全重原子帧（借用 ESMFlow 帧的基准）")}
    curves = {}
    for gen, scen in {("BioEmu", "sidechain_plus_H"), ("ESMFlow", "sidechain_plus_H")}:
        c = B[(B.generator == gen) & (B.scenario == scen)].sort_values("n_res")
        curves[(gen, scen)] = (np.log(c.n_res.values.astype(float)), np.log(c.cpu_core_s_per_frame.values.astype(float)))

    def interp(key, n):
        x, y = curves[key]; ln = np.log(float(n))
        if x[0] <= ln <= x[-1]:
            return float(np.exp(np.interp(ln, x, y))), "插值"
        i = (0, 1) if ln < x[0] else (-2, -1)
        sl = (y[i[1]] - y[i[0]]) / (x[i[1]] - x[i[0]])
        return float(np.exp(y[i[0] if ln < x[0] else i[1]] + sl * (ln - (x[i[0]] if ln < x[0] else x[i[1]])))), "外推（链长超出 86–223 aa 基准范围）"
    import mdtraj as md
    frames = {}
    for g in ("BioEmu", "ESMFlow", "Boltz2", "Str2Str", "ShortMD"):
        for d in glob.glob(os.path.join(INP["gen_dir"], g, "*", "")):
            p_ = os.path.basename(d.rstrip("/"))
            if p_ in L and os.path.exists(os.path.join(d, "samples.xtc")):
                with md.open(os.path.join(d, "samples.xtc")) as f:
                    frames[(g, p_)] = len(f)
    est = []
    for (g, p_), n in frames.items():
        if g not in TYPE:
            est.append(dict(generator=g, protein=p_, n_res=L[p_], n_frames=n, frame_type="含氢全原子帧", est_core_s_per_frame=NR, est_core_h=NR,
                            method="未测：基准没有包含这种帧类型，不外推")); continue
        gen_b, scen_b, ft = TYPE[g]
        c, how = interp((gen_b, scen_b), L[p_])
        est.append(dict(generator=g, protein=p_, n_res=L[p_], n_frames=n, frame_type=ft, est_core_s_per_frame=round(c, 3), est_core_h=round(c * n / 3600, 3), method=how))
    E = pd.DataFrame(est).sort_values(["generator", "protein"])
    f_est = os.path.join(TAB, "S3_bv_scoring_cost_estimate.csv"); E.to_csv(f_est, index=False)
    Ee = E[E.est_core_h != NR].assign(est_core_h=lambda d: d.est_core_h.astype(float))
    tot = Ee.groupby("generator").est_core_h.sum()
    add(generator="BV 打分（主文 12 蛋白，已测帧类型的 4 类生成器）", task="按基准外推的总打分成本（估计，不是实测）", scope=f"{len(Ee)} 个（生成器×蛋白）、{int(Ee.n_frames.sum())} 帧；ShortMD（含氢帧）未测，未计入",
        n_targets=len(Ee), samples_per_target=NR, hardware=meta["cpu_model"] + "（单核折算）", n_units=1, wall_clock_s=round(float(Ee.est_core_h.sum()) * 3600, 0),
        unit_hours=round(float(Ee.est_core_h.sum()), 2), unit="core-h", wall_basis="每帧 CPU 时间按帧类型、对链长做对数-对数插值 × 实际 xtc 帧数",
        source="tables/S3_bv_scoring_cost_estimate.csv", note="各生成器 core-h: " + "; ".join(f"{k} {v:.1f}" for k, v in tot.items()) +
        "。各次重复之间的逐帧耗时相关性弱（见工作文档），单点估计的不确定度大；这里只用于量级说明。原打分用 21_score_parallel.py 多进程并行，此处折算成单核时间。")
else:
    print("警告：benchmark_summary.csv 不存在，打分成本行未生成")

C = pd.DataFrame(rows)
f_cost = os.path.join(TAB, "S3_compute_cost.csv"); C.to_csv(f_cost, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(C[["generator", "task", "scope", "n_targets", "n_units", "wall_clock_s", "unit_hours", "unit", "per_target_s_median"]].to_string(index=False))
log.finish([f_cost] + ([os.path.join(TAB, "S3_bv_scoring_cost_estimate.csv")] if os.path.exists(BS) else []), note=f"rows={len(C)}")
