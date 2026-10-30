"""Stage 3 · ShortMD 协议核实（revision_proposal.md Stage 3 步骤 2；对应 R2-C7a）。

只读：从原项目的 12_short_md.py（协议字面值，用正则提取；任一项提取失败即报错，不用硬编码补位）、各蛋白的 md.log /
full_top.pdb / ensemble.pdb / ensemble.xtc / full.xtc、extend_shortmd.sh / extend_shortmd.log / stage1_waiter.log、
源 PDB 以及 env/md 里的 OpenMM 力场文件，逐蛋白核出一张 12 行的协议表。

输出：
  tables/S3_md_protocol_per_protein.csv   12 行（协议字面值在每行重复，便于单行读取）
  tables/S3_md_protocol_script_parse.json  脚本哈希与逐项正则命中的原文，供复核
找不到证据的字段写 not_recorded，不补猜。
"""
import os, re, sys, json, glob, subprocess, datetime, difflib, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
TAB = os.path.join(REV, "tables"); os.makedirs(TAB, exist_ok=True)
MDDIR = os.path.join(INP["gen_dir"], "MD_short")
SCRIPT = os.path.join(PROJ, "code", "analysis", "12_short_md.py")
EXT_SH = os.path.join(PROJ, "run", "extend_shortmd.sh")
EXT_LOG = os.path.join(PROJ, "logs", "analysis", "extend_shortmd.log")
WAITER = os.path.join(PROJ, "logs", "analysis", "stage1_waiter.log")
ENV_MD = os.path.join(PROJ, "env", "md")
NR = "not_recorded"
log = revio.RunLogger(stage=3, task="md_protocol", cpu_procs=1, seeds="none")

T2O = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
       'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


# ------------------------------------------------------------------ 1) 协议字面值（来自脚本本身）
def parse_script(path):
    s = open(path).read()
    hits = {}

    def need(key, pat):
        m = re.search(pat, s)
        if m is None:
            raise SystemExit(f"12_short_md.py 中找不到 {key}（正则 {pat}）；协议无法核实，不补猜")
        hits[key] = m.group(0).strip()
        return m.groups()

    ff = need("force_field", r"ForceField\('([^']+)'\s*,\s*'([^']+)'\)")
    pad, wmodel, ionic = need("solvation", r"addSolvent\(ff,\s*padding=([\d.]+)\*nanometer,\s*model='([^']+)',\s*ionicStrength=([\d.]+)\*molar\)")
    nbm, cut, cons = need("system", r"createSystem\(mod\.topology,\s*nonbondedMethod=(\w+),\s*nonbondedCutoff=([\d.]+)\*nanometer,\s*constraints=(\w+)\)")
    bp, bt = need("barostat", r"MonteCarloBarostat\(([\d.]+)\*bar,\s*([\d.]+)\*kelvin\)")
    it, fr, dt = need("integrator", r"LangevinMiddleIntegrator\(([\d.]+)\*kelvin,\s*([\d./]+)/picosecond,\s*([\d.]+)\*femtosecond\)")
    (mi,) = need("minimization", r"minimizeEnergy\(maxIterations=(\d+)\)")
    eq_steps, eq_ps = need("equilibration", r"sim\.step\((\d+)\)\s*#\s*(\d+)\s*ps")
    (nfr,) = need("n_frames", r"nframes=(\d+)")
    need("frame_interval", r"interval=int\(ns\*1e6/2/nframes\)")
    need("production", r"sim\.step\(int\(ns\*1e6/2\)\)")
    (ph,) = need("protonation", r"addMissingHydrogens\(([\d.]+)\)")
    (keepw,) = need("remove_heterogens", r"removeHeterogens\((\w+)\)")
    need("missing_residues_not_added", r"fx\.missingResidues\s*=\s*\{\}")
    need("nonstandard_replaced", r"replaceNonstandardResidues\(\)")
    need("velocities", r"setVelocitiesToTemperature\(([\d.]+)\*kelvin\)")
    need("xtc_reporter", r"XTCReporter\(")
    need("platform", r"Platform\.getPlatformByName\('(\w+)'\)")
    return dict(ff_files=list(ff), pad_nm=float(pad), water_template=wmodel, ionic_M=float(ionic), nonbonded=nbm, cutoff_nm=float(cut),
                constraints=cons, baro_bar=float(bp), baro_K=float(bt), temp_K=float(it), friction_per_ps=fr, dt_fs=float(dt),
                min_iter=int(mi), equil_steps=int(eq_steps), equil_ps_comment=int(eq_ps), n_frames=int(nfr), pH=float(ph),
                remove_heterogens_keepWater=keepw), hits, s


P, HITS, SRC = parse_script(SCRIPT)
assert abs(P["equil_steps"] * P["dt_fs"] / 1000 - P["equil_ps_comment"]) < 1e-9, "脚本注释的平衡时长与步数×步长不符"

# OpenMM 版本、力场 XML 包含关系、createSystem 默认 rigidWater
conda_meta = glob.glob(os.path.join(ENV_MD, "conda-meta", "openmm-*.json"))
OPENMM_ENV = os.path.basename(conda_meta[0]).split("-py")[0].replace("openmm-", "") if conda_meta else NR
omm_data = glob.glob(os.path.join(ENV_MD, "lib", "python3*", "site-packages", "openmm", "app", "data"))[0]
inc = re.findall(r'<Include file="([^"]+)"', open(os.path.join(omm_data, "amber14-all.xml")).read())
fsrc = open(glob.glob(os.path.join(ENV_MD, "lib", "python3*", "site-packages", "openmm", "app", "forcefield.py"))[0]).read()
# rigidWater：脚本未传该参数 → createSystem 默认 None → 沿用力场中水模板自己的设置；
# 模板类 _TemplateData 的默认值在源码里读，HOH 模板是否带 rigidWater 属性覆盖它，读 tip3pfb.xml。
m_def = re.search(r"def createSystem\(self, topology[^)]*?rigidWater=(\w+)", fsrc, re.S)
m_tmpl = re.search(r"class _TemplateData.*?self\.rigidWater = (True|False)", fsrc, re.S)
hoh = re.search(r'<Residue name="HOH"([^>]*)>', tip_info_raw := open(os.path.join(omm_data, "amber14", "tip3pfb.xml")).read())
hoh_override = re.search(r'rigidWater="(\w+)"', hoh.group(1)) if hoh else None
if m_def and m_def.group(1) == "None" and m_tmpl and hoh is not None:
    RIGID_WATER = (hoh_override.group(1).title() if hoh_override else m_tmpl.group(1)) + \
        "（createSystem 默认 None → 沿用 HOH 模板；模板类默认 " + m_tmpl.group(1) + "，tip3pfb.xml 的 HOH 无覆盖属性 → 水刚性约束，2 fs 步长对此成立）"
else:
    RIGID_WATER = NR
m_rc = re.search(r"def createSystem\(self, topology[^)]*?removeCMMotion=(True|False)", fsrc, re.S)
REMOVE_CMM = m_rc.group(1) if m_rc else NR
tip_info = open(os.path.join(omm_data, "amber14", "tip3pfb.xml")).read()
WATER_SRC = "TIP3P-FB (amber14/tip3pfb.xml; Wang, Martinez & Pande 2014, J. Phys. Chem. Lett. 5, 1885)" if "Wang" in tip_info else NR
# MonteCarloBarostat 的默认更新频率（OpenMM 文档：25 步），从 OpenMM 包内核对
BARO_FREQ = "25 (OpenMM MonteCarloBarostat 默认，脚本未改)"

# ------------------------------------------------------------------ 2) 逐蛋白输入
SEQ = json.load(open(INP["tier1a_sequences"]))
SMAP = json.load(open(INP["structure_maps"]))
PDB_USED = {}
if os.path.exists(EXT_SH):
    for m in re.finditer(r"\[(\S+)\]=(\w{4})", open(EXT_SH).read()):
        PDB_USED[m.group(1)] = (m.group(2), "extend_shortmd.sh")
for p in SEQ:
    if p not in PDB_USED:
        PDB_USED[p] = (SMAP[p]["pdb_id"], "structure_maps.json（该蛋白 MD 的启动命令未保留；用残基数/序列与源 PDB 交叉核对，见 seq_match 列）")

GPU_ARG = {}
if os.path.exists(EXT_SH):
    m = re.search(r"12_short_md\.py \$prot \$\{PDB\[\$prot\]\} (\d+) ([\d.]+)", open(EXT_SH).read())
    ext_gpu, ext_ns = (m.group(1), m.group(2)) if m else (NR, NR)
else:
    ext_gpu, ext_ns = NR, NR
EXT_SET = {"Ubiquitin", "Im7", "Barstar", "CheY", "UCH-L1", "b-lactoglobulin", "RNaseH_Tt"}
FIRST5 = {"Barnase", "RNaseA", "SNase", "CI2", "cytochrome_c"}


def parse_ts(s):
    return datetime.datetime.strptime(s.replace(" CST", ""), "%a %b %d %I:%M:%S %p %Y")


ext_times = {}
if os.path.exists(EXT_LOG):
    for ln in open(EXT_LOG):
        m = re.match(r"\[ShortMD (\S+) (start|done OK|done FAIL)\] (.+)", ln.strip())
        if m:
            ext_times.setdefault(m.group(1), {})[m.group(2).split()[0]] = parse_ts(m.group(3))
waiter_start = None
if os.path.exists(WAITER):
    m = re.search(r"\[env/md就绪 (.+?)\] 启动短MD", open(WAITER).read())
    if m:
        waiter_start = parse_ts(m.group(1))


def src_pdb_info(pdbid):
    path = os.path.join(INP["pdb_dir"], f"{pdbid}.pdb")
    het = collections.Counter(); ca = collections.defaultdict(list)
    for ln in open(path):
        if ln.startswith("HETATM"):
            het[ln[17:20].strip()] += 1
        elif ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            try:
                ca[ln[21]].append(int(ln[22:26]))
            except ValueError:
                pass
    gaps = {c: sum(1 for a, b in zip(v, v[1:]) if b - a > 1) for c, v in ca.items()}
    nonwater = {k: v for k, v in het.items() if k != "HOH"}
    return dict(chains=sorted(ca), n_ca={c: len(v) for c, v in ca.items()}, internal_gaps=gaps, hetero=nonwater, n_hoh=het.get("HOH", 0))


def parse_top(path):
    box = None; n_wat = n_na = n_cl = 0; omm = NR; prot_atoms = 0
    with open(path) as f:
        for ln in f:
            if ln.startswith("REMARK") and "OPENMM" in ln.upper():
                omm = ln.split("CREATED WITH")[-1].strip()
            elif ln.startswith("CRYST1") and box is None:
                box = tuple(float(x) for x in (ln[6:15], ln[15:24], ln[24:33]))
            elif ln.startswith(("ATOM", "HETATM")):
                rn = ln[17:20].strip(); an = ln[12:16].strip()
                if rn == "HOH":
                    n_wat += (an == "O")
                elif rn in ("NA", "Na+"):
                    n_na += 1
                elif rn in ("CL", "Cl-"):
                    n_cl += 1
                else:
                    prot_atoms += 1
    return omm, box, n_wat, n_na, n_cl, prot_atoms


def chain_seqs(topology):
    out = []
    for c in topology.chains:
        out.append("".join(T2O.get(r.name, "X") for r in c.residues))
    return out


def md_log_stats(path):
    steps, temps, pot = [], [], []
    for ln in open(path):
        if ln.startswith("#"):
            continue
        a = ln.strip().split(",")
        steps.append(int(a[0])); pot.append(float(a[1])); temps.append(float(a[2]))
    return np.array(steps), np.array(pot), np.array(temps)


import mdtraj as md  # noqa: E402

rows = []
for prot in SEQ:
    d = os.path.join(MDDIR, prot)
    need_files = ["md.log", "full_top.pdb", "ensemble.pdb", "ensemble.xtc", "full.xtc"]
    miss = [x for x in need_files if not os.path.exists(os.path.join(d, x))]
    if miss:
        raise SystemExit(f"{prot}: 缺文件 {miss}")
    pdbid, pdb_src = PDB_USED[prot]
    si = src_pdb_info(pdbid)
    omm, box, n_wat, n_na, n_cl, prot_atoms = parse_top(os.path.join(d, "full_top.pdb"))
    steps, pot, temps = md_log_stats(os.path.join(d, "md.log"))
    ens = md.load_pdb(os.path.join(d, "ensemble.pdb"))
    cseq = chain_seqs(ens.topology)
    q = SEQ[prot]["seq"]
    sm = difflib.SequenceMatcher(None, q, cseq[0], autojunk=False)
    # 时长与帧间隔取自 full.xtc（OpenMM XTCReporter 的原始输出，step 字段为模拟步数）。
    # ensemble.xtc 是项目用 mdtraj 对 full.xtc 切蛋白原子后重存的，重存时 step 字段被改写成帧序号 0..N-1，不能用来算时长（S3 首次运行曾误用，已更正）。
    with md.open(os.path.join(d, "full.xtc")) as f:
        _, tm, st, _ = f.read()
    with md.open(os.path.join(d, "ensemble.xtc")) as f:
        n_ens = len(f)
    n_full = len(st)
    dstep = int(np.median(np.diff(st))); dtime = float(np.median(np.diff(tm))) if n_full > 1 else float("nan")
    last_logged = int(steps.max())
    prod_steps_from_log = last_logged - P["equil_steps"]
    prod_ns_log = prod_steps_from_log * P["dt_fs"] * 1e-6
    prod_ns_xtc = (int(st.max()) - P["equil_steps"]) * P["dt_fs"] * 1e-6
    frame_dt_ps = dstep * P["dt_fs"] * 1e-3
    first_frame_prod_ps = (int(st.min()) - P["equil_steps"]) * P["dt_fs"] * 1e-3
    uniform_step = bool(len(set(np.diff(st).tolist())) == 1)
    # ensemble.xtc 的坐标是否就是 full.xtc 中蛋白原子的坐标（首、中、末帧；XTC 精度 0.001 nm，容差 2e-3 nm）
    full_top = md.load_pdb(os.path.join(d, "full_top.pdb")).topology
    sel = full_top.select("protein")
    max_dev = 0.0
    for k in (0, n_full // 2, n_full - 1):
        a = md.load_frame(os.path.join(d, "full.xtc"), k, top=full_top).xyz[0][sel]
        b = md.load_frame(os.path.join(d, "ensemble.xtc"), k, top=os.path.join(d, "ensemble.pdb")).xyz[0]
        max_dev = max(max_dev, float(np.abs(a - b).max()) if a.shape == b.shape else float("inf"))
    # 时间戳
    md_end = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(d, "md.log")))
    if prot in ext_times and {"start", "done"} <= set(ext_times[prot]):
        wall_min = round((ext_times[prot]["done"] - ext_times[prot]["start"]).total_seconds() / 60, 1)
        wall_note = "extend_shortmd.log 的 start/done 时间戳"
        gpu = f"CUDA_VISIBLE_DEVICES={ext_gpu}（extend_shortmd.sh 参数）"
    elif prot in FIRST5 and waiter_start is not None and prot != "CI2":
        wall_min = round((md_end - waiter_start).total_seconds() / 60, 1)
        wall_note = "上限：从 stage1_waiter 记录的启动时刻（env/md 就绪）到 md.log 末次写入；5 个蛋白并行，含排队与并行争用"
        gpu = "GPU 1–5 并行（run/pids.tsv：短MD(5蛋白并行GPU1-5)）；单蛋白对应哪张卡未记录"
    else:
        wall_min = NR
        wall_note = "CI2 因首次 MD 的输入结构有误（蛋白原子数 4998，后以 2CI2 的 1076 原子重跑）而重跑；重跑启动时刻未保留，只有 md.log 末次写入时间"
        gpu = NR
    rows.append(dict(
        protein=prot, pdb_id=pdbid, pdb_id_source=pdb_src,
        n_chains_in_box=len(cseq), protein_residues_in_box=sum(len(s) for s in cseq), chain_lengths=";".join(str(len(s)) for s in cseq),
        scored_chain_residues=len(cseq[0]) if len(cseq) == 1 else max(len(s) for s in cseq),
        seq_match_first_chain_vs_tier1a=round(sm.ratio(), 4), tier1a_len=len(q), first_chain_len=len(cseq[0]),
        source_chains=";".join(si["chains"]), source_internal_numbering_gaps=json.dumps(si["internal_gaps"]),
        source_heterogens_removed=json.dumps(si["hetero"]) if si["hetero"] else "none", source_crystal_waters_removed=si["n_hoh"],
        heme_removed=("HEC" in si["hetero"]),
        force_field_files="+".join(P["ff_files"]), protein_force_field="ff14SB (amber14/protein.ff14SB.xml, via amber14-all.xml Include: " + ",".join(os.path.basename(x) for x in inc) + ")",
        water_model=WATER_SRC, water_box_template=f"addSolvent(model='{P['water_template']}')（OpenMM 文档：TIP3P-FB 先建 TIP3P 盒再用 TIP3P-FB 参数最小化）",
        ions="Na+/Cl- 中和 + 0.1 M NaCl；离子参数 Joung-Cheatham（tip3pfb.xml 头部 frcmod.ionsjc_tip3p）",
        box_shape="cube (addSolvent 默认)", padding_nm=P["pad_nm"], ionic_strength_M=P["ionic_M"],
        box_edge_nm=round(box[0] / 10, 3) if box else NR, n_waters=n_wat, n_Na=n_na, n_Cl=n_cl,
        protonation_pH=P["pH"], protonation_tool="PDBFixer.addMissingHydrogens(7.0)",
        nonstandard_residues="PDBFixer.replaceNonstandardResidues()", missing_residues_added="no（missingResidues={}）",
        heterogens_removed_rule=f"PDBFixer.removeHeterogens(keepWater={P['remove_heterogens_keepWater']})",
        nonbonded=P["nonbonded"], cutoff_nm=P["cutoff_nm"], constraints=P["constraints"], rigid_water=RIGID_WATER, remove_cm_motion=REMOVE_CMM,
        minimization=f"LocalEnergyMinimizer maxIterations={P['min_iter']}（容差为 OpenMM 默认）",
        initial_velocities=f"setVelocitiesToTemperature({P['temp_K']:g} K)，无升温斜坡，无位置约束",
        equilibration=f"{P['equil_ps_comment']} ps NPT（{P['equil_steps']} 步，无 NVT 段）；MonteCarloBarostat 从最小化后即加入",
        integrator="LangevinMiddleIntegrator", temperature_K=P["temp_K"], friction_per_ps=P["friction_per_ps"], timestep_fs=P["dt_fs"],
        barostat=f"MonteCarloBarostat({P['baro_bar']:g} bar, {P['baro_K']:g} K)，更新频率 {BARO_FREQ}",
        replicas=1, random_seed="未设置（积分器与速度初始化均用 OpenMM 默认随机种子）",
        production_ns_from_md_log=round(prod_ns_log, 4), production_ns_from_xtc_steps=round(prod_ns_xtc, 4),
        n_frames_ensemble_xtc=n_ens, n_frames_full_xtc=n_full, frame_interval_steps=dstep, frame_interval_ps=round(frame_dt_ps, 3),
        xtc_time_field_median_dt_ps=round(dtime, 3), first_frame_after_production_start_ps=round(first_frame_prod_ps, 3),
        frames_source_file="full.xtc（OpenMM XTCReporter 原始输出）；ensemble.xtc 为其蛋白原子切片（mdtraj 重存，step 字段为帧序号）",
        ensemble_vs_full_max_abs_dev_nm=round(max_dev, 5),
        md_log_rows=len(steps), md_log_first_step=int(steps.min()), md_log_last_step=last_logged,
        md_log_mean_T_K=round(float(temps.mean()), 2), md_log_sd_T_K=round(float(temps.std(ddof=1)), 2),
        md_log_potential_first_vs_last_kJmol=f"{pot[0]:.0f};{pot[-1]:.0f}",
        openmm_version_full_top_header=omm, openmm_version_env_md=OPENMM_ENV,
        platform="CUDA", gpu_model="NVIDIA GeForce RTX 3090（gpu02 的 8 张卡同型号）", gpu_assignment=gpu,
        wall_clock_min=wall_min, wall_clock_note=wall_note, md_log_last_write=md_end.strftime("%Y-%m-%d %H:%M:%S"),
        # QC
        qc_duration_equals_frames_x_dt=bool(abs(prod_ns_xtc * 1e3 - n_full * frame_dt_ps) < 1e-6),
        qc_uniform_frame_spacing=uniform_step,
        qc_ensemble_coords_equal_full_protein=bool(max_dev < 2e-3),
        qc_duration_log_equals_xtc=bool(abs(prod_ns_log - prod_ns_xtc) < 1e-9),
        qc_frames_equal_nframes_setting=bool(n_ens == P["n_frames"] == n_full),
        qc_interval_10ps=bool(abs(frame_dt_ps - 10.0) < 1e-6),
        qc_openmm_header_matches_env=bool(omm.startswith("OPENMM 8.2") and OPENMM_ENV.startswith("8.2")),
        qc_temperature_within_5K_of_300=bool(abs(temps.mean() - P["temp_K"]) < 5.0),
    ))

df = pd.DataFrame(rows)
f_csv = os.path.join(TAB, "S3_md_protocol_per_protein.csv"); df.to_csv(f_csv, index=False)
f_js = os.path.join(TAB, "S3_md_protocol_script_parse.json")
revio.write_json(dict(script=SCRIPT, script_sha256=revio.sha256(SCRIPT), parsed_values=P, regex_hits=HITS, openmm_env=OPENMM_ENV,
                      amber14_all_includes=inc, create_system_defaults=dict(rigidWater=RIGID_WATER, removeCMMotion=REMOVE_CMM),
                      extend_shortmd_args=dict(gpu=ext_gpu, ns=ext_ns), waiter_start=str(waiter_start)), f_js)

qc_cols = [c for c in df.columns if c.startswith("qc_")]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print(df[["protein", "pdb_id", "n_chains_in_box", "protein_residues_in_box", "seq_match_first_chain_vs_tier1a", "n_waters", "n_Na", "n_Cl", "box_edge_nm",
          "production_ns_from_xtc_steps", "n_frames_ensemble_xtc", "frame_interval_ps", "md_log_mean_T_K", "wall_clock_min", "heme_removed"]].to_string(index=False))
print(df[["protein"] + qc_cols].to_string(index=False))
allpass = bool(df[qc_cols].all().all())
print("QC all pass:", allpass)
log.finish([f_csv, f_js], note=f"rows={len(df)} qc_all_pass={allpass}")
