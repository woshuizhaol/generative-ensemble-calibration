"""Stage 2：训练集重叠审计（revision_proposal.md Stage 2；对应 R2-C7d）。

查询：12 个 Tier-1a 蛋白的成熟构建序列（tier1a_sequences.json，即各生成器的输入序列）。
目标库（只取与各生成器训练数据相关的）：
  ds23    MEGAscale Dataset2/3 的全部唯一 aa_seq（含突变体）—— BioEmu 的 PPFT 微调数据（预印本 S.1.7：“Dataset2_Dataset3”）
  ds1     MEGAscale Dataset1：dna_seq 按读码框 0 翻译后去重 —— 不属 BioEmu 微调集（S.1.7），按方案一并比对作完整性
  atlas   ATLAS train / val / test 链序列（repos/alphaflow/splits/atlas.csv + 各划分）—— ESMFlow（AlphaFlow-MD）训练数据
  cath2 / cath1 / megasim   BioEmu 的 MD 训练系统（Zenodo；序列取自各系统 topology.pdb）；megasim 突变体
                            （21458 个：14287 个单点、7171 个双点突变）全部来自 271 个 megasim_merge 亲本，已核对，故只比对亲本
  bbtest  bioemu-benchmarks 的全部测试集（commit 63edbfa）—— 预印本称训练集已按“与这些测试蛋白相似度 ≤40%”过滤
不可判定：BioEmu 预训练的 AFDB 子集（161k 结构，无公开清单）；Str2Str / ESMFold 一类以 PDB 结构训练的模型无实验折叠自由能训练数据，
          其 PDB 训练集包含各蛋白的沉积结构属定义上必然，不逐条比对。

两条并行比对（方案 S2 步骤 3）：
  (1) MMseqs2 easy-search：-s 7.5 --min-seq-id 0 -e 10 -c 0（不设覆盖阈值；方案写 cov-mode 2，而 -c 默认 0.0 时 cov-mode 不起作用，
      覆盖度改由输出的 qcov/tcov 事后判读）--max-seqs 5000（避免默认 300 条命中截断）；
  (2) 精确 16-mer 子串匹配。一个 16-mer 若出现在 >MAX_PARENTS(=5) 个不同亲本骨架里，视为多个亲本骨架共有的片段（如融合骨架里重复出现的同一模块），不计入重叠（偏离 D2-2：方案未设此上限；
      它对 12×7 个判定没有任何影响，见 S2_overlap_calls_plan_original_rule.csv）。
判定：strong = 存在命中满足 fident≥0.9 且 alnlen≥30，或存在计入的 16-mer 命中；
  homolog = 否则存在命中满足 fident≥0.3 且 (qcov≥0.5 或 tcov≥0.5) 且 E≤E_HOMOLOG（默认 1e-3）；其余 none。
  E_HOMOLOG 的由来（偏离记录 D2-1）：方案原规则只写 fident≥0.3 与覆盖度≥0.5，未设 E 值上限，而搜索本身用 -e 10。
  首次运行时，阴性对照（每个蛋白只打乱 1 次）中就有一条打乱的 Ubiquitin 对 ATLAS 被判为 homolog（fident 0.305，E=4.5），
  方案要求阴性对照必须全部为 none，因此规则判为过于宽松。修正办法是加 E 值上限，并把阴性对照扩大到每个蛋白打乱 50 次
  （共 600 条打乱序列）用来估计零分布：只要有任一打乱序列在 E≤1e-3 通过，就要继续收紧。1e-3 是序列搜索里常用的显著性阈值，
  稳定窗口：真实结果里非 strong 的 homolog 最大 E 约为 4.6e-10，被取消的随机水平比对 E≥0.10，打乱零分布的最小 E 为 0.0021；
  阈值取 1e-3 落在 (4.6e-10, 0.0021) 这一很宽的窗口内，窗口内任何取值给出相同的判定，因此结论不依赖这个取值。
对照：阳性（ubiquitin→ds23，barstar→atlas 必须 strong）；阴性（每个查询序列氨基酸随机打乱 50 次，必须全为 none）。
"""
import os, re, sys, csv, json, glob, shutil, subprocess, collections, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
TAB = os.path.join(REV, "tables"); RAW = os.path.join(REV, "raw_outputs", "s2"); LOGD = os.path.join(REV, "logs", "s2")
DB = os.path.join(RAW, "db"); os.makedirs(DB, exist_ok=True); os.makedirs(TAB, exist_ok=True); os.makedirs(LOGD, exist_ok=True)
EXT = os.path.join(REV, "data", "external")
MMSEQS = INP["mmseqs"]; THREADS = int(os.environ.get("S2_THREADS", "16"))
MAX_PARENTS = 5; KMER = 16
E_HOMOLOG = float(os.environ.get("S2_E_HOMOLOG", "1e-3"))   # 见 docstring “判定” 段
log = revio.RunLogger(stage=2, task="overlap_audit", cpu_procs=THREADS, seeds="shuffle control seed 20261224")
outs = []

# ------------------------------------------------------------------ 查询
seqs = json.load(open(INP["tier1a_sequences"]))
queries = {p: v["seq"] for p, v in seqs.items()}
rng = np.random.default_rng(20261224)
N_SHUF = 50
shuf = {f"{p}__shuf{k:02d}": "".join(rng.permutation(list(s))) for p, s in queries.items() for k in range(N_SHUF)}
q_real = os.path.join(DB, "query_real.fasta"); q_shuf = os.path.join(DB, "query_shuffled.fasta")
for fn, dd in ((q_real, queries), (q_shuf, shuf)):
    with open(fn, "w") as f:
        for k, s in dd.items():
            f.write(f">{k}\n{s}\n")

# ------------------------------------------------------------------ 目标库
CODON = {}
bases = "TCAG"; aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
for i, a in enumerate(aas):
    CODON[bases[i // 16] + bases[(i // 4) % 4] + bases[i % 4]] = a


def translate(dna):
    return "".join(CODON.get(dna[i:i + 3], "X") for i in range(0, len(dna) - len(dna) % 3, 3))


def parent_ds(name):
    return re.split(r"\.pdb", name)[0]


libs = {}   # lib -> list of (name, seq, parent)
t0 = time.time()
# ds23
d23 = pd.read_csv(INP["tsuboyama_ds23"], low_memory=False, usecols=["name", "WT_name", "aa_seq"])
d23 = d23.drop_duplicates("aa_seq")
libs["ds23"] = [(n, s, parent_ds(w)) for n, w, s in zip(d23.name, d23.WT_name, d23.aa_seq)]
print("ds23 unique aa_seq:", len(libs["ds23"]), f"{time.time() - t0:.0f}s", flush=True)
# ds1（翻译；全部为读码框 0 且无终止密码子，S2 预查：2000/2000）
seen = {}; stop_n = 0; n_rows = 0
for ch in pd.read_csv(INP["tsuboyama_ds1"], usecols=["name", "dna_seq"], chunksize=300000):
    for n, dna in zip(ch.name, ch.dna_seq):
        n_rows += 1; s = translate(dna)
        if "*" in s:
            stop_n += 1; continue
        if s not in seen:
            seen[s] = n
libs["ds1"] = [(n, s, parent_ds(n)) for s, n in seen.items()]
print("ds1 rows", n_rows, "unique aa", len(seen), "with stop codon skipped", stop_n, f"{time.time() - t0:.0f}s", flush=True)
# atlas
spl = INP["atlas_splits_dir"]; atl = []
for part, fn in (("train", "atlas_train.csv"), ("val", "atlas_val.csv"), ("test", "atlas_test.csv")):
    t = pd.read_csv(os.path.join(spl, fn))
    atl += [(f"{n}|{part}", s, n[:4].lower()) for n, s in zip(t.name, t.seqres)]
libs["atlas"] = atl
# BioEmu MD 训练系统
for lib, fn in (("cath2", "cath2_systems.csv"), ("cath1", "cath1_systems.csv"), ("megasim", "megasim_merge_systems.csv")):
    t = pd.read_csv(os.path.join(EXT, "bioemu_md_training_systems", fn))
    libs[lib] = [(s, q, s) for s, q in zip(t.system, t.sequence)]
# bioemu-benchmarks 测试集
bb = []
for fn in sorted(glob.glob(os.path.join(EXT, "bioemu_benchmarks_63edbfa", "assets", "*.csv"))):
    t = pd.read_csv(fn); base = os.path.basename(fn).replace("_benchmark_0.1", "").replace(".csv", "")
    if "sequence" in t.columns:
        idc = "test_case" if "test_case" in t.columns else ("name" if "name" in t.columns else t.columns[0])
        bb += [(f"{base}:{n}", s, str(n).split("__")[0]) for n, s in zip(t[idc], t["sequence"]) if isinstance(s, str)]
seen_b = {}
for n, s, p in bb:
    seen_b.setdefault(s, (n, s, p))
libs["bbtest"] = list(seen_b.values())
for k, v in libs.items():
    fa = os.path.join(DB, f"{k}.fasta")
    with open(fa, "w") as f:
        for i, (n, s, p) in enumerate(v):
            f.write(f">{i}\n{s}\n")
    json.dump({str(i): [n, p] for i, (n, s, p) in enumerate(v)}, open(os.path.join(DB, f"{k}.names.json"), "w"))
    print("library", k, len(v), "sequences", flush=True)

# ------------------------------------------------------------------ MMseqs2
FMT = "query,target,fident,alnlen,nident,qlen,tlen,qcov,tcov,evalue,bits,qstart,qend,tstart,tend"
ver = subprocess.run([MMSEQS, "version"], capture_output=True, text=True).stdout.strip()


def search(qfa, lib, tag):
    out = os.path.join(RAW, f"{tag}_{lib}.m8"); tmp = os.path.join(RAW, f"tmp_{tag}_{lib}")
    cmd = [MMSEQS, "easy-search", qfa, os.path.join(DB, f"{lib}.fasta"), out, tmp, "-s", "7.5", "--min-seq-id", "0", "-e", "10",
           "-c", "0", "--max-seqs", "5000", "--threads", str(THREADS), "--format-output", FMT]
    with open(os.path.join(REV, "run", "commands.sh"), "a") as cf_:
        cf_.write(f"# [S2] mmseqs {ver} | {tag}/{lib}\n" + " ".join(cmd) + "\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(os.path.join(LOGD, f"mmseqs_{tag}_{lib}.log"), "w").write(" ".join(cmd) + "\n" + r.stdout + r.stderr)
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode != 0:
        raise SystemExit(f"mmseqs failed for {tag}/{lib}: {r.stderr[-400:]}")
    cols = FMT.split(",")
    return pd.read_csv(out, sep="\t", header=None, names=cols) if os.path.getsize(out) else pd.DataFrame(columns=cols)


hits = {}
for lib in libs:
    for qfa, tag in ((q_real, "real"), (q_shuf, "shuf")):
        t1 = time.time(); hits[(tag, lib)] = search(qfa, lib, tag)
        print(f"mmseqs {tag}/{lib}: {len(hits[(tag, lib)])} hits, {time.time() - t1:.0f}s", flush=True)

# ------------------------------------------------------------------ 16-mer
def kmer_index(qd):
    idx = collections.defaultdict(list)
    for p, s in qd.items():
        for i in range(len(s) - KMER + 1):
            idx[s[i:i + KMER]].append((p, i))
    return idx


def kmer_scan(qd, lib):
    idx = kmer_index(qd); found = collections.defaultdict(lambda: collections.defaultdict(set))  # (protein,kmer) -> parents / names
    names = {}
    for n, s, par in libs[lib]:
        for i in range(len(s) - KMER + 1):
            w = s[i:i + KMER]
            if w in idx:
                for p, pos in idx[w]:
                    d = found[(p, w)]
                    d["parents"].add(par)
                    if len(d["names"]) < 6:
                        d["names"].add(n)
    rows = []
    for (p, w), d in found.items():
        rows.append(dict(protein=p, kmer=w, n_parents=len(d["parents"]), parents=";".join(sorted(d["parents"]))[:200],
                         examples=";".join(sorted(d["names"]))[:200], counted=len(d["parents"]) <= MAX_PARENTS))
    return pd.DataFrame(rows)


km = {}
for lib in libs:
    for tag, qd in (("real", queries), ("shuf", shuf)):
        t1 = time.time(); km[(tag, lib)] = kmer_scan(qd, lib)
        print(f"16-mer {tag}/{lib}: {len(km[(tag, lib)])} kmer hits ({int(km[(tag, lib)].counted.sum()) if len(km[(tag, lib)]) else 0} counted), {time.time() - t1:.0f}s", flush=True)

# ------------------------------------------------------------------ 判定
NAMES = {lib: json.load(open(os.path.join(DB, f"{lib}.names.json"))) for lib in libs}


def call_for(tag, protein, lib, e_hom=None, cap=True):
    e_hom = E_HOMOLOG if e_hom is None else e_hom
    h = hits[(tag, lib)]; h = h[h["query"] == protein]
    k = km[(tag, lib)]; k = k[(k.protein == protein)] if len(k) else k
    best = None
    if len(h):
        h = h.assign(nres=h.fident * h.alnlen)
        strong_h = h[(h.fident >= 0.9) & (h.alnlen >= 30)]
        hom_h = h[(h.fident >= 0.3) & ((h.qcov >= 0.5) | (h.tcov >= 0.5)) & (h.evalue <= e_hom)]
        b = (strong_h if len(strong_h) else (hom_h if len(hom_h) else h)).sort_values(["nres", "fident"], ascending=False).iloc[0]
        best = b
    counted = (k[k.counted] if cap else k) if len(k) else k
    strong = (best is not None and best.fident >= 0.9 and best.alnlen >= 30) or (len(h) and ((h.fident >= 0.9) & (h.alnlen >= 30)).any()) or len(counted) > 0
    homolog = len(h) and ((h.fident >= 0.3) & ((h.qcov >= 0.5) | (h.tcov >= 0.5)) & (h.evalue <= e_hom)).any()
    call = "strong_overlap" if strong else ("homolog" if homolog else "none")
    row = dict(protein=re.sub(r"__shuf\d+$", "", protein), library=lib, call=call, n_mmseqs_hits=len(h),
               best_fident=None if best is None else round(float(best.fident), 3), best_alnlen=None if best is None else int(best.alnlen),
               best_qcov=None if best is None else round(float(best.qcov), 3), best_tcov=None if best is None else round(float(best.tcov), 3),
               best_evalue=None if best is None else float(best.evalue),
               best_target=None if best is None else NAMES[lib][str(int(best.target))][0],
               n_strong_targets=int(((h.fident >= 0.9) & (h.alnlen >= 30)).sum()) if len(h) else 0,
               n_kmer16_hits_total=len(k), n_kmer16_hits_counted=len(counted),
               kmer16_example_targets=(counted.examples.iloc[0] if len(counted) else ""),
               kmer16_common_motif_hits=int((~k.counted).sum()) if len(k) else 0)
    return row


rows = []; ctrl = []
for lib in libs:
    for p in queries:
        rows.append(call_for("real", p, lib))
        for k in range(N_SHUF):
            ctrl.append(dict(call_for("shuf", f"{p}__shuf{k:02d}", lib), shuffle_id=k, control="shuffled_query_negative"))
T = pd.DataFrame(rows); C = pd.DataFrame(ctrl)
# 敏感性：方案原规则（homolog 无 E 值上限即 E≤10；16-mer 无亲本数上限），仅用于说明偏离 D2-1/D2-2 对判定的影响
orig_rows = [call_for("real", p, lib, e_hom=10.0, cap=False) for lib in libs for p in queries]
O = pd.DataFrame(orig_rows)
M_ = T[["protein", "library", "call"]].merge(O[["protein", "library", "call"]], on=["protein", "library"], suffixes=("_final", "_plan_original_rule"))
M_["changed"] = M_.call_final != M_.call_plan_original_rule
M_.to_csv(os.path.join(TAB, "S2_overlap_calls_plan_original_rule.csv"), index=False); outs.append(os.path.join(TAB, "S2_overlap_calls_plan_original_rule.csv"))
print("calls changed by D2-1/D2-2 (final vs plan original rule):", int(M_.changed.sum()))
print(M_[M_.changed].to_string(index=False))
# 阳性对照
def get(df, p, lib):
    return df[(df.protein == p) & (df.library == lib)].iloc[0]
pos = []
for p, lib in (("Ubiquitin", "ds23"), ("Barstar", "atlas")):
    r = get(T, p, lib); pos.append(dict(control="positive", protein=p, library=lib, call=r.call, best_fident=r.best_fident, best_alnlen=r.best_alnlen,
                                        best_target=r.best_target, passed=r.call == "strong_overlap"))
neg_fail = C[C.call != "none"]
ctl = pd.DataFrame(pos + [dict(control=f"negative_shuffled_{N_SHUF}x12_all_libraries", protein=f"({12 * N_SHUF} shuffled queries x {len(libs)} libraries)",
                               library="(all)", call="all none" if len(neg_fail) == 0 else f"{len(neg_fail)} non-none of {len(C)}",
                               best_fident=None, best_alnlen=None, best_target="", passed=len(neg_fail) == 0)])
# 零分布：打乱查询下最小的 E 值（用于说明 E_HOMOLOG 的选取）
_hs = [(lib, hits[("shuf", lib)]) for lib in libs]
_allE = pd.concat([h.evalue for _, h in _hs]) if _hs else pd.Series(dtype=float)
nullE = dict(n_searches=len(C), n_queries=12 * N_SHUF, n_hits_total=int(len(_allE)), min_evalue=float(_allE.min()) if len(_allE) else None,
             q001=float(_allE.quantile(0.001)) if len(_allE) else None, q01=float(_allE.quantile(0.01)) if len(_allE) else None,
             n_hits_evalue_le_1e3=int((_allE <= E_HOMOLOG).sum()), E_HOMOLOG=E_HOMOLOG)
json.dump(nullE, open(os.path.join(TAB, "S2_overlap_null_evalue.json"), "w"), indent=1); outs.append(os.path.join(TAB, "S2_overlap_null_evalue.json"))
print("shuffle-null E-values:", nullE)
C.to_csv(os.path.join(TAB, "S2_overlap_negative_control_detail.csv"), index=False); outs.append(os.path.join(TAB, "S2_overlap_negative_control_detail.csv"))
ctl.to_csv(os.path.join(TAB, "S2_overlap_controls.csv"), index=False); outs.append(os.path.join(TAB, "S2_overlap_controls.csv"))
# AFDB 不可判定行
extra = [dict(protein=p, library="afdb_bioemu_pretrain", call="not_determinable", note="BioEmu 预训练的 AFDB 子集（161k 结构）无公开序列清单") for p in queries]
extra += [dict(protein=p, library="pdb_structure_models(Str2Str/ESMFold)", call="not_applicable",
               note="以 PDB 结构训练，无实验折叠自由能训练数据；沉积结构属训练集定义上必然") for p in queries]
extra += [dict(protein=p, library="mdcath_boltz2_training", call="not_determinable",
               note="Boltz-2 预印本 A.1.2 的 MD 训练数据之一；方案 §1.3 明确不做 mdCATH 比对") for p in queries]
extra += [dict(protein=p, library="misato_boltz2_training", call="not_determinable",
               note="Boltz-2 预印本 A.1.2 的 MD 训练数据之一（蛋白-配体复合物）；未比对") for p in queries]
T2 = pd.concat([T, pd.DataFrame(extra)], ignore_index=True)
T2.to_csv(os.path.join(TAB, "S2_overlap_tier1a.csv"), index=False); outs.append(os.path.join(TAB, "S2_overlap_tier1a.csv"))
# 逐命中原始表（透明）
for (tag, lib), h in hits.items():
    if tag == "real" and len(h):
        h2 = h.copy(); h2["target_name"] = h2.target.map(lambda x: NAMES[lib][str(int(x))][0]); h2["target_parent"] = h2.target.map(lambda x: NAMES[lib][str(int(x))][1])
        h2[(h2.fident >= 0.3)].to_csv(os.path.join(RAW, f"hits_real_{lib}_fident30.csv"), index=False)
kms = [km[("real", lib)].assign(library=lib) for lib in libs if len(km[("real", lib)])]
if kms:
    pd.concat(kms).to_csv(os.path.join(TAB, "S2_overlap_kmer16_hits.csv"), index=False); outs.append(os.path.join(TAB, "S2_overlap_kmer16_hits.csv"))

# ------------------------------------------------------------------ 生成器层汇总
GEN = [("BioEmu (经实验折叠自由能微调；来源数据集 MEGAscale Dataset2/3，所用 v1.1 约 502k 条是否含该蛋白无法判定)", "ds23", "trained_on_dGfold=True"),
       ("BioEmu (MD 训练系统 CATH2)", "cath2", "MD"), ("BioEmu (MD 训练系统 CATH1)", "cath1", "MD"), ("BioEmu (MD 训练系统 MegaSim)", "megasim", "MD"),
       ("ESMFlow / AlphaFlow-MD (ATLAS train+val+test)", "atlas", "MD"),
       ("MEGAscale Dataset1（不在 BioEmu 微调集内，预印本 S.1.7）", "ds1", "reference"),
       ("bioemu-benchmarks 测试集（BioEmu 称训练集已按 ≤40% 相似度过滤）", "bbtest", "held_out_test")]
S = []
for label, lib, kind in GEN:
    sub = T[T.library == lib]
    S.append(dict(group=label, library=lib, kind=kind, n_strong=int((sub.call == "strong_overlap").sum()),
                  strong_proteins=";".join(sub[sub.call == "strong_overlap"].protein), n_homolog=int((sub.call == "homolog").sum()),
                  homolog_proteins=";".join(sub[sub.call == "homolog"].protein), n_none=int((sub.call == "none").sum())))
S = pd.DataFrame(S); S.to_csv(os.path.join(TAB, "S2_overlap_summary.csv"), index=False); outs.append(os.path.join(TAB, "S2_overlap_summary.csv"))

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(T.pivot(index="protein", columns="library", values="call").to_string())
print(S[["group", "n_strong", "strong_proteins", "n_homolog", "homolog_proteins"]].to_string(index=False))
print(ctl.to_string(index=False))
json.dump(dict(mmseqs=ver, kmer=KMER, max_parents=MAX_PARENTS, library_sizes={k: len(v) for k, v in libs.items()},
               ds1_rows=n_rows, ds1_unique=len(seen), ds1_stop_skipped=stop_n),
          open(os.path.join(TAB, "S2_overlap_meta.json"), "w"), indent=1); outs.append(os.path.join(TAB, "S2_overlap_meta.json"))
shutil.rmtree(os.path.join(RAW, "tmp_dummy"), ignore_errors=True)
log.finish(outs, note=f"strong: {', '.join(f'{r.library}:{r.n_strong}' for r in S.itertuples())}")
