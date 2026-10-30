"""Stage 1：数据账目、全局域集重建与承诺项核查（revision_proposal.md Stage 1）。

对应关切：R2-C4d（65 域判据）、R2-m13（删失/Pro/N 端处理）、R2-C7c（额外参照集）、R2-m10（n=129 vs 101）。
步骤：
 1 Tier-1a 残基账目：mmc2.xlsx 逐表 + classics；N/A 分解（Pro / N 端 / 其他）；下限值与合并值；最终 n；
   编号核对（Pro 位点必为 N/A；保留残基的残基名与沉积结构一致）。
 2 Tsuboyama 过滤级联：Dataset2/3 → wt → WT_name 去重 → 30% 聚类（复现现有 cluster.tsv）→ 40–72 aa。
 3 D_new：源数据定义的可靠性（dG_ML 为数值）+ 每 30% 聚类一个代表（判据冻结在 docs/tier3_criterion.md）。
 4 D_legacy65 来源说明。
 5 Start2Fold 依赖 grep。
 6 SNase n=129 账目。
"""
import os, sys, re, json, glob, subprocess, collections, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import bv, revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
TAB = os.path.join(REV, "tables"); os.makedirs(TAB, exist_ok=True)
LOGD = os.path.join(REV, "logs", "s1"); os.makedirs(LOGD, exist_ok=True)
log = revio.RunLogger(stage=1, task="data_accounting", cpu_procs=1, seeds="none")
outs = []

# =============================== 1. Tier-1a 残基账目 ===============================
import openpyxl
# sheet(PDB) → 蛋白名，取自原 code/pipeline/01_parse_tier1a.py:9–20 的 META（不复制其余元数据，只用于命名）
SHEET2PROT = {"1A2P": "Barnase", "1AYI": "Im7", "1BTA": "Barstar", "1CYE": "CheY", "1FS3": "RNaseA", "1SNO": "SNase",
              "1TM1": "CI2", "1UBQ": "Ubiquitin", "2ETL": "UCH-L1", "3BLG": "b-lactoglobulin"}
seqs = json.load(open(INP["tier1a_sequences"]))
allp = pd.read_parquet(INP["tier1a_all"])
maps = json.load(open(INP["structure_maps"]))
wb = openpyxl.load_workbook(os.path.join(INP["tier1a_raw_nguyen"], "mmc2.xlsx"), read_only=True, data_only=True)

acc = []; resrows = []
for sheet in wb.sheetnames:
    prot = SHEET2PROT[sheet]; seq = seqs[prot]["seq"]
    data = list(wb[sheet].iter_rows(values_only=True)); hdr = data[0]; body = [r for r in data[1:] if r is not None]
    n_raw = len(body); na = [r for r in body if not isinstance(r[-1], (int, float))]
    na_vals = collections.Counter(repr(r[-1]) for r in na)
    resi_na = [int(r[0]) for r in na]
    pro_pos = {i + 1 for i, a in enumerate(seq) if a == "P"}
    n_pro_na = sum(1 for r in resi_na if r in pro_pos)
    pro_with_value = sorted(int(r[0]) for r in body if isinstance(r[-1], (int, float)) and int(r[0]) in pro_pos)
    n_nterm_na = sum(1 for r in resi_na if r == 1)
    num = [r for r in body if isinstance(r[-1], (int, float))]
    acc.append(dict(protein=prot, source="Nguyen2022 mmc2.xlsx sheet " + sheet, pdb_sheet=sheet, header=" | ".join(map(str, hdr)),
                    n_rows_raw=n_raw, n_NA=len(na), NA_markers=json.dumps(dict(na_vals)), n_NA_proline=n_pro_na,
                    n_NA_Nterminal=n_nterm_na, n_NA_other=len(na) - n_pro_na - n_nterm_na,
                    n_lower_bound_kept=0, n_approx_kept=0, n_merged_unf_fl=0, n_removed_unprotected=0,
                    n_final=len(num), n_final_tier1a_all=int((allp.protein_id == prot).sum()),
                    pro_positions_with_value=json.dumps(pro_with_value), seq_len=len(seq), n_seq_pro=len(pro_pos),
                    nterm_has_value=any(int(r[0]) == 1 for r in num)))
# cytochrome c（Milne 1998）与 RNase H*（Hollien 1999）
cyt = pd.read_csv(os.path.join(INP["tier1a_classics_dir"], "cytc_Milne1998.csv"))
fl = cyt.flag.fillna("")
acc.append(dict(protein="cytochrome_c", source="Milne1998 ProtSci Table 1 (tier1a_classics/cytc_Milne1998.csv)", pdb_sheet="1HRC",
                header="resi,resn,logP,dGHX_kcal,flag", n_rows_raw=len(cyt), n_NA=0, NA_markers="{}", n_NA_proline=0, n_NA_Nterminal=0,
                n_NA_other=0, n_lower_bound_kept=int((fl == "lb").sum()), n_approx_kept=int((fl == "~").sum()), n_merged_unf_fl=0,
                n_removed_unprotected=int((fl == "unprot").sum()), n_final=int((fl != "unprot").sum()),
                n_final_tier1a_all=int((allp.protein_id == "cytochrome_c").sum()), pro_positions_with_value="[]",
                seq_len=len(seqs["cytochrome_c"]["seq"]), n_seq_pro=seqs["cytochrome_c"]["seq"].count("P"),
                nterm_has_value=bool((cyt.resi == 1).any())))
rh = pd.read_csv(os.path.join(INP["tier1a_classics_dir"], "rnaseH_Hollien1999.csv"))
both = rh.dGunf.notna() & rh.dGfl.notna(); only_u = rh.dGunf.notna() & rh.dGfl.isna(); only_f = rh.dGunf.isna() & rh.dGfl.notna()
RTh = 0.0019872041 * (25 + 273.15)
comb = -RTh * np.log(np.exp(-rh.dGunf / RTh) + np.exp(-rh.dGfl / RTh))
acc.append(dict(protein="RNaseH_Tt", source="Hollien1999 PNAS Table 1 (tier1a_classics/rnaseH_Hollien1999.csv)", pdb_sheet="1RIL",
                header="resi,resn,dGunf,dGfl,dGopen_kcal", n_rows_raw=len(rh), n_NA=0, NA_markers="{}", n_NA_proline=0, n_NA_Nterminal=0,
                n_NA_other=0, n_lower_bound_kept=0, n_approx_kept=0, n_merged_unf_fl=int(both.sum()), n_removed_unprotected=0,
                n_final=len(rh), n_final_tier1a_all=int((allp.protein_id == "RNaseH_Tt").sum()), pro_positions_with_value="[]",
                seq_len=len(seqs["RNaseH_Tt"]["seq"]), n_seq_pro=seqs["RNaseH_Tt"]["seq"].count("P"),
                nterm_has_value=bool((rh.resi == 1).any()),
                ))
A = pd.DataFrame(acc)
A["T_C"] = A.protein.map(allp.groupby("protein_id").T_C.first()); A["pH"] = A.protein.map(allp.groupby("protein_id").pH.first())
A["T_assumed"] = A.protein.map(allp.groupby("protein_id").T_assumed.first())
# 原 01_parse_tier1a.py:17 的元数据注释写明 Ubiquitin “条件 unknown(默认25C)”，但 T_assumed 列为 False（因 META 显式写了 25.0）
A["T_note"] = A.protein.map({"CheY": "T/pH 未知，按 25 °C 假定（01_parse_tier1a.py:13,29）",
                              "Ubiquitin": "01_parse_tier1a.py:17 注释“条件unknown(默认25C)”；T_assumed 列记为 False 但实为默认值"}).fillna("")
A["grade"] = A.protein.map(allp.groupby("protein_id").grade.first()); A["ex2"] = A.protein.map(allp.groupby("protein_id").ex2.first())
A["pdb_scoring"] = A.protein.map(lambda p: maps[p]["pdb_id"]); A["offset_deposited"] = A.protein.map(lambda p: maps[p].get("offset", 0))
A["gen_offset"] = A.protein.map(lambda p: maps[p].get("gen_offset", 0))
A["rnaseH_merge_rule_maxabs"] = np.where(A.protein == "RNaseH_Tt", float(np.abs(rh.dGopen_kcal[both] - comb[both]).max()), np.nan)
A["rnaseH_single_source_equal"] = np.where(A.protein == "RNaseH_Tt",
                                           bool(np.allclose(rh.dGopen_kcal[only_u], rh.dGunf[only_u]) and np.allclose(rh.dGopen_kcal[only_f], rh.dGfl[only_f])), None)

# 编号核对 1：Pro 位点必为 N/A（Nguyen 表按 1 起始编号时成立；否则说明编号错位）
A["check_all_pro_NA"] = [(json.loads(x) == []) if s.startswith("Nguyen") else None for x, s in zip(A.pro_positions_with_value, A.source)]

# 编号核对 2：保留残基的残基名 vs 沉积结构（经与打分完全相同的 fix_frame 处理后，resSeq = resi + offset）
import mdtraj as md
ident = []
for _, r in A.iterrows():
    p = r.protein; pdb = r.pdb_scoring; off = int(r.offset_deposited)
    t = md.load(os.path.join(INP["pdb_dir"], f"{pdb}.pdb")); t = t[0] if t.n_frames > 1 else t
    tf = bv.fix_frame(t, True)
    st = {res.resSeq: bv.T2O.get(res.name, "X") for res in tf.topology.residues}
    g = allp[allp.protein_id == p]
    if p in ("cytochrome_c", "RNaseH_Tt") or g.resn.astype(str).str.strip().ne("").all():
        expected = {int(a.resi): bv.to1(a.resn) for a in g.itertuples()}
        src = "tier1a_all.resn"
    else:
        seq = seqs[p]["seq"]; expected = {int(a.resi): seq[int(a.resi) - 1] for a in g.itertuples()}
        src = f"tier1a_sequences.json[{seqs[p]['pdb']}] (resi→seq[resi−1])"
    matched = sum(1 for k, v in expected.items() if st.get(k + off) == v)
    present = sum(1 for k in expected if (k + off) in st)
    mism = [(k, v, st.get(k + off)) for k, v in expected.items() if (k + off) in st and st[k + off] != v]
    ident.append(dict(protein=p, pdb=pdb, offset=off, expected_source=src, n_kept=len(expected), n_present_in_structure=present,
                      n_resname_match=matched, pct_match=round(100 * matched / max(present, 1), 1), mismatches=json.dumps(mism[:20])))
    # offset 扫描：在打分实际使用的编号（fix_frame 输出，OpenMM 写出时重编号 1..N）上找身份最佳 offset
    scan = {o: sum(1 for k, v in expected.items() if st.get(k + o) == v) for o in range(-8, 9)}
    ident[-1].update(best_offset_identity=max(scan, key=lambda o: (scan[o], -abs(o))), best_offset_matches=max(scan.values()),
                     raw_pdb_first_residue=f"{list(t.topology.residues)[0].name}{list(t.topology.residues)[0].resSeq}",
                     fixed_first_residue=f"{list(tf.topology.residues)[0].name}{list(tf.topology.residues)[0].resSeq}")
I = pd.DataFrame(ident)
I["offset_used_is_identity_optimal"] = I.offset == I.best_offset_identity
# 返修统一编号映射：沉积结构一侧同时给出原稿 offset（主口径，§0.3 规则 5）与残基身份最优 offset（敏感性）
align_map = {r.protein: dict(pdb=r.pdb, deposited_offset_manuscript=int(r.offset), deposited_offset_identity=int(r.best_offset_identity),
                             deposited_identity_pct_manuscript=float(r.pct_match),
                             gen_offset_structure_maps=int(maps[r.protein].get("gen_offset", 0)))
             for r in I.itertuples()}
revio.write_json(align_map, os.path.join(TAB, "S1_alignment_map.json")); outs.append(os.path.join(TAB, "S1_alignment_map.json"))
fI = os.path.join(TAB, "S1_deposited_structure_numbering.csv"); I.to_csv(fI, index=False); outs.append(fI)
A = A.merge(I[["protein", "n_present_in_structure", "n_resname_match", "pct_match", "expected_source", "mismatches",
               "best_offset_identity", "best_offset_matches", "offset_used_is_identity_optimal"]], on="protein", how="left")

# 生成器系综一侧的编号核对（与排行榜 22_leaderboard.py 同一 best_offset 规则；只读 eval/pred 的 resn）
gid = []
for gname in ["BioEmu", "ESMFlow", "Boltz2", "ShortMD", "Str2Str"]:
    for p in sorted(allp.protein_id.unique()):
        pr = pd.read_csv(os.path.join(INP["eval_pred_dir"], gname, f"{p}.csv")); g = allp[allp.protein_id == p]
        pres = {int(a.resSeq): bv.to1(a.resn) for a in pr.itertuples()}
        go = maps.get(p, {}).get("gen_offset", maps.get(p, {}).get("offset", 0))
        bo = bv.best_offset(pres, g, go)
        if g.resn.astype(str).str.strip().ne("").all():
            expected = {int(a.resi): bv.to1(a.resn) for a in g.itertuples()}
        else:
            seq = seqs[p]["seq"]; expected = {int(a.resi): seq[int(a.resi) - 1] for a in g.itertuples()}
        for lab, o in (("gen_offset", go), ("best_offset", bo)):
            pres_n = sum(1 for k in expected if (k + o) in pres)
            gid.append(dict(generator=gname, protein=p, rule=lab, offset=o, n_present=pres_n,
                            n_match=sum(1 for k, v in expected.items() if pres.get(k + o) == v)))
G = pd.DataFrame(gid); G["pct_match"] = (100 * G.n_match / G.n_present.clip(lower=1)).round(1)
fG = os.path.join(TAB, "S1_generator_numbering.csv"); G.to_csv(fG, index=False); outs.append(fG)

# CI2 三个 PDB ID 的用途（方案 S1 / D2）
ci2 = pd.DataFrame([
    dict(pdb_id="1TM1", role="Nguyen 2022 mmc2.xlsx 的工作表名（实验数据来源标识）；tier1a_all.pdb_id 列", used_for_coordinates=False),
    dict(pdb_id="3CI2", role="tier1a_sequences.json 的序列来源（64 aa，成熟序列 1 起始编号；生成器输入序列）", used_for_coordinates=False),
    dict(pdb_id="2CI2", role="沉积单结构打分与短 MD 起始结构（structure_maps.pdb_id；offset +1）", used_for_coordinates=True)])
fC = os.path.join(TAB, "S1_ci2_pdb_roles.csv"); ci2.to_csv(fC, index=False); outs.append(fC)
f1 = os.path.join(TAB, "S1_tier1a_residue_accounting.csv"); A.to_csv(f1, index=False); outs.append(f1)

# 逐残基最终表（522 行）+ 标记
fin = allp.copy()
fin["censor_flag"] = ""; fin["merge_flag"] = ""
cmap = dict(zip(cyt.resi.astype(int), fl))
m = fin.protein_id == "cytochrome_c"
fin.loc[m, "censor_flag"] = fin.loc[m, "resi"].map(lambda x: {"lb": "lower_bound", "~": "approximate"}.get(cmap.get(int(x), ""), ""))
rmap = {int(a.resi): ("dGunf+dGfl_combined" if (pd.notna(a.dGunf) and pd.notna(a.dGfl)) else ("dGunf_only" if pd.notna(a.dGunf) else "dGfl_only"))
        for a in rh.itertuples()}
m = fin.protein_id == "RNaseH_Tt"; fin.loc[m, "merge_flag"] = fin.loc[m, "resi"].map(lambda x: rmap.get(int(x), ""))
f2 = os.path.join(TAB, "S1_tier1a_residues_final.parquet"); fin.to_parquet(f2, index=False); outs.append(f2)

# =============================== 2. Tsuboyama 过滤级联 ===============================
ds = pd.read_csv(INP["tsuboyama_ds23"], low_memory=False,
                 usecols=["name", "WT_name", "mut_type", "aa_seq", "deltaG", "deltaG_95CI_high", "deltaG_95CI_low", "dG_ML", "ddG_ML"])
casc = [dict(step="Dataset2/3 CSV rows", n=len(ds), note="Tsuboyama2023_Dataset2_Dataset3_20230416.csv")]
casc.append(dict(step="unique WT_name (all rows)", n=ds.WT_name.nunique(), note="含一个仅出现在突变行的 WT_name"))
wt = ds[ds.mut_type.astype(str).str.lower() == "wt"].copy()
casc.append(dict(step="mut_type=='wt' rows", n=len(wt), note="同一 WT 有多条重复测量行（_wtm/_wte/_wty/_wth）"))
wt_all_rows = wt.copy()
wt = wt.drop_duplicates(subset=["WT_name"]).reset_index(drop=True)          # 与原 03_parse_tier3.py:9 相同（保留文件中首条 wt 行）
casc.append(dict(step="drop_duplicates(WT_name) → WT domains", n=len(wt),
                 note=f"原 03_parse_tier3.py:5,9；稿件 478。取文件中首条 wt 行：{int((wt.name == wt.WT_name).sum())}/478 为规范行 name==WT_name，"
                      f"例外 {wt[wt.name != wt.WT_name][['WT_name', 'name']].values.tolist()}；"
                      f"{int((wt_all_rows.groupby('WT_name').size() > 1).sum())} 个 WT 有重复测量行，重复行不参与判定"))
# 30% 聚类：用原 MMseqs2 二进制在 478 WT 上复现（与现有 cluster.tsv 逐行一致才使用）
mm = os.path.join(REV, "raw_outputs", "s1", "mmseqs"); os.makedirs(mm, exist_ok=True)
fasta = os.path.join(PROJ, "data", "interim", "tier3_wt.fasta")
cmd = [INP["mmseqs"], "easy-cluster", fasta, os.path.join(mm, "clu30"), os.path.join(mm, "tmp"), "--min-seq-id", "0.3", "-c", "0.7", "--threads", "8"]
r = subprocess.run(cmd, capture_output=True, text=True); open(os.path.join(mm, "mmseqs_easycluster.log"), "w").write(r.stdout + r.stderr)
ver = subprocess.run([INP["mmseqs"], "version"], capture_output=True, text=True).stdout.strip()
newc = pd.read_csv(os.path.join(mm, "clu30_cluster.tsv"), sep="\t", header=None, names=["rep", "mem"])
oldc = pd.read_csv(os.path.join(PROJ, "data", "interim", "tier3_clust30_cluster.tsv"), sep="\t", header=None, names=["rep", "mem"])
same = sorted(map(tuple, newc.values.tolist())) == sorted(map(tuple, oldc.values.tolist()))
fa_names = [x.split("\n")[0] for x in open(fasta).read().split(">")[1:]]
casc.append(dict(step="MMseqs2 easy-cluster --min-seq-id 0.3 -c 0.7 (cov-mode 0) on 478 WT", n=newc.rep.nunique(),
                 note=f"mmseqs {ver}; 与现有 tier3_clust30_cluster.tsv 排序后逐行一致（成员→代表映射相同；行序不同，非逐字节相同）={same}; "
                      f"输入 fasta=478 条且顺序/序列与 CSV WT 一致={fa_names == wt.WT_name.tolist()}"))
if not same:
    raise SystemExit("MMseqs2 复现与现有聚类不一致，停止（方案 S1 风控：先调参数复现）")
import shutil
shutil.rmtree(os.path.join(mm, "tmp"), ignore_errors=True)   # 清理 MMseqs2 临时目录（保持目录整洁）
clu = dict(zip(newc.mem, newc.rep))
wt["cluster_rep"] = wt.WT_name.map(clu); wt["L"] = wt.aa_seq.str.len()
wt["origin"] = np.where(wt.WT_name.astype(str).str.match(r"^[0-9][A-Za-z0-9]{3}", na=False), "natural", "de_novo")  # 原 03:8
wt["dG"] = pd.to_numeric(wt.deltaG, errors="coerce")
wt["ci_lo"] = pd.to_numeric(wt.deltaG_95CI_low, errors="coerce"); wt["ci_hi"] = pd.to_numeric(wt.deltaG_95CI_high, errors="coerce")
wt["ci_width"] = wt.ci_hi - wt.ci_lo
wt["dGML_str"] = wt.dG_ML.astype(str)
wt["dGML_numeric"] = pd.to_numeric(wt.dG_ML, errors="coerce").notna()
ds3 = ds.assign(ddg_num=pd.to_numeric(ds.ddG_ML, errors="coerce").notna()).groupby("WT_name").ddg_num.any()
wt["in_dataset3"] = wt.WT_name.map(ds3).fillna(False).astype(bool)
L = wt[(wt.L >= 40) & (wt.L <= 72)].copy()
_rep = wt_all_rows.assign(num=pd.to_numeric(wt_all_rows.dG_ML, errors="coerce").notna())
_rep = _rep[_rep.WT_name.isin(set(L.WT_name))].groupby("WT_name").num.agg(["any", "all"])
_incons = sorted(_rep[_rep["any"] != _rep["all"]].index)
casc.append(dict(step="length 40–72 aa", n=len(L),
                 note=f"原 03:14；落在 {L.cluster_rep.nunique()} 个 30% 聚类中；其中 {len(_incons)} 个 WT 的 dG_ML 在重复行间不一致（判定只看首行，见 tier3_criterion.md）"))
pd.DataFrame(dict(WT_name=_incons)).to_csv(os.path.join(TAB, "S1_tier3_replicate_dGML_inconsistent.csv"), index=False)
outs.append(os.path.join(TAB, "S1_tier3_replicate_dGML_inconsistent.csv"))
R_ = L[L.dGML_numeric]
casc.append(dict(step="reliability: dG_ML numeric (excludes '>5' and '–')", n=len(R_),
                 note=f"排除 '>5' {int((L.dGML_str == '>5').sum())}、'–/-' {int(L.dGML_str.isin(['-', '–']).sum())}；"
                      f"落在 {R_.cluster_rep.nunique()} 个聚类"))

# =============================== 3. D_new（判据冻结） ===============================
def choose_rep(g):
    rep = g.cluster_rep.iloc[0]
    if rep in set(g.WT_name):
        return g[g.WT_name == rep].iloc[0], "mmseqs_representative"
    g2 = g.sort_values(["ci_width", "WT_name"], kind="mergesort")
    return g2.iloc[0], "narrowest_95CI_member"


rows = []
for rep, g in R_.groupby("cluster_rep", sort=True):
    row, how = choose_rep(g)
    d = row.to_dict(); d["rep_rule"] = how; d["n_reliable_members_in_cluster"] = len(g)
    d["n_members_in_cluster_40_72"] = int((L.cluster_rep == rep).sum()); rows.append(d)
D = pd.DataFrame(rows).sort_values("WT_name").reset_index(drop=True)
casc.append(dict(step="one representative per 30% cluster → D_new", n=len(D),
                 note=f"rep_rule: mmseqs_representative {int((D.rep_rule == 'mmseqs_representative').sum())}, "
                      f"narrowest_95CI_member {int((D.rep_rule == 'narrowest_95CI_member').sum())}"))
# 与原 80 / 65 域的关系 + 已有系综 + AF 模型
sub = json.load(open(INP["tier3_subset"]))
name2t3 = {v["name"]: k for k, v in sub.items()}
def has_ens(gen, t3):
    return bool(t3) and os.path.exists(os.path.join(INP["gen_dir"], gen, t3, "operator_f.npz"))
D["legacy_T3_id"] = D.WT_name.map(name2t3)
D["in_legacy80"] = D.legacy_T3_id.notna(); D["in_legacy65"] = D.legacy_T3_id.map(lambda x: bool(x) and int(x.split("_")[1]) < 65 if isinstance(x, str) else False)
GENS = ["BioEmu_T3", "ESMFlow_T3", "Boltz2_T3", "Str2Str_T3", "Str2Str_T3_d035"]
for gname in GENS:
    D[f"has_ens_{gname}"] = [has_ens(gname, x) if isinstance(x, str) else False for x in D.legacy_T3_id]
D["has_xtc_BioEmu_T3"] = [isinstance(x, str) and os.path.exists(os.path.join(INP["gen_dir"], "BioEmu_T3", x, "samples.xtc")) for x in D.legacy_T3_id]
afdir = os.path.join(REV, "data", "external", "tsuboyama_af", "unz", "AlphaFold_model_PDBs"); af = set(os.listdir(afdir))
def afname(n):
    for c in (n, n.replace("|", ":"), n.replace("|", "_")):
        if c in af:
            return c
    return ""
D["af_model_file"] = D.WT_name.map(afname)
D["af_seq_matches"] = [bool(a) and md.load(os.path.join(afdir, a)).topology.to_fasta()[0] == s for a, s in zip(D.af_model_file, D.aa_seq)]
casc.append(dict(step="D_new with AF model (sequence-identical)", n=int(D.af_seq_matches.sum()),
                 note=f"缺 AF 模型 {int((D.af_model_file == '').sum())} 个"))

# S5B 起始结构映射（D_new ∪ legacy65）：精确同名 AF 模型；否则亲本 AF 模型 + 序列差异位点（偏离 D1-1）
leg65_names = {v["name"] for k, v in sub.items() if int(k.split("_")[1]) < 65}
union = sorted(set(D.WT_name) | leg65_names)
afm = []
for n in union:
    s = wt.set_index("WT_name").loc[n, "aa_seq"]; a = afname(n)
    if a:
        ps = md.load(os.path.join(afdir, a)).topology.to_fasta()[0]
        afm.append(dict(WT_name=n, start="exact_AF", af_file=a, parent_af_file="", n_seq_diff=int(sum(x != y for x, y in zip(ps, s))) if len(ps) == len(s) else -1,
                        seq_diffs="[]", len_match=len(ps) == len(s), in_Dnew=n in set(D.WT_name), in_legacy65=n in leg65_names))
        continue
    parent = n.split(".pdb")[0] + ".pdb"; fp = os.path.join(afdir, parent)
    ps = md.load(fp).topology.to_fasta()[0] if os.path.exists(fp) else ""
    diffs = [(i + 1, x, y) for i, (x, y) in enumerate(zip(ps, s)) if x != y] if len(ps) == len(s) else None
    afm.append(dict(WT_name=n, start="parent_AF_mutated" if diffs is not None else "unresolved", af_file="", parent_af_file=parent if ps else "",
                    n_seq_diff=len(diffs) if diffs is not None else -1, seq_diffs=json.dumps(diffs), len_match=len(ps) == len(s),
                    in_Dnew=n in set(D.WT_name), in_legacy65=n in leg65_names))
AFM = pd.DataFrame(afm)
fA = os.path.join(TAB, "S1_tier3_af_start_map.csv"); AFM.to_csv(fA, index=False); outs.append(fA)
casc.append(dict(step="S5B start structures for D_new ∪ legacy65", n=len(AFM),
                 note=f"exact_AF {int((AFM.start == 'exact_AF').sum())}; parent_AF_mutated {int((AFM.start == 'parent_AF_mutated').sum())} "
                      f"(D_new {int(((AFM.start == 'parent_AF_mutated') & AFM.in_Dnew).sum())}, legacy65 {int(((AFM.start == 'parent_AF_mutated') & AFM.in_legacy65).sum())}); "
                      f"unresolved {int((AFM.start == 'unresolved').sum())}"))
casc.append(dict(step="D_new ∩ original 80 / ∩ original 65", n=int(D.in_legacy80.sum()), note=f"∩65 = {int(D.in_legacy65.sum())}"))
casc.append(dict(step="D_new ∩ Dataset 3 (WT ΔG < 4.75, group 0)", n=int(D.in_dataset3.sum()), note="仅作敏感性子集定义"))
keep = ["WT_name", "cluster_rep", "rep_rule", "n_reliable_members_in_cluster", "n_members_in_cluster_40_72", "L", "origin", "aa_seq",
        "dG", "ci_lo", "ci_hi", "ci_width", "dGML_str", "in_dataset3", "legacy_T3_id", "in_legacy80", "in_legacy65"] + \
       [f"has_ens_{g}" for g in GENS] + ["has_xtc_BioEmu_T3", "af_model_file", "af_seq_matches"]
D = D[keep]
f3 = os.path.join(TAB, "S1_tier3_Dnew.csv"); D.to_csv(f3, index=False); outs.append(f3)
C = pd.DataFrame(casc); f4 = os.path.join(TAB, "S1_tier3_cascade.csv"); C.to_csv(f4, index=False); outs.append(f4)

# =============================== 4. D_legacy65 来源 ===============================
t3w = wt.set_index("WT_name")
leg = []
order = sorted(sub, key=lambda k: int(k.split("_")[1]))
for i, k in enumerate(order):
    v = sub[k]; n = v["name"]
    xtc = os.path.join(INP["gen_dir"], "BioEmu_T3", k, "samples.xtc")
    leg.append(dict(T3_id=k, WT_name=n, dG_json=v["deltaG"], dG_csv=float(t3w.loc[n, "dG"]), length=v["length"],
                    block_of_16=i // 16 + 1, cluster_rep=clu.get(n), is_cluster_rep=(clu.get(n) == n), dG_ML=str(t3w.loc[n, "dG_ML"]),
                    in_dataset3=bool(t3w.loc[n, "in_dataset3"]), origin=t3w.loc[n, "origin"], in_legacy65=int(k.split("_")[1]) < 65,
                    **{f"scored_{g}": has_ens(g, k) for g in GENS},
                    bioemu_xtc_exists=os.path.exists(xtc),
                    bioemu_xtc_mtime=pd.Timestamp(os.path.getmtime(xtc), unit="s", tz="UTC").tz_convert("Asia/Shanghai").strftime("%m-%d %H:%M:%S") if os.path.exists(xtc) else "",
                    in_Dnew=n in set(D.WT_name)))
LG = pd.DataFrame(leg)
npz_t = [os.path.getmtime(f) for f in glob.glob(os.path.join(INP["gen_dir"], "BioEmu_T3", "*", "operator_f.npz"))]
summ = dict(n_subset=len(LG), blocks=[f"{LG[LG.block_of_16 == b].dG_json.min():.2f}–{LG[LG.block_of_16 == b].dG_json.max():.2f}" for b in range(1, 6)],
            n_scored_each_generator={g: int(LG[f"scored_{g}"].sum()) for g in GENS},
            scored_ids_equal_T3_000_064=all(set(LG[LG[f"scored_{g}"]].T3_id) == {f"T3_{i:03d}" for i in range(65)} for g in GENS),
            excluded15_dG=f"{LG[~LG.in_legacy65].dG_json.min():.2f}–{LG[~LG.in_legacy65].dG_json.max():.2f}",
            excluded15_dGML=collections.Counter(LG[~LG.in_legacy65].dG_ML.map(lambda s: "numeric" if re.match(r"^-?\d", s) else s)),
            legacy65_all_dGML_numeric=bool(LG[LG.in_legacy65].dG_ML.map(lambda s: bool(re.match(r"^-?\d", s))).all()),
            n_clusters_80=int(LG.cluster_rep.nunique()), n_clusters_65=int(LG[LG.in_legacy65].cluster_rep.nunique()),
            n_cluster_reps_80=int(LG.is_cluster_rep.sum()),
            bioemu_all80_xtc=bool(LG.bioemu_xtc_exists.all()),
            last_scored_npz_time=pd.Timestamp(max(npz_t), unit="s", tz="UTC").tz_convert("Asia/Shanghai").strftime("%m-%d %H:%M:%S"),
            json_dG_equals_csv=bool(np.allclose(LG.dG_json, LG.dG_csv)), n_legacy65_in_Dnew=int(LG[LG.in_legacy65].in_Dnew.sum()))
LG.attrs = {}
f5 = os.path.join(TAB, "S1_tier3_legacy65_provenance.csv"); LG.to_csv(f5, index=False); outs.append(f5)
revio.write_json(dict(summ, excluded15_dGML=dict(summ["excluded15_dGML"])), os.path.join(TAB, "S1_tier3_legacy65_summary.json"))
outs.append(os.path.join(TAB, "S1_tier3_legacy65_summary.json"))

# =============================== 5. Start2Fold 依赖 ===============================
gl = []
for root in ["code", "figures/scripts", "results", "eval", "operator", "verify"]:
    rp = os.path.join(PROJ, root)
    r = subprocess.run(["grep", "-rIl", "-i", "-E", r"start2fold|tier2|STF0", rp], capture_output=True, text=True)
    for fn in r.stdout.split():
        gl.append(os.path.relpath(fn, PROJ))
with open(os.path.join(LOGD, "start2fold_dependency_grep.txt"), "w") as f:
    f.write("grep -rIl -i -E 'start2fold|tier2|STF0' over code/ figures/scripts/ results/ eval/ operator/ verify/\n")
    f.write("\n".join(gl) + "\n")
    f.write(f"N_FILES={len(gl)}\n")
outs.append(os.path.join(LOGD, "start2fold_dependency_grep.txt"))

# =============================== 6. SNase n=129 账目 ===============================
t = md.load(os.path.join(INP["pdb_dir"], "1SNO.pdb")); tf = bv.fix_frame(t[0], True)
cc = pd.read_csv(os.path.join(INP["results_stats"], "hdxer_crosscheck_SNase.csv"))
mine = pd.read_csv(os.path.join(INP["results_stats"], "hdxer_mine_SNase.csv"))
ex = set(allp[allp.protein_id == "SNase"].resi.astype(int))
sn = []
for res in tf.topology.residues:
    if res.name not in bv.T2O:
        continue
    sn.append(dict(resSeq_fixed=res.resSeq, resn=bv.T2O[res.name], is_proline=res.name == "PRO",
                   in_hdxer_mine_130=res.resSeq in set(mine.resSeq), in_hdxer_crosscheck_129=res.resSeq in set(cc.resSeq),
                   has_experiment_101=res.resSeq in ex))
SN = pd.DataFrame(sn)
f6 = os.path.join(TAB, "S1_snase_n129_map.csv"); SN.to_csv(f6, index=False); outs.append(f6)
n129 = dict(n_residues_fixed=len(SN), n_pro=int(SN.is_proline.sum()), n_mine=int(SN.in_hdxer_mine_130.sum()),
            n_crosscheck=int(SN.in_hdxer_crosscheck_129.sum()), crosscheck_missing_vs_mine=SN[SN.in_hdxer_mine_130 & ~SN.in_hdxer_crosscheck_129].resSeq_fixed.tolist(),
            n_exp=int(SN.has_experiment_101.sum()), n_exp_and_crosscheck=int((SN.has_experiment_101 & SN.in_hdxer_crosscheck_129).sum()),
            exp_not_in_crosscheck=SN[SN.has_experiment_101 & ~SN.in_hdxer_crosscheck_129].resSeq_fixed.tolist(),
            pdb_resSeq_range_raw=[int(min(r.resSeq for r in t.topology.residues if r.name in bv.T2O)), int(max(r.resSeq for r in t.topology.residues if r.name in bv.T2O))],
            note="OpenMM PDBFile.writeFile(keepIds=False) 把 fix_frame 输出重编号为 1..N；SNase 构建体编号 1..136 与之对齐（offset 0）")
revio.write_json(n129, os.path.join(TAB, "S1_snase_n129_summary.json")); outs.append(os.path.join(TAB, "S1_snase_n129_summary.json"))

# ------------------------------- 汇总打印 -------------------------------
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(A[["protein", "n_rows_raw", "n_NA", "n_NA_proline", "n_NA_Nterminal", "n_NA_other", "n_lower_bound_kept", "n_approx_kept",
         "n_merged_unf_fl", "n_removed_unprotected", "n_final", "n_final_tier1a_all", "check_all_pro_NA", "nterm_has_value",
         "n_present_in_structure", "n_resname_match", "pct_match", "T_C", "pH", "T_assumed"]].to_string(index=False))
print("SUM n_final =", int(A.n_final.sum()), "| SUM tier1a_all =", int(A.n_final_tier1a_all.sum()))
print(C.to_string(index=False)); print(json.dumps(summ, default=str, ensure_ascii=False, indent=1)); print(json.dumps(n129, ensure_ascii=False))
print("Start2Fold grep files:", gl)
print("D_new origin:", D.origin.value_counts().to_dict(), "dG range", round(D.dG.min(), 2), round(D.dG.max(), 2))
log.finish(outs, note=f"D_new={len(D)}; sum_final={int(A.n_final.sum())}")
