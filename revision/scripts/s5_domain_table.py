"""Stage 5B · 全局域集（D_new ∪ D_legacy65，212 个域）的统一清单：序列、域编号、AF 起点、各生成器已有/待补系综。
域编号：在原项目 80 个锚点域里的用原 T3_id；其余（D_new 独有）用 DN_<清理后的 WT_name>。
输出 tables/S5_domain_table.csv（212 行）与 tables/S5_domain_table_summary.json。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, re, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
log = revio.RunLogger(stage=5, task="domain_table", cpu_procs=1, seeds="none")
T = lambda n: pd.read_csv(os.path.join(REV, "tables", n))
M, L, D = T("S1_tier3_af_start_map.csv"), T("S1_tier3_legacy65_provenance.csv"), T("S1_tier3_Dnew.csv")
sub = json.load(open(INP["tier3_subset"]))
seq_new = dict(zip(D.WT_name, D.aa_seq)); t3id = dict(zip(L.WT_name, L.T3_id))
seq_old = {L.set_index("T3_id").loc[k, "WT_name"]: v["seq"] for k, v in sub.items() if k in set(L.T3_id)}
rows = []
for r in M.itertuples():
    w = r.WT_name; did = t3id.get(w) or "DN_" + re.sub(r"[^A-Za-z0-9_.-]", "_", w)
    sq = seq_new.get(w) or seq_old.get(w)
    chk = (seq_new.get(w) == seq_old.get(w)) if (w in seq_new and w in seq_old) else None
    ens = {g: os.path.exists(os.path.join(INP["gen_dir"], g, did, "samples.xtc")) if did.startswith("T3_") else False
           for g in ("BioEmu_T3", "ESMFlow_T3", "Boltz2_T3", "Str2Str_T3", "Str2Str_T3_d035")}
    rows.append(dict(domain_id=did, WT_name=w, length=len(sq), seq=sq, seq_agrees_between_old_and_new_tables=chk, in_legacy65=bool(r.in_legacy65), in_Dnew=bool(r.in_Dnew), af_start=r.start,
                     af_file=r.af_file if isinstance(r.af_file, str) else r.parent_af_file, **{f"have_{k}": v for k, v in ens.items()}))
X = pd.DataFrame(rows)
for g in ("BioEmu_T3", "ESMFlow_T3", "Boltz2_T3"):
    X[f"need_{g}"] = ~X[f"have_{g}"]
X["need_Str2Str_AF"] = X.af_start.eq("exact_AF")          # 19 个只有亲本模型的域不做（方案原预案）
X["parent_AF_only_no_Str2Str_AF"] = X.af_start.eq("parent_AF_mutated")   # 只有亲本 AF 模型的域，不做 Str2Str_AF
f = os.path.join(REV, "tables", "S5_domain_table.csv"); X.to_csv(f, index=False)
S = dict(n_domains=len(X), length_min=int(X.length.min()), length_max=int(X.length.max()), seq_mismatch_between_tables=int((X.seq_agrees_between_old_and_new_tables == False).sum()),
         n_seq_cross_checked=int(X.seq_agrees_between_old_and_new_tables.notna().sum()), **{c: int(X[c].sum()) for c in X.columns if c.startswith(("have_", "need_", "deferred_"))},
         unique_ids=int(X.domain_id.nunique()), unique_seq=int(X.seq.nunique()))
json.dump(S, open(os.path.join(REV, "tables", "S5_domain_table_summary.json"), "w"), ensure_ascii=False, indent=1); print(json.dumps(S, ensure_ascii=False, indent=1))
log.finish([f], note=f"domains={len(X)}")
