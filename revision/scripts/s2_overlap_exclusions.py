"""Stage 2 产出：逐生成器的训练集重叠剔除名单（供 S9 的重叠敏感性分析使用）。

输入：tables/S2_overlap_tier1a.csv（s2_overlap_audit.py 的判定）。
规则（事先写定，与 S2 判定一致）：
  - 主名单 strong：对该生成器的任一训练库判为 strong_overlap；
  - 严格名单 strong_or_homolog：在 strong 之外，加上 homolog（fident≥0.3、覆盖度≥0.5、E≤1e-3）。
生成器与其训练库的对应（依据：BioEmu 预印本 S.1.7 与 S1 的 MODEL_CARD；AlphaFlow/ESMFlow 论文）：
  BioEmu        ds23（MEGAscale Dataset2/3 的 PPFT 微调）、cath2 / cath1 / megasim（MD 训练系统）
  ESMFlow       atlas（ATLAS train+val+test）
  Boltz-2       PDB 结构 + 三套 MD 轨迹：MISATO、ATLAS（1,284 个蛋白，每个取 100 帧）、mdCATH（预印本 A.1.2）；全文无 MEGAscale 或
                折叠自由能数据 → ATLAS 部分与 ESMFlow 同样比对；mdCATH、MISATO 按方案 §1.3“明确不做”未比对，记为 not_determinable
  Str2Str       PDB 结构训练，无实验折叠自由能训练数据 → 不逐条比对（not_applicable）
  ShortMD       本项目自行模拟，无训练集
ds1（MEGAscale Dataset1）与 bbtest（bioemu-benchmarks 测试集）不是任何生成器的训练库，只作参照，不进入剔除名单。
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths()
TAB = os.path.join(REV, "tables")
log = revio.RunLogger(stage=2, task="overlap_exclusions", cpu_procs=1, seeds="none")
T = pd.read_csv(os.path.join(TAB, "S2_overlap_tier1a.csv"))
GEN = {"BioEmu": ["ds23", "cath2", "cath1", "megasim"], "ESMFlow": ["atlas"], "Boltz2": ["atlas"]}
UNDET = {"Boltz2": "mdCATH 与 MISATO 未比对（not_determinable；方案 §1.3 明确不做）。Boltz-2 用的 ATLAS 取自同一份 ATLAS 清单，"
                   "但它按自己的划分留出测试簇，这里 train+val+test 全部计入"}
NA = {"Str2Str": "not_applicable（PDB 结构训练，无实验折叠自由能训练数据）",
      "ShortMD": "no_training_set（本项目自行模拟）"}
prots = sorted(T.protein.unique())
rows = []
for g, libs in GEN.items():
    for p in prots:
        sub = T[(T.protein == p) & (T.library.isin(libs))]
        strong = sub[sub.call == "strong_overlap"]; hom = sub[sub.call == "homolog"]
        rows.append(dict(generator=g, protein=p, strong=len(strong) > 0, strong_libraries=";".join(strong.library),
                         strong_best_target=";".join(f"{r.library}:{r.best_target}" for r in strong.itertuples()),
                         homolog=len(hom) > 0, homolog_libraries=";".join(hom.library),
                         homolog_best_target=";".join(f"{r.library}:{r.best_target}(fident={r.best_fident},E={r.best_evalue:.1e})" for r in hom.itertuples()),
                         exclude_strong=len(strong) > 0, exclude_strict=(len(strong) > 0) or (len(hom) > 0), note=UNDET.get(g, "")))
for g, why in NA.items():
    for p in prots:
        rows.append(dict(generator=g, protein=p, strong=False, strong_libraries="", strong_best_target="", homolog=False, homolog_libraries="",
                         homolog_best_target="", exclude_strong=False, exclude_strict=False, note=why))
E = pd.DataFrame(rows)
f = os.path.join(TAB, "S2_overlap_exclusions.csv"); E.to_csv(f, index=False)
L = []
for g in list(GEN) + list(NA):
    s = E[(E.generator == g) & E.exclude_strong].protein.tolist(); t = E[(E.generator == g) & E.exclude_strict].protein.tolist()
    L.append(dict(generator=g, exclude_strong=";".join(s), n_strong=len(s), exclude_strict=";".join(t), n_strict=len(t)))
L = pd.DataFrame(L); f2 = os.path.join(TAB, "S2_overlap_exclusion_lists.csv"); L.to_csv(f2, index=False)
pd.set_option("display.width", 220); print(L.to_string(index=False))
log.finish([f, f2], note="; ".join(f"{r.generator}: strong={r.exclude_strong or '-'}" for r in L.itertuples()))
