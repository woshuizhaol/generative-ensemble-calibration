"""Stage 9 追加（U-S3-2）· 步骤 3c：把 Seminario 力常数与类比参数组装成 `param/heme_c.frcmod`，并写出参数来源表 `param/heme_c_param_provenance.tsv`。
基础参数：Amber 自带的血红素 b 参数 `frcmod.hemall`（Giammona / Bayly，已复制为 param/frcmod.hemall_orig）负责卟啉大环内部（键、角、二面角、非键的 FE、NP、NO 等）；
本文件只补 tleap 报缺的项：
  (1) Fe 位点的键与角（FE–S、S–FE–NP、S–FE–NO、NB–FE–S、CT/2C–S–FE）：取自 Gaussian 小模型的 Seminario 力常数与优化几何（`seminario_fe_site.tsv`，K_Amber = k/2，r0、θ0 取小模型优化值），
      同一原子类型对的多个值取平均并记录标准差；
  (2) 硫醚（Cys 的 SG–CAB / SG–CAC 与卟啉 C3B / C3C 上的 CH(S)CH3）：S–CT–CB 沿用 ff14SB 的 S–CT–CT（50.0 kcal/mol/rad²，108.6°），CB–CT–H1 沿用血红素 b 参数里的 CB–CT–HC（35.0，109.5°）；
  (3) 扭转：X–S–FE–X 取零势垒（与 frcmod.hemall 里 X–NB/NP/NO–FE–X 一致）；
  (4) 近线性的角（θ0 > 170°）：Seminario 值若 >0 也把力常数置 0（Amber 的角项在 180° 附近数值不稳定，且 cis 角已足够约束八面体），在来源表里标注。
参数来源表每行：term、K、r0 或 θ0、source、n_values、sd。
用法：python s9d_cytc_heme_frcmod.py [--sem param/seminario_fe_site.tsv]"""
import os, sys, json, argparse, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths(); P = os.path.join(REV, "md_cytc_heme", "param")
ap = argparse.ArgumentParser(); ap.add_argument("--sem", default=os.path.join(P, "seminario_fe_site.tsv")); a = ap.parse_args()
log = revio.RunLogger(stage=9, task="cytc_heme_frcmod", cpu_procs=1, seeds="none")
S = pd.read_csv(a.sem, sep="\t")
# 模型里的标签 → Amber 原子类型
TYPE = {"FE": "FE"}
for r in "AC": TYPE[f"HEC.N{r}"] = "NP"
for r in "BD": TYPE[f"HEC.N{r}"] = "NO"
for r in "ABCD": TYPE[f"HEC.C1{r}"] = "CC"; TYPE[f"HEC.C4{r}"] = "CC"
TYPE["HID18.NE2"] = "NB"; TYPE["MET80.SD"] = "S"; TYPE["MET80.CG"] = "2C"; TYPE["MET80.CE"] = "CT"; TYPE["HID18.CE1"] = "CR"; TYPE["HID18.CD2"] = "CV"


def pad(key):
    return "-".join(x.ljust(2) for x in key.split("-"))


def ty(label):
    return TYPE.get(label, label)


rows = collections.defaultdict(list)
for r in S.itertuples():
    parts = r.atoms.split("-"); t = [ty(x) for x in parts]; key = "-".join(t if r.kind == "bond" or t[0] <= t[-1] else t[::-1]); rows[(r.kind, key)].append((r.K_amber, r.geom))
prov = []; bond_lines = []; angle_lines = []
for (kind, key), v in sorted(rows.items()):
    K = np.array([x[0] for x in v]); g = np.array([x[1] for x in v]); k_mean = float(K.mean()); g_mean = float(g.mean())
    lin = kind == "angle" and g_mean > 170.0
    k_use = 0.0 if lin else k_mean
    line = f"{pad(key):<10} {k_use:>9.3f} {g_mean:>10.4f}"
    (bond_lines if kind == "bond" else angle_lines).append(line)
    prov.append(dict(term=key, kind=kind, K=round(k_use, 3), geom=round(g_mean, 4), source="Seminario (g16 UB3LYP/6-31G* small model)" + ("; linear angle, K set to 0" if lin else ""), n_values=len(v), sd=round(float(K.std(ddof=1)), 3) if len(v) > 1 else 0.0, K_seminario_mean=round(k_mean, 3)))
EXTRA_ANGLES = [("S-CT-CB", 50.0, 108.6, "ff14SB S–CT–CT 类比（parm10）"), ("CB-CT-H1", 35.0, 109.5, "血红素 b 参数 CB–CT–HC 类比（frcmod.hemall）")]
for t, K, th, src in EXTRA_ANGLES:
    angle_lines.append(f"{pad(t):<10} {K:>9.3f} {th:>10.4f}"); prov.append(dict(term=t, kind="angle", K=K, geom=th, source=src, n_values=0, sd=0.0, K_seminario_mean=np.nan))
with open(os.path.join(P, "heme_c.frcmod"), "w") as fh:
    fh.write("Heme c / Fe(III) low-spin site: Seminario force constants from the g16 small model, plus thioether analogues\n")
    fh.write("BOND\n" + "\n".join(bond_lines) + "\n\nANGLE\n" + "\n".join(angle_lines) + "\n\nDIHE\nX -S -FE-X    1       0.000       0.000       2.000   Fe-S(Met) torsion: no barrier (same as X-NB/NP/NO-FE-X in frcmod.hemall)\n\nIMPROPER\n\n")   # 不写 MASS / NONBON：tleap 要求二者同时出现或同时缺失；S 与 FE 的质量和范德华参数已在 ff14SB 与 frcmod.hemall 里
pd.DataFrame(prov).to_csv(os.path.join(P, "heme_c_param_provenance.tsv"), sep="\t", index=False)
print(open(os.path.join(P, "heme_c.frcmod")).read()); log.finish([os.path.join(P, "heme_c.frcmod"), os.path.join(P, "heme_c_param_provenance.tsv")])
