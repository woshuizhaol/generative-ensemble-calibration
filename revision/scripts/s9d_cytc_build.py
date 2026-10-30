"""Stage 9 追加（U-S3-2）· 步骤 5：用 tleap 构建 cyt c 溶剂化体系（无血红素对照 / 带血红素）。
力场与原 ShortMD 逐项对应：蛋白 ff14SB（leaprc.protein.ff14SB）；水 TIP3P-FB（leaprc.water.fb3，对应 amber14/tip3pfb.xml）；
离子 Joung–Cheatham（frcmod.ionsjc_tip3p，对应 tip3pfb.xml 头部的 frcmod.ionsjc_tip3p；leaprc.water.fb3 默认带的是 Li–Merz 离子，所以显式覆盖）。
盒子：立方，边长取 OpenMM Modeller.addSolvent(padding=1.0 nm) 的规则：溶质坐标范围中点到最远原子的距离 r，边长 = 2r + 1.0 nm（原协议 cyt c 为 5.374 nm）。tleap 的 iso 选项会把盒子放大约 28%（水分子数 7520 对 4200），所以不用；
  改为按每个方向的缓冲 (边长 − 溶质范德华包围盒)/2 调用 solvateBox（缓冲给成向量时，tleap 把它当作每个方向的距离）。
离子：0.1 M NaCl（盐对数 = round(水分子数 × 0.1 / 55.4)，与 OpenMM Modeller.addSolvent 的取法相同）加中和离子；先跑一遍 tleap 数水分子，再加离子。
带血红素时读 param/ 下的 HEC.mol2（含 RESP 电荷）、FE.mol2 与 heme_c.frcmod（Fe 位点由 Seminario 得到的力常数 + 补充项），并按晶体结构建 Fe–N(卟啉)×4、Fe–NE2(His18)、Fe–SD(Met80)、SG(Cys14)–CAB、SG(Cys17)–CAC 共 8 个键。
输出（param/）：<name>_solv.prmtop、<name>_solv.inpcrd、<name>_solv.pdb、<name>_build_report.json。
用法：python s9d_cytc_build.py --name cytc_apo|cytc_heme"""
import os, sys, re, json, argparse, subprocess, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402

PROJ, REV, INP = revio.load_paths(); D = os.path.join(REV, "md_cytc_heme"); P = os.path.join(D, "param"); os.makedirs(P, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True, choices=["cytc_apo", "cytc_heme"]); a = ap.parse_args()
log = revio.RunLogger(stage=9, task="cytc_build_" + a.name, cpu_procs=1, seeds="none")
AMB = "/opt/amber/amber22"; env = dict(os.environ, AMBERHOME=AMB, PATH=f"{AMB}/bin:" + os.environ["PATH"])
heme = a.name == "cytc_heme"
start = os.path.join(D, "prep", "cytc_heme_start.pdb" if heme else "cytc_apo_start.pdb")
head = textwrap.dedent(f"""\
    source leaprc.protein.ff14SB
    source leaprc.water.fb3
    loadamberparams frcmod.ionsjc_tip3p
    loadOff fb3box.off
    """)
if heme:
    head += textwrap.dedent("""\
        loadamberparams frcmod.hemall_orig
        loadamberparams heme_c.frcmod
        HEC = loadmol2 HEC.mol2
        FE = loadmol2 FE.mol2
        """)
head += f"x = loadpdb {start}\n"
if heme:
    head += textwrap.dedent("""\
        bond x.105.FE x.106.NA
        bond x.105.FE x.106.NB
        bond x.105.FE x.106.NC
        bond x.105.FE x.106.ND
        bond x.105.FE x.18.NE2
        bond x.105.FE x.80.SD
        bond x.14.SG x.106.CAB
        bond x.17.SG x.106.CAC
        """)
head += "check x\n"


import numpy as np
xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(start) if l.startswith(("ATOM", "HETATM"))])
center = 0.5 * (xyz.min(0) + xyz.max(0)); radius = float(np.linalg.norm(xyz - center, axis=1).max()); WIDTH_A = 2 * radius + 10.0   # 1.0 nm 填充


def tleap(script, tag):
    open(os.path.join(P, f"{a.name}_{tag}.in"), "w").write(script)
    r = subprocess.run(["tleap", "-s", "-f", f"{a.name}_{tag}.in"], cwd=P, env=env, capture_output=True, text=True)
    open(os.path.join(P, f"{a.name}_{tag}.out"), "w").write(r.stdout + r.stderr); return r.stdout + r.stderr


out0 = tleap(head + "solvateBox x FB3BOX 1.0\nquit\n", "pass0")
vdw = [float(v) for v in re.search(r"Solute vdw bounding box:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out0).groups()]
buf = " ".join(f"{(WIDTH_A - v) / 2:.4f}" for v in vdw); SOLV = f"solvateBox x FB3BOX {{{buf}}}\n"
out1 = tleap(head + SOLV + f"saveamberparm x {a.name}_wet.prmtop {a.name}_wet.inpcrd\nquit\n", "pass1")
m = re.search(r"Added (\d+) residues", out1)
assert m, "tleap 输出里没有找到加入的水分子数：\n" + out1[-800:]
nW = int(m.group(1))
q = float(re.search(r"The unperturbed charge of the unit \(([-\d.]+)\)", out1).group(1))
pairs = round(nW * 0.1 / 55.4); net = int(round(q)); n_na = pairs + max(0, -net); n_cl = pairs + max(0, net)
out2 = tleap(head + SOLV + f"addIonsRand x Na+ {n_na}\naddIonsRand x Cl- {n_cl}\ncheck x\nsaveamberparm x {a.name}_solv.prmtop {a.name}_solv.inpcrd\nsavepdb x {a.name}_solv.pdb\nquit\n", "pass2")
err = [l for l in out2.split("\n") if "FATAL" in l or "Could not find" in l or re.search(r"Errors = [1-9]", l)]
box = re.findall(r"Total bounding box for atom centers:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out2)
rep = dict(name=a.name, start=start, box_rule_openmm_width_A=round(WIDTH_A, 3), solute_vdw_box_A=vdw, n_waters=nW, unperturbed_charge=q, salt_pairs=pairs, n_Na=n_na, n_Cl=n_cl, box_A=box[-1] if box else None, tleap_errors=err[:10])
json.dump(rep, open(os.path.join(P, f"{a.name}_build_report.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(rep, ensure_ascii=False))
log.finish([os.path.join(P, f"{a.name}_solv.prmtop")] if os.path.exists(os.path.join(P, f"{a.name}_solv.prmtop")) else [], exit_code=1 if err else 0, status="ok" if not err else "tleap_errors")
sys.exit(1 if err else 0)
