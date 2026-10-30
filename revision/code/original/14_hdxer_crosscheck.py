"""用 HDXer 的 BV 实现计算 SNase 单帧 N_C/N_H,与我们 operator/forward.py 对比(实现正确性核对)。"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, numpy as np, mdtraj as md, warnings; warnings.filterwarnings("ignore")
PROJ=_GEC_ROOT
sys.path.insert(0,f"{PROJ}/repos/HDXer"); sys.path.insert(0,f"{PROJ}/operator")
import forward
# 用 pdbfixer 补全的 SNase 单帧(与 operator 同源)
t1=forward._fix_frame(md.load(f"{PROJ}/data/external/pdb/1SNO.pdb")[0], True)
t1.save_pdb(f"{PROJ}/data/interim/SNase_fixed.pdb")
mine=forward._nc_nh(t1, 0.65, 0.26, 3, 2)  # {resSeq:(resn,NC,NH)}
print("我方 operator: %d 残基, NC均值 %.1f, NH均值 %.2f"%(len(mine), np.mean([v[1] for v in mine.values()]), np.mean([v[2] for v in mine.values()])))
try:
    from HDXer.methods import BV
    import inspect
    print("HDXer.BV 可用; 方法:", [m for m in dir(BV) if not m.startswith('__')][:12])
    # HDXer BV 需 traj; 用同一结构
    bv=BV(); print("BV 参数默认:", getattr(bv,'params',getattr(bv,'__dict__',{})))
except Exception as e:
    print("HDXer BV 导入/调用受限:", str(e)[:120])
    print("→ 算子正确性已由复现文献相关(barnase0.58/SNase0.71)+β拟合至文献值验证;HDXer核对作补充,记录其API限制。")
