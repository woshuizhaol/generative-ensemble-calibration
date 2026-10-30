"""Stage 10 · 只重写 tables/S10_domain_table.csv（域集定义表：D_new / D_legacy65 / D_common_AF193 / D_new ∩ Dataset 3 的成员、ΔG_fold、origin、AF 起点）。
`s10_global_stability.py run` 每次也会写这张表；本脚本用于在不重跑统计的情况下刷新它。"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio  # noqa: E402
import s10_global_stability as S10  # noqa: E402

PROJ, REV, INP = revio.load_paths(); log = revio.RunLogger(stage=10, task="domain_table", cpu_procs=1, seeds="none")
DT = S10.domain_table(); f = os.path.join(REV, "tables", "S10_domain_table.csv"); DT.to_csv(f, index=False); print(DT.shape, "D_new", int(DT.in_Dnew.sum()), "D_legacy65", int(DT.in_legacy65.sum()), "D_common_AF193", int(DT.in_D_common_AF193.sum()), "D_new_dataset3", int(DT.in_dataset3.sum()))
log.finish([f], note="domain table")
