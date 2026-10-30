"""Stage 13 · 生成 ShortMD 副本的任务清单与 GPU 分配（可重入，不依赖随机数）。
蛋白顺序与 PDB 编号取自 tables/S3_md_protocol_per_protein.csv；seed_md = 13000 + 10·i + k（i = 蛋白序号 1–12，k = 副本号 2/3；预登记 PREREG_replicates.md 第 2 节）。
分配：先 r2 后 r3，按体系大小（3×水分子数 + 蛋白原子数的粗估）降序，贪心分到当前累计成本最小的卡（LPT），使两张卡大致同时结束，且 r2 集合先于 r3 集合完整。
输出：md_replicates/jobs_all.tsv 与 md_replicates/jobs_gpu<id>.tsv。用法：python s13_make_md_jobs.py --gpus 0,7"""
import os, argparse
import pandas as pd
from lib import revio

PROJ, REV, INP = revio.load_paths()
ap = argparse.ArgumentParser(); ap.add_argument("--gpus", default="0,7"); a = ap.parse_args()
gpus = a.gpus.split(","); S = pd.read_csv(os.path.join(REV, "tables", "S3_md_protocol_per_protein.csv"))
jobs = []
for i, r in enumerate(S.itertuples(), 1):
    for k in (2, 3):
        cost = 3 * int(r.n_waters) + int(r.protein_residues_in_box) * 17
        jobs.append(dict(protein=r.protein, pdb=r.pdb_id, protein_index=i, replicate=k, seed=13000 + 10 * i + k, est_cost=cost))
J = pd.DataFrame(jobs).sort_values(["replicate", "est_cost"], ascending=[True, False]).reset_index(drop=True)
load = {g: 0 for g in gpus}; assign = []
for r in J.itertuples():
    g = min(gpus, key=lambda x: load[x]); load[g] += r.est_cost; assign.append(g)
J["gpu"] = assign; J["order_in_lane"] = J.groupby("gpu").cumcount() + 1
J.to_csv(os.path.join(REV, "md_replicates", "jobs_all.tsv"), sep="\t", index=False)
for g in gpus:
    J[J.gpu == g].to_csv(os.path.join(REV, "md_replicates", f"jobs_gpu{g}.tsv"), sep="\t", index=False)
print(J.to_string()); print({g: load[g] for g in gpus})
