"""Stage 5C 诊断 · 笛卡尔噪声的散布—σ 关系（排查 16/60 个条件未落入 ±5% 是校准算法的问题还是物理上不可达）。
对未收敛最严重的条件（默认 cyt c×ShortMD、CI2×Str2Str_xtal、CI2×BioEmu），在 σ 网格上用与正式运行相同的约束最小化（noise.Minimizer）生成 n 个构象，
报告最小化后 CA 成对 RMSD（在与目标的公共 CA 上，×10 得 Å）。用校准种子 101（不用最终种子 21–25，避免和正式系综混用）。
目的是看关系是否单调、目标散布 s_G 是否落在可达范围内、正式运行最终采用的 σ 对应什么散布。不产生任何进入分析的系综。
输出：tables/S5_noise_cart_diagnostic.csv"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS"):
    os.environ.setdefault(_k, "1")
import sys, argparse, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import revio, geom, noise  # noqa: E402
import s5_noise_ensembles as N  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", default="cytochrome_c:ShortMD,CI2:Str2Str_xtal,CI2:BioEmu"); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sigmas", default="0.3,0.5,0.7,0.9,1.2,1.6,2.0,3.0,4.0"); a = ap.parse_args()
    PROJ, REV, INP = revio.load_paths(); log = revio.RunLogger(stage=5, task="noise_cart_diagnostic", cpu_procs=1, seeds="calibration seed 101")
    MPmap = pd.read_csv(os.path.join(REV, "tables", "S4_old_column_to_frame_map.csv")); cal = pd.read_csv(os.path.join(REV, "tables", "S5_noise_calibration.csv")); rows = []
    for case in a.cases.split(","):
        p, cond = case.split(":"); R = N.setup_ref(p, REV, INP); pos, ca_idx = R["pos"], R["ca_idx"]; M = noise.Minimizer(R["top"])
        X, sq = N.load_target(cond, p, REV, INP, MPmap); a_idx, b_idx = N.common_idx(R["seq_ref"], sq); s_t = geom.mean_pairwise_rmsd(X[:, b_idx]) * 10
        c = cal[(cal.protein == p) & (cal.noise_type == "cart") & (cal.condition == cond)].iloc[0]
        for sg in [float(x) for x in a.sigmas.split(",")]:
            t0 = time.time(); rng = np.random.default_rng(101); pre = [noise.cart_noise(pos, R["heavy_idx"], R["h_parent"], sg / 10.0, rng) for _ in range(a.n)]
            Y = np.array([M.minimize(x) for x in pre]); sp = geom.mean_pairwise_rmsd(Y[:, ca_idx][:, a_idx]) * 10
            sp_pre = geom.mean_pairwise_rmsd(np.array(pre)[:, ca_idx][:, a_idx]) * 10
            rows.append(dict(protein=p, condition=cond, sigma_A=sg, n_conformers=a.n, spread_before_minimization_A=float(sp_pre), spread_after_minimization_A=float(sp), compression=float(sp / sp_pre), target_spread_A=float(s_t),
                             formal_run_param_final_A=float(c.param_final), formal_run_realized_A=float(c.realized_mean_A), wall_s=time.time() - t0))
            print(rows[-1], flush=True)
    D = pd.DataFrame(rows); f = os.path.join(REV, "tables", "S5_noise_cart_diagnostic.csv"); D.to_csv(f, index=False)
    for (p, cond), d in D.groupby(["protein", "condition"]):
        print(p, cond, "target %.3f A; monotonic increasing after minimization: %s" % (d.target_spread_A.iloc[0], bool(np.all(np.diff(d.spread_after_minimization_A.values) > 0))))
    log.finish([f], note="cart diagnostic")


if __name__ == "__main__":
    main()
