"""Stage 5B · ESMFlow 推理的带种子包装（alphaflow 的 predict.py 没有种子参数，噪声来自全局 torch RNG，S3 已核对）。
不修改原仓库：先 torch/numpy/random 设种子，再用 runpy 执行原仓库的 predict.py（参数原样从命令行传入）。
用法：S5_SEED=11 python s5_esmflow_run.py --mode esmfold --input_csv … --weights … --samples 50 --outpdb … --no_overwrite"""
import os, sys, runpy, random
seed = int(os.environ["S5_SEED"]); repo = os.environ["ALPHAFLOW_REPO"]
os.chdir(repo); sys.path.insert(0, repo)
import numpy as np, torch
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
print(f"[s5_esmflow_run] seed={seed} cwd={os.getcwd()} argv={sys.argv[1:]}", flush=True)
sys.argv = [os.path.join(repo, "predict.py")] + sys.argv[1:]
runpy.run_path("predict.py", run_name="__main__")
