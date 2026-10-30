"""Stage 5 · Str2Str 推理的带种子包装（原仓库 src/eval.py 在采样路径上不设种子，train.py 才有 seed_everything）。
不修改原仓库：先 lightning.seed_everything(seed, workers=True)，再调用原仓库的 main()。其余 hydra 覆盖项照常从命令行传入。
用法：S5_SEED=11 CUDA_VISIBLE_DEVICES=0 PROJECT_ROOT=<repo> ... python s5_str2str_run.py task_name=inference ckpt_path=... model.inference.delta_min=0.25 ..."""
import os, sys
seed = int(os.environ["S5_SEED"])
repo = os.environ["PROJECT_ROOT"]; os.chdir(repo); sys.path.insert(0, repo)
import lightning as L
L.seed_everything(seed, workers=True)
print(f"[s5_str2str_run] seed_everything({seed}) cwd={os.getcwd()}", flush=True)
from src.eval import main
main()
