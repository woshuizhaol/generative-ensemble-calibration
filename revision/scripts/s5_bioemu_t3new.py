"""Stage 5B · BioEmu 补跑：给 D_new ∪ D_legacy65 中还没有 BioEmu 系综的域（143 个，含 T3_078）生成系综。
设置与原 T3 运行一致（S3 溯源表）：bioemu-v1.1、请求 200 个样本、批大小 --batch_size_100 10、过滤开启（默认）。方案要求"若命令行支持种子参数，统一设为 11"：--base_seed 11。
与原运行的唯一差别：检查点用 R/cache/ckpt 里经哈希核对的本地副本（--ckpt_path/--model_config_path），因为 hf-mirror 不返回 BioEmu 取检查点所需的 HEAD 元数据（工程日志）；权重与 HF 公布值一致（S3）。
多泳道：每个域用 os.mkdir(claims/<域>) 原子领取，可以随时再开一条泳道（不同 GPU）。v2（S5 返工）：只领取嵌入已缓存的域（采样完全离线，去掉代理变量），没有就绪的就等；与 s5_bioemu_prefetch.py B 的并行嵌入重叠进行。用法：python s5_bioemu_t3new.py <gpu>
输出：raw_outputs/s5/gen/BioEmu_T3new/<域>/{samples.xtc, topology.pdb, …}；logs/s5/bioemu_t3new/<域>.log；raw_outputs/s5/runs/bioemu_<gpu>.tsv（每域一行：命令、GPU、种子、起止时间、帧数、退出码）。"""
import os, sys, time, subprocess, datetime
import pandas as pd
gpu = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from lib import revio  # noqa: E402
P, R, _ = revio.load_paths(); os.chdir(R)   # 路径唯一来源：config/paths.yaml
dom = pd.read_csv("tables/S5_domain_table.csv"); dom = dom[dom.need_BioEmu_T3].sort_values("length", ascending=False)
for d in ("raw_outputs/s5/gen/BioEmu_T3new", "logs/s5/bioemu_t3new", "raw_outputs/s5/runs", "raw_outputs/s5/claims/bioemu"):
    os.makedirs(d, exist_ok=True)
env = dict(os.environ, HOME=R + "/home_bioemu", XDG_CACHE_HOME=R + "/cache", HF_HOME=R + "/cache/huggingface", HF_ENDPOINT="https://hf-mirror.com", TORCH_HOME=R + "/cache/torch",
           MPLCONFIGDIR=R + "/cache/mpl", COLABFOLD_DIR=R + "/home_bioemu/.cache/colabfold", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1")
for _k in ("all_proxy", "https_proxy", "http_proxy", "ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):   # v2：采样完全离线；若误用网络会立刻失败而不是悄悄走隧道
    env.pop(_k, None)
runs = f"raw_outputs/s5/runs/bioemu_gpu{gpu}.tsv"
if not os.path.exists(runs):
    open(runs, "w").write("domain_id\tlength\tgpu\tbase_seed\tnum_samples_requested\tstart\tend\twall_s\texit_code\tn_frames_kept\tcommand\n")
import mdtraj as md
import hashlib
ready = lambda sq: os.path.exists(f"cache/embeds_cache/{hashlib.sha256(sq.encode()).hexdigest()}.fasta")   # v2：嵌入完成才领取（.fasta 是 get_colabfold_embeds 最后写的完成标记）
def next_domain():
    """返回下一个嵌入已就绪且未被领取的域；全部领取完则返回 None；没有就绪的则等待。"""
    while True:
        left = 0
        for q in dom.itertuples():
            if os.path.exists(f"raw_outputs/s5/claims/bioemu/{q.domain_id}"):
                continue
            left += 1
            if ready(q.seq):
                try:
                    os.mkdir(f"raw_outputs/s5/claims/bioemu/{q.domain_id}"); return q
                except FileExistsError:
                    continue
        if left == 0:
            return None
        time.sleep(60)
while True:
    r = next_domain()
    if r is None:
        break
    out = f"raw_outputs/s5/gen/BioEmu_T3new/{r.domain_id}"
    cmd = [P + "/env/bioemu/bin/python", "-m", "bioemu.sample", "--sequence", r.seq, "--num_samples", "200", "--output_dir", out, "--batch_size_100", "10",
           "--ckpt_path", R + "/cache/ckpt/bioemu-v1.1_checkpoint.ckpt", "--model_config_path", R + "/cache/ckpt/bioemu-v1.1_config.yaml", "--base_seed", "11",
           "--cache_so3_dir", R + "/cache/sampling_so3_cache", "--cache_embeds_dir", R + "/cache/embeds_cache"]
    t0 = time.time(); s = datetime.datetime.now().isoformat(timespec="seconds")
    rc = subprocess.run(cmd, env=env, stdout=open(f"logs/s5/bioemu_t3new/{r.domain_id}.log", "w"), stderr=subprocess.STDOUT).returncode
    nf = -1
    if rc == 0 and os.path.exists(out + "/samples.xtc"):
        nf = md.load(out + "/samples.xtc", top=out + "/topology.pdb").n_frames
    open(runs, "a").write("\t".join(map(str, [r.domain_id, r.length, gpu, 11, 200, s, datetime.datetime.now().isoformat(timespec="seconds"), round(time.time() - t0), rc, nf, " ".join(cmd[:4]) + " --sequence <seq> …"])) + "\n")
