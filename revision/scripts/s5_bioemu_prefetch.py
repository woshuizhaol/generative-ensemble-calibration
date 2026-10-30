"""Stage 5B · BioEmu 的 MSA 与嵌入预取，目的是让后续采样完全不依赖本机的反向隧道。
BioEmu 唯一联网的步骤是向 ColabFold 的 MMseqs2 服务器取 MSA；嵌入缓存按 sha256(序列) 命名，bioemu.sample 会先查缓存。
阶段 A（联网，线程池）：对每条还没有嵌入的序列调用 bioemu 自己的 _get_a3m_string（与 bioemu.sample 里的取法完全相同），把 A3M 存成 cache/msa/<sha>.a3m。
阶段 B（本地 CPU，分区并行；服务器上的 jax 是 CPU 版）：用 get_colabfold_embeds(seq, msa_file=<上面的 a3m>) 算嵌入并写入 cache/embeds_cache（A3M 文本与在线取到的逐字相同，所以嵌入不变）。
顺序：按长度升序（与泳道的降序相反），跳过已有嵌入和已被泳道领取的域，两端不会撞车。可重入。
用法：python s5_bioemu_prefetch.py A [threads] | B <k> <n>"""
import os, sys, time, hashlib, datetime
import pandas as pd
phase = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from lib import revio  # noqa: E402
P, R, _ = revio.load_paths(); os.chdir(R)   # 路径唯一来源：config/paths.yaml
os.environ.update(HOME=R + "/home_bioemu", XDG_CACHE_HOME=R + "/cache", COLABFOLD_DIR=R + "/home_bioemu/.cache/colabfold", MPLCONFIGDIR=R + "/cache/mpl", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  all_proxy="socks5h://127.0.0.1:18099", https_proxy="socks5h://127.0.0.1:18099", http_proxy="socks5h://127.0.0.1:18099")
os.makedirs("cache/msa", exist_ok=True); os.makedirs("raw_outputs/s5/runs", exist_ok=True)
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
dom = pd.read_csv("tables/S5_domain_table.csv"); dom = dom[dom.need_BioEmu_T3].sort_values("length")
have_emb = lambda s: os.path.exists(f"cache/embeds_cache/{sha(s)}_single.npy") and os.path.exists(f"cache/embeds_cache/{sha(s)}_pair.npy")
claimed = lambda d: os.path.exists(f"raw_outputs/s5/claims/bioemu/{d}")
todo = [(r.domain_id, r.seq) for r in dom.itertuples() if not have_emb(r.seq) and not claimed(r.domain_id)]
log = open(f"raw_outputs/s5/runs/bioemu_prefetch_{phase}{sys.argv[2] if phase == 'B' else ''}.tsv", "a")
def w(*a):
    log.write("\t".join(map(str, a)) + "\n"); log.flush()
if phase == "A":
    from concurrent.futures import ThreadPoolExecutor
    from bioemu.get_embeds import _get_a3m_string
    def fetch(item):
        d, s = item; f = f"cache/msa/{sha(s)}.a3m"
        if os.path.exists(f):
            return d, "cached"
        for k in range(4):
            try:
                t0 = time.time(); a3m = _get_a3m_string(s, None, None)
                open(f + ".tmp", "w").write(a3m); os.replace(f + ".tmp", f); w(d, "A", "ok", round(time.time() - t0), datetime.datetime.now().isoformat(timespec="seconds"), a3m.count(">")); return d, "ok"
            except Exception as e:
                w(d, "A", f"retry{k}", repr(e)[:120], datetime.datetime.now().isoformat(timespec="seconds")); time.sleep(20 * (k + 1))
        return d, "FAILED"
    with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as ex:
        res = list(ex.map(fetch, todo))
    print({k: sum(1 for _, v in res if v == k) for k in ("ok", "cached", "FAILED")}, "of", len(res))
else:
    # 服务器上的 jax 是 CPU 版（jaxlib 无 CUDA），嵌入每条约 330 s；分区并行：第 k 个工作进程处理 todo[k::n]。用法：B <k> <n>
    for _k in ("all_proxy", "https_proxy", "http_proxy"):
        os.environ.pop(_k, None)
    os.environ.update(CUDA_VISIBLE_DEVICES="")
    kk, nn = int(sys.argv[2]), int(sys.argv[3]); todo = todo[kk::nn]
    from bioemu.get_embeds import get_colabfold_embeds
    n = 0
    for d, s in todo:
        f = f"cache/msa/{sha(s)}.a3m"
        if not os.path.exists(f) or have_emb(s) or claimed(d):
            continue
        t0 = time.time(); get_colabfold_embeds(s, cache_embeds_dir="cache/embeds_cache", msa_file=f); n += 1
        w(d, "B", "ok", round(time.time() - t0), datetime.datetime.now().isoformat(timespec="seconds"))
    print("embeddings computed:", n)
