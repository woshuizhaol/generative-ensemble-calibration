"""返修共享 I/O：路径解析（读 config/paths.yaml，禁止脚本内硬编码绝对路径）、哈希、运行登记。"""
import os, json, hashlib, time, socket, getpass, subprocess, sys
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
REV_ROOT_DEFAULT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def load_paths():
    """返回 (project_root, revision_root, inputs dict[绝对路径])。环境变量可覆盖根目录。"""
    cfg = yaml.safe_load(open(os.path.join(REV_ROOT_DEFAULT, "config", "paths.yaml")))
    proj = os.environ.get("GEC_PROJECT_ROOT", cfg["project_root"])
    rev = os.environ.get("GEC_REVISION_ROOT", cfg.get("revision_root", REV_ROOT_DEFAULT))
    proj = proj if os.path.isabs(proj) else os.path.abspath(os.path.join(REV_ROOT_DEFAULT, proj))   # 相对路径相对 revision 根目录解析：存档包里 project_root=inputs、revision_root=. 即可搬迁
    rev = rev if os.path.isabs(rev) else os.path.abspath(os.path.join(REV_ROOT_DEFAULT, rev))
    inp = {k: os.path.join(proj, v) for k, v in cfg["inputs"].items()}
    return proj, rev, inp


def load_seeds():
    return yaml.safe_load(open(os.path.join(REV_ROOT_DEFAULT, "config", "seeds.yaml")))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_dir(path, pattern_ext=(".py",)):
    """目录内指定扩展名文件的联合哈希（按相对路径排序），用于登记脚本版本。"""
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for fn in sorted(files):
            if fn.endswith(pattern_ext):
                fp = os.path.join(root, fn)
                h.update(os.path.relpath(fp, path).encode())
                h.update(sha256(fp).encode())
    return h.hexdigest()


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)


class RunLogger:
    """把一次运行登记到 run/run_registry.tsv 与 run/commands.sh（revision_proposal.md §4.2）。"""

    COLS = ["run_id", "stage", "task", "start_iso", "end_iso", "wall_s", "host", "cpu_procs",
            "gpu_ids", "env", "script", "script_sha256", "config_sha256", "seeds", "inputs_sha256",
            "outputs", "exit_code", "status", "note"]

    def __init__(self, stage, task, cpu_procs=1, gpu_ids="", seeds="", note=""):
        _, self.rev, _ = load_paths()
        self.stage, self.task = stage, task
        self.cpu_procs, self.gpu_ids, self.seeds, self.note = cpu_procs, gpu_ids, seeds, note
        self.t0 = time.time()
        self.start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.run_id = f"S{stage}_{task}_{time.strftime('%Y%m%d%H%M%S')}"
        self.script = os.path.abspath(sys.argv[0])
        self.cmd = " ".join([sys.executable] + sys.argv)
        os.makedirs(os.path.join(self.rev, "run"), exist_ok=True)   # 存档包里没有 run/ 目录（独立复现测试发现），首次运行时创建并写登记表表头
        _reg = os.path.join(self.rev, "run", "run_registry.tsv")
        if not os.path.exists(_reg):
            with open(_reg, "w") as f:
                f.write("\t".join(self.COLS) + "\n")
        with open(os.path.join(self.rev, "run", "commands.sh"), "a") as f:
            f.write(f"# ==== [S{stage}] {task} | {self.start_iso} | host={socket.gethostname()} | user={getpass.getuser()}\n")
            f.write(f"# cwd={os.getcwd()}\n# env={sys.prefix} python={sys.version.split()[0]}\n")
            f.write(f"# script_sha256={sha256(self.script)} seeds={seeds}\n{self.cmd}\n")

    def finish(self, outputs, exit_code=0, status="ok", inputs_sha256="", note=None):
        end_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        wall = round(time.time() - self.t0, 1)
        row = dict(run_id=self.run_id, stage=self.stage, task=self.task, start_iso=self.start_iso,
                   end_iso=end_iso, wall_s=wall, host=socket.gethostname(), cpu_procs=self.cpu_procs,
                   gpu_ids=self.gpu_ids, env=sys.prefix, script=self.script,
                   script_sha256=sha256(self.script), config_sha256=sha256(os.path.join(self.rev, "config", "paths.yaml")),
                   seeds=self.seeds, inputs_sha256=inputs_sha256, outputs=";".join(outputs),
                   exit_code=exit_code, status=status, note=note if note is not None else self.note)
        with open(os.path.join(self.rev, "run", "run_registry.tsv"), "a") as f:
            f.write("\t".join(str(row[c]).replace("\t", " ").replace("\n", " ") for c in self.COLS) + "\n")
        with open(os.path.join(self.rev, "run", "commands.sh"), "a") as f:
            f.write(f"# end={end_iso} wall_s={wall} exit={exit_code} status={status} outputs={';'.join(outputs)}\n\n")
        return row
