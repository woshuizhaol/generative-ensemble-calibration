"""原项目脚本（存档包 code/original）的输入引用静态扫描：提取 `{PROJ}/…`、`join(PROJ, "…")`、`_GEC_ROOT + "/…"` 等路径模板的静态前缀，
检查它在原项目与存档包 inputs/ 里是否存在、大小多少。被 s11_build_release.py（补齐包内输入）与 s11_original_inputs_scan.py（输出核查表）共用。"""
import os, re
import pandas as pd

NAMES = r"(?:PROJ|ROOT|P|BASE|PROJECT|proj|_GEC_ROOT)"
PAT = re.compile(r"""f?["']\{""" + NAMES + r"""\}/([^"']+)["']""")
PAT_ADD = re.compile(NAMES + r"""\s*\+\s*f?["']/([^"']+)["']""")
PAT_JOIN = re.compile(r"""(?:os\.path\.join|join)\(\s*""" + NAMES + r"""\s*,\s*((?:["'][^"']+["']\s*,?\s*)+)\)""")


def dsize(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) else -1


def scan(orig_dir, project, inp_dir):
    rows = []
    for fn in sorted(os.listdir(orig_dir)):
        if not fn.endswith(".py"):
            continue
        txt = open(os.path.join(orig_dir, fn), errors="ignore").read(); seen = set()
        for m in PAT.finditer(txt):
            seen.add(m.group(1))
        for m in PAT_ADD.finditer(txt):
            seen.add(m.group(1))
        for m in PAT_JOIN.finditer(txt):
            seen.add("/".join(re.findall(r"""["']([^"']+)["']""", m.group(1))))
        for t in sorted(seen):
            pref = re.split(r"[{\*]", t)[0]
            if not pref:
                continue
            pth = pref if os.path.exists(os.path.join(project, pref)) else os.path.dirname(pref)
            pth = pth.rstrip("/")
            rows.append(dict(script=fn, template=t, static_prefix=pth, in_package=bool(pth) and os.path.exists(os.path.join(inp_dir, pth)), in_project=bool(pth) and os.path.exists(os.path.join(project, pth)),
                             size_bytes=dsize(os.path.join(project, pth)) if pth else -1))
    return pd.DataFrame(rows).drop_duplicates(["script", "static_prefix"]).reset_index(drop=True)
