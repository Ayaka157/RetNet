"""并行下载 HF 语料（hf-mirror 镜像 + 断点续传）。

为什么不用原来的顺序 curl：
- 官方源会 302 到 `us.aws.cdn.hf.co`，经 AutoDL 的学术代理实测只有 5 KB/s–1.6 MB/s，
  抖动极大；`hf-mirror.com` 单路 0.4 MB/s，**4 路并发合计能到 2.6 MB/s**。
- 顺序下载下一个 13 GB 要十几小时，赶不上计划的阶段切换；并行 6 路约 1.5 小时。

安全边界（与合作的数据监督器一致）：
- 大小等于期望 → 跳过
- 大小**大于**期望 → 确定损坏（旧的 snapshot_download 半成品），先删再下
- 大小**小于**期望 → 正在下载，用 `curl -C -` 续传，绝不删
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/root/autodl-tmp/RetNet")
MANIFEST = ROOT / "logs/_dl_manifest.tsv"
BASE = ROOT / "data/hf"
JOBS = int(os.environ.get("JOBS", "6"))
MIRROR_FROM = "https://huggingface.co"
MIRROR_TO = "https://hf-mirror.com"
RETRIES = 6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_manifest():
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        p = line.rstrip("\r").split("\t")
        if len(p) < 4:
            continue
        url, local, fname, want = p[0], p[1], p[2], int(p[3])
        rows.append((url.replace(MIRROR_FROM, MIRROR_TO), local, fname, want))
    return rows


def fetch(url: str, out: Path, want: int) -> tuple[str, int, str]:
    """下一个文件，返回 (名字, 最终大小, 状态)。"""
    if out.exists():
        have = out.stat().st_size
        if have == want:
            return out.name, have, "skip"
        if have > want:
            out.unlink()
            log(f"  删除超大文件 {out.name}（have={have} want={want}）")
    out.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, RETRIES + 1):
        cmd = ["curl", "-L", "-C", "-", "-sS",
               "--retry", "3", "--retry-delay", "3",
               "--connect-timeout", "20", "--max-time", "3600",
               "--speed-limit", "512", "--speed-time", "180",
               "-o", str(out), url]
        rc = subprocess.call(cmd)
        have = out.stat().st_size if out.exists() else 0
        if have == want:
            return out.name, have, "ok"
        log(f"  {out.name} 第 {attempt} 次: rc={rc} have={have}/{want}")
        time.sleep(min(30, 3 * attempt))
    have = out.stat().st_size if out.exists() else 0
    return out.name, have, "fail"


def main() -> int:
    rows = read_manifest()
    if not rows:
        log("清单为空")
        return 1
    todo = []
    for url, local, fname, want in rows:
        out = BASE / local / fname
        if out.exists() and out.stat().st_size == want:
            continue
        todo.append((url, out, want))
    log(f"清单 {len(rows)} 个文件，待下载 {len(todo)} 个，并发 {JOBS}")
    if not todo:
        log("全部完整")
        return 0

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futs = [ex.submit(fetch, u, o, w) for u, o, w in todo]
        for i, f in enumerate(as_completed(futs), 1):
            name, have, st = f.result()
            if st in ("ok", "skip"):
                ok += 1
            else:
                fail += 1
            if st == "fail":
                log(f"  ❌ {name} 失败 have={have}")
    log(f"完成：成功 {ok} / 失败 {fail} / 共 {len(todo)}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
