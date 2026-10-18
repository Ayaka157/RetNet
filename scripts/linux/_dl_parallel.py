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

    resume_ok = True
    for attempt in range(1, RETRIES + 1):
        # ⚠️ 每一次重试前都要重新检查「超大」：curl 对超大文件返回 rc=0，
        #    光看返回码会一直重试却永远不收敛。实测 train_00010 比期望多 32768 字节，
        #    6 次重试后 have 一动不动（2026-10-18）。
        #    超大说明之前从错误偏移续写过，只能删掉从头下。
        if out.exists() and out.stat().st_size > want:
            log(f"  超大文件，删除重下 {out.name}（have={out.stat().st_size} want={want}）")
            out.unlink()
        # rc=33 = 服务端不支持 Range：带 -C - 会一直失败，必须丢掉残缺文件从头下
        resume_flag = ["-C", "-"] if resume_ok else []
        cmd = ["curl", "-L", *resume_flag, "-sS",
               "--retry", "3", "--retry-delay", "3",
               "--connect-timeout", "20", "--max-time", "3600",
               "--speed-limit", "512", "--speed-time", "180",
               "-o", str(out), url]
        rc = subprocess.call(cmd)
        have = out.stat().st_size if out.exists() else 0
        if have == want:
            return out.name, have, "ok"
        log(f"  {out.name} 第 {attempt} 次: rc={rc} have={have}/{want}")
        if rc == 33:                     # 不支持断点续传 -> 下一轮从头下
            resume_ok = False
            if out.exists():
                out.unlink()
        time.sleep(min(30, 3 * attempt))
    have = out.stat().st_size if out.exists() else 0
    return out.name, have, "fail"


def main() -> int:
    # ⚠️ 单实例锁：两个下载器同时写同一批文件会交错追加，得到「比期望还大」的坏文件。
    #    2026-10-18 实测：手工重启下载器时没发现监督器又拉起了一个，两个实例并存
    #    十几分钟，train_00009/00013 都被写超（+6.7 MB / +64 MB），只能删了重下。
    #    用 flock 保证全局唯一，比「pgrep 数进程」可靠。
    import fcntl
    lock_fh = open(ROOT / "logs/.dl_parallel.lock", "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("已有一个下载器实例在跑，本实例退出")
        return 0

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
