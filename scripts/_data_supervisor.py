"""数据准备监督器：边训边把后续语料变成训练能直接吃的 flat 分片，并清理原始包。

它解决三件事（对应「训练中定时推送新数据 + 清理数据盘」）：

1. **完整性校验**：按 `logs/_dl_manifest.tsv` 的期望字节数逐个比对，删掉不符的文件
   —— `snapshot_download` 与 `curl -C -` 混用时会产生「比期望还大」的坏文件，
   而 curl 见到文件存在就不再重下，不主动删就会永远卡住。
2. **分词**：某个数据集下载齐全后，调 `build_flat_corpus.py --text-field auto`
   产出 `data_cache/hf_pack_*/shard_*.npy`。训练计划的 flat 阶段直接读这些分片，
   数据一到就绪，回调会自动接力。
3. **清盘**：分片产物校验通过后删除原始 parquet/tar.gz。⚠️ 顺序不能反——
   必须先确认 shard 数量和 meta.json 都在，才删源。

用法：
    setsid nohup /root/miniconda3/bin/python -u scripts/_data_supervisor.py \
        > logs/linux/data_supervisor.out 2>&1 &
"""
import glob
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/RetNet")
TOKENIZER = ROOT / "data_cache/chat_pack_v13/bpe_tokenizer.json"
MANIFEST = ROOT / "logs/_dl_manifest.tsv"
BASE = ROOT / "data/hf"
PY = "/root/miniconda3/bin/python"

#: 每个目标包：源目录、产物目录、期望文件数、是否需要先解 tar.gz
#: cap = 分词产出上限（token）。计划只需要这么多，多分是白费时间与磁盘。
#: math 不在计划里（余额只够跑 pretrain/网页/代码/教材/退火），所以**不分词**，省下 1 个多小时。
SPECS = [
    dict(name="code",     src="ling_coder",        out="hf_pack_code",        n=27, extract=False,
         cap=2_500_000_000),
    dict(name="web",      src="chinese_web_hq",    out="hf_pack_chinese_web", n=13, extract=False,
         cap=700_000_000),
    dict(name="textbook", src="fineweb_edu_zh_v3", out="hf_pack_textbook",    n=18, extract=True,
         cap=400_000_000),
]

INTERVAL = 300          # 每轮的间隔秒数
MAX_ROUNDS = 120        # 最多跑 10 小时


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_manifest() -> dict[str, list[tuple[str, int]]]:
    """local_dir -> [(filename, expected_bytes)]"""
    out: dict[str, list[tuple[str, int]]] = {}
    if not MANIFEST.exists():
        return out
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\r").split("\t")
        if len(parts) < 4:
            continue
        local, fname, want = parts[1], parts[2], int(parts[3])
        out.setdefault(local, []).append((fname, want))
    return out


def check_and_clean(man: dict[str, list[tuple[str, int]]], local: str):
    """校验该数据集；删掉大小不符的文件。返回 (完整?, 已有, 应有)。"""
    files = man.get(local, [])
    if not files:
        return False, 0, 0
    have = 0
    for fname, want in files:
        p = BASE / local / fname
        if not p.exists():
            continue
        size = p.stat().st_size
        if size == want:
            have += 1
        elif size > want:
            # 只有**比期望还大**才是确定损坏：snapshot_download 写了一半的文件
            # 会让 curl -C - 从错误偏移续写，多出来一截。
            # 比期望**小**说明还在下载中，绝不能删 —— 删了它 curl 就得从头再来，
            # 两边每 5 分钟互相打架，永远下不完（2026-10-18 实测踩过）。
            log(f"  删除损坏文件（超大）{local}/{fname} (have={size} want={want})")
            p.unlink()
    return have == len(files), have, len(files)


def pack_ready(out_dir: Path) -> bool:
    shards = sorted(out_dir.glob("shard_*.npy"))
    meta = out_dir / "meta.json"
    if not shards or not meta.exists():
        return False
    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
        return int(d.get("total_tokens", 0)) > 0 and len(d.get("shards", [])) == len(shards)
    except Exception:                                          # noqa: BLE001
        return False


def prepare(spec) -> bool:
    """把一个数据集变成 flat 分片。返回是否已就绪。"""
    out_dir = ROOT / "data_cache" / spec["out"]
    if pack_ready(out_dir):
        return True

    src_dir = BASE / spec["src"]
    # 需要先解压的（教材是 tar.gz 三视图）
    if spec["extract"]:
        tars = sorted(src_dir.rglob("*.tar.gz"))
        if not tars:
            return False
        ex_dir = src_dir / "_extracted"
        if not ex_dir.exists() or not any(ex_dir.rglob("*.json*")):
            ex_dir.mkdir(parents=True, exist_ok=True)
            for t in tars:
                try:
                    with tarfile.open(t, "r:gz") as tf:
                        tf.extractall(ex_dir)
                    log(f"  解压 {t.name}")
                except Exception as exc:                        # noqa: BLE001
                    log(f"  解压 {t.name} 失败: {exc}")
        patterns = [str(ex_dir / "**" / "*.jsonl"), str(ex_dir / "**" / "*.json")]
        kind = "--jsonl" if any(glob.glob(p, recursive=True) for p in patterns) else "--parquet"
        if kind == "--parquet":
            patterns = [str(ex_dir / "**" / "*.parquet")]
    else:
        patterns = [str(src_dir / "**" / "*.parquet")]
        kind = "--parquet"

    files = [f for p in patterns for f in glob.glob(p, recursive=True)]
    if not files:
        return False

    log(f"  开始分词 {spec['name']}：{len(files)} 个文件 -> {out_dir}")
    cmd = [PY, "-u", "scripts/build_flat_corpus.py",
           "--tokenizer", str(TOKENIZER), "--out", str(out_dir), kind, *patterns,
           "--text-field", "auto", "--tokens-per-shard", "250000000"]
    if spec.get("cap"):
        cmd += ["--max-tokens", str(spec["cap"])]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0 or not pack_ready(out_dir):
        log(f"  分词失败 rc={rc}")
        return False
    d = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    log(f"  ✅ {spec['name']} 完成：{d['total_tokens'] / 1e9:.3f}B token / "
        f"{len(d['shards'])} 分片")
    return True


def release_raw(spec) -> None:
    """分片齐全后删掉原始下载，腾磁盘。"""
    src_dir = BASE / spec["src"]
    freed = 0
    for p in list(src_dir.rglob("*")):
        if p.is_file():
            freed += p.stat().st_size
            p.unlink()
    for p in sorted(src_dir.rglob("*"), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass
    if freed:
        log(f"  已清理 {spec['src']} 原始包，释放 {freed / 1e9:.2f} GB")


def downloader_running() -> bool:
    """curl 下载器是否活着（用 ps 查，不依赖 pid 文件）。"""
    try:
        out = subprocess.check_output(["bash", "-c",
            "ps -eo args | grep -c '[s]cripts/linux/_dl_parallel.py'"], text=True, timeout=10)
        return int(out.strip() or 0) > 0
    except Exception:                                          # noqa: BLE001
        return False


def ensure_downloader() -> None:
    """没在下载就拉起来。

    为什么需要：curl 一轮只按清单顺序走一遍，**不会回头**；而监督器先前把
    半成品删掉过（已修），也可能有网络中断导致某轮提前退出。没有这一层，
    缺的文件就永远补不回来。
    """
    if downloader_running():
        return
    log("  下载器未运行，重新拉起")
    # 并行版：hf-mirror 镜像 + 6 路并发。顺序 curl 在抖动代理下只有 5 KB/s-1.6 MB/s，
    # 并行 6 路实测合计 2.6 MB/s，13 GB 从十几小时缩到约 1.5 h。
    subprocess.Popen([PY, "-u", "scripts/linux/_dl_parallel.py"],
                     cwd=str(ROOT),
                     stdout=open(ROOT / "logs/linux/dl_parallel.out", "a"),
                     stderr=subprocess.STDOUT,
                     start_new_session=True)


def main() -> int:
    log(f"数据监督器启动；每 {INTERVAL}s 一轮，最多 {MAX_ROUNDS} 轮")
    for rnd in range(1, MAX_ROUNDS + 1):
        man = read_manifest()
        pending = []
        for spec in SPECS:
            out_dir = ROOT / "data_cache" / spec["out"]
            if pack_ready(out_dir):
                # 包已就绪 → 顺手清原始下载
                if (BASE / spec["src"]).exists():
                    release_raw(spec)
                continue
            local_full = f"data/hf/{spec['src']}"
            complete, have, want = check_and_clean(man, local_full.replace("data/hf/", ""))
            if not complete:
                pending.append(f"{spec['name']} {have}/{want}")
                continue
            log(f"=== {spec['name']} 下载齐全（{have}/{want}），开始分词 ===")
            if prepare(spec):
                release_raw(spec)
            else:
                pending.append(f"{spec['name']} 分词未完成")

        free = os.statvfs(str(ROOT)).f_bavail * os.statvfs(str(ROOT)).f_frsize
        log(f"第 {rnd} 轮结束；待处理: {pending or '无'}；磁盘剩余 {free / 1e9:.1f} GB")
        if not pending:
            log("全部语料已就绪")
            return 0
        ensure_downloader()      # 自愈：下载器挂了就重拉
        time.sleep(INTERVAL)
    log("达到最大轮数，退出")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
