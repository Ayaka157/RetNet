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
import shutil
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
    # need_gb：够分词所需的**完整**字节数（cap / 1.1，实测约 1.17 token/字节）。
    # 不必等整个数据集下完：代码只要 2.07B token，9 个分片就够，
    # 等剩下 18 个白等到段切换之后（2026-10-18 实测）。
    #
    # text_field（docs/00 E10）：**必须显式**，不再用 auto。
    # auto 用字段名黑名单猜正文，会把 prog_zh 的 10 个数值元数据列拼进训练流。
    # 这里的 "text" 是按已知 schema 写的；若某个数据集不是这个列名，
    # build_flat_corpus 现在会在读第一条记录时立刻退出并打印真实字段名，
    # 不会先跑几小时再以错误率失败。
    dict(name="code",     src="ling_coder",        out="hf_pack_code",        n=27, extract=False,
         cap=2_500_000_000, need_gb=2.0,   text_field="text"),
    dict(name="web",      src="chinese_web_hq",    out="hf_pack_chinese_web", n=13, extract=False,
         cap=700_000_000,   need_gb=0.64,  text_field="text"),
    # textbook 解压出 696,600 个文件，其中 2/3 是 stats.json（非文本），
    # 不跳过会被算成 66% 错误率，触发分词脚本自带的「错误率 > 1% 判失败」守卫。
    dict(name="textbook", src="fineweb_edu_zh_v3", out="hf_pack_textbook",    n=18, extract=True,
         cap=400_000_000,   need_gb=0.36,  text_field="text", skip="stats.json"),
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
    """校验该数据集；删掉大小不符的文件。

    返回 (完整字节数, 期望字节数, 完整文件数, 总文件数)。
    """
    files = man.get(local, [])
    if not files:
        return 0, 0, 0, 0
    have = 0
    have_bytes = 0
    want_bytes = 0
    for fname, want in files:
        p = BASE / local / fname
        if not p.exists():
            continue
        want_bytes += want
        size = p.stat().st_size
        if size == want:
            have += 1
            have_bytes += size
        elif size > want:
            # 只有**比期望还大**才是确定损坏：snapshot_download 写了一半的文件
            # 会让 curl -C - 从错误偏移续写，多出来一截。
            # 比期望**小**说明还在下载中，绝不能删 —— 删了它 curl 就得从头再来，
            # 两边每 5 分钟互相打架，永远下不完（2026-10-18 实测踩过）。
            log(f"  删除损坏文件（超大）{local}/{fname} (have={size} want={want})")
            p.unlink()
    return have_bytes, want_bytes, have, len(files)


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



def scrub_corrupt(spec) -> int:
    """分词失败后，找出内容损坏（但大小正确）的 parquet 并删掉，让下载器重下。

    为什么要单独一遍：`check_and_clean` 只比字节数，而两写入者交错写出的文件
    大小可能恰好正确、内容却是坏的，读的时候才抛
    `Corrupt snappy compressed data` / `ArrowInvalid`。
    """
    import pyarrow.parquet as pq
    src_dir = BASE / spec["src"]
    if not src_dir.exists():
        return 0
    bad = 0
    for f in sorted(src_dir.rglob("*.parquet")):
        try:
            pf = pq.ParquetFile(f)
            for _ in pf.iter_batches(batch_size=64):
                break
        except Exception:                                      # noqa: BLE001
            log(f"  删除内容损坏的文件 {f.name}")
            try:
                f.unlink()
                bad += 1
            except OSError:
                pass
    if bad:
        log(f"  共删除 {bad} 个损坏文件，等待重下")
    return bad


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
           # 显式正文列 + 文档边界，两者都不能再靠默认值（docs/00 E09/E10）
           "--text-field", spec["text_field"],
           "--add-eos",
           "--tokens-per-shard", "250000000"]
    if spec.get("cap"):
        cmd += ["--max-tokens", str(spec["cap"])]
    if spec.get("skip"):
        cmd += ["--skip", spec["skip"]]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0 or not pack_ready(out_dir):
        log(f"  分词失败 rc={rc}")
        scrub_corrupt(spec)
        return False
    d = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    log(f"  ✅ {spec['name']} 完成：{d['total_tokens'] / 1e9:.3f}B token / "
        f"{len(d['shards'])} 分片")
    drop_from_manifest(spec["src"])
    return True


def drop_from_manifest(src: str) -> None:
    """把已分词的数据集从下载清单里摘掉。

    否则 release_raw 删了原始包之后，下载器下一轮又看到「文件缺失」，
    把几 GB 重新下一遍 —— 白耗带宽，还会在磁盘紧张时把刚腾出的空间吃回去。
    """
    if not MANIFEST.exists():
        return
    keep = [ln for ln in MANIFEST.read_text(encoding="utf-8").splitlines()
            if ln.strip() and ln.split("\t")[1] != src]
    if len(keep) != len(MANIFEST.read_text(encoding="utf-8").splitlines()):
        # 先杀掉下载器：它在启动时把清单读进内存，不重启就还会按旧清单继续下
        subprocess.call(["bash", "-c", "pkill -f 'dl_parallel[.]py' || true"])
        time.sleep(1)
        subprocess.call(["bash", "-c", "pkill -P 1 curl || true"])   # 连它的 curl 子进程一起清
        MANIFEST.write_text("\n".join(keep) + "\n", encoding="utf-8")
        log(f"  已从下载清单移除 {src}（分词完成，不再重下，下载器将按新清单重启）")


def release_raw(spec) -> None:
    """分片齐全后删掉原始下载，腾磁盘。

    ⚠️ 只删**大小与清单完全一致**的文件。下载器可能还在往同目录写后续分片，
    整个 rmtree 会把在写的文件删掉（curl 会继续往已 unlink 的 inode 写，
    最后得到一个查不到的文件，白下一场）。
    """
    src_dir = BASE / spec["src"]
    # 清单的 key 是 "ling_coder"；但 fname 可能带子目录（"data/train_00000..."），
    # 而 rglob 拿到的 p.name 只有文件名 —— 必须两边都取 basename 才能匹配。
    # 2026-10-18 实测：直接用 dict(get()) 导致命中 0 个文件，释放 0 GB。
    want_map = {Path(f).name: w for f, w in read_manifest().get(spec["src"], [])}
    freed = 0
    for p in list(src_dir.rglob("*")):
        if not p.is_file():
            continue
        w = want_map.get(p.name)
        if w is None or p.stat().st_size != w:
            continue        # 不认识或还没下完，留着
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
    # ⚠️ 先清孤儿 curl：上一次下载器被 kill 时，它的 curl 子进程会被 init 收养并继续写
    #    同一批文件；新实例一启动就和它们交错追加，得到「比期望还大」的坏文件。
    #    2026-10-18 实测：13 个 curl 里 6 个是孤儿，下载量在倒退（463 -> 425 MB）。
    subprocess.call(["bash", "-c", "pkill -P 1 curl || true"])
    log("  下载器未运行，重新拉起（已清孤儿 curl）")
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
                # 包已就绪 → 整个源目录都可以删（含解压产物与未下完的部分）。
                # 2026-10-18 实测：只调 release_raw 时，textbook 解压出的 3.8 GB
                # 留在 data/hf 里没人管，因为它不在下载清单中。
                src_dir0 = BASE / spec["src"]
                if src_dir0.exists():
                    release_raw(spec)
                    shutil.rmtree(src_dir0, ignore_errors=True)
                    log(f"  已删除 {spec['src']} 整个目录（含解压产物）")
                continue
            local_full = f"data/hf/{spec['src']}"
            have_b, want_b, have, want = check_and_clean(
                man, local_full.replace("data/hf/", ""))
            # ⚠️ 不能超过数据集本身的体积：textbook 全量只有 0.28 GB，
            #    而 need_gb 写的是 0.4（按 cap 反推），永远凑不够 → 它永远不分词。
            need_b = spec.get("need_gb", 0) * 1e9
            if want_b:
                need_b = min(need_b, want_b)
            enough = have_b >= need_b if need_b else have == want
            if not enough:
                pending.append(f"{spec['name']} {have}/{want} "
                               f"({have_b / 1e9:.1f}/{need_b / 1e9:.1f} GB 够用)")
                continue
            log(f"=== {spec['name']} 下载齐全（{have}/{want}），开始分词 ===")
            if prepare(spec):
                release_raw(spec)
                # 分词已完成且已出清单 → 整个目录都没用了，连没下完的部分一起删。
                # 2026-10-18 实测：只删「完整」文件会留下 2.0 GB 残缺包白占磁盘，
                # 而且下载器重启后还会按旧清单继续往里写。
                src_dir2 = BASE / spec["src"]
                if src_dir2.exists():
                    shutil.rmtree(src_dir2, ignore_errors=True)
                    log(f"  已删除 {spec['src']} 整个目录（含未下完的部分）")
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
