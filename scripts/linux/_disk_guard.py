"""磁盘守卫：定时检查两个盘，逼近阈值就主动腾空间，并留下状态文件给人看。

覆盖的盘：
- **数据盘** 50 GB（/root/autodl-tmp）：checkpoint 12 GB + 语料下载 + 分词产物
- **系统盘** 30 GB（/）：torch.compile 的 /tmp 缓存能长到 8 GB 以上

腾空间的手段（按"损失从小到大"排序，且都**不动正在跑的训练**）：
1. 删已分词完成的数据集原始包（分词产物还在，删了不影响）
2. 把 checkpoint 数量从 N 降到 2（保留最近两个可回滚点）
3. 系统盘紧张时清 /tmp 下的 inductor 编译缓存（已加载的 kernel 在显存里，不受影响）

⚠️ 绝不删的：当前输出目录里最新的 checkpoint、data_cache 里 plan 正在用的分片、
       `tasks/plan.json`、`logs/`。

用法：
    setsid nohup /root/miniconda3/bin/python -u scripts/linux/_disk_guard.py \
        > logs/linux/disk_guard.out 2>&1 &
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/RetNet")
DATA_DISK = Path("/root/autodl-tmp")
SYS_DISK = Path("/")
STATUS = ROOT / "logs/linux/disk_status.txt"

INTERVAL = 120              # 检查间隔（秒）
DATA_WARN = 10.0            # 数据盘低于此值开始清（GB）
DATA_CRIT = 5.0             # 数据盘低于此值进一步清
SYS_WARN = 5.0              # 系统盘低于此值清缓存
KEEP_CKPT = 2               # 数据盘紧张时保留的 checkpoint 个数


def free_gb(p: Path) -> float:
    try:
        return shutil.disk_usage(str(p)).free / 1e9
    except Exception:                                          # noqa: BLE001
        return -1.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def out_dir() -> Path | None:
    """从运行中的 train.py 命令行里找出 --output_dir。"""
    try:
        out = subprocess.check_output(
            ["bash", "-c", "ps -eo args | grep -m1 'retnet_hf/train.py' || true"],
            text=True, timeout=10)
    except Exception:                                          # noqa: BLE001
        return None
    parts = out.split()
    for i, tok in enumerate(parts):
        if tok == "--output_dir" and i + 1 < len(parts):
            p = Path(parts[i + 1])
            return p if p.is_absolute() else ROOT / p
    return None


def trim_checkpoints(keep: int) -> int:
    """把最新 checkpoint 之外的旧目录删掉，返回释放字节数。"""
    d = out_dir()
    if d is None or not d.exists():
        return 0
    cks = sorted((p for p in d.glob("checkpoint-*") if p.is_dir()),
                 key=lambda p: int(p.name.split("-")[-1]))
    if len(cks) <= keep:
        return 0
    freed = 0
    for p in cks[:-keep]:
        try:
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p)
            freed += sz
            log(f"  删除旧 checkpoint {p.name}（{sz / 1e9:.2f} GB）")
        except Exception as exc:                               # noqa: BLE001
            log(f"  删除 {p.name} 失败: {exc}")
    return freed


def drop_tokenized_raw() -> int:
    """删掉「分词产物已就绪」的数据集原始下载。"""
    freed = 0
    pairs = [
        ("data/hf/ling_coder", "data_cache/hf_pack_code"),
        ("data/hf/chinese_web_hq", "data_cache/hf_pack_chinese_web"),
        ("data/hf/math_zh", "data_cache/hf_pack_math"),
        ("data/hf/fineweb_edu_zh_v3", "data_cache/hf_pack_textbook"),
    ]
    for src, pack in pairs:
        sp, pp = ROOT / src, ROOT / pack
        if not sp.exists():
            continue
        try:
            meta = json.loads((pp / "meta.json").read_text(encoding="utf-8"))
            ready = int(meta.get("total_tokens", 0)) > 0 and len(meta.get("shards", [])) > 0
        except Exception:                                      # noqa: BLE001
            ready = False
        if not ready:
            continue
        for p in list(sp.rglob("*")):
            if p.is_file():
                freed += p.stat().st_size
                p.unlink()
        log(f"  清理 {src} 原始包（分词已完成）")
    return freed


def clean_sys_cache() -> int:
    """清 /tmp 下 torch.compile 的 inductor 缓存与 pip 缓存。"""
    freed = 0
    for pat in ("torchinductor_*", "torch_compile_*"):
        for p in Path("/tmp").glob(pat):
            try:
                sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p)
                freed += sz
                log(f"  清理系统盘 /tmp/{p.name}（{sz / 1e9:.2f} GB）")
            except Exception as exc:                           # noqa: BLE001
                log(f"  清理 /tmp/{p.name} 失败: {exc}")
    return freed


def main() -> int:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    log(f"磁盘守卫启动：每 {INTERVAL}s 检查；数据盘警告线 {DATA_WARN} GB / 危险线 {DATA_CRIT} GB")
    while True:
        d_free = free_gb(DATA_DISK)
        s_free = free_gb(SYS_DISK)
        d_used = 100 * (1 - d_free / 50.0) if d_free >= 0 else -1
        action = ""

        if d_free >= 0 and d_free < DATA_WARN:
            log(f"数据盘 {d_free:.1f} GB 低于警告线 {DATA_WARN} GB，开始清理")
            freed = drop_tokenized_raw()
            freed += trim_checkpoints(KEEP_CKPT)
            action = f"数据盘清理释放 {freed / 1e9:.2f} GB"
            if d_free < DATA_CRIT:
                more = trim_checkpoints(1)
                action += f"；危险线再释放 {more / 1e9:.2f} GB"
            d_free = free_gb(DATA_DISK)

        if s_free >= 0 and s_free < SYS_WARN:
            freed = clean_sys_cache()
            action += f"；系统盘清理释放 {freed / 1e9:.2f} GB"
            s_free = free_gb(SYS_DISK)

        # 状态文件：给人/给未来的我看
        try:
            ck = out_dir()
            n_ck = len(list(ck.glob("checkpoint-*"))) if ck and ck.exists() else 0
            STATUS.write_text(
                f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"数据盘: 剩余 {d_free:.1f} GB / 50 GB（已用 {d_used:.0f}%）\n"
                f"系统盘: 剩余 {s_free:.1f} GB / 30 GB\n"
                f"checkpoint: {n_ck} 个（{ck}）\n"
                f"本轮动作: {action or '无'}\n",
                encoding="utf-8")
        except Exception as exc:                               # noqa: BLE001
            log(f"写状态文件失败: {exc}")

        if action:
            log(action)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
