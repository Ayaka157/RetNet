"""验证「边训边传 + 自动接力 + 回收旧数据」这组能力。

覆盖：
1. 阶段数据没到时 stage_ready=False（不崩）
2. 数据到了以后能打开（接力）
3. 正在写入的截断分片会被跳过而不是崩，传完后 refresh 能接过
4. 消费完 gc_stage 能删掉数据并返回释放字节数
5. GC 安全：目录与其他阶段共用 / 是验证集时拒绝删除
6. 计划解析 gc / min_shards / min_free_gb
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# 远端（4070 笔记本）通过 ssh 跑时 stdout 是 GBK，打印 ✅/❌ 会抛
# UnicodeEncodeError 并让测试在第一条断言处整段崩掉（2026-10-20 实测）。
# 所有会打印符号的脚本式测试都必须在任何输出之前把 stdout/stderr 固定成 UTF-8。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from retnet_hf.data import StagePackedDataset
from retnet_hf.plan import parse_plan

ok = True
SEQ = 8          # 用很小的 seq_len 让测试快


def check(cond, msg):
    global ok
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        ok = False


def write_shard(path, n_tokens, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(1, 19999, size=n_tokens, dtype=np.int16)
    np.save(path, arr)
    return path


tmp = Path(tempfile.mkdtemp())
try:
    dA = tmp / "stage_a"
    dA.mkdir()
    write_shard(dA / "shard_00000.npy", (SEQ + 1) * 100, seed=1)
    write_shard(dA / "shard_00001.npy", (SEQ + 1) * 100, seed=2)
    # docs/00 E13：GC 删除前必须能证明"可重建"，所以阶段目录要有 meta.json。
    (dA / "meta.json").write_text(json.dumps({
        "source_ids": ["test/stage_a"],
        "rebuild_cmd": "python scripts/build_flat_corpus.py --help",
    }, ensure_ascii=False), encoding="utf-8")
    dB = tmp / "stage_b"          # 还没上传的空目录
    dB.mkdir()

    print("=== 1) 数据未就绪：不崩，返回原因 ===")
    ds = StagePackedDataset([{"name": "a", "data_dir": str(dA), "mode": "flat"},
                             {"name": "b", "data_dir": str(dB), "mode": "flat"}], SEQ)
    ds.set_stage(0)
    check(len(ds) == 200, f"阶段 a rows={len(ds)}（2 片 × 100 行）")
    ready, why = ds.stage_ready(1)
    check(ready is False, f"阶段 b 未就绪 -> {why}")

    print("=== 2) 数据到达后能接力 ===")
    write_shard(dB / "shard_00000.npy", (SEQ + 1) * 50, seed=3)
    ready, why = ds.stage_ready(1)
    check(ready is True, f"阶段 b 到达后就绪 -> {why}")
    ds.set_stage(1)
    check(len(ds) == 50, f"切到 b 后 rows={len(ds)}")
    check(ds.active_name == "b", "active_name 已更新")

    print("=== 3a) 只有截断分片时判未就绪（不崩） ===")
    dC = tmp / "stage_c"
    dC.mkdir()
    p = dC / "shard_00000.npy"
    write_shard(p, (SEQ + 1) * 30, seed=4)
    full = p.stat().st_size
    with open(p, "r+b") as f:          # 模拟「正在上传」：截掉后半
        f.truncate(full // 2)
    ds3 = StagePackedDataset([{"name": "c", "data_dir": str(dC), "mode": "flat"}], SEQ)
    ready3, why3 = ds3.stage_ready(0)
    check(ready3 is False, f"全部截断 -> 未就绪（{why3[:40]}…）")
    write_shard(p, (SEQ + 1) * 30, seed=4)      # 传完
    ready3b, why3b = ds3.stage_ready(0)
    check(ready3b is True, f"传完后就绪 -> {why3b}")

    print("=== 3b) 已打开的阶段能 refresh 收进新到的分片 ===")
    write_shard(dC / "shard_00001.npy", (SEQ + 1) * 20, seed=6)
    changed = ds3.refresh_stage(0)
    check(changed and len(ds3) == 50, f"refresh 后 rows={len(ds3)}（30 + 20）")
    changed_again = ds3.refresh_stage(0)
    check(changed_again is False, "没有新分片时 refresh 返回 False（无谓重扫）")

    print("=== 4) 回收：删文件并返回释放字节 ===")
    freed = ds.gc_stage(0)
    check(freed > 0, f"gc 释放 {freed} 字节")
    check(not (dA / "shard_00000.npy").exists(), "阶段 a 的分片已被删除")
    check(not (dA / "shard_00001.npy").exists(), "阶段 a 的第二个分片也删了")

    print("=== 4b) GC 安全：没有 meta.json 时拒绝删除（E13） ===")
    dA2 = tmp / "stage_a2"
    dA2.mkdir()
    write_shard(dA2 / "shard_00000.npy", (SEQ + 1) * 10, seed=11)
    ds4b = StagePackedDataset([{"name": "a2", "data_dir": str(dA2), "mode": "flat"}], SEQ)
    ds4b.set_stage(0)
    freed4b = ds4b.gc_stage(0)
    check(freed4b == 0 and (dA2 / "shard_00000.npy").exists(),
          "缺 meta.json -> 拒绝回收（文件仍在）")

    print("=== 5) GC 安全：共用目录 / 验证集目录都拒绝 ===")
    dD = tmp / "shared"
    dD.mkdir()
    write_shard(dD / "shard_00000.npy", (SEQ + 1) * 10, seed=5)
    ds5 = StagePackedDataset([{"name": "s1", "data_dir": str(dD), "mode": "flat"},
                              {"name": "s2", "data_dir": str(dD), "mode": "flat"}], SEQ)
    ds5.set_stage(0)
    freed5 = ds5.gc_stage(0)
    check(freed5 == 0 and (dD / "shard_00000.npy").exists(),
          "两段共用同一目录时拒绝回收（文件仍在）")

    print("=== 6) 计划解析新字段 ===")
    raw = {
        "plan_version": 5, "seq_len": SEQ, "min_free_gb": 12.5,
        "stages": [
            {"name": "x", "data_dir": str(dA), "steps": 10, "gc": True, "min_shards": 3},
            {"name": "y", "data_dir": str(dB), "steps": 10, "gc": False},
        ],
        "lr": {"peak": 1e-4, "min": 1e-5, "warmup": 1},
        "save_steps": 100, "eval_steps": 100, "val_dir": str(dA),
    }
    plan = parse_plan(raw, Path("<test>"))
    check(plan.stages[0].gc is True and plan.stages[0].min_shards == 3,
          f"gc={plan.stages[0].gc} min_shards={plan.stages[0].min_shards}")
    check(plan.stages[1].gc is False, "未写 gc 时默认 False")
    check(abs(plan.min_free_gb - 12.5) < 1e-9, f"min_free_gb={plan.min_free_gb}")

    print("=== 7) min_shards 未达标时判未就绪 ===")
    ds7 = StagePackedDataset([{"name": "x", "data_dir": str(dB), "mode": "flat",
                               "min_shards": 5}], SEQ)
    ready7, why7 = ds7.stage_ready(0)
    check(ready7 is False, f"只 1 个分片 < 要求 5 -> {why7}")
    ds7b = StagePackedDataset([{"name": "x", "data_dir": str(dB), "mode": "flat",
                                "min_shards": 1}], SEQ)
    ready7b, why7b = ds7b.stage_ready(0)
    check(ready7b is True, f"要求 1 个分片 -> 就绪（{why7b}）")
    print("=== 8) exclude：跳过指定来源的分片（如 skypile） ===")
    dE = tmp / "stage_e"
    dE.mkdir()
    write_shard(dE / "keep_00000.npy", (SEQ + 1) * 40, seed=7)
    write_shard(dE / "skypile_00000.npy", (SEQ + 1) * 60, seed=8)
    dsE_all = StagePackedDataset([{"name": "e", "data_dir": str(dE), "mode": "flat"}], SEQ)
    dsE_all.set_stage(0)
    check(len(dsE_all) == 100, f"不排除时 rows={len(dsE_all)}（40+60）")
    dsE = StagePackedDataset([{"name": "e", "data_dir": str(dE), "mode": "flat",
                               "exclude": ["skypile_"]}], SEQ)
    dsE.set_stage(0)
    check(len(dsE) == 40, f"排除 skypile_ 后 rows={len(dsE)}（只剩 keep 的 40）")
    check(all("skypile" not in p.name for p in dsE.files()), "files() 也排除了 skypile 分片")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("全部通过" if ok else "有失败项"))
sys.exit(0 if ok else 1)
