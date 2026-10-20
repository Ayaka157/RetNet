"""动态训练计划的单元测试：解析 / LR 曲线 / 多阶段数据集 / 热加载 / 坏文件容错。

**自带临时夹具**，不依赖仓库里的 tasks/plan.json 与真实语料——那个计划会随
训练进度变化（阶段可能还没分词好），拿它当测试夹具会让测试跟着环境漂。
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# 同 test_plan_gc.py：ssh 到 GBK 控制台时，打印 ✅/❌ 会 UnicodeEncodeError。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from retnet_hf.data import StagePackedDataset
from retnet_hf.plan import PlanWatcher, parse_plan

ok = True
SEQ = 8


def check(cond, msg):
    global ok
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        ok = False


def write_flat(d: Path, name: str, n_rows: int, seed=0):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(3, 19999, size=(SEQ + 1) * n_rows, dtype=np.int16)
    arr[::32] = 2      # docs/00 E12：必须有 eos，否则 _FlatStage 拒绝打开
    np.save(d / name, arr)


def write_packed(d: Path, n_rows: int, seed=0):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = rng.integers(1, 19999, size=(n_rows, SEQ + 1), dtype=np.int16)
    wc = rng.integers(0, 4, size=(n_rows, SEQ + 1), dtype=np.uint8)
    np.save(d / "train.ids.npy", ids)
    np.save(d / "train.wcode.npy", wc)
    np.save(d / "val.ids.npy", ids[:2])
    np.save(d / "val.wcode.npy", wc[:2])


tmp = Path(tempfile.mkdtemp())
try:
    dA = tmp / "flat_a"
    write_flat(dA, "shard_00000.npy", 100, 1)
    write_flat(dA, "skypile_00000.npy", 60, 2)      # 用来测 exclude
    dB = tmp / "packed_b"
    write_packed(dB, 40, 3)

    base_raw = {
        "plan_version": 1, "seq_len": SEQ, "min_free_gb": 9,
        "stages": [
            {"name": "flat", "data_dir": str(dA), "mode": "flat",
             "exclude": ["skypile_"], "steps": 30, "gc": True, "min_shards": 2},
            {"name": "packed", "data_dir": str(dB), "mode": "packed", "steps": 20},
        ],
        "lr": {"peak": 1e-4, "min": 1e-5, "warmup": 5},
        "save_steps": 100, "eval_steps": 100, "val_dir": str(dB),
    }
    plan_path = tmp / "plan.json"
    plan_path.write_text(json.dumps(base_raw), encoding="utf-8")

    print("=== 1) 解析 ===")
    w = PlanWatcher(plan_path, verbose=False)
    plan = w.load(force=True)
    check(plan is not None, "加载成功")
    check(plan.total_steps == 50, f"total_steps={plan.total_steps}")
    check(plan.cumulative() == [30, 50], f"累计边界 {plan.cumulative()}")
    check(plan.stage_index_at(0) == 0, "step 0 -> 段 0")
    check(plan.stage_index_at(29) == 0, "step 29 -> 段 0")
    check(plan.stage_index_at(30) == 1, "step 30 -> 段 1")
    check(plan.min_free_gb == 9, f"min_free_gb={plan.min_free_gb}")
    check(plan.stages[0].gc is True and plan.stages[0].exclude == ["skypile_"],
          "gc / exclude 解析正确")

    print("=== 2) LR 曲线 ===")
    lrs = [plan.lr_at(s) for s in (0, 4, 5, 27, 49, 50)]
    check(lrs[0] < lrs[1] <= lrs[2], "warmup 单调升到 peak（warmup 末尾与余弦起点同为 peak）")
    check(abs(lrs[2] - plan.lr_peak) < 1e-9, "warmup 结束 = peak")
    check(lrs[-1] <= plan.lr_min * 1.001, "末尾降到 min")

    print("=== 3) 多阶段数据集 ===")
    ds = StagePackedDataset(
        [{"name": s.name, "data_dir": str(s.resolved_dir()), "mode": s.mode,
          "gc": s.gc, "min_shards": s.min_shards, "exclude": s.exclude}
         for s in plan.stages], plan.seq_len)
    check(len(ds._cache) == 0, "初始化时不打开任何阶段（惰性）")
    ds.set_stage(0)
    check(len(ds) == 100, f"段0 rows={len(ds)}（100 行；skypile 的 60 行被 exclude）")
    item = ds[0]
    check(item["input_ids"].shape == (SEQ,) and item["wcodes"].shape == (SEQ,), "行形状正确")
    ds.set_stage(1)
    check(len(ds) == 40, f"段1 rows={len(ds)}（packed）")
    check(len(set(ds[0]["wcodes"].tolist())) > 1, "packed 段 wcode 有多档")
    check(ds[10 ** 9]["input_ids"].shape == (SEQ,), "超界索引取模不崩")

    print("=== 4) 热加载：改 steps 立刻反映 ===")
    check(w.maybe_reload() is None, "未改动时返回 None")
    raw2 = json.loads(json.dumps(base_raw))
    raw2["plan_version"] = 2
    raw2["stages"][0]["steps"] = 45
    time.sleep(0.02)
    plan_path.write_text(json.dumps(raw2), encoding="utf-8")
    p2 = w.maybe_reload()
    check(p2 is not None, "改动后返回新计划")
    check(p2.total_steps == 65, f"total_steps 50 -> {p2.total_steps}")
    check(p2.stage_index_at(30) == 0, "加长后 step 30 重新落在段 0")

    print("=== 5) 坏计划文件不崩训练 ===")
    plan_path.write_text("{ not json", encoding="utf-8")
    check(w.maybe_reload() is None, "坏文件返回 None")
    check(w.plan is not None and w.plan.total_steps == 65, "沿用上一份可用计划")

    print("=== 6) 校验拒绝明显错误 ===")
    for bad, why in [
        ({"stages": []}, "空 stages"),
        ({"stages": [{"name": "a", "data_dir": "x", "steps": 0}]}, "所有 steps 都是 0"),
        ({"stages": [{"name": "a", "data_dir": "x", "steps": 10}],
          "lr": {"peak": 1e-5, "min": 1e-4}}, "min > peak"),
        ({"stages": [{"name": "a", "data_dir": "x", "steps": 10}],
          "lr": {"warmup": 100}}, "warmup >= total"),
    ]:
        try:
            parse_plan(bad, Path("<test>"))
            check(False, f"{why} 应该被拒绝")
        except ValueError:
            check(True, f"{why} 被正确拒绝")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("全部通过" if ok else "有失败项"))
sys.exit(0 if ok else 1)
