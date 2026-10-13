"""把 torch.compile 留下的 ``_orig_mod.`` 前缀从 safetensors 里剥掉。

## 背景（这是个会静默毁掉整个训练成果的 bug）

``torch.compile(model)`` 返回的是 ``OptimizedModule``，它把原模型放在 ``_orig_mod``
子模块下，于是 ``state_dict()`` 的**每一个键**都带上 ``_orig_mod.`` 前缀。

``retnet_hf/train.py`` 为了真正启用编译，把编译后的模块同时赋给了 ``trainer.model``
和 ``trainer.model_wrapped``。HF 的 ``save_model`` 取的是 ``self.model.state_dict()``
—— 所以存出来的检查点全带前缀。

而 ``RetNetForCausalLM.from_pretrained`` 加载时**不会报错**：它把 ``_orig_mod.*``
当成"多余的键"丢掉，把 ``blocks.*`` 当成"缺失的键"随机初始化。结果是
**"加载成功"但拿到一个随机模型**：

    v14 checkpoint-10000   前向 loss=nan   logits 全 nan
    v13 终版               前向 loss=2.5767  logits finite

评测侧的症状是 ``torch.multinomial`` 抛 ``CUDA error: device-side assert triggered``
—— 极难从表象追到根因。

## 用法

    python scripts/strip_compile_prefix.py ^
        --src checkpoints/retnet_272m_v14/checkpoint-10000 ^
        --dst checkpoints/retnet_272m_v14/checkpoint-10000-clean

默认只重写 ``model.safetensors`` 并复制 ``config.json``（评测够用）；
``--all`` 会把其余文件也复制过去（续训才需要 optimizer 状态）。
若无前缀，则原样复制，脚本会说明。
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

PREFIX = "_orig_mod."


def read_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return hdr


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--all", action="store_true",
                    help="连同 optimizer.pt 等一起复制（续训用）")
    ap.add_argument("--in-place", action="store_true",
                    help="就地修复（先备份成 model.safetensors.orig）")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    wf = src / "model.safetensors"
    if not wf.exists():
        print(f"!! 找不到 {wf}", file=sys.stderr)
        return 1

    hdr = read_header(wf)
    keys = [k for k in hdr if k != "__metadata__"]
    n_pre = sum(1 for k in keys if k.startswith(PREFIX))
    print(f"源  {src}")
    print(f"    {len(keys)} 个键，其中 {n_pre} 个带 {PREFIX!r} 前缀")

    if n_pre == 0:
        print("已经干净，无需处理。", flush=True)
        if args.in_place:
            return 0
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and (args.all or f.name in ("config.json",)):
                shutil.copy2(f, dst / f.name)
        print(f"仅复制了配置文件到 {dst}")
        return 0

    from safetensors.torch import load_file, save_file

    print("  正在读取权重 ...", flush=True)
    sd = load_file(str(wf))
    stripped = {k[len(PREFIX):] if k.startswith(PREFIX) else k: v
                for k, v in sd.items()}
    dup = len(stripped) != len(sd)
    print(f"  剥离前缀：{len(sd)} -> {len(stripped)} 个张量"
          + ("（有键冲突！）" if dup else ""))

    if args.in_place:
        dst = src
        backup = wf.parent / (wf.name + ".orig")
        wf.rename(backup)
        print(f"  原文件已备份为 {backup.name}")

    dst.mkdir(parents=True, exist_ok=True)
    # 保留 __metadata__
    meta = hdr.get("__metadata__")
    print(f"  写出 {dst / 'model.safetensors'} ...", flush=True)
    save_file(stripped, str(dst / "model.safetensors"), metadata=meta)

    if dst.resolve() == src.resolve():
        print(f"完成（就地修复）-> {dst}")
    else:
        for f in src.iterdir():
            if not f.is_file() or f.name == "model.safetensors":
                continue
            if args.all or f.name in ("config.json", "trainer_state.json",
                                      "training_args.bin", "scheduler.pt",
                                      "rng_state.pth"):
                shutil.copy2(f, dst / f.name)
        print(f"完成 -> {dst}")
    print("  验证：用 RetNetForCausalLM.from_pretrained 加载，"
          "看是否还报「not initialized」并检查 loss 是否有限")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
