"""Linux（AutoDL）实例上机自检：一条命令回答「这台机器现在能不能开训」。

设计目标：**把 debug 从付费的 GPU 时段里挪出来**。
它逐项检查并明确打印 PASS/FAIL，失败时给出该看什么、怎么修；
任何一项 FAIL 都以非零码退出，这样可以直接用在 `&&` 链里。

    python scripts/linux/env_check.py                 # 默认查 checkpoints/retnet_500m + chat_pack_v13
    python scripts/linux/env_check.py --quick         # 只查环境，不跑前向/反向

检查项：
  1. Python / torch / CUDA / GPU 型号与显存
  2. transformers 版本是否 == 4.57.6（**跳版本会静默改变 loss 归一化**）
  3. bitsandbytes 可导入（`optim=adamw_bnb_8bit` 依赖它）
  4. 数据集文件与形状（seq_len 必须等于训练配置里的 2048）
  5. 模型目录可加载、参数量是否符合预期
  6. **一次真实的前向+反向**（小 seq）——这一项能一次性抓出
     bnb / compile / 内核 / 数据类型 的绝大多数问题
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPECT_TRANSFORMERS = "4.57.6"
EXPECT_PARAMS = 504_179_896
EXPECT_SEQ = 2048

fails: list[str] = []
warns: list[str] = []


def check(cond: bool, msg: str, hint: str = "") -> bool:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        fails.append(msg)
        if hint:
            print(f"         ↳ {hint}")
    return bool(cond)


def warn(cond: bool, msg: str, hint: str = "") -> None:
    if not cond:
        print(f"  [WARN] {msg}")
        if hint:
            print(f"         ↳ {hint}")
        warns.append(msg)
    else:
        print(f"  [PASS] {msg}")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                              # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="checkpoints/retnet_500m")
    ap.add_argument("--data-dir", default="data_cache/chat_pack_v13")
    ap.add_argument("--quick", action="store_true", help="只查环境，不跑前向/反向")
    args = ap.parse_args()

    print("=" * 74)
    print("1. 运行时环境")
    print("=" * 74)
    print(f"  python  : {sys.version.split()[0]}  ({sys.executable})")
    check(sys.version_info >= (3, 10), f"Python >= 3.10",
          "AutoDL 的 PyTorch 2.8.0 镜像给的是 3.12，符合")

    try:
        import torch
    except Exception as exc:                                           # noqa: BLE001
        check(False, f"import torch", f"{type(exc).__name__}: {exc}")
        print("\n❌ torch 都导入不了，后面的检查没有意义。")
        return 1
    print(f"  torch   : {torch.__version__}  (built for CUDA {torch.version.cuda})")
    if "+cu" in torch.__version__:
        print(f"            注意：torch 的 CUDA 代号 = {torch.__version__.split('+')[1]}")

    print()
    print("=" * 74)
    print("2. GPU")
    print("=" * 74)
    ok_cuda = check(torch.cuda.is_available(), "torch.cuda.is_available()",
                    "无卡模式开机时这里必然是 FAIL —— 那是预期行为，不是故障")
    if ok_cuda:
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 2**30
        print(f"  设备    : {name}")
        print(f"  显存    : {total:.2f} GiB")
        check(total >= 20, f"显存 >= 20 GiB（{total:.1f}）",
              "8 GB 卡跑不了「免检查点」的 0.5B，24 GB 卡才可以")
        free, tot = torch.cuda.mem_get_info(i)
        print(f"  当前空闲: {free / 2**30:.2f} / {tot / 2**30:.2f} GiB")
        warn(free > 15 * 2**30, f"空闲显存 > 15 GiB（实得 {free / 2**30:.2f} GiB）",
             "有别的进程占着卡？关掉再跑，否则步时不准")

    print()
    print("=" * 74)
    print("3. transformers / bitsandbytes 版本")
    print("=" * 74)
    try:
        import transformers
        tv = transformers.__version__
        check(tv == EXPECT_TRANSFORMERS,
              f"transformers == {EXPECT_TRANSFORMERS}（实得 {tv}）",
              f"pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
              f"transformers=={EXPECT_TRANSFORMERS}")
    except Exception as exc:                                           # noqa: BLE001
        check(False, "import transformers", f"{type(exc).__name__}: {exc}")

    try:
        import bitsandbytes as bnb
        print(f"  bitsandbytes: {bnb.__version__}")
        check(hasattr(bnb.optim, "Adam8bit"), "bitsandbytes 提供 Adam8bit",
              "optim=adamw_bnb_8bit 依赖它")
    except Exception as exc:                                           # noqa: BLE001
        check(False, "import bitsandbytes", f"{type(exc).__name__}: {exc}")

    print()
    print("=" * 74)
    print("4. 数据集")
    print("=" * 74)
    import numpy as np
    dd = ROOT / args.data_dir
    n_train = None
    for fn in ("train.ids.npy", "train.wcode.npy", "val.ids.npy", "val.wcode.npy",
               "bpe_tokenizer.json", "meta.json"):
        p = dd / fn
        if not check(p.exists(), f"{fn} 存在",
                     "用 rsync/scp 传上来；见 docs/24 的上传清单"):
            continue
        print(f"           {p.stat().st_size:,} 字节")
    if (dd / "train.ids.npy").exists():
        a = np.load(dd / "train.ids.npy", mmap_mode="r")
        seq = a.shape[1] - 1
        n_train = int(a.shape[0])
        print(f"  train.ids.npy: {n_train:,} 行 × {seq} token")
        check(seq == EXPECT_SEQ, f"seq_len == {EXPECT_SEQ}（实得 {seq}）",
              "seq_len 对不上说明传的不是 chat_pack_v13")
        if (dd / "meta.json").exists():
            m = json.loads((dd / "meta.json").read_text(encoding="utf-8"))
            print(f"  meta.json    : train_tokens={m.get('train_tokens')} "
                  f"r1_max_chars={m.get('r1_max_chars')} "
                  f"strip_think={m.get('strip_think')}")

    print()
    print("=" * 74)
    print("5. 模型")
    print("=" * 74)
    md = ROOT / args.model_dir
    if check(md.is_dir(), f"模型目录存在：{md}"):
        need = list(md.glob("*.safetensors")) + list(md.glob("*.bin"))
        check(bool(need), "目录里有权重文件",
              "grow_500m.py 的产物；若训练中请指向最新的 checkpoint-N")
        if need:
            print(f"            {need[0].name}  {need[0].stat().st_size:,} 字节")
        try:
            from retnet_hf.modeling_retnet import RetNetForCausalLM
            m = RetNetForCausalLM.from_pretrained(str(md))
            n = sum(p.numel() for p in m.parameters())
            cfg = m.config
            print(f"  config  : d={cfg.d_model} L={cfg.num_layers} h={cfg.num_heads} "
                  f"vd={cfg.value_dim} mode={cfg.mode} chunk={cfg.chunk_size} "
                  f"use_gla={cfg.use_gla} embed_scale={cfg.embed_scale}")
            check(n == EXPECT_PARAMS, f"参数量 == {EXPECT_PARAMS:,}（实得 {n:,}）")
            g = m.blocks[0].msr.gamma.detach().float()
            check(float(g.min()) > 0.0,
                  f"gamma 最小值 > 0（{float(g.min()):.6f}）",
                  "gamma=0 会让批量 chunkwise 前向 NaN；见 docs/23 §8.9")
            gn = [p.detach() for nm, p in m.named_parameters()
                  if nm.endswith("group_norm.weight")]
            check(bool(gn) and float(gn[0].min()) != 0.0,
                  "group_norm.weight 不含 0")

            if not args.quick and ok_cuda:
                print()
                print("  --- 真实前向 + 反向（小 seq，能抓出 bnb/compile/内核问题）---")
                import torch
                m.config.mode = "chunkwise"
                m.config.chunk_size = 128
                m.train()
                m = m.to("cuda")
                ids = torch.randint(1, cfg.vocab_size, (1, 256), device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = m(input_ids=ids, labels=ids)
                loss = out.loss
                check(torch.isfinite(loss), f"loss 有限（{float(loss.detach()):.4f}）")
                loss.backward()
                nz = sum(1 for p in m.parameters()
                         if p.grad is not None and torch.isfinite(p.grad).all()
                         and p.grad.abs().max().item() > 0)
                check(nz > 0, f"有 {nz} 个参数拿到非零梯度",
                      "梯度全 0 常见于 mask 把 loss 压成了 0")
                peak = torch.cuda.max_memory_reserved() / 2**30
                print(f"           峰值 reserved {peak:.2f} GiB（小 seq，仅供参考）")
                del m
                torch.cuda.empty_cache()
        except Exception as exc:                                       # noqa: BLE001
            import traceback
            check(False, "模型加载/前向", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    print()
    print("=" * 74)
    if fails:
        print(f"❌ {len(fails)} 项 FAIL —— 先修这些，别急着开训：")
        for f in fails:
            print(f"   - {f}")
        if warns:
            print(f"（另有 {len(warns)} 项 WARN，不阻塞）")
        return 1
    print(f"✅ 全部通过。" + (f"（{len(warns)} 项 WARN，不阻塞）" if warns else ""))
    print("   可以开训：bash scripts/linux/run_train.sh --config tasks/v15_500m.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
