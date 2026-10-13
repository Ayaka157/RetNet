"""在实例上就地造一个**同架构**的 504M 模型并保存成 checkpoint —— 用于零上传测速。

## 为什么需要它

换卡后第一件要量的是「新卡在这套 workload 上的真实步时」。项目里的 `bench_trainer.py`
强制要 `--model_dir`，而真模型 1.9 GB 要传 20 分钟 —— 但**步时只取决于形状**：
`docs/23` §4.7(a) 已实测「显存由形状决定，与权重值无关，四次复测峰值逐位一致」。
所以按同一个 `config.json` 就地造一个同架构模型，就能把这段时间省掉。

## 安全要求

初始化必须走模型自己的 `_init_weights`。**这里是本项目踩过的坑**（`docs/23` §8.9）：
`_init_weights` 曾把唯一的 1-D Parameter（`gamma`）置零 → `log(0) = -inf` → `-inf*0 = NaN`，
前向全 NaN。已修，但本脚本仍**自带自检**：gamma 必须为正、短序列前向必须有限，
否则拒绝保存 —— 免得又拿一个坏探针去测速。

用法：
    python scripts/linux/make_synth_ckpt.py \
        --config checkpoints/retnet_500m/config.json --out checkpoints/_synth_500m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="真实模型的 config.json（552 字节）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=128, help="自检用的短序列长度")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    for k in ("architectures", "transformers_version", "_name_or_path"):
        cfg.pop(k, None)
    config = RetNetHFConfig(**cfg)

    torch.manual_seed(1234)
    model = RetNetForCausalLM(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"构造完成：{n:,} 参数（对比真实模型 504,179,896）")

    # ---- 自检 1：gamma 必须为正（否则 log(0) -> -inf -> NaN）----
    gammas = [(nm, p) for nm, p in model.named_parameters() if nm.endswith("gamma")]
    if not gammas:
        print("!! 找不到任何 gamma 参数 —— 架构与预期不符，拒绝保存")
        return 2
    bad = [nm for nm, p in gammas if not torch.isfinite(p).all() or (p <= 0).any()]
    if bad:
        print(f"!! gamma 非法（<=0 或非有限）: {bad[:3]} —— 拒绝保存")
        return 2
    print(f"自检 1 通过：{len(gammas)} 个 gamma，全部 > 0，"
          f"范围 [{min(float(p.min()) for _, p in gammas):.4f}, "
          f"{max(float(p.max()) for _, p in gammas):.4f}]")

    # ---- 自检 2：短序列前向必须有限 ----
    # ⚠️ 第一版这里写成 `(1, a.seq + 1)`，而 seq+1=129 不是 chunk_size(64) 的倍数，
    # chunkwise 直接抛 ValueError。真实数据是 (N, 2049) 宽，collator 会把它切成
    # input_ids/labels 各 2048 —— 所以自检要用 **2048**（= seq），不是 2049。
    if a.seq % int(cfg.get("chunk_size", 64)) != 0:
        print(f"!! --seq {a.seq} 必须是 chunk_size ({cfg.get('chunk_size')}) 的整数倍")
        return 2
    model.eval()
    ids = torch.randint(0, config.vocab_size, (1, a.seq))
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    loss = float(out.loss)
    if not torch.isfinite(torch.tensor(loss)):
        print(f"!! 短序列前向 loss 非有限：{loss} —— 拒绝保存")
        return 2
    print(f"自检 2 通过：seq={a.seq} 前向 loss = {loss:.4f}")

    model.to(torch.float32)
    model.save_pretrained(a.out)
    model.config.save_pretrained(a.out)
    print(f"已保存到 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
