"""搜索 0.5B / 0.6B 的候选配置 —— 用 meta device 精确数参数，不手推公式。

    python scripts/design_500m.py

保持项目既有比例：key_dim = d_model/num_heads = 64、value_dim = 2*d_model、
ffn_dim = 2*d_model、vocab 20000、tie_weights。
"""
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402


def nparams(L, d, heads, vd, ffn, vocab=20000, swiglu=False):
    """在 meta device 上实例化，精确数参数量（不分配真实内存）。"""
    cfg = RetNetHFConfig(
        vocab_size=vocab, d_model=d, num_layers=L, num_heads=heads,
        value_dim=vd, ffn_dim=ffn, tie_weights=True, use_gla=True,
        use_swiglu=swiglu, use_growth_ln=False,   # 新模型不需要 growth LN
    )
    with torch.device("meta"):
        m = RetNetForCausalLM(cfg)
    return sum(p.numel() for p in m.parameters())


def main():
    print("=== 校准：现有 272M 配置的实算参数量 ===")
    got = nparams(20, 1024, 16, 2048, 2048)
    print(f"  算出来 {got:,}   实测 checkpoint 272,304,448   差 {got-272304448:+,}")

    print("\n=== 扫描（key_dim 固定 64，value_dim = ffn_dim = 2*d_model）===")
    print(f"  {'d':>6}{'L':>4}{'heads':>7}{'参数量':>14}{'vs 272M':>10}{'注意':>10}")
    cands = []
    for d in (1152, 1216, 1280, 1344, 1408, 1472, 1536, 1664):
        heads = d // 64
        if d % 64:
            continue
        for L in range(18, 34, 2):
            p = nparams(L, d, heads, 2 * d, 2 * d)
            if 440e6 <= p <= 680e6:
                cands.append((p, L, d, heads))
    cands.sort()
    for p, L, d, heads in cands:
        tag = ""
        if 480e6 <= p <= 530e6:
            tag = "<= 0.5B 候选"
        elif 570e6 <= p <= 630e6:
            tag = "<= 0.6B 候选"
        print(f"  {d:>6}{L:>4}{heads:>7}{p/1e6:>12.1f}M"
              f"{p/272304448:>9.2f}x{'  ' + tag if tag else '':>12}")

    print("\n=== 候选配置的完整预算表（单一口径，全部可复算）===")
    import importlib.util as _iu
    _spec = _iu.spec_from_file_location(
        "scale_speed", str(Path(__file__).resolve().parent / "scale_speed.py"))
    ss = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(ss)
    alpha, _ = ss.method_b_fit()

    TOTAL, DESKTOP, SAFETY = 8188, 390, 205
    proc_budget = TOTAL - DESKTOP - SAFETY
    B_PER_PARAM = 10.004

    picks = [
        ("0.5B 宽浅", 20, 1408, 22),
        ("0.5B 窄深", 26, 1216, 19),
        ("0.6B 宽浅", 24, 1408, 22),
        ("0.6B 窄深", 32, 1216, 19),
    ]
    print(f"  进程可用预算 = {TOTAL} - {DESKTOP}(桌面) - {SAFETY}(安全垫) = {proc_budget} MiB")
    print("  （步时单位 ms；method_a / method_b 返回值本身就是毫秒，不要再乘 1000）")
    print()
    hdr = (f"  {'方案':<10}{'d/L/heads':>14}{'参数量':>10}{'状态':>9}{'余量':>9}"
           f"{'层输入':>8}{'A乐观':>9}{'A悲观':>9}{'B':>9}")
    print(hdr)
    for name, L, d, heads in picks:
        p = nparams(L, d, heads, 2 * d, 2 * d)
        st = p * B_PER_PARAM / 2**20
        avail = proc_budget - st
        stored = L * 2048 * d * 2 / 2**20
        a_lo = ss.method_a(p, 0.85, 0.50)
        a_hi = ss.method_a(p, 1.00, 1.00)
        b = ss.method_b(p, alpha)
        print(f"  {name:<10}{f'{d}/{L}/{heads}':>14}{p/1e6:>9.1f}M{st:>8.0f}M{avail:>8.0f}M"
              f"{stored:>7.0f}M{a_lo:>8.0f}ms{a_hi:>8.0f}ms{b:>8.0f}ms")
    print()
    print("  前提（每条候选都成立才有效）：")
    print("  1) 训练态每参数 10.004 B（fp32 权重+fp32 梯度+8-bit Adam），已用 272M 校准")
    print("  2) 必须开梯度检查点；不开的话非状态内存按参数量线性外推到 8~9.6 GiB，装不下")
    print("  3) batch 1 × accum 4、seq 2048（seq 来自 chat_pack_v13 的 2048）")
    print("  4) 步时含梯度检查点的 1.266x（实测），并含笔记本/桌面 1.190x 设备差")
    print("  5) 方法 B 的指数来自旧实现两档 tqdm，batch 6 vs 4 未修，方向偏高")
    print("  6) 「层输入」只含每层存下来的输入，不含重算瞬时量 —— 后者未实测")


if __name__ == "__main__":
    main()
