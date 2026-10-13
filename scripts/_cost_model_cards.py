"""换卡是否更省：基于 3090 实测基线的成本模型与 break-even 敏感性分析。

    python scripts/_cost_model_cards.py

## 模型（唯一一条公式）

    每 token 成本 ∝ 单价 / 达成算力，  达成算力 = 峰值 × MFU
    R（换卡/3090 的成本比） = (P_new/P_3090) × (peak_3090/peak_new) × (MFU_3090/MFU_new)

它在「同一份 token 预算」下成立，前提是**算力需求与卡无关**：
本模型是稠密前向+反向、无重算（grad_checkpoint=false），所以
`FLOPs/token = 2 × 3 × κ·P_active` 只由模型与序列长度决定，换卡不改变分子。

## 口径与来源

- 实测基线（直接引用，不重估）：581 ms/优化步、8192 token/步、1 epoch=67,390 步
  = 552,330,489 token = 11.21 h = ¥16.6（¥1.48/时）；3090 峰值按 71 TFLOPS(bf16 dense)。
- 评测/存档开销按实测建模（不是忽略）：每次评测 833×2048 token 的前向，
  每次存档 60 s 写盘（**与卡速无关**，是磁盘时间）。
- 除基线外的峰值/单价都是**题给假设值**，不是实测。
"""
from __future__ import annotations

# ---------------- 输入：实测基线 ----------------
P_ACTIVE = 504_179_896
KAPPA = 1.0367                      # MAC/token = κ·P_active（前向，含非嵌入项）
FWD_MULT = 3                        # 训练 ≈ 3× 前向（2 fwd + 4 bwd 除以 2）
EPOCH_STEPS = 67_390
TOK_PER_STEP = 8192                 # batch 4 × seq 2048
EPOCH_TOK = 552_330_489
BASE_MS_PER_STEP = 581.0
BASE_HOURS_MEASURED = 11.21
BASE_PRICE = 1.48
BASE_PEAK = 71.0                    # RTX 3090 bf16 dense TFLOPS
EVAL_TOK = 833 * 2048               # val 全集
N_EVAL = 13                         # 每 epoch 评测次数（save_steps=5000）
SAVE_SEC = 60.0                     # 每次存档写盘秒数（磁盘时间，与卡无关）

# ---------------- 输入：候选卡（题给假设） ----------------
CARDS = [
    ("RTX 3090 24GB",      1.48, [71.0]),
    ("A800 80GB",          5.59, [312.0]),
    ("RTX PRO 6000 96GB",  6.98, [210.0, 375.0, 500.0]),
]
MFUS = [0.30, 0.40, 0.50, 0.62]

MAC_FWD = KAPPA * P_ACTIVE
FLOP_PER_TOK = 2.0 * FWD_MULT * MAC_FWD          # FLOP/token
FLOP_EPOCH = EPOCH_TOK * FLOP_PER_TOK            # FLOP/epoch
FLOP_EVAL = EVAL_TOK * 2.0 * MAC_FWD             # 一次评测的前向 FLOP


def epoch_hours(peak_tflops: float, mfu: float) -> tuple[float, float, float]:
    """返回 (纯计算 h, 评测 h, 存档 h)。"""
    achieved = peak_tflops * 1e12 * mfu              # FLOP/s
    h_compute = FLOP_EPOCH / achieved / 3600
    h_eval = N_EVAL * FLOP_EVAL / achieved / 3600
    h_save = N_EVAL * SAVE_SEC / 3600
    return h_compute, h_eval, h_save


def total_hours(peak, mfu):
    return sum(epoch_hours(peak, mfu))


def breakeven_mfu(price: float, peak: float, target_hours: float) -> float:
    """解 price × (compute+eval+save) = baseline_cost 时的 MFU。

    存档时间是固定项，先扣掉；compute+eval 与 1/MFU 成正比。
    """
    fixed_h = N_EVAL * SAVE_SEC / 3600
    variable_budget_s = (target_hours - fixed_h) * 3600
    # variable = (FLOP_EPOCH + N_EVAL·FLOP_EVAL) / (peak·mfu)  = budget
    flops_var = FLOP_EPOCH + N_EVAL * FLOP_EVAL
    return flops_var / (peak * 1e12 * variable_budget_s)


def main() -> None:
    base_cost = BASE_HOURS_MEASURED * BASE_PRICE

    print("=" * 78)
    print("0. 基线自检：模型能否复现实测的 11.21 h / ¥16.6")
    print("=" * 78)
    mfu_base = (TOK_PER_STEP / (BASE_MS_PER_STEP / 1000)) * FLOP_PER_TOK / (BASE_PEAK * 1e12)
    hc, he, hs = epoch_hours(BASE_PEAK, mfu_base)
    print(f"  FLOP/token            = {FLOP_PER_TOK/1e9:.3f} GFLOP")
    print(f"  FLOP/epoch            = {FLOP_EPOCH:.4e}")
    print(f"  实测 token/s          = {TOK_PER_STEP/(BASE_MS_PER_STEP/1000):,.0f}")
    print(f"  → 反解 MFU            = {mfu_base:.1%}")
    print(f"  模型算得：纯计算 {hc:.2f} h + 评测 {he*60:.1f} min + 存档 {hs*60:.1f} min"
          f" = {hc+he+hs:.2f} h")
    print(f"  实测：{BASE_HOURS_MEASURED} h  →  模型偏差 {(hc+he+hs)/BASE_HOURS_MEASURED-1:+.1%}")
    print(f"  成本：模型 {total_hours(BASE_PEAK,mfu_base)*BASE_PRICE:.1f} 元"
          f" vs 实测 {base_cost:.1f} 元")

    print()
    print("=" * 78)
    print("1. break-even 表：每 epoch 成本 / 相对 3090 倍数 / ¥每 1B token")
    print(f"   基准：3090 = {BASE_HOURS_MEASURED} h × ¥{BASE_PRICE} = ¥{base_cost:.2f}/epoch"
          f" = ¥{base_cost/(EPOCH_TOK/1e9):.1f}/1B token")
    print("=" * 78)
    hdr = f"{'卡':<20}{'峰值':>7}{'MFU':>6}{'达成TFLOP/s':>12}{'小时':>8}{'元/epoch':>10}{'倍数':>7}{'元/1B':>8}"
    print(hdr)
    print("-" * len(hdr) + "-" * 10)
    for name, price, peaks in CARDS:
        for pk in peaks:
            for m in MFUS:
                h = total_hours(pk, m)
                cost = h * price
                ratio = cost / base_cost
                per1b = cost / (EPOCH_TOK / 1e9)
                tag = f"{name.split()[0]} {pk:.0f}TF" if len(peaks) > 1 else name
                print(f"{tag:<20}{pk:>7.0f}{m:>6.0%}{pk*m:>12.1f}{h:>8.2f}"
                      f"{cost:>10.1f}{ratio:>7.2f}{per1b:>8.1f}")
        print()

    print("=" * 78)
    print("2. break-even MFU：新卡要跑到多少 MFU 才与 3090 打平（¥16.59/epoch）")
    print("=" * 78)
    print(f"{'卡':<20}{'峰值':>7}{'纯算力口径':>12}{'含存档固定开销':>16}{'可达性':>10}")
    for name, price, peaks in CARDS[1:]:
        for pk in peaks:
            t_pure = base_cost / price                 # 目标小时数
            m_pure = breakeven_mfu(price, pk, t_pure)
            # 含固定开销 = 上面这个函数已经扣掉了存档；把它标成"含"再看纯口径
            flops_var = FLOP_EPOCH + N_EVAL * FLOP_EVAL
            m_pure_only = flops_var / (pk * 1e12 * t_pure * 3600)
            verdict = "不可能" if m_pure > 1.0 else ("偏高" if m_pure > 0.62 else "合理")
            print(f"{name:<20}{pk:>7.0f}{m_pure_only:>11.1%}{m_pure:>15.1%}{verdict:>10}")
    print("  （'纯算力口径'= 忽略存档写盘时间；'含存档固定开销'= 存档的 13×60 s 不随卡速缩短，")
    print("    所以需要更高的 MFU 才能补回来 —— 这就是模型自身假设带来的 2~5 个点差异。）")

    print()
    print("=" * 78)
    print("3. Chinchilla 20×（10.0836 B token = 18.256 epoch）：总成本与墙钟")
    print("=" * 78)
    n_epoch = 10.0836e9 / EPOCH_TOK
    print(f"  epoch 数 = {n_epoch:.3f}；3090 实测锚点：¥{base_cost:.2f}/epoch、{BASE_HOURS_MEASURED} h/epoch")
    print(f"{'卡':<20}{'MFU':>6}{'小时/epoch':>12}{'总成本':>10}{'墙钟天数':>10}{'vs 3090':>9}")
    for name, price, peaks in CARDS:
        for pk in peaks:
            for m in ([mfu_base] if name.startswith("RTX 3090") else MFUS):
                if name.startswith("RTX 3090"):
                    h, cost = BASE_HOURS_MEASURED, base_cost
                else:
                    h = total_hours(pk, m)
                    cost = h * price
                tag = f"{name.split()[0]} {pk:.0f}TF" if len(peaks) > 1 else name
                print(f"{tag:<20}{m:>6.0%}{h:>12.2f}{cost*n_epoch:>10.0f}"
                      f"{h*n_epoch/24:>10.2f}{cost/base_cost:>9.2f}")
        print()

    print("=" * 78)
    print("4. 4× RTX 3090 DDP（每张 ¥1.48/时，合计 ¥5.92/时）")
    print("=" * 78)
    print(f"{'扩展效率':<10}{'有效加速':>10}{'小时/epoch':>12}{'元/epoch':>10}{'元/1B':>8}{'vs 1×3090':>11}")
    for eff in (1.00, 0.85, 0.70):
        speed = 4 * eff
        h = BASE_HOURS_MEASURED / speed
        cost = h * 4 * BASE_PRICE
        print(f"{eff:<10.0%}{speed:>10.2f}{h:>12.2f}{cost:>10.1f}"
              f"{cost/(EPOCH_TOK/1e9):>8.1f}{cost/base_cost:>11.2f}")
    print("  注：1.00 行是理想线性扩展 —— 成本与单卡完全相同（时间买到 1/4，钱也付 4 倍）。")

    print()
    print("=" * 78)
    print("5. 名义性价比（¥/TFLOP·h = 单价/峰值），以及它为什么不能直接用来选卡")
    print("=" * 78)
    for name, price, peaks in CARDS:
        for pk in peaks:
            tag = f"{name.split()[0]} {pk:.0f}TF" if len(peaks) > 1 else name
            print(f"  {tag:<20} ¥{price/pk:.5f} /TFLOP·h"
                  f"   （按 62% MFU 达成 → ¥{price/(pk*0.62):.5f}）")


if __name__ == "__main__":
    main()
