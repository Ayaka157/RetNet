"""500M / 600M 的步时预估 —— 两种独立方法 + 实证锚点。

    python scripts/scale_speed.py

方法 A（自下而上）：用 docs/19 §6.3 实测的**内核构成**，对每个分量按各自的标度律外推。
方法 B（自上而下）：用项目里两档真实模型的 tqdm 实测，拟合每 token 成本的经验指数。

两种方法**都给**，因为它们各自的偏差方向不同：
  A 假定 GEMM 效率随矩阵变大而改善 -> 偏乐观
  B 只有两档、且 batch 与梯度检查点都不同 -> 有系统偏差，方向不定

两种都不是测量。真正定论要跑一次 40 步实测。
"""

MIB = 2 ** 20

# ---------------------------------------------------------------- 实测锚点

# 笔记本上**正在跑**的 272M（chunk 128, batch 1 x accum 4, **关**检查点, 8-bit Adam, compile,
# seq 2048）。2026-10-10 从日志 10500->13200 三点内插：788 ms/步。
LAPTOP_272_MS = 788.0
TOK_PER_STEP = 1 * 4 * 2048          # batch x accum x seq = 8192

# 桌面上同一配置（C'）的实测与内核构成，docs/19 §6.3：
# 3 个微步共 470 ms busy：GEMM 312 / Triton elementwise 118 / Triton reduce 20 / elementwise 20
DESKTOP_272_MS = 662.0
PROFILE_BUSY_MS = {"gemm": 312.0, "triton_el": 118.0, "triton_red": 20.0, "torch_el": 20.0}
PROFILE_MICROSTEPS = 3
# 一次优化步 = 4 个微步；把构成按比例放大到一次优化步
SCALE_TO_OPTSTEP = 4 / PROFILE_MICROSTEPS

P_REF = 272_304_448

# 设备差：同一配置，笔记本 788 ms vs 桌面 662 ms
LAPTOP_OVER_DESKTOP = LAPTOP_272_MS / DESKTOP_272_MS

# 梯度检查点的代价：docs/19 §6.2 实测 B=903ms（开）vs C=713ms（关）=> 1.266x
GRAD_CKPT = 903.0 / 713.0

# 旧实现的实证点（**配置不同，仅作旁证**）：
#   rp_143m.log: 142.87M, seq 512, batch 6, **关**检查点 -> 2.66 it/s = 376 ms/步 = 3072 tok
#   rp_272m.log: 272.30M, seq 512, batch 4, **开**检查点 -> 1.50 it/s = 667 ms/步 = 2048 tok
EMP = [("143M", 142.87e6, 512 * 6, 376.0, False),
       ("272M", 272.30e6, 512 * 4, 667.0, True)]


def method_a(p, gamma_gemm=0.90, beta_other=0.65):
    """自下而上：分量按各自标度律外推，并乘上设备差与梯度检查点代价。"""
    r = p / P_REF
    t = PROFILE_BUSY_MS["gemm"] * SCALE_TO_OPTSTEP * (r ** gamma_gemm)
    for k in ("triton_el", "triton_red", "torch_el"):
        t += PROFILE_BUSY_MS[k] * SCALE_TO_OPTSTEP * (r ** beta_other)
    # 剩余（发射间隙等）按常数处理
    t += (DESKTOP_272_MS - sum(PROFILE_BUSY_MS.values()) * SCALE_TO_OPTSTEP)
    return t * LAPTOP_OVER_DESKTOP * GRAD_CKPT


def method_b_fit():
    """用两个旧实现实测点拟合每 token 成本的经验指数（先修掉梯度检查点）。"""
    pts = []
    for name, p, tok, ms, ckpt in EMP:
        per_tok = ms / tok
        if ckpt:
            per_tok /= GRAD_CKPT        # 换算成"关检查点"等价
        pts.append((p, per_tok))
    (p1, c1), (p2, c2) = pts
    alpha = __import__("math").log(c2 / c1) / __import__("math").log(p2 / p1)
    return alpha, pts


def method_b(p, alpha):
    """把经验指数作用在当前 272M 的实测每 token 成本上。"""
    base_per_tok = LAPTOP_272_MS / TOK_PER_STEP     # 当前已是"关检查点"
    return base_per_tok * ((p / P_REF) ** alpha) * TOK_PER_STEP * GRAD_CKPT


def main():
    print("=== 锚点 ===")
    print(f"  笔记本 272M（关检查点）实测   = {LAPTOP_272_MS:.0f} ms/步"
          f"  ({TOK_PER_STEP} token/步 -> {TOK_PER_STEP/(LAPTOP_272_MS/1000):.0f} tok/s)")
    print(f"  桌面   272M（同配置）实测     = {DESKTOP_272_MS:.0f} ms/步"
          f"  -> 笔记本/桌面 = {LAPTOP_OVER_DESKTOP:.3f}x")
    print(f"  梯度检查点代价（B/C 实测）    = {GRAD_CKPT:.3f}x")
    print(f"  一次优化步的内核构成（桌面）  = "
          + " / ".join(f"{k} {v*SCALE_TO_OPTSTEP:.0f}ms" for k, v in PROFILE_BUSY_MS.items()))

    alpha, pts = method_b_fit()
    print("\n=== 方法 B 的经验拟合（旧实现两档，配置不同，仅作旁证）===")
    for (p, c), (n, p0, tok, ms, ck) in zip(pts, EMP):
        print(f"  {n}: {p/1e6:.2f}M  每 token {c:.4f} ms"
              f"  (原始 {ms:.0f}ms / {tok}tok = {ms/tok:.4f} ms, "
              f"{'已除' if ck else '未除'}检查点 {GRAD_CKPT:.3f}x)")
    print(f"  -> 每 token 成本 ∝ P^{alpha:.3f}")
    print("  ⚠️ 该拟合**有系统偏差且方向是高估**：272M 那档的 batch 是 4、143M 是 6，")
    print("     固定开销摊到更少的 token 上，会把 272M 的每-token 成本抬高。")

    print("\n=== 预估 ===")
    print(f"  {'参数量':>7} {'方法A(乐观)':>14} {'方法A(悲观)':>14} {'方法B(经验)':>14}"
          f" {'tok/s 区间':>16} {'相对现在':>10}")
    for name, p in (("400M", 400e6), ("500M", 500e6), ("600M", 600e6), ("700M", 700e6)):
        a_lo = method_a(p, 0.85, 0.50)      # GEMM 改善 + 其余次线性 -> 乐观
        a_hi = method_a(p, 1.00, 1.00)      # 全线性 -> 悲观
        b = method_b(p, alpha)
        lo, hi = min(a_lo, b), max(a_hi, b)
        tps_hi, tps_lo = TOK_PER_STEP / (lo / 1000), TOK_PER_STEP / (hi / 1000)
        print(f"  {name:>7} {a_lo:11.0f} ms {a_hi:11.0f} ms {b:11.0f} ms"
              f" {tps_lo:7.0f}~{tps_hi:.0f} tok/s {lo/LAPTOP_272_MS:8.2f}~{hi/LAPTOP_272_MS:.2f}x")

    print("\n=== 换算成「每天能喂多少 token」与「跑完 3B 要多久」 ===")
    print(f"  {'参数量':>7} {'tok/s(中位)':>12} {'每天 token':>14} {'跑 3B token':>14}")
    now = TOK_PER_STEP / (LAPTOP_272_MS / 1000)
    print(f"  {'272M(现状)':>7} {now:10.0f} {now*86400/1e9:11.2f} B/天"
          f" {3e9/now/86400:11.1f} 天")
    for name, p in (("500M", 500e6), ("600M", 600e6)):
        a_lo = method_a(p, 0.85, 0.50)
        a_hi = method_a(p, 1.00, 1.00)
        b = method_b(p, alpha)
        mid = (min(a_lo, b) + max(a_hi, b)) / 2
        tps = TOK_PER_STEP / (mid / 1000)
        print(f"  {name:>7} {tps:10.0f} {tps*86400/1e9:11.2f} B/天 {3e9/tps/86400:11.1f} 天")

    print("\n=== 方法 A 的两个边界（说明乐观/悲观从哪来）===")
    print("  乐观：GEMM 效率随矩阵变大改善（指数 0.85），elementwise 按面积 L*d 次线性（0.50）")
    print("  悲观：所有分量都按参数量线性（指数 1.00）")
    print("  方法 B 只用旧实现两档的 tqdm，未修 batch 6->4 的差异，**有系统偏差**。")


if __name__ == "__main__":
    main()
