"""8 GB 能装多大模型 —— 严格记账。

    python scripts/vram_budget.py

所有常数标注来源。**不跑 GPU，纯算术。**

早先一版有一个方法错误：把「笔记本 nvidia-smi 的 7226 MiB」减去
「桌面的 reserved 6.61 GiB」得出 0.45 GiB 的"分配器外开销"——
那是**跨两台机器相减**，不能作为常数。本版改用笔记本（真正在训练的那台）
的 nvidia-smi 实测值作为基准，不再跨机器拼数。
"""

# ---------------------------------------------------------------- 实测常数

TOTAL_MIB = 8188              # RTX 4070 Laptop / RTX 3070 都是 8188 MiB

# 272M 的 C' 配置（chunk 128, batch 1 x accum 4, **关**梯度检查点, 8-bit AdamW, compile）
# 在**笔记本**上的真实占用。2026-10-10 22:41-22:46 采样 100 次全部 7226 MiB（整卡稳态）。
#
# 关键：memory.used 是**整卡**占用。按进程拆开后（Windows GPU 计数器
# Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory）：
#   PID 32572 python（训练本体）= 6940 MiB
#   dwm 167 + WindowsTerminal 43 + LegionZone 37 + csrss 34 + explorer 32 + 其余 ~77
#   -> 桌面/系统合计约 390 MiB
# 所以真正属于训练进程的是 6940 MiB，不是 7226。
PROC_USED_MIB = 6940
DESKTOP_MIB = 390
LAPTOP_USED_MIB = 7226        # 整卡，留作对照

# 每参数常驻（训练态）
B_WEIGHT = 4.0      # fp32 权重：safetensors 365 个张量全 F32，1089.2 MB = 272.3M x 4B
B_GRAD = 4.0        # fp32 梯度
B_OPT = 2.004       # 8-bit AdamW：m/v 各 1 B/参数 + absmax(块 2048, fp32)
B_PER_PARAM = B_WEIGHT + B_GRAD + B_OPT

P_REF = 272_304_448
SAFETY_MIB = 205              # 0.20 GiB 余量，别贴着墙

GIB = 2 ** 30
MIB = 2 ** 20


def state_mib(p):
    """权重 + 梯度 + 优化器状态。"""
    return p * B_PER_PARAM / MIB


def main():
    print("=== 实测基准（笔记本，真正在训练的那台）===")
    print(f"  显存总量              = {TOTAL_MIB} MiB")
    print(f"  整卡 memory.used      = {LAPTOP_USED_MIB} MiB"
          f"  ({100*LAPTOP_USED_MIB/TOTAL_MIB:.1f}%)")
    print(f"  **训练进程本体**      = {PROC_USED_MIB} MiB"
          f"  ({100*PROC_USED_MIB/TOTAL_MIB:.1f}%)   <- 按进程拆出来的")
    print(f"  桌面/系统进程         = {DESKTOP_MIB} MiB"
          f"（dwm 167 + terminal 43 + LegionZone 37 + csrss 34 + explorer 32 + 其余）")

    st = state_mib(P_REF)
    print(f"\n  其中 权重+梯度+优化器 = {st:.0f} MiB"
          f"  ({P_REF*B_PER_PARAM/1e6:.1f} MB)")
    print(f"  训练进程的非状态内存  = {PROC_USED_MIB - st:.0f} MiB")
    print("    <- 这是**关梯度检查点**的激活 + workspace + CUDA context。")
    print("       项目报告 §6.2：B->C 快 1.26x 正是因为腾出显存后**关掉了**梯度检查点。")
    print("       而结构估算的「**开**检查点」激活在 272M 只有 262~467 MiB，")
    print("       比值 9.3~16.6x —— 与梯度检查点通常省一个数量级相符，")
    print("       但**开检查点的激活值本项目从未实测**，这只是自洽性检查。")

    proc_budget = TOTAL_MIB - DESKTOP_MIB - SAFETY_MIB
    print(f"\n  训练进程可用预算      = {TOTAL_MIB} - {DESKTOP_MIB}(桌面) - {SAFETY_MIB}(安全垫)"
          f" = {proc_budget} MiB")

    print("\n=== 维持「关检查点」时，还能加多少参数 ===")
    head = TOTAL_MIB - PROC_USED_MIB - DESKTOP_MIB - SAFETY_MIB
    per_m = 1e6 * B_PER_PARAM / MIB
    print(f"  可用余量            = {head} MiB")
    print(f"  每 1M 参数的常驻成本 = {per_m:.2f} MiB")
    print(f"  -> 在「其余内存不变」的假设下可再加约 {head/per_m:.0f} M 参数")
    print(f"  -> 即总计约 {(P_REF/1e6) + head/per_m:.0f} M")
    print("     ⚠️ 乐观算法：关检查点时激活值随参数量线性涨，真实上限更低。")
    print("        **结论：不打开梯度检查点，几乎上不去。**")

    print("\n=== 逐参数量核算（要求：**打开**梯度检查点、batch 1、seq 2048）===")
    print(f"  进程可用预算 = {proc_budget} MiB（= 整卡 - 桌面/系统 - 安全垫）")
    print("  非状态内存需要多少，本项目**没有实测过**。给三个档位看判定如何翻转：")
    print()
    hdr = f"  {'P':>7}  {'状态':>9}  {'可用余量':>9}"
    for label in ("非状态 500M", "非状态 1000M", "非状态 1500M"):
        hdr += f"  {label:>13}"
    print(hdr)

    buckets = (500, 1000, 1500)
    for p in (272_304_448, 400e6, 500e6, 600e6, 700e6, 750e6, 800e6, 1e9):
        s = state_mib(p)
        avail = proc_budget - s
        row = f"  {p/1e6:7.0f}M  {s:7.0f} MiB  {avail:7.0f} MiB"
        for b in buckets:
            net = avail - b
            if net < 0:
                row += f"  {'装不下':>13}"
            else:
                row += f"  {net:>10.0f} MiB 余"
        print(row)

    print("\n=== 结构估算的「激活」单独是多少（不含 context/workspace）===")
    print("  (a) 每层存下来的输入 L x seq x d x 4B")
    print("  (b) 重算一层时的瞬时张量；272M 时按 q/k/v/g + chunkwise 中间量 + MLP 隐层"
          "估为 0.10~0.30 GiB，随后按 L*d 线性放大")
    print(f"  {'配置':<14}{'L':>4}{'d':>6}{'(a)':>10}{'(b) 估':>14}{'激活合计(估)':>16}")
    act = {}
    for name, L, d in (("272M", 20, 1024), ("400M", 24, 1280),
                       ("600M", 28, 1408), ("700M", 32, 1536)):
        stored = L * 2048 * d * 4 / MIB
        scale = (L * d) / (20 * 1024)
        blo, bhi = 0.10 * scale * 1024, 0.30 * scale * 1024
        lo, hi = stored + blo, stored + bhi
        act[name] = (lo, hi)
        print(f"  {name:<14}{L:>4}{d:>6}{stored:8.0f} MiB{blo:7.0f}~{bhi:.0f} MiB"
              f"{lo:9.0f}~{hi:.0f} MiB")

    # context + cuBLAS/Triton workspace：本项目未测，取 400~500 MiB 一档
    CTX_LO, CTX_HI = 400, 500

    print("\n=== 结论（程序判定，不手写数字）===")
    for name, p in (("400M", 400e6), ("500M", 500e6), ("600M", 600e6),
                    ("700M", 700e6), ("750M", 750e6), ("800M", 800e6)):
        budget = proc_budget - state_mib(p)
        alo, ahi = act.get(name, (float("nan"), float("nan")))
        if alo != alo:
            # 没有结构示例的档位，用 700M 的比例外推
            sc = p / 700e6
            alo, ahi = act["700M"][0] * sc, act["700M"][1] * sc
        nlo, nhi = alo + CTX_LO, ahi + CTX_HI
        if nhi <= budget:
            v = f"**可以**（需求上限 {nhi:.0f} < 预算 {budget:.0f}）"
        elif nlo <= budget:
            v = f"**不能保证**（下限 {nlo:.0f} 过，上限 {nhi:.0f} 超预算 {budget:.0f}）"
        else:
            v = f"**不行**（下限 {nlo:.0f} 已超预算 {budget:.0f}）"
        print(f"  {name:<5} 状态 {state_mib(p):5.0f} MiB  预算 {budget:5.0f} MiB  "
              f"非状态需求 {nlo:.0f}~{nhi:.0f} MiB  {v}")

    print("\n=== 可选的杠杆（都未实测）===")
    for name, bpp in (("现状 8-bit AdamW", B_PER_PARAM),
                      ("paged_adamw_8bit（优化器分页到 CPU）", B_WEIGHT + B_GRAD + 0.004),
                      ("bf16 权重 + fp32 master（不省）", 2 + 4 + B_OPT + 2)):
        s7 = 700e6 * bpp / MIB
        print(f"  {name:<40} {bpp:6.3f} B/参数 -> 700M 状态 {s7:5.0f} MiB，"
              f"进程余量 {proc_budget - s7:5.0f} MiB")


if __name__ == "__main__":
    main()
