"""把「参数量 → 步时」做成可复算的模型，并给出 参数/算力 解耦的 Pareto 前沿。

    python scripts/flops_model.py            # 校核 + 前沿表
    python scripts/flops_model.py --check    # 只跑校核（三条独立锚点）

## 这个脚本要回答的问题

用户的目标是「把参数量提到 0.5B~0.6B，同时**守住训练速度**」。
`docs/16` §4.1 断言「每步成本随参数量线性涨」，但那是**陈述**，不是**推导**。
本脚本把它推成一条可检查的恒等式，并顺带找出唯一的逃逸口。

## 核心恒等式（本脚本 §1 校核）

对固定形状族（`vdim/d`、`ffn/d`、`key_dim` 恒定）：

    前向 MAC/token = κ(d, C) × P_active

`κ` **不是**普适常数，但散布很窄（实算，C=128）：d1024/L20 = 1.0494、
d1408/L20 = 1.0367、d1408/L24 = 1.0370、d1216/L26 = 1.0427。
即 **跨配置散布 ±1%**，所以 `MAC/token ∝ P_active` 的**线性关系是精确的**；
但 κ 随 `C` 涨（O(C·d) 块内项）、随 d 略降。
（本文早期版本写「κ = 1.033，与 d、L、heads 都无关」是**错**的——1.033 对应 C=64 的窄深配置。
线性结论不受影响，那句话已撤回。）

于是 `T_step ≈ launch_floor + 6·κ·P_active·tokens / TFLOPs`，
**只有「每 token 激活参数量」P_active 出现在成本里，总参数量 P_total 不出现。**

推论：要守住步时就必须守住 P_active。让 P_total > P_active 的**唯一**手段是
**条件计算**（有的参数不参与每个 token 的前向）。这也正是 `docs/16` §7.1 评估 MoE 时
**问反了的问题**：它问「MoE 能不能省 FLOPs」（答：不能，要付显存），
而用户问的是「能不能加参数不加 FLOPs」（答：MoE 是唯一能做到的机制）。

## 口径（严禁混用）

* 单位：**MiB**（1 MiB = 2^20 B）。凡是写 GB/GiB 的地方都显式标注。
* 每参数训练态字节数 `10.004 B`：fp32 权重 4 + fp32 梯度 4 + 8-bit Adam 2.004。
  已用 272M 的 2,598 MiB 实测校准（见 `docs/22` §0.1）。
* 显存预算：整卡 8,188 − 桌面 390 − 安全垫 205 = **7,593 MiB**（`scripts/vram_budget.py` 口径）。
  `docs/16` §1.3 用 `audit_mem.py` 量到 7,715 MiB（含安全垫），比本口径**松 122 MiB**。
  本脚本一律用紧的 7,593，因此结论偏保守。
* 步时锚点全部来自**笔记本 4070**、`data_cache/chat_pack_v13`、seq 2048、
  等效 batch 4 = 8,192 token/优化步。桌面 3070 的数一律不参与。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402

# ---------------------------------------------------------------- 常量与锚点
VOCAB = 20000
B_PER_PARAM = 10.004          # fp32 w + fp32 g + 8-bit Adam
TOTAL_MIB = 8188              # 整卡
DESKTOP_MIB = 390             # 桌面 + 系统
SAFETY_MIB = 205
BUDGET_MIB = TOTAL_MIB - DESKTOP_MIB - SAFETY_MIB     # 7,593

SEQ = 2048
TOKENS_PER_STEP = 8192        # batch1 × accum4 × seq2048

# --- 实测锚点（笔记本 4070）-------------------------------------------------
# docs/15 §1 表：C = batch1×accum4 + chunk128 + compile + 关检查点 + 8-bit Adam
#   微基准稳态口径（预热后、不含评测/存盘/取数）
ANCHOR_C_MS = 713.0
# v14 真实训练的**端到端**墙钟口径：1.27 it/s → 787 ms/步（读 logs/v14_272m_cont.log）
#   两者差 10.5%，差额 = 取数 + 评测摊销 + 日志 + 存盘。**都是对的，但不同口径，
#   不许互相减**。本文所有倍率都以 713 为分母（微基准口径），端到端换算见 §3。
ANCHOR_TRAIN_MS = 788.0
# docs/15 §3.1 表：同 batch1×accum4 + 检查点 = 936 ms（**当时还是 fp32 AdamW**）
ANCHOR_CKPT_MS = 936.0
# 检查点倍率的两个口径（详见 docstring）
CKPT_LO, CKPT_HI = 1.27, 1.31
# 发射/空闲地板：由上面两个锚点 + 「重算使 FLOPs ×4/3」联立反解出的**区间**。
# 第三条独立约束（见 docs/22 §3.2）也落在这里：docs/15 §0.2 的 profiler 记录
# C 配置一步发射 10,116 个内核、GPU 空闲 20.8%。按空闲率折算 0.208×713 = 148 ms；
# 按内核数折算 148 ms / 10,116 = 14.6 µs/内核的有效暴露成本（裸发射延迟是
# 17.7~23.7 µs，差额被内核排队重叠吃掉，物理上说得通）。两条路径都给 ~148 ms。
# 但 floor=148 会让 936 那个点反解出 ckpt=1.264，低于 [1.27,1.31]——这个张力
# 三种解释都能自圆其说，**只有 E1a/E1b 能分开**。所以区间保持覆盖它的全部后果。
FLOOR_LO_MS, FLOOR_HI_MS = 92.0, 148.0

# docs/16 已发表的、可交叉核对的两个 FLOPs 数（每 token 前向 MAC，**chunk 64**）
DOC16_DENSE_MACS = 281.3e6
DOC16_S1_MACS = 214.5e6

# 步时模型的**分母**必须用本项目自己的 272M/chunk128（= 实测锚点 713 ms 的那次跑的
# 形状）。用 docs/16 的 chunk64 数会让锚点行算出 722 ms 而不是 713 —— 那是把两个
# 不同 chunk 的口径相除，属于本会话已经犯过四次的错。chunk64 的数只用于 §1 校核 C/D。
BASE_MACS = None      # 由 main() 在导入后填：dense_macs_per_token(20,1024,16,2048,2048,128)


# ---------------------------------------------------------------- §1 计算量模型
def layer_macs(d: int, heads: int, vdim: int, ffn: int, chunk: int, seq: int):
    """单层**前向** MAC/token 的逐项分解。全部按形状数，不查表。"""
    kd = d // heads              # key_dim
    vdh = vdim // heads          # value_head_dim
    nc = seq // chunk
    return {
        # 五个投影：各有 1 个 MAC / 参数
        "q_proj": d * d,
        "k_proj": d * d,
        "v_proj": d * vdim,
        "g_proj": d * vdim,
        "out_proj": vdim * d,
        # retention 核（chunkwise 形式，O(L·C) 不是 O(L²)）
        "U = k^T (v·decay)": d * vdim // heads,
        "T @ U 闭式扫描": d * vdim * (nc + 1) // (heads * chunk),
        "inner = (q k^T)·mask": chunk * d,
        "inner @ v": chunk * vdim,
        "cross = q·decay @ S": d * vdim // heads,
        # FFN
        "ffn": 2 * d * ffn,
        # GroupNorm / silu 等逐元素算子（非 MAC，按等价 MAC 计，量级极小）
        "逐元素": 3 * vdim + 2 * d,
    }


def layer_params(d: int, vdim: int, ffn: int) -> int:
    """单层可训练参数（无 bias；两个 LayerNorm 各 d 个 weight）。"""
    return 2 * d * d + 3 * d * vdim + 2 * d * ffn + 2 * d


def dense_macs_per_token(L: int, d: int, heads: int, vdim: int, ffn: int,
                         chunk: int, seq: int = SEQ) -> int:
    """全模型**前向** MAC/token：L 层 + LM head 只算一次。"""
    per = sum(layer_macs(d, heads, vdim, ffn, chunk, seq).values())
    return L * per + VOCAB * d


def nparams_dense(L: int, d: int, heads: int, vdim: int, ffn: int) -> int:
    """meta device 上实例化后精确数参数量（不分配真实内存）。"""
    cfg = RetNetHFConfig(
        vocab_size=VOCAB, d_model=d, num_layers=L, num_heads=heads,
        value_dim=vdim, ffn_dim=ffn, tie_weights=True, use_gla=True,
        use_swiglu=False, use_growth_ln=False,
    )
    with torch.device("meta"):
        m = RetNetForCausalLM(cfg)
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- §2 解耦配方
class Recipe:
    """一个候选配置：总参数、每 token 计算量、显存、步时。

    两个轴必须分开报，混起来就是误导：
      * `p_total`    —— 「名义参数量」，用户要的那个数字；
      * `macs_active`—— 每 token 真正要算多少，**只有它决定步时**。
    两者的比值才是「加参数不加算力」的度量。
    """

    def __init__(self, name, L, d, heads, vdim, ffn, chunk,
                 experts=1, topk=1, mod_layers=0, mod_capacity=1.0,
                 note=""):
        assert topk <= experts
        assert ffn % topk == 0, "激活 FFN 必须与稠密逐位相同"
        self.name = name
        self.L, self.d, self.heads = L, d, heads
        self.vdim, self.ffn, self.chunk = vdim, ffn, chunk
        self.experts, self.topk = experts, topk
        self.mod_layers, self.mod_capacity = mod_layers, mod_capacity
        self.note = note

        ffn_expert = ffn // topk
        self.ffn_expert = ffn_expert

        # --- 参数：稠密骨架（= 1 个专家）+ (E−1) 份额外专家 ---
        # 额外专家的参数量用「加宽 ffn」的实测差分公式加（§1 校核 B 已验）
        self.p_dense_skel = nparams_dense(L, d, heads, vdim, ffn)
        self.p_total = self.p_dense_skel + (experts - 1) * 2 * d * ffn_expert * L
        self.p_active = self.p_dense_skel

        # --- 计算量 ---------------------------------------------------------
        per = sum(layer_macs(d, heads, vdim, ffn, chunk, SEQ).values())
        self.macs_total = L * per + VOCAB * d            # 稠密等效（不含 MoD 跳层）
        self.macs_router = L * d * experts               # 路由器：可忽略，但记账
        # MoD：被跳过的层按 capacity 折算。**跳层只减算力，不减参数。**
        mod_saving = mod_layers * (1.0 - mod_capacity)
        self.macs_active = int(self.macs_total * (1 - mod_saving / L)) + self.macs_router
        # 「每单位算力换到的参数」相对稠密基准归一，>1 才叫「加参数不加算力」
        self.param_per_flop = (self.p_total / self.macs_active)

    def rho(self) -> float:
        """平均一个 token 走到的参数占总参数的比例。

        两个机制不同源，所以**不强行合成一个公式**：
          * MoE（E 专家 top-1）：注意力参数全走，FFN 只走 1/E 份 → P_skel/P_total
          * MoD（部分层按容量跳层）：参数一个不少，但 token 少走若干层 → 1 − 跳掉的比例
        """
        if self.experts > 1:
            return self.p_active / self.p_total
        if self.mod_layers > 0:
            return 1.0 - self.mod_layers * (1.0 - self.mod_capacity) / self.L
        return 1.0

    # --- 显存 -------------------------------------------------------------
    def memory_mib(self) -> dict:
        state = self.p_total * B_PER_PARAM / 2 ** 20
        block_in = self.L * SEQ * self.d * 2 / 2 ** 20        # 每层存下的块输入 (bf16)
        # 单层重算瞬态：最大活着的中间张量，∝ seq × max(d, ffn_expert)
        transient = SEQ * max(self.d, self.ffn_expert) * 2 * 4 / 2 ** 20
        losshead = SEQ * VOCAB * (2 + 4 + 4) / 2 ** 20        # logits bf16+float+log_softmax
        experts_buf = (SEQ * self.d * 2 * 2 +
                       SEQ * self.ffn_expert * 2 * 2) / 2 ** 20 * self.topk
        frag = 0.15 * 1024
        return {"state": state, "block_in": block_in, "transient": transient,
                "losshead": losshead, "experts_buf": experts_buf, "frag": frag,
                "total": state + block_in + transient + losshead + experts_buf + frag}

    # --- 步时 -------------------------------------------------------------
    def overhead_band(self):
        """FLOPs 模型**没有**建模的额外开销倍率 (低, 高)。

        上界可**推导**，不是猜的（`scripts/` 里这条已验证）：

        * padding：固定容量路由要按 `capacity_factor` 算满。独立均匀路由下每个专家的
          负载 ~ Binomial(seq, 1/E)，相对标准差 `sqrt((E−1)/seq)`。本项目 seq=2048、
          E≤6 → 相对 sdev ≤ 4.9%，取 max(z≈1.6) 得 **cf ≈ 1.04~1.09**；
          Switch 的工程默认是 1.25。padding 只作用在 FFN 上，而 FFN 占单层
          MAC 的 **31.6%** → 开销 **(cf−1) × 31.6% = 2.5%~7.9%**。
          **seq=2048 是本项目的有利条件**：Switch 需要 cf=1.25 是因为它的序列短、E 大。
        * 路由器 + gather/scatter 内核：≈ +1~3%（未实测）
        * 优化器要更新 P_total 份参数：E=5 时 +336M 参数 × ~15 B ≈ +5 GB 流量 ≈ +25 ms
          （≈ +2.8%）

        合计 ≈ +6~13%。本函数给 (1.05, 1.15) 作为**保守包络**（低端略高于推导，
        因为第 2、3 项没有实测）。
        """
        if self.experts > 1:
            return (1.05, 1.15)
        if self.mod_layers > 0:
            return (1.03, 1.10)
        return (1.0, 1.0)

    def step_ms(self, floor_ms: float, ckpt: float, with_overhead: bool = True) -> float:
        """只用两个实测锚点 + 计算量模型，不含任何按参数量的幂律外推。

        T = floor·(L/20) + (713 − floor) · (MAC_active / BASE_MACS)   [关检查点]
        再乘检查点倍率与条件计算开销带。
        """
        assert BASE_MACS is not None
        ratio = self.macs_active / BASE_MACS
        base = floor_ms * (self.L / 20.0) + (ANCHOR_C_MS - floor_ms) * ratio
        t = base * ckpt
        if with_overhead:
            t *= self.overhead_band()[0]
        return t

    def step_ms_hi(self, floor_ms: float, ckpt: float) -> float:
        assert BASE_MACS is not None
        ratio = self.macs_active / BASE_MACS
        base = floor_ms * (self.L / 20.0) + (ANCHOR_C_MS - floor_ms) * ratio
        return base * ckpt * self.overhead_band()[1]


def check() -> bool:
    """§1 校核：四条独立锚点。任何一条不过就返回 False。"""
    ok = True
    print("=" * 78)
    print("§1 计算量/参数模型校核（全部为独立锚点，不是拟合）")
    print("=" * 78)

    # A. 参数量必须逐位等于真实 checkpoint
    p = nparams_dense(20, 1024, 16, 2048, 2048)
    good = (p == 272_304_448)
    ok &= good
    print(f"A 272M 参数量        : 模型 {p:,}  实测 272,304,448  "
          f"{'✅ 逐位相同' if good else '❌ 差 ' + format(p - 272304448, '+')}")

    # B. 单层参数公式必须与 meta device 之差一致（MoE 额外专家的算法依赖它）
    p1 = nparams_dense(20, 1024, 16, 2048, 2048)
    p2 = nparams_dense(20, 1024, 16, 2048, 4096)     # ffn_dim 加倍
    diff, expect = p2 - p1, 2 * 1024 * 2048 * 20
    good = (diff == expect)
    ok &= good
    print(f"B 单层 FFN=2·d·ffn  : 加倍 ffn 后差 {diff:,}  预期 {expect:,}  "
          f"{'✅' if good else '❌'}   （MoE 额外专家参数量按此式加，已验）")

    # C. 前向 MAC/token 必须复现 docs/16 §4.3 的 281.3M
    m = dense_macs_per_token(20, 1024, 16, 2048, 2048, 64)
    err = abs(m - DOC16_DENSE_MACS) / DOC16_DENSE_MACS
    good = err < 0.02
    ok &= good
    print(f"C 272M MAC/token     : 模型 {m/1e6:.1f}M  docs/16 {DOC16_DENSE_MACS/1e6:.1f}M  "
          f"差 {err*100:.2f}%  {'✅' if good else '❌'}")

    # D. S1（value_dim 减半）必须复现 docs/16 的 214.5M（−23.7%）
    m1 = dense_macs_per_token(20, 1024, 16, 1024, 2048, 64)
    err = abs(m1 - DOC16_S1_MACS) / DOC16_S1_MACS
    good = err < 0.02
    ok &= good
    print(f"D S1 MAC/token       : 模型 {m1/1e6:.1f}M  docs/16 {DOC16_S1_MACS/1e6:.1f}M  "
          f"差 {err*100:.2f}%  {'✅' if good else '❌'}")

    # E. 步时模型必须同时复现 713（关检查点）与 936（开检查点，fp32 AdamW 上界）
    print("\n§1b 步时模型自洽性（用实测锚点反解，再看能否同时命中）")
    print(f"  锚点：关检查点 {ANCHOR_C_MS:.0f} ms；开检查点 {ANCHOR_CKPT_MS:.0f} ms "
          f"（**当时是 fp32 AdamW，不是 8-bit** —— 口径不完全可比，见 docstring）")
    print(f"  {'floor(ms)':>10}{'反解 ckpt 倍率':>16}{'落在区间?':>12}")
    for f in (FLOOR_LO_MS, 120.0, FLOOR_HI_MS):
        # 713 = f + C ;  936 = (f + C·4/3)   ← 重算使 FLOPs ×4/3
        C = ANCHOR_C_MS - f
        ck = (f + C * 4 / 3) / ANCHOR_C_MS
        inband = "✅ 是" if CKPT_LO <= ck <= CKPT_HI else "❌ 否"
        print(f"  {f:>10.0f}{ck:>16.3f}{inband:>12}")
    print(f"  区间 [{CKPT_LO}, {CKPT_HI}] 与反解值一致 → floor 与 ckpt 倍率不能同时定死，")
    print(f"  **必须实测**。这正是 E1a/E1b 要钉的量（无人值守驱动已挂）。")

    print("\n§1 校核结果：", "✅ 全部通过" if ok else "❌ 有不通过项")
    return ok


# ---------------------------------------------------------------- §3 前沿表
def capacity_scan() -> None:
    """§2b：开检查点时的稠密容量上限，以及**在给定步时预算下 MoE 能换到多少名义参数**。

    `docs/16` §4.1 在 d=1024 上算到 L≤54 → ~700M，用的是 7,715 MiB 的松预算。
    这里用紧的 7,593 MiB，并且**宽度也一起扫**（因为 c_layer ∝ d²）。
    """
    print("\n" + "=" * 78)
    print("§2b 容量上限（开检查点）与步时预算下的最大名义参数")
    print("=" * 78)
    print(f"  预算 {BUDGET_MIB} MiB（docs/16 §1.3 用的是 7,715，松 122 MiB）")

    print(f"\n  (a) 稠密：给定 d，最多能加到几层")
    print(f"  {'d':>6}{'heads':>7}{'P_total':>10}{'L_max':>7}{'MAC/tok':>10}{'步时ms':>16}{'vs713':>11}")
    for d in (1024, 1216, 1408, 1664):
        if d % 64:
            continue
        L_max, P_at_max = None, None
        for L in range(20, 70, 2):
            r = Recipe("scan", L, d, d // 64, 2 * d, 2 * d, 128)
            if r.memory_mib()["total"] <= BUDGET_MIB:
                L_max, P_at_max = L, r
            else:
                break
        if P_at_max is None:
            continue
        t_lo = P_at_max.step_ms(FLOOR_HI_MS, CKPT_LO)
        t_hi = P_at_max.step_ms_hi(FLOOR_LO_MS, CKPT_HI)
        print(f"  {d:>6}{d//64:>7}{P_at_max.p_total/1e6:>9.0f}M{L_max:>7}"
              f"{P_at_max.macs_active/1e6:>9.0f}M{t_lo:>8.0f}~{t_hi:<8.0f}"
              f"{t_lo/ANCHOR_C_MS:>7.2f}~{t_hi/ANCHOR_C_MS:<5.2f}")

    print(f"\n  (b) 固定步时预算，MoE 能换到多少名义参数（骨架固定 d1024/L20）")
    print(f"  {'E':>4}{'P_total':>10}{'ACTIVE':>10}{'步时ms':>16}{'vs713':>11}  判定")
    for E in range(1, 9):
        r = Recipe(f"E={E}", 20, 1024, 16, 2048, 2048, 128, experts=E, topk=1)
        mem = r.memory_mib()
        if E == 1:
            t_lo = t_hi = r.step_ms(FLOOR_HI_MS, 1.0, with_overhead=False)
            note = "（稠密基线，关检查点）"
        else:
            t_lo, t_hi = r.step_ms(FLOOR_HI_MS, CKPT_LO), r.step_ms_hi(FLOOR_LO_MS, CKPT_HI)
            note = "✅ 装得下" if mem["total"] <= BUDGET_MIB else f"❌ 超 {mem['total']-BUDGET_MIB:.0f} MiB"
        print(f"  {E:>4}{r.p_total/1e6:>9.0f}M{r.p_active/1e6:>9.0f}M"
              f"{t_lo:>8.0f}~{t_hi:<8.0f}{t_lo/ANCHOR_C_MS:>7.2f}~{t_hi/ANCHOR_C_MS:<5.2f}  {note}")

    print("\n  读法：MoE 每加 1 个专家 +83.9M 名义参数、+0 算力，代价是 +0.84 GiB 状态。")
    print("  墙在 E=6 与 E=7 之间（E=6 只剩 313 MiB，E=7 超 487 MiB）。")
    print("  **也就是说：步时预算内 MoE 能把名义参数推到 ~0.69B，而稠密在 0.6B 就吃掉 2.6× 步时。**")

    print("\n  端到端口径换算：v14 真实训练实测 1.27 it/s = 787 ms/步（")
    print(f"  与微基准 {ANCHOR_C_MS:.0f} ms 差 {(ANCHOR_TRAIN_MS/ANCHOR_C_MS-1)*100:.1f}%，差额是取数+评测摊销+日志+存盘）。")
    print("  两口径的**倍率**近似相同（那些开销按步数而非参数量计），所以：")
    print(f"    0.6B 稠密 ≈ {ANCHOR_TRAIN_MS*2.51/1000:.2f}~{ANCHOR_TRAIN_MS*2.68/1000:.2f} s/步；"
          f"0.5B 稠密 ≈ {ANCHOR_TRAIN_MS*2.10/1000:.2f}~{ANCHOR_TRAIN_MS*2.26/1000:.2f} s/步（端到端）")
    print("  但评测时间本身 ∝ 参数量，所以端到端倍率会比微基准倍率**略高**，不是略低。")


def dilution_bound() -> None:
    """§2c 方向性修正：§2/§4 的步时模型**偏保守**（把逐元素算力也按 d² 缩放了）。

    模型的构造是「compute 部分整体 × MAC 比」。但真实步时里有一大块是
    **带宽受限的逐元素/归一化/损失** 算子，它们的成本随**张量元素数**走（∝ d），
    不随 d² 走（GEMM 才 ∝ d²）。所以加宽时这一块被模型**高估**了。

    ⚠️ **前提的强度不一样，必须分开看**：
      * 「GEMM ∝ d²、逐元素 ∝ d」——【推导】，可靠。
      * 「逐元素占 compute 时间的比例」——【转引】`docs/15` §0.3 写「~51%」。
        但该表的**基准含糊**：按 565 ms 忙时算，GEMM 应占 49~54%；
        而 10.28 TFLOP ÷ 277 ms = 37 TFLOPs，**超过了 batch1 实测形状峰值
        (26~42，见 docs/15 §0.4)**，说明 49% 这个数本身有不确定性。
        所以这里给**区间**，不给点值。
      * 「损失占逐元素的比例」——**没有数据**，给 0~40% 的区间（损失 ∝ V·seq，与 d 无关）。

    因此本节产出的是**保守方向的界**，用来提示「2.10~2.26× 可能偏悲观」，
    **不替换头条数字**。真正的裁决在 E3a/E3b。
    """
    print("\n" + "=" * 78)
    print("§2c 方向性修正：§4 的步时模型**偏保守**（单侧界）")
    print("=" * 78)
    print("  **稳健的部分（只需前提 1，【推导】）**：")
    print("    GEMM 成本 ∝ d²，而非 GEMM 的逐元素/归一化/损失 成本 ∝ 张量元素数（∝ d）。")
    print("    §4 的模型把**全部** compute 按 MAC 比（∝d² 那个）缩放 → 系统性**高估**。")
    print("    所以真实步时 ≤ 模型预测（在相同 floor/ckpt 下）。**方向是确定的，永远更快。**")
    print()
    print("  **量级（需要前提 2，【转引】强度较弱）**：")
    print("    docs/15 §0.3 记「GEMM ~49% / 逐元素 ~51%」，但它没写清分母是墙钟还是忙时：")
    print("      · 若分母是忙时 565 ms → GEMM 占 b = 0.49")
    print("      · 由 10.28 TFLOP 与 batch1 形状峰值 26~42 TFLOPs（docs/15 §0.4）反推，")
    print("        GEMM 时间 ∈ [245, 395] ms → b ∈ [0.43, 0.70]")
    print("    取 b ∈ [0.49, 0.70] 作为可辩护区间；再给损失占逐元素的 0~40%（无数据）。")

    G_LO, G_HI = 0.49, 0.70
    F_LO, F_HI = 0.0, 0.40

    # 272M 的 compute 部分（关检查点）
    base_compute = ANCHOR_C_MS - FLOOR_HI_MS          # 用 floor=148 的分解
    print(f"\n  272M 的 compute 部分 = {ANCHOR_C_MS:.0f} − {FLOOR_HI_MS:.0f} = {base_compute:.0f} ms")

    rows = [
        ("0.5B 宽浅", 20, 1408, 22, 2816, 2816, 128),
        ("0.5B 窄深", 26, 1216, 19, 2432, 2432, 128),
        ("0.6B 宽浅", 24, 1408, 22, 2816, 2816, 128),
        ("0.6B 窄深", 32, 1216, 19, 2432, 2432, 128),
    ]
    print(f"\n  在**同一个 floor/ckpt 端点** (148 ms, 1.27) 下比较，隔离出稀释效应：")
    print(f"  {'方案':<12}{'MAC比':>8}{'元素比':>8}{'模型 ms':>10}{'vs713':>8}"
          f"{'稀释后 ms':>14}{'vs713':>14}{'降幅':>12}")
    for name, L, d, heads, vdim, ffn, C in rows:
        r = Recipe(name, L, d, heads, vdim, ffn, C)
        mac_ratio = r.macs_active / BASE_MACS
        elem_ratio_act = (L / 20.0) * (d / 1024.0)        # 激活元素数 ∝ L·d
        model_compute = base_compute * mac_ratio
        t_model = (FLOOR_HI_MS * (L / 20.0) + model_compute) * CKPT_LO
        est = []
        for b in (G_LO, G_HI):
            for f in (F_LO, F_HI):
                elem_ratio = (1 - f) * elem_ratio_act + f   # 损失部分与 d 无关
                tc = base_compute * (b * mac_ratio + (1 - b) * elem_ratio)
                t = (FLOOR_HI_MS * (L / 20.0) + tc) * CKPT_LO
                est.append((tc, t, t / ANCHOR_C_MS))
        tc_lo = min(e[0] for e in est); tc_hi = max(e[0] for e in est)
        t_lo = min(e[1] for e in est);  t_hi = max(e[1] for e in est)
        mx_lo = min(e[2] for e in est); mx_hi = max(e[2] for e in est)
        drop_lo = (1 - t_lo / t_model) * 100
        drop_hi = (1 - t_hi / t_model) * 100
        print(f"  {name:<12}{mac_ratio:>8.3f}{elem_ratio_act:>8.3f}"
              f"{t_model:>9.0f}{t_model/ANCHOR_C_MS:>7.2f}×"
              f"{t_lo:>9.0f}~{t_hi:<4.0f}{mx_lo:>8.2f}~{mx_hi:<5.2f}×"
              f"{drop_hi:>7.1f}~{drop_lo:<5.1f}%")

    print("\n  用法（**不要替换 §4.3 的头条区间**）：")
    print("  §4.3 的头条是 0.5B 2.10~2.26× / 0.6B 2.51~2.68×。稀释效应是**单侧**的，")
    print("  所以完整的诚实区间应把低端下移（数值取自上面表格，不自造）：")
    print("    **0.5B ≈ 1.79~2.26×，0.6B ≈ 2.10~2.68×**")
    print("  下限 = 稀释后最低（同 floor/ckpt 端点）与 floor/ckpt 低端的最优组合；")
    print("  上限 = 未稀释的 floor/ckpt 最差端点。这是**更大的**不确定带，")
    print("  不是更好的结果预告——E3a/E3b 是唯一裁决。")
    print()
    print("  它**不改变** §2 的恒等式，也**不改变**「稠密没有同速解」：")
    print("  稀释只让斜率比 d² **略缓**，不会让斜率变成 0 —— 前面那个 1 附近的")
    print("  「参数/算力比」列才是根因。")


def pareto() -> None:
    print("\n" + "=" * 78)
    print("§2 参数/算力解耦前沿：P_total（名义）与 MAC/token（成本）必须分开看")
    print("=" * 78)
    print("  前提（每一条都成立上表才有效）：")
    print("  1) 训练态 10.004 B/参数，用 272M 的 2,598 MiB 实测校准过")
    print("  2) 一律**开梯度检查点**（0.5B 不开装不下：非状态内存按参数线性外推 ≈ 7.2 GiB）")
    print("  3) seq 2048、等效 batch 4 = 8,192 token/步、chunk 128、8-bit Adam、compile")
    print("  4) MoE 用固定容量路由（top-1 或 top-2/E、每专家 ffn/topk），")
    print("     保证**激活 FFN 计算量与稠密逐位相同**；额外只有路由器与 gather/scatter")
    print("  5) MoE/MoD 的参数量按 §1 校核 B 的公式加，**代码尚未实现**，故步时是推导值")

    recipes = [
        Recipe("现状 272M 稠密", 20, 1024, 16, 2048, 2048, 128,
               note="当前在跑的 v14 配置；**必须关检查点**才拿到 713 ms"),
        Recipe("0.5B 稠密宽浅", 20, 1408, 22, 2816, 2816, 128,
               note="docs/22 §2.1 候选"),
        Recipe("0.5B 稠密窄深", 26, 1216, 19, 2432, 2432, 128,
               note="同参数、更多层：激活 ∝ √(P·L) 更差，仅作对照"),
        Recipe("0.6B 稠密宽浅", 24, 1408, 22, 2816, 2816, 128,
               note="docs/22 §2.1 候选"),
        # --- MoE 家族：E 从 2 到 6，每次加一整份 FFN（ffn=2d，与项目比例一致）---
        Recipe("MoE E=2 top1", 20, 1024, 16, 2048, 2048, 128, experts=2, topk=1,
               note="+1 份额外 FFN"),
        Recipe("MoE E=4 top1", 20, 1024, 16, 2048, 2048, 128, experts=4, topk=1,
               note="+3 份"),
        Recipe("MoE E=5 top1", 20, 1024, 16, 2048, 2048, 128, experts=5, topk=1,
               note="**名义 0.6B，激活仍 272M** → 步时≈现状"),
        Recipe("MoE E=6 top1", 20, 1024, 16, 2048, 2048, 128, experts=6, topk=1,
               note="名义 0.69B；逼近显存墙"),
        Recipe("MoE E=7 top1", 20, 1024, 16, 2048, 2048, 128, experts=7, topk=1,
               note="**预期装不下**：用来定位墙在哪"),
        Recipe("0.5B 稠密+MoD", 20, 1408, 22, 2816, 2816, 128,
               mod_layers=10, mod_capacity=0.5,
               note="10/20 层按 50% 容量跳层：参数全保留，算力 −25%"),
    ]

    base = recipes[0]
    print(f"\n  归一基准 = 现状行：MAC/token {base.macs_active/1e6:.0f}M、"
          f"{ANCHOR_C_MS:.0f} ms、显存 {base.memory_mib()['total']:.0f} MiB")
    print("  「参数/算力」= (P_total/MAC_active) / 基准同量 —— >1 才叫「加参数不加算力」")
    print("  步时区间 = floor∈[92,148] × ckpt∈[1.27,1.31] × 条件计算开销带 的端点组合\n")
    print(f"  {'方案':<16}{'d/L/h':>10}{'P_total':>9}{'MAC/tok':>9}{'参数/算力':>10}"
          f"{'显存GiB':>9}{'余量MiB':>9}{'步时ms':>17}{'vs713':>11}")
    for r in recipes:
        mem = r.memory_mib()
        if r.name.startswith("现状"):
            # 现状是**关**检查点的实测点，不乘检查点倍率，也不乘开销带
            t_lo = r.step_ms(FLOOR_HI_MS, 1.0, with_overhead=False)
            t_hi = r.step_ms(FLOOR_HI_MS, 1.0, with_overhead=False)
        else:
            t_lo = r.step_ms(FLOOR_HI_MS, CKPT_LO)
            t_hi = r.step_ms_hi(FLOOR_LO_MS, CKPT_HI)
        fit = "✅" if mem["total"] <= BUDGET_MIB else "❌OOM"
        print(f"  {r.name:<16}{f'{r.d}/{r.L}/{r.heads}':>10}"
              f"{r.p_total/1e6:>8.0f}M{r.macs_active/1e6:>8.0f}M"
              f"{r.param_per_flop/base.param_per_flop:>10.2f}"
              f"{mem['total']/1024:>9.2f}"
              f"{BUDGET_MIB - mem['total']:>9.0f}"
              f"{t_lo:>8.0f}~{t_hi:<8.0f}{t_lo/ANCHOR_C_MS:>7.2f}~{t_hi/ANCHOR_C_MS:<5.2f} {fit}")

    print(f"\n  显存预算 = {TOTAL_MIB} − {DESKTOP_MIB} − {SAFETY_MIB} = {BUDGET_MIB} MiB "
          f"= {BUDGET_MIB/1024:.2f} GiB")
    print("  步时区间来自 floor ∈ [92,148] ms × ckpt ∈ [1.27,1.31] 的端点组合，")
    print("  端点取**同向**是为了给出最宽的诚实误差带，不是置信区间。")

    print("\n  显存分项（MiB）—— 用来检查是不是状态主导：")
    print(f"  {'方案':<16}{'状态':>9}{'块输入':>9}{'瞬态':>9}{'损失头':>9}{'专家缓冲':>10}{'碎片':>8}")
    for r in recipes:
        m = r.memory_mib()
        print(f"  {r.name:<16}{m['state']:>9.0f}{m['block_in']:>9.0f}"
              f"{m['transient']:>9.0f}{m['losshead']:>9.0f}"
              f"{m['experts_buf']:>10.0f}{m['frag']:>8.0f}")

    print("\n  **读法**：前四行 P_total ≈ MAC/tok（「参数翻倍 = FLOPs 翻倍」的直接后果），")
    print("  MoE 行的 MAC/tok 不动而 P_total 涨，MoD 行反过来。")
    print("  「参数/算力」列才是这里唯一的自变量：>1 就是「加参数不加算力」。")

    print("\n" + "=" * 78)
    print("§3 结论")
    print("=" * 78)
    print("  1) 稠密路线（相对现状 713 ms）：0.5B = 2.1~2.3×，0.6B = 2.5~2.7×。")
    print("     docs/22 §2.3 的「method A 乐观 1.96×」**偏低**，因为用了 0.85/0.50 的")
    print("     次线性幂律——那没有依据。固定形状族下 FLOPs 对 MAC 是**严格线性**。")
    print("  2) 穷举之后，让 P_total > P_active 的机制只有**条件计算**：")
    print("     MoE（P_total↑，P_active 平）、MoD（P_total 平，P_active↓）。")
    print("     权重共享（减唯一参数）/ 量化（不改 FLOPs）/ GQA（无 KV cache）")
    print("     **都不改变「每 token 要算多少」**，所以都到不了用户要的那个方向。")
    print("  3) `docs/16` §7.1 对 MoE 的否决**问的是另一个问题**：")
    print("     它问「MoE 能不能省 FLOPs」→ 答「不能，要 +31% 显存」。")
    print("     用户问「能不能加参数不加 FLOPs」→ MoE 是**唯一**能做到的机制。")
    print("     在步时预算内：MoE 名义参数上限 ~0.69B（E=6，1.33~1.51×），")
    print("     稠密要到 0.6B 就得付 2.5~2.7×。**MoE 每单位步时换到的名义参数是稠密的 ~2.5 倍。**")
    print("     但见 §4：名义参数 ≠ 稠密容量，这条不构成「应该做 MoE」的结论。")
    print("  4) 宽度 > 深度，现在有了**步时**层面的量化：同为 ~700M，")
    print("     d1664/L20 = 2.80~3.04×，而 d1024/L54 = 3.31~3.40×（快 15~18%）。")
    print("     原因是发射地板 ∝ 层数，深度要付两次：激活 ∝ √(P·L)，发射 ∝ L。")


def caveats() -> None:
    print("\n" + "=" * 78)
    print("§4 MoE 路线的反面证据（不写这些就是误导）")
    print("=" * 78)
    items = [
        ("名义参数 ≠ 稠密容量",
         "MoE 的 scaling law（arXiv:2402.07871）在**等激活 FLOPs** 下优于稠密，"
         "但同 P_total 的 MoE 明显弱于同 P_total 的稠密。\n"
         "     所以 0.6B/272M-active 的 MoE **不等于** 0.6B 稠密。"
         "docs/04 §4 那条「数学/结构化输出要 ≥0.5–1B」的证据\n"
         "     是**稠密**尺度律，用它来论证 MoE 的名义参数量是**无效**的。"),
        ("每专家数据被稀释 top-1×E",
         "top-1 路由下每个专家只见到 1/E 的 token。E=5 时专家的等效数据量降到 1/5。\n"
         "     docs/16 §4.2 已经指出 272M 只有 ~2.5 token/参数；再除 5 = **0.5 token/参数**。"),
        ("torch.compile 会被动态形状打断",
         "本项目 2.41× 的提速全部来自 compile + 融合。数据相关的专家分桶会让形状变化，\n"
         "     触发图断裂或反复重编译。**必须用固定容量路由**（Switch 式 capacity factor），\n"
         "     代价是溢出 token 被丢弃或加残差旁路。这是最大的工程风险。"),
        ("单卡 + batch 1",
         "本项目 batch=1 × accum4。没有专家并行；按专家分桶后每个专家 GEMM 的 M 从 2048\n"
         "     降到 ~2048/E。docs/15 §0.3 已经把「occupancy/split」列为已知问题，这正好加剧它。"),
        ("可能需要 +5~15% 步时的额外开销",
         "路由器、gather/scatter 内核、以及优化器步要更新 P_total 份参数（608M 对 272M：\n"
         "     优化器显存流量 +5 GB ≈ +25 ms）。这些**不在**上面的 FLOPs 模型里。"),
        ("本项目零证据",
         "模型代码里没有 MoE。要做得先加实现 + 负载均衡损失 + 固定容量，再跑 A/B。\n"
         "     在 272M 尺度上没有任何本项目自己的数据支持它值得做。"),
    ]
    for i, (t, d) in enumerate(items, 1):
        print(f"  {i}. {t}\n     {d}")
    print("\n  结论：MoE 能兑现「0.6B 名义参数 @ 现状步时」，但它兑现的**不是**用户要的那个 0.6B。")
    print("  真正该先做的是 §5 的三件事。")


def next_steps() -> None:
    print("\n" + "=" * 78)
    print("§5 建议的执行顺序（按 期望收益/代价 排）")
    print("=" * 78)
    rows = [
        ("1", "E1a/E1b：同口径测出检查点倍率与 floor",
         "已在跑的无人值守驱动会给出。它同时钉住本模型唯一的自由参数。"),
        ("2", "E2a/E2b：目标尺寸上的分层检查点逐档扫描",
         "唯一能把「必须开检查点」的代价换回来的手段。docs/15 只在 272M 上否定过它。"),
        ("3", "先跑 0.5B 稠密（而不是 0.6B）",
         "0.5B 余量 2.1 GiB、步时 2.2×；0.6B 余量 1.2 GiB、步时 2.6×。"
         "先证实尺度收益存在，再决定要不要付 0.6B 的钱。"),
        ("4", "MoE 只在 3 证实「尺度确实换来能力」之后才做",
         "否则会同时付两份代价：名义参数没换来能力，工程复杂度还吃掉 compile 的收益。"),
        ("5", "同时把数据量对齐",
         "docs/16 §4.2：272M 只有 2.5 token/参数，文献位置在 20+。"
         "这是**改变分母**的唯一一条，且不需要改架构。"),
    ]
    for n, a, b in rows:
        print(f"  {n}. {a}\n     {b}")


def main():
    global BASE_MACS
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只跑校核")
    a = ap.parse_args()
    # 锚点分母：必须与实测 713 ms 那次跑同样的 chunk（128）
    BASE_MACS = dense_macs_per_token(20, 1024, 16, 2048, 2048, 128)
    ok = check()
    if not a.check:
        pareto()
        capacity_scan()
        dilution_bound()
        caveats()
        next_steps()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
