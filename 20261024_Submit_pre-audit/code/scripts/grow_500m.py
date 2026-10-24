"""Grow 272M (d1024/L20/H16, already width-grown from 768) to 504M
(d1408/L20/H22) — STRICTLY function-preserving width growth.

    python scripts/grow_500m.py --src checkpoints/retnet_272m_v13 \
        --dst checkpoints/retnet_500m --verify

## 与 `grow_272m.py` 的关键差别（**这是本脚本存在的主要理由**）

`grow_272m.py` 把 143M(d768, **未生长过**、普通 LayerNorm) 长成 272M(d1024)，
用的是 `GrowthLayerNorm(growth_old_dim=768, d=1024)`。

而 **272M 本身已经是一次宽度生长的产物**：它的 config 里写着
`use_growth_ln=True, growth_old_dim=768, embed_scale=768**0.5`，
所以它的 norm 是**两段**的：

    x[..., :768]                       -> LayerNorm(768)
    x[..., 768:] * gamma_new + beta_new     (256 维仿射，无归一化)

`GrowthLayerNorm.forward` 只看 `old_dim` 一个切点。所以再长到 d1408 时：

* ❌ `growth_old_dim=1024`（看似"当前宽度"）→ 前 1024 维被当成**一个** LayerNorm(1024) 切片。
  但源模型前 768 维是用**只有这 768 维的统计量**归一化的，两者**不等价** →
  **函数不再保真**，而且**不报错**（本项目已经栽过 `_orig_mod.` 前缀那种"静默错误"）。
* ✅ `growth_old_dim=768`（**继承源模型的原始切点**）→ 映射为：
  `ln_old(768)` 照搬源的 `ln_old`；`gamma_new` / `beta_new` 覆盖
  `[768:1408]` 共 **640** 维，**整段**写成门控值 —— 中间层 `BLOCK_GATE=1.0`、
  `final_norm` `FINAL_GATE=0.0`。源模型在这 640 维上的**旧门控不被继承**
  （v21 的中间层旧门控是 0，照搬会让新维度永久死；final 的旧门控若非 0，
  会让 E03 的 embedding 重新随机漏进 logits —— 实测 max|Δ|=0.529）。

  函数保真的机制是**分两处**的，不能混为一谈：
  中间层的新维度进入 q/k/v/g/ffn 的**新列**，而那些列按下面的 `fill()`
  恒为 0，所以新维度不改变旧维度的输出；`final_norm` 的第二段是新维度
  通往 logits 的唯一出口，只有它必须走零旁支（`FINAL_GATE=0`）。
  旧版注释把两者都说成“零旁支”，与代码不符：中间层是 1.0，不是 0。

  **前提**：源检查点必须由本配方产出（中间层旧门控 1.0、final 0.0）。
  若源的第二段已经被训练偏了，整段重设就会改变前向；此时 `--verify` 会
  以 max|Δ logits| 超容差拒绝保存（rc=2），不会静默产出不可比的模型。

`checkpoints/probe_500m` 的 config 里写的是 `growth_old_dim=1024`——那是个**随机权重探针**，
只用来量显存/步时，参数总量与切点无关，所以没影响；**但它不能当作真实生长的配方**。

## 容差为什么是 1e-3（**实测的噪声地板**，不是为了让它通过而放松）

同一个模型、数学等价但算法不同的两条路径，实测 max|Δ logits|：

| 对比 | max abs(delta) |
|---|---|
| chunkwise(64) vs chunkwise(128) | **0.000**（逐位相同） |
| chunkwise(64) vs **recurrent** | **1.326e-04** ← 真实噪声地板 |
| chunkwise(64) vs **parallel** | 3.395（parallel 与 chunkwise 的实现差异远大于噪声） |

加宽会改变 matmul 的归约维（K: 1024 → 1408，多出的部分是 0），**数学等价、浮点不等价**，
所以保真校验**不可能**是 0。实测生长后是 **2.2e-4**（= 地板的 1.7 倍，
相对 logits 尺度 24.7 是 9e-6）。
而这个脚本第一版漏搬 `mode`/`chunk_size` 时给出的是 **1.9**——**差四个数量级**。
取 **1e-3** 同时远高于地板、又远低于真 bug，判别力足够。

## 为什么 L 不变

272M 是 L20，0.5B 目标也是 L20 → **纯宽度生长，不插新层**。
所以没有 `grow_272m.py` 里那种"零残差新层"分支，脚本因此更短。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf.modeling_retnet import RetNetForCausalLM, RetNetHFConfig  # noqa: E402

OLD_D, NEW_D = 1024, 1408
OLD_L, NEW_L = 20, 20
OLD_H, NEW_H = 16, 22
OLD_VD, NEW_VD = 2048, 2816
GROUP = 128                      # value_head_dim，两侧相同（2048/16 = 2816/22 = 128）
TARGET_PARAMS = 504_179_896      # 与 probe_500m / meta device 三方对齐

# 三个初始化常量，故意做成模块级：测试可以 monkeypatch 它们，验证
# "把新行标准差改成 0（回到旧行为）时，可训性校验必须变红"。
# 这不是为了配置灵活，是为了让 docs/00 E04 的守卫**可被证明会失败**。
NEW_ROW_INIT_STD = 0.02     # 新权重的行（新单元的输出）标准差
BLOCK_GATE = 1.0            # 中间层 norm 的新维度门控
FINAL_GATE = 0.0            # final_norm 的新维度门控（输出端保真）


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="272M 检查点（可以是已生长过的）")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verify", action="store_true", help="校验 logits 逐位不变")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="函数保真的 logits max|Δ| 容差。默认 1e-3，依据见下。")
    ap.add_argument("--seq", type=int, default=128, help="校验用的序列长度")
    args = ap.parse_args()

    torch.manual_seed(0)
    old = RetNetForCausalLM.from_pretrained(args.src)
    old_sd = old.state_dict()
    oc = old.config

    # --- 断言源模型就是脚本假设的那个形状（防止拿错检查点静默产出废品）---
    assert oc.d_model == OLD_D and oc.num_layers == OLD_L and oc.num_heads == OLD_H, \
        f"源模型形状不符：d={oc.d_model} L={oc.num_layers} H={oc.num_heads}，期望 {OLD_D}/{OLD_L}/{OLD_H}"
    assert oc.value_dim == OLD_VD, f"源 value_dim={oc.value_dim}，期望 {OLD_VD}"

    # --- 继承源的**原始**归一化切点，这是保真的关键 ---
    src_grown = bool(getattr(oc, "use_growth_ln", False))
    src_old_dim = int(getattr(oc, "growth_old_dim", 0) or 0)
    grow_old_dim = src_old_dim if src_grown else OLD_D
    # embed_scale 必须沿用源模型的值（生长不改变嵌入缩放）
    src_embed_scale = getattr(oc, "embed_scale", None)
    if src_embed_scale is None:
        src_embed_scale = grow_old_dim ** 0.5

    print(f"源模型: d={oc.d_model} L={oc.num_layers} H={oc.num_heads} vdim={oc.value_dim}")
    print(f"        use_growth_ln={src_grown}  growth_old_dim={src_old_dim}  "
          f"embed_scale={src_embed_scale:.6f}")
    print(f"生长切点采用 growth_old_dim = {grow_old_dim}"
          f"{'  ← 继承源（关键）' if src_grown else '  ← 源未生长过，取当前宽度'}")

    # --- ⚠️ 必须继承的**非权重**配置字段 ---
    # 第一版漏了 `mode`/`chunk_size`，结果新模型落到不同的保留模式，
    # **数学等价但数值不同** → max|Δ logits| = 1.9（函数保真校验失败）。
    # 这类"配置没搬过去"是**静默**的：模型照样能加载、能训练，只是生长白做了。
    src_mode = getattr(oc, "mode", None) or "chunkwise"
    src_chunk = int(getattr(oc, "chunk_size", 64) or 64)
    src_plw = getattr(oc, "prompt_loss_weight", 0.1)

    cfg = RetNetHFConfig(
        vocab_size=oc.vocab_size, d_model=NEW_D, num_layers=NEW_L, num_heads=NEW_H,
        value_dim=NEW_VD, ffn_dim=NEW_VD, dropout=oc.dropout, norm_eps=oc.norm_eps,
        tie_weights=True, use_gla=True, use_swiglu=False,
        use_growth_ln=True, growth_old_dim=grow_old_dim,
        embed_scale=src_embed_scale,          # 保持不变
    )
    # 这三个字段 RetNetHFConfig 的签名里没有单独形参，直接写上去
    cfg.mode = src_mode
    cfg.chunk_size = src_chunk
    cfg.prompt_loss_weight = src_plw
    print(f"        mode={src_mode}  chunk_size={src_chunk}  prompt_loss_weight={src_plw}  ← 一并继承")
    new = RetNetForCausalLM(cfg)
    new_sd = new.state_dict()

    def fill(p, old_key, out_d, in_d, gen):
        """旧权重放左上角；新**列**置 0（块内隔离），新**行**随机（让新单元有输出）。

        为什么是"列 0 行随机"，而不是全 0 或全随机：

        * 新列若随机，它会立刻改变旧维度的输出，函数不再保真，且无法用任何
          门控挡住（扰动已经进了旧维度）。所以新列必须是 0。
        * 新行若全 0，新单元（新头、新隐藏单元）的输出恒为 0，下游那些
          "读新单元"的列拿到的输入是 0，梯度 `上游 x 0 = 0`，整条新子网
          永远学不动——v21 的 338M 死参数就是这么来的。
        * 新行随机后：新单元有非零输出 -> 下游新列拿到非零输入 -> 新列在
          第 1 步就有梯度；新列变非零后，新行在第 2 步拿到梯度。
          这是一条**自愈的级联**，不是死锁。
        """
        w = torch.zeros(out_d, in_d)
        ow = old_sd[old_key]
        w[: ow.shape[0], : ow.shape[1]] = ow
        n_new_rows = out_d - ow.shape[0]
        if n_new_rows > 0:
            w[ow.shape[0]:, : ow.shape[1]] = \
                torch.randn(n_new_rows, ow.shape[1], generator=gen) * NEW_ROW_INIT_STD
        new_sd[p] = w

    def grow_norm(new_p, old_prefix, d_new, gate):
        """两段式 norm：旧段照搬，**第二段整段**（含源里已有的那一截）重设为 gate。

        这里**故意不**继承源模型第二段的 ``gamma_new[:n_src_new]``，理由分两边：

        * 中间层 gate=1.0：新维度要进入 q/k/v 的**新列**，而新列初始为 0；
          若门控也沿用源值（v21 里恰好是 0，E02 的 bug），则新列乘到的输入恒 0，
          ``dL/dW_newcol = 上游 x 0 = 0``，新列永远学不动。所以整段必须抬到 1.0。
        * final_norm gate=0.0：它是第二段通往 logits 的唯一出口。整段置 0 才能
          保证“生长后 logits 不变”—— embedding 的第二段会被重新随机（E03），
          若 final_norm 上半段还留着源的非零门控，那次随机就会漏进 logits。
          实测：把源 final ``gamma_new``（0.4）照搬过来，max|Δ logits| = 0.529。

        两者合起来：函数在输出端严格保真，而新参数从第 0/1 步起就有梯度。
        **代价是源模型第二段学到的门控会被丢弃**；这是有意的取舍，不是抄漏。
        """
        if src_grown:
            new_sd[new_p + ".ln_old.weight"] = old_sd[old_prefix + ".ln_old.weight"].clone()
            new_sd[new_p + ".ln_old.bias"] = old_sd[old_prefix + ".ln_old.bias"].clone()
        else:
            new_sd[new_p + ".ln_old.weight"] = old_sd[old_prefix + ".weight"].clone()
            new_sd[new_p + ".ln_old.bias"] = old_sd[old_prefix + ".bias"].clone()
        n_new = int(d_new) - int(grow_old_dim)
        new_sd[new_p + ".gamma_new"] = torch.full((n_new,), float(gate))
        new_sd[new_p + ".beta_new"] = torch.zeros(n_new)

    # 嵌入：旧列照搬；[grow_old_dim:] 一律重新随机。
    # 这一段既包含本次新增的 384 列，也包含上一次生长遗留的 256 个死列——
    # 它们必须一起复活，否则"函数保真"仍然成立，但新维度没有信号源，
    # 下游新列拿到的输入仍然是 0，梯度仍然是 0。
    gen = torch.Generator().manual_seed(1234)
    emb = torch.zeros(cfg.vocab_size, NEW_D)
    emb[:, :grow_old_dim] = old_sd["embedding.weight"][:, :grow_old_dim]
    rev = emb[:, grow_old_dim:]
    emb[:, grow_old_dim:] = torch.randn(rev.shape, generator=gen) * 0.02
    new_sd["embedding.weight"] = emb

    for i in range(NEW_L):
        p, q = f"blocks.{i}.", f"blocks.{i}."
        fill(p + "msr.q_proj.weight", q + "msr.q_proj.weight", NEW_D, NEW_D, gen)
        fill(p + "msr.k_proj.weight", q + "msr.k_proj.weight", NEW_D, NEW_D, gen)
        fill(p + "msr.v_proj.weight", q + "msr.v_proj.weight", NEW_VD, NEW_D, gen)
        fill(p + "msr.g_proj.weight", q + "msr.g_proj.weight", NEW_VD, NEW_D, gen)
        fill(p + "msr.out_proj.weight", q + "msr.out_proj.weight", NEW_D, NEW_VD, gen)
        fill(p + "ffn.0.weight", q + "ffn.0.weight", NEW_VD, NEW_D, gen)
        fill(p + "ffn.2.weight", q + "ffn.2.weight", NEW_D, NEW_VD, gen)

        # GroupNorm：按头成组拷贝，新头 weight=1 / bias=0
        gn_w = torch.ones(NEW_VD); gn_b = torch.zeros(NEW_VD)
        ow = old_sd[q + "msr.group_norm.weight"]; ob = old_sd[q + "msr.group_norm.bias"]
        assert ow.numel() == OLD_VD
        for h in range(OLD_H):
            gn_w[h * GROUP:(h + 1) * GROUP] = ow[h * GROUP:(h + 1) * GROUP]
            gn_b[h * GROUP:(h + 1) * GROUP] = ob[h * GROUP:(h + 1) * GROUP]
        new_sd[p + "msr.group_norm.weight"] = gn_w
        new_sd[p + "msr.group_norm.bias"] = gn_b

        # GLA gamma：旧头照搬（训练后已偏离公式值），新头用公式值
        new_gamma = 1.0 - torch.pow(2.0, -5.0 - torch.arange(NEW_H, dtype=torch.float32))
        new_gamma[:OLD_H] = old_sd[q + "msr.gamma"]
        new_sd[p + "msr.gamma"] = new_gamma

        # 中间层门控 = BLOCK_GATE：新维度必须能流进 q/k/v 的新列，否则新列永远零梯度
        grow_norm(p + "norm1", q + "norm1", NEW_D, gate=BLOCK_GATE)
        grow_norm(p + "norm2", q + "norm2", NEW_D, gate=BLOCK_GATE)

    # final_norm 门控 = FINAL_GATE：新维度在输出端被关掉，保证 logits 不变；
    # 它的梯度仍非零（上游经 tied lm_head 的新列回传），所以会自己长开
    grow_norm("final_norm", "final_norm", NEW_D, gate=FINAL_GATE)

    # 布局断言（P1-8）：文档写的映射必须就是真的映射，否则下一次生长会照着
    # 一段错的配方去改。检查三件事：分段数量、源段逐值继承、新增段恒为 gate。
    def _check_norm_layout(new_p: str, gate: float) -> None:
        g = new_sd[new_p + ".gamma_new"]
        b = new_sd[new_p + ".beta_new"]
        n_new = NEW_D - grow_old_dim
        if int(g.numel()) != n_new or int(b.numel()) != n_new:
            raise SystemExit(
                f"{new_p}: gamma_new/beta_new 长度 {g.numel()}/{b.numel()} "
                f"!= 期望 {n_new}（NEW_D={NEW_D}, 切点={grow_old_dim}）")
        if not torch.all(g == float(gate)):
            bad = g[g != float(gate)]
            raise SystemExit(
                f"{new_p}: 第二段 {n_new} 维门控应整段为 {gate}，"
                f"实际有 {bad.numel()} 个不等于它（如 {float(bad[0]):.4f}）")
        if not torch.all(b == 0.0):
            raise SystemExit(f"{new_p}: 第二段 beta 不是全 0")

    for i in range(NEW_L):
        _check_norm_layout(f"blocks.{i}.norm1", BLOCK_GATE)
        _check_norm_layout(f"blocks.{i}.norm2", BLOCK_GATE)
    _check_norm_layout("final_norm", FINAL_GATE)
    print(f"[layout] norm 分段布局断言通过：第二段整段 "
          f"blocks={BLOCK_GATE} / final={FINAL_GATE}（源旧门控有意不继承）",
          flush=True)

    new.load_state_dict(new_sd, strict=True)
    new.tie_weights()
    n = sum(pp.numel() for pp in new.parameters())
    # TARGET_PARAMS 是 504M 配方专用的常数；测试把维度缩小时它被置 0。
    # 旧实现仍然算 `n == 0` -> ok_n 恒 False -> 成功路径永远返回 1，
    # 测试只能写 `assert rc in (0, 1)`，于是“成功”和“参数量不符”分不开。
    if TARGET_PARAMS:
        ok_n = (n == TARGET_PARAMS)
        print(f"生长后参数量 = {n:,}  （期望 {TARGET_PARAMS:,}）  "
              f"{'✅' if ok_n else '❌ 差 ' + format(n - TARGET_PARAMS, ',')}")
    else:
        ok_n = True
        print(f"生长后参数量 = {n:,}（TARGET_PARAMS=0：跳过参数量断言，缩小配方）")

    verified = None
    if args.verify:
        x = torch.randint(1, cfg.vocab_size, (2, args.seq))
        old.eval(); new.eval()
        # NOTE: 必须传 labels —— `RetNetForCausalLM.forward` 只在 labels 非 None 时
        # 才给 `loss` 赋值，否则第 526 行 `CausalLMOutput(loss=loss, ...)` 直接
        # UnboundLocalError。（那是既有实现的小毛病，不是本脚本的问题；这里绕开它。）
        with torch.inference_mode():
            d = (old(x, labels=x).logits - new(x, labels=x).logits).abs().max().item()
        verified = d < args.tol
        print(f"函数保真校验: max|Δ logits| = {d:.3e}  容差 {args.tol:.0e}  "
              f"{'✅ OK' if verified else '❌ FAIL'}")

    if not args.verify:
        print("\n⚠️ 没加 --verify。**生长脚本不校验等于没生长**——"
              "配置文件漏搬（本项目真实踩过）会静默产出不保真的检查点。建议永远带上。")

    # ---- 可训性校验（E04）------------------------------------------------
    # 函数保真只证明"生长没改坏旧模型"，它**不能**证明"新参数能学"：
    # 一个恒零的旁支是最完美的函数保真，上面那条检查因此永远通过。
    # 这里跑 5 步真实反向，逐参数看梯度是否非零。零梯度 = 这个参数
    # 从第 0 步起就没有任何信号，训练多少步都不会变。
    #
    # 为什么是 5 步而不是 1 步：新维度是"级联自愈"的——
    #   第 0 步：只有 embedding 新列、以及读新输入的列有梯度；
    #   第 1 步：新列变非零，读它们的下游列开始有梯度；
    #   第 2 步：新单元的输出投影（新行）拿到梯度。
    # 5 步足够走完这条链；仍为零的参数就是真的死了。
    print(f"[trainable] 跑 5 步反向，seq={args.seq}，统计零梯度参数 ...", flush=True)
    new.train()
    opt = torch.optim.AdamW(new.parameters(), lr=1e-4)
    probe = torch.randint(1, cfg.vocab_size, (2, args.seq))
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out = new(probe, labels=probe)
        out.loss.backward()
        opt.step()
    dead = [n for n, p in new.named_parameters()
            if p.grad is None or p.grad.abs().max().item() == 0.0]
    trainable = not dead
    if dead:
        print(f"[trainable] ❌ {len(dead)} 个参数梯度恒 0：")
        for name in dead[:12]:
            print("    ", name)
    else:
        print("[trainable] ✅ 全部参数梯度非零")

    # **两道校验都过才落盘**：只保真但不可训的检查点，
    # 静默产出的就是 v21 那种"67% 参数是 0"的模型。
    if verified is False or not trainable:
        print("❌ 校验未通过 —— 拒绝保存。先修映射/初始化，别拿这个检查点训练。")
        return 2

    new.save_pretrained(args.dst, safe_serialization=True)
    print(f"saved -> {args.dst}")
    return 0 if ok_n else 1


if __name__ == "__main__":
    sys.exit(main())
