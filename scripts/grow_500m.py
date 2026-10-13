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
* ✅ `growth_old_dim=768`（**继承源模型的原始切点**）→ 前三段映射为上表的
  `ln_old(768)` + `gamma_new[0:256]`（源值）+ `gamma_new[256:640]`（**零**）。
  新 384 维是零旁支，**函数严格保真**。

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

    def fill(p, old_key, out_d, in_d):
        """把旧权重放进新矩阵的左上角，新增行列保持 0。"""
        w = torch.zeros(out_d, in_d)
        ow = old_sd[old_key]
        w[: ow.shape[0], : ow.shape[1]] = ow
        new_sd[p] = w

    def grow_norm(new_p, old_prefix, d_new):
        """把源的一段/两段 norm 映射到新的两段 norm，新段补零。"""
        if src_grown:
            new_sd[new_p + ".ln_old.weight"] = old_sd[old_prefix + ".ln_old.weight"].clone()
            new_sd[new_p + ".ln_old.bias"] = old_sd[old_prefix + ".ln_old.bias"].clone()
            src_gn, src_bn = old_sd[old_prefix + ".gamma_new"], old_sd[old_prefix + ".beta_new"]
        else:
            new_sd[new_p + ".ln_old.weight"] = old_sd[old_prefix + ".weight"].clone()
            new_sd[new_p + ".ln_old.bias"] = old_sd[old_prefix + ".bias"].clone()
            src_gn = torch.zeros(0); src_bn = torch.zeros(0)
        n_new = d_new - grow_old_dim
        gn = torch.zeros(n_new); bn = torch.zeros(n_new)
        if src_gn.numel():
            k = src_gn.numel()
            assert k <= n_new, f"源的 gamma_new({k}) 比新的({n_new})还大"
            gn[:k] = src_gn            # 源的新段原样搬过来
            bn[:k] = src_bn
        # 其余保持 0 → 新增维度是零旁支
        new_sd[new_p + ".gamma_new"] = gn
        new_sd[new_p + ".beta_new"] = bn

    # 嵌入：旧列照搬，新列 0
    emb = torch.zeros(cfg.vocab_size, NEW_D)
    emb[:, :OLD_D] = old_sd["embedding.weight"]
    new_sd["embedding.weight"] = emb

    for i in range(NEW_L):
        p, q = f"blocks.{i}.", f"blocks.{i}."
        fill(p + "msr.q_proj.weight", q + "msr.q_proj.weight", NEW_D, NEW_D)
        fill(p + "msr.k_proj.weight", q + "msr.k_proj.weight", NEW_D, NEW_D)
        fill(p + "msr.v_proj.weight", q + "msr.v_proj.weight", NEW_VD, NEW_D)
        fill(p + "msr.g_proj.weight", q + "msr.g_proj.weight", NEW_VD, NEW_D)
        fill(p + "msr.out_proj.weight", q + "msr.out_proj.weight", NEW_D, NEW_VD)
        fill(p + "ffn.0.weight", q + "ffn.0.weight", NEW_VD, NEW_D)
        fill(p + "ffn.2.weight", q + "ffn.2.weight", NEW_D, NEW_VD)

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

        grow_norm(p + "norm1", q + "norm1", NEW_D)
        grow_norm(p + "norm2", q + "norm2", NEW_D)

    grow_norm("final_norm", "final_norm", NEW_D)

    new.load_state_dict(new_sd, strict=True)
    new.tie_weights()
    n = sum(pp.numel() for pp in new.parameters())
    ok_n = (n == TARGET_PARAMS)
    print(f"生长后参数量 = {n:,}  （期望 {TARGET_PARAMS:,}）  "
          f"{'✅' if ok_n else '❌ 差 ' + format(n - TARGET_PARAMS, ',')}")

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

    # **校验失败就不落盘**：静默产出一个"生长过但不保真"的检查点，
    # 会让后面无法区分 loss 尖峰来自生长还是来自别的（本项目已栽过同类静默错误）。
    if verified is False:
        print("❌ 校验未通过 —— 拒绝保存。请先修映射，别拿这个检查点训练。")
        return 2

    new.save_pretrained(args.dst, safe_serialization=True)
    print(f"saved -> {args.dst}")
    return 0 if ok_n else 1


if __name__ == "__main__":
    sys.exit(main())
