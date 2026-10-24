"""Generate docs/01_function_reference.md.

Principle: **code blocks are extracted from the source with AST**, never copied
by hand, so they stay byte-identical; explanations come from the EXPL map, and
the generator fails if any function is missing. English prose is loaded from
``doc_i18n/expl_en.json`` and overrides every EXPL entry.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

from _refs import cite, refs_md
from fix_status import FIX_NOTE, FIX_STATUS

#: Default target is the main repository; point the snapshot at its own tree
#: (code lives under <root>/code):
#:   RETNET_GEN_ROOT=<snapshot>/code RETNET_GEN_OUT=<snapshot>/docs
ROOT = Path(os.environ.get("RETNET_GEN_ROOT", r"E:\Workspaces\RetNet"))
OUT_DIR = Path(os.environ.get("RETNET_GEN_OUT", str(ROOT / "docs")))
OUT = OUT_DIR / "01_function_reference.md"

#: English prose shipped in the documentation. The Chinese EXPL entries below
#: are kept as the translation source; this file overrides every one of them so
#: the generated document is English-only. Missing keys abort generation.
I18N = Path(__file__).resolve().parent / "doc_i18n" / "expl_en.json"

FILES = [
    "retnet_hf/train.py",
    "retnet_hf/modeling_retnet.py",
    "retnet_hf/retention_fast.py",
    "retnet_hf/data.py",
    "retnet_hf/fused_ce.py",
    "run_task.py",
    "scripts/grow_500m.py",
]

# key = "相对路径::限定名"，value = (解释, [00 文档的编号])
EXPL: dict[str, tuple[str, list[str]]] = {}


def E(text: str, refs: list[str]) -> tuple[str, list[str]]:
    return text, refs


EXPL["retnet_hf/train.py::CompileSafeTrainer.get_decay_parameter_names"] = E(
    "决定哪些参数吃 weight decay。父类的默认规则只排除 bias 与 LayerNorm.weight，"
    "而 msr.gamma 是裸 nn.Parameter，两头都不沾，于是被 wd=0.1 一路乘性压小。"
    "本方法在开启 _no_decay_1d 时把所有一维参数（norm 的 scale、门控、gamma）移出衰减集合，"
    "并在首次调用时打印被移出的数量。它是 E06 的修复点：默认值已改为开启，"
    "并把 gamma 是否仍在衰减集合里作为开训前的硬断言。",
    ["E06", "E07"],
)

EXPL["retnet_hf/train.py::CompileSafeTrainer.save_model"] = E(
    "绕开 torch.compile 的 OptimizedModule 前缀问题。当模型被 OptimizedModule 包裹时，"
    "state_dict 的每个键都会带上 _orig_mod. 前缀，用 from_pretrained 加载这种检查点不会报错，"
    "只会把所有键当成多余键丢掉，最终得到一个随机模型。本方法在保存前临时把 self.model "
    "换回内层模型，保存后再换回来，从而保证落盘的键名干净，resume 与导出两条路径同时正确。",
    [],
)

EXPL["retnet_hf/train.py::verify_export"] = E(
    "导出后的自检：读 safetensors 的头部 JSON，检查键名里没有 _orig_mod. 前缀，"
    "并确认文件存在。它只验证键名的形状，不验证数值，因此查不出全零参数或死维度——"
    "E08 说的就是这一点：这个检查通过并不代表模型能训、能用，只代表加载不会静默失败。"
    "新增的活参数统计与梯度覆盖检查，才是补上这一块缺口的地方。",
    ["E08", "E23"],
)

EXPL["retnet_hf/train.py::_sync_resume_state"] = E(
    "续训前的状态对齐。HF 的 Trainer 会用检查点 trainer_state.json 里的 batch size "
    "覆盖命令行参数，而且缺少 optimizer.pt 时连学习率调度器都不恢复。本函数只比较"
    "**确实会被 HF 采纳**的四个字段：train_batch_size、logging_steps、eval_steps、"
    "save_steps；不一致时先备份原文、再就地改写，避免“命令行写 16、实际跑 1”的静默降级。"
    "grad_accum 与 max_steps 不在这份列表里——它们不会被 trainer_state.json 覆盖，"
    "写进对齐表只会制造“改了但其实没用”的错觉。",
    [],
)

EXPL["retnet_hf/train.py::_preflight"] = E(
    "开训前的硬检查，在模型加载之后、构造 Trainer 之前调用。它做两件事："
    "第一，扫权重数值统计“非零参数占比”，低于 95% 直接 SystemExit，"
    "因为参数量是从 config 数出来的，哪怕 67% 精确为 0 也会打印 504M（E08）；"
    "第二，收集所有 msr.gamma 并打印分布，出现 <=0 直接拒绝，避免 log(0) 让整层 NaN（E06）。"
    "它只查开训前能查的东西；训练中 gamma 是否被衰减，由单独的 step 比值断言负责。",
    ["E06", "E08"],
)

EXPL["retnet_hf/modeling_retnet.py::_chunked_weighted_ce"] = E(
    "分块加权交叉熵，替代原来对整段 logits 做 .float() 的写法（E22）。"
    "旧实现 logits_f = logits.float() 会为 B=16/L=2048/V=20000 物化约 2.6 GB 的 fp32 副本；"
    "本函数按**字节预算**反推每个分块的 token 数（默认 64 MB，V=20000 时 838 token/块），"
    "每块只把 [chunk,V] 转 fp32，生产形状 B=16/L=2048 因此切出 40 块而不是旧实现的一块。"
    "权重码语义与旧实现完全一致：pad=0 不计、reply/plain=1、prompt=prompt_loss_weight，"
    "最后除以权重和；没有 wcodes 时退化为普通均值。显式传 chunk_tokens 可覆盖预算（测试用）。",
    ["E22"],
)

EXPL["retnet_hf/modeling_retnet.py::build_causal_lm_output"] = E(
    "把 CausalLMOutput 的构造收成一个函数，让 forward 里只剩一处返回。"
    "它本身不改变数值，只是把 transformers 的输出类型依赖集中到一个地方，"
    "便于在测试里替换或在未来版本里改字段。E22 改 forward 时顺手抽出来，"
    "避免在 forward 尾部再出现局部 import。",
    ["E22"],
)

EXPL["retnet_hf/train.py::live_param_ratio"] = E(
    "统计模型里非零参数的比例（E08）。旧代码只打印 numel 之和，"
    "504,179,896 哪怕 67% 是精确 0 也照样显示 504M。这个函数把“名义参数”"
    "和“可学参数”分开，供 _preflight 判断是否拒绝开训，也被 test_live_params.py 直接测试。"
    "注意 GroupNorm 的 bias 初始就是 0，所以新模型的比例也不是 100%，绝对值没有意义，"
    "要看的量级是“是不是掉了一大截”。",
    ["E08"],
)

EXPL["retnet_hf/train.py::assert_val_disjoint"] = E(
    "开训前断言验证集目录与任何训练阶段都不同（E14）。上一轮 val_dir 是 chat_pack_v13，"
    "而最后 14,000 步的 sft_mix 里 40% 就是它；训练期 eval_loss 在末段从 4.55 掉到 2.4450，"
    "那不是模型变好，是它看到了验证集。函数在构造数据集之前抛 SystemExit，"
    "不给“先跑起来再说”的机会。",
    ["E14", "E15"],
)

EXPL["retnet_hf/data.py::_FlatStage._check_boundaries"] = E(
    "抽查 flat 分片里有没有 eos，没有就拒绝打开（E12）。flat 流按固定宽度切片，"
    "切点与文档边界无关；如果分词时没加 eos，模型看到的是“上一篇末行 + 下一篇开头”，"
    "于是学会忽略远距离内容。这里只抽查前 2M token（一片有 2.5e8，全扫太贵），"
    "足以发现“整包没有边界”；确属有意为之的流，必须在计划里显式写 require_eos: false。",
    ["E12"],
)

EXPL["retnet_hf/train.py::main"] = E(
    "训练入口：解析参数、加载模型与计划、构造数据集、创建 TrainingArguments、"
    "挂控制/衰减/梯度覆盖回调、就地编译 forward、可选续训，最后 trainer.train()。"
    "E06 对应 no_decay_1d 默认开启与 DecayGuardCallback 的优化器分组断言；"
    "E07 对应 GradCoverageCallback（第 1 步后统计零梯度，超阈值停机）；"
    "E08 对应 _preflight 的活参数判定（新模型 <95% 拒绝，剪枝检查点用 --allow_dead_params "
    "或历史基线放行）；E12 对应 plan_stage_specs 把 require_eos 带进数据集；"
    "E14 对应 assert_val_disjoint 的三层检查；E15 是 load_best_model_at_end=False；"
    "E17 的默认值来自 retnet_hf/defaults.py 单一常量；P2-1 把 --eval_batches 落成真正的"
    "评测行数上限。训练成功后在 rank 0 写 live_params.json 作为下次续训的基线。",
    ["E06", "E07", "E08", "E12", "E14", "E15", "E20"],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetHFConfig.__init__"] = E(
    "模型配置。默认值给出 vocab_size、d_model、num_layers、num_heads、dropout、norm_eps，"
    "value_dim 与 ffn_dim 未指定时取 2 倍 d_model；use_gla 控制 gamma 是可学习参数还是固定缓冲；"
    "use_growth_ln 与 growth_old_dim 决定归一化是否采用两段式；checkpoint_layers 控制"
    "分层梯度检查点。这些字段会随 config.json 一起落盘，生长脚本必须原样继承，"
    "漏搬任何一个都会让生长后的模型落到不同的前向路径。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetHFConfig.key_dim"] = E(
    "计算单个注意力头的 key 维度，等于 d_model 整除 num_heads。RetNet 的 retention "
    "分数是 q 与 k 的逐头内积，key_dim 决定每个头的表示宽度；它同时被旋转位置编码"
    "（xPos）用来确定频率表的长度，因此这个属性必须在 d_model 与 num_heads 都确定之后"
    "才可读，配置里不允许单独覆写。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::rotate_every_two"] = E(
    "把最后一维的相邻两个分量 (x0, x1) 旋转成 (-x1, x0)，是实现复数乘法 e^{iθ}·(a+bi) "
    "的实数等价写法。RetNet 用旋转位置编码把绝对位置信息注入 q 与 k，而旋转角由位置与"
    "频率表共同决定。这个函数只做形状变换，不引入任何可学习参数，因此不会出现在优化器状态里。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::theta_shift"] = E(
    "对张量施加旋转位置编码：先按 rotate_every_two 旋转，再与 cos 相乘，并加上原值与 sin "
    "的乘积，得到 (x·cos + rotate(x)·sin) 的等价复数乘法结果。q 与 k 都经过它，"
    "于是两者的内积只依赖相对位置差，这正是 RetNet 相对位置编码的来源。频率表在初始化时"
    "以 buffer 形式注册，不参与训练。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention.__init__"] = E(
    "构建一个多尺度 retention 层：按头计算 key_dim 与 value_head_dim，注册每头独立的"
    "衰减率 gamma（use_gla 时是可学习参数，初值严格取 1-2^(-5-h)），注册 xPos 的频率表，"
    "并建立 q/k/v/g/out 五个无 bias 线性层与一个 GroupNorm。gamma 的初值公式决定了"
    "22 个头从短记忆到长记忆的分布，任何把它整体置零或置一的改动都会破坏多尺度设计。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention._init_weights"] = E(
    "本层权重的初始化：q/k/v/g 投影用 gain=2^-2.5 的 Xavier，out 投影用 gain=1 的 Xavier，"
    "bias 置零，而 gamma 被显式跳过。注释里记录了一次真实事故：早期版本用“一维参数一律置零”"
    "兜底，结果唯一的 1 维参数就是 gamma，门控被置零后 log(0) 产生 NaN。"
    "这条注释与本项目 E02 的教训方向一致，但生长脚本绕过了这个初始化函数，"
    "所以它保护不了生长路径。",
    ["E02"],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention._rotations"] = E(
    "按给定长度与起始偏移生成 sin/cos 旋转表：位置用 arange(offset, offset+length)，"
    "与频率表做外积得到相位，再分别取 sin 与 cos。偏移量让自回归解码在逐 token 生成时"
    "能继续使用正确的位置编号，而不是每次都从 0 开始；这也是 prefill 与 decode 能衔接的"
    "前提之一。整个计算在 fp32 下完成后再转回输入精度。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention._project_and_rotate"] = E(
    "把输入 x 投影成 q/k/v/g 四个张量，重塑为 [B,H,L,dim]，对 q 与 k 施加旋转位置编码，"
    "并按 key_dim 的平方根倒数缩放 k。v 与 g 不做位置旋转，因为它们不参与位置相关的内积。"
    "返回的四个张量分别进入 retention 计算与输出门控；缩放写在 k 上而不是分数上，"
    "是为了让 chunkwise 与 recurrent 两条路径共享同一份缩放。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention._head_group_norm"] = E(
    "把 [B,H,L,vhd] 转置并展平成 [B*L, H*vhd]，交给 GroupNorm 按头分组归一化，再还原形状。"
    "注释记录了一次实测：把归一化挪到 permute 之前会**反而慢 9.2%**（190 vs 174 ms/微步）——"
    "参考布局把 permute 放在归一化之前，那次拷贝被逐元素归一化核吸收；挪到之后它成为 out_proj "
    "这个外部 cuBLAS 调用的输入，再也融不进任何东西。等价性由测试保证（fp32 相对差 1.6e-7）。"
    "这个布局是性能敏感的，改动前必须先跑基准。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention._decay_mask"] = E(
    "构造 [H,L,L] 的衰减掩码：delta 是位置差，causal 保证只看过去，"
    "gamma 的 delta 次幂给出指数衰减，非因果位置显式置 0。当前实现不再对掩码做任何"
    "按行归一化——2026-10-20 删掉了原来的 mask/sqrt(sum(mask))，因为那一步只存在于"
    "parallel 路径，会让训练（chunkwise）与 prefill（parallel）算的不是同一个函数（E16）。"
    "掩码只依赖 gamma 与位置差，本身不含可学习参数。",
    ["E16"],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention._parallel_retention"] = E(
    "并行路径：先算 q 与 k 转置的内积，再乘衰减掩码，最后乘 v。2026-10-20 删掉了"
    "原来那句 scores 除以按行绝对值和的归一化——它是 parallel 独有的第二次归一化，"
    "chunkwise 与 recurrent 都没有，制造了训练与推理之间的不等价（E16）。"
    "现在这条路径与 chunkwise 只差分块方式、不差数学；但 prefill 走 parallel、"
    "解码走 recurrent 的拼接仍需要一条数值回归测试来锁住。",
    ["E16"],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention._parallel_final_state"] = E(
    "从一段完整序列一次性算出结束时的递归状态：对每个位置按 gamma 的(tgt_len-1-i)次幂"
    "加权后做 einsum，得到 [B,H,kd,vd]；若已有历史状态，再按 gamma^tgt_len 衰减后相加。"
    "它只在 return_state 为真时被调用，供 prefill 之后切换到逐 token 解码使用。"
    "注意它用的是整段长度作为衰减指数，与 chunkwise 的逐块衰减在数学上一致。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention._recurrent_retention"] = E(
    "逐 token 的递归路径：状态按 gamma 衰减后加上当前 token 的 k⊗v，再用 q 与状态求内积。"
    "它是 O(1) 推理的实现基础，也是三条路径里最直观的一条。E21 增加了 tgt_len==1 的快路径："
    "解码时每次只有一步，直接算，省掉 Python 循环与 list/stack 的物化，数学与循环版逐位相同；"
    "长序列仍走循环，只用于罕见场景。",
    ["E21"],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention._chunkwise_retention"] = E(
    "训练主路径：把序列切成 chunk，块内用并行衰减掩码，块间用递归状态传递。"
    "状态更新为 S=gamma^C·S+k^T·(v·gamma^{C-1-j})，跨块贡献为 (q·gamma^{j+1})·S。"
    "它要求 tgt_len 能被 chunk_size 整除，否则直接抛错。E16 之外，这条路径本身是正确的，"
    "但它依赖 gamma 的长度：gamma^128 在实测中位数下约等于 1e-32，跨块信息等于被清零。",
    ["E16"],
)

EXPL["retnet_hf/modeling_retnet.py::MultiScaleRetention.forward"] = E(
    "按 mode 分派到 parallel、recurrent 或 chunkwise，再把输出交给 GroupNorm、"
    "与 silu(g) 做逐元素门控、过 dropout 与 out_proj。三条路径共享同一组权重；"
    "2026-10-20 删除 parallel 独有的两次归一化之后，它们在数学上一致，"
    "差别只剩分块与循环的实现方式（E16）。训练走 chunkwise，生成 prefill 走 parallel，"
    "解码走 recurrent。",
    ["E16", "E21"],
)

EXPL["retnet_hf/modeling_retnet.py::GrowthLayerNorm.__init__"] = E(
    "两段式 LayerNorm 的构造：旧维度用一个普通的 LayerNorm(old_dim)，新维度用两个可学习向量"
    " gamma_new 与 beta_new 做逐通道仿射。默认 gamma_new=1.0、beta_new=0，这是“新维度可以学”"
    "的默认值；E02 的问题恰恰是生长脚本把它覆写成 0，于是新维度被乘法门关死，"
    "梯度也一起被关死。默认值是对的，错在覆盖它的那一行。",
    ["E02", "E05"],
)

EXPL["retnet_hf/modeling_retnet.py::GrowthLayerNorm.forward"] = E(
    "前向分两段：前 old_dim 维走 ln_old 得到归一化结果，后 new_dim-old_dim 维只做"
    " x*gamma_new+beta_new 的仿射（不做归一化），最后拼接。新维度没有归一化，"
    "因此 gamma_new 就是唯一的缩放门；当它为 0 且输入也为 0 时，这一段恒为 0，"
    "反向链路上每一环都含 0 因子。这就是 E05 描述的恒零结构，"
    "修复不改这里的数学，只保证 gamma_new 初值非零、并且训练中不被衰减。",
    ["E05", "E02"],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetBlock._norm"] = E(
    "按配置选择归一化实现：use_growth_ln 为真时返回 GrowthLayerNorm 并传入 growth_old_dim，"
    "否则返回普通 LayerNorm。它是生长模型与普通模型共用的唯一入口，"
    "意味着 growth_old_dim 一旦配错，整层的归一化切点都会错位。生长脚本必须继承源模型的"
    "原始切点（而不是当前宽度），原因就在这里。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetBlock.__init__"] = E(
    "构建一个 RetNet 块：一个 MultiScaleRetention、两个归一化（注意顺序是 pre-LN）、"
    "一个 FFN。FFN 在 use_swiglu 为真时用 gate/up/down 三个投影实现 SwiGLU，"
    "否则用两层线性加 GELU。checkpoint_layers 记录本层是否做梯度检查点，"
    "默认 -1 表示跟随全局开关。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetBlock._block_forward"] = E(
    "块的实际计算：先对 x 做 norm1 再进 retention，得到 y 后加残差；"
    "再对 y 做 norm2 进 FFN，加残差与 dropout。这是标准的 pre-LN 残差结构。"
    "两条残差都直接相加、没有缩放系数，因此任何往残差里注入非零新维度的操作"
    "都会立刻改变输出——这也是宽度生长必须靠门控或置零来保函数的原因。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetBlock.forward"] = E(
    "在 _block_forward 外面包一层梯度检查点：仅当训练模式、且本层编号**小于**检查点阈值"
    "（checkpoint_layers）时才启用；层号 >= 阈值时关闭。用 use_reentrant=False 的非重入版本，"
    "避免与 torch.compile 的图捕获冲突。检查点只影响显存与重算，不改变数学；"
    "关闭时直接调用 _block_forward。",
    ["E22"],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetForCausalLM.__init__"] = E(
    "组装完整模型：词嵌入、若干 RetNetBlock、末期归一化、输出头。tie_weights 为真时"
    "把 lm_head.weight 指向 embedding.weight，二者共享同一份参数，因此 embedding 的新列"
    "同时也是分类器的新列——生长时必须同时考虑输入与输出两条语义。"
    "构造完成后调用 _init_weights_local 做初始化。",
    ["E03"],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetForCausalLM._init_weights_local"] = E(
    "模型级初始化：embedding 用 N(0,0.02)，FFN 用 Xavier，retention 层调用自己的 _init_weights，"
    "未共享权重时输出头也用 Xavier。它只在“从零构造”时生效，from_pretrained 会用文件里的数值覆盖，"
    "生长脚本也是先构造再覆盖 state_dict，所以这个函数保护不了生长路径——"
    "这正是 E01–E03 能在它眼皮底下发生的原因。",
    ["E01", "E02", "E03"],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetForCausalLM.tie_weights"] = E(
    "在 from_pretrained 之后重新建立权重共享关系。HF 的加载流程会按 state_dict 重建参数，"
    "可能把原本共享的两个张量拆成两份；本方法把 lm_head.weight 重新指向 embedding.weight，"
    "保证训练与推理由同一份参数驱动。若忘记调用，模型会多出一份输出头，"
    "参数量与显存都对不上，且训练只更新其中一份。",
    [],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetForCausalLM.set_checkpoint_layers"] = E(
    "设置“只对前 n 层做梯度检查点”的策略：把 n 写进 config，并同步到每个 block。"
    "必须在 HF 打开全局梯度检查点之后调用，否则会被全局开关覆盖。"
    "分层的目的是在固定显存下换更大的物理 batch，属于性能旋钮，不改变模型数学。",
    ["E22"],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetForCausalLM._forward_hidden"] = E(
    "模型主干：embedding 乘 embed_scale、过 dropout，逐层调用 block，最后过 final_norm。"
    "return_hidden 为真时提前返回隐藏态，交给融合交叉熵，避免物化 [B,L,V]；"
    "否则过 lm_head 得到 logits。states 用于递归/分块推理时传递每层状态，"
    "长度必须与层数一致，否则直接报错。",
    ["E22"],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetForCausalLM.forward"] = E(
    "Trainer 兼容的前向：解析 mode 与 chunk_size（None 时回落到 config），"
    "按是否融合交叉熵选择损失路径；非融合路径交给 _chunked_weighted_ce 分块计算（E22），"
    "不再对整段 logits 做 .float()。E14/E15 的验证集问题最终也体现在这里的 loss 上："
    "只要 val_dir 与训练目录相交，这个数就不可信。",
    ["E22", "E14", "E15"],
)

EXPL["retnet_hf/modeling_retnet.py::RetNetForCausalLM.generate"] = E(
    "自回归生成：长度大于 1 时用 parallel 做 prefill 并取最后一位 logits，之后每个 token "
    "用 recurrent 走一步并递增 offset。采样支持温度、top-k、top-p、重复惩罚与 n-gram 禁选，"
    "并可在每个 token 上回调。2026-10-20 之前 parallel 比训练路径多两次归一化，"
    "prefill 与训练分布不一致（E16）；归一化已删，但 prefill 与 recurrent 的拼接"
    "仍应有一条数值回归测试锁住。",
    ["E16", "E21"],
)

EXPL["retnet_hf/retention_fast.py::_index_tables"] = E(
    "缓存只依赖 chunk_size 的三张索引表：causal 掩码、clamp 后的指数、以及位置向量。"
    "原实现每次调用都重算 arange/sub/ge/clamp，是四次纯开销的内核发射；"
    "而本算子的成本几乎全是发射延迟，所以缓存它的收益是实打实的。"
    "缓存键包含设备字符串，避免跨设备复用导致错误。",
    [],
)

EXPL["retnet_hf/retention_fast.py::_decay_tables"] = E(
    "由每头 gamma 生成四张衰减表：块内掩码、跨块衰减、值衰减与整块衰减。"
    "关键实现细节是全程走对数空间：exp(logγ·delta)，因为因式分解的 γ^i·γ^-j 在 γ<1、"
    "指数为负时会溢出 fp32 变 Inf，而这个项目曾经真的出现过整层 NaN。"
    "gamma 先夹一个极小正数，保证 γ=0 时 0^0=1、0^k=0，与参考实现语义一致。",
    ["E16"],
)

EXPL["retnet_hf/retention_fast.py::chunkwise_batched"] = E(
    "批量 chunkwise 内核：把序列按 chunk 切块，用一次批量 matmul 算出所有块的状态贡献，"
    "再用闭式下三角矩阵 T 做 exclusive scan 得到每块之前的状态，最后分别算块内与跨块贡献。"
    "总共 5 次批量 matmul（U、S_all、qk、inner、cross），与 chunk 数无关；"
    "闭式解避免了 nc 次串行循环，这是它比参考实现快一个数量级的原因。"
    "数学上与 _chunkwise_retention 是**同一个函数、浮点舍入级等价，不是逐位相等**"
    "（实测 max|Δ| 1.9e-6 @γ=0.5、1.5e-5 @γ=0.99，tests/test_fast_chunkwise.py）；"
    "它只读 gamma，不做任何归一化。",
    ["E16"],
)

EXPL["retnet_hf/data.py::PackedChatDataset.__init__"] = E(
    "加载已打包的二维 ids 与 wcode，并断言形状为 (rows, seq_len+1) 且两者一致。"
    "行宽是 seq_len+1 而不是 seq_len，因为同一行要同时提供 input 与 shift 后的 label。"
    "断言在这里非常关键：v22 的第一次失败就是验证集宽度 2049 与期望的 16385 不匹配，"
    "当时只抛了一个 AssertionError，根因却是数据包用错了。",
    ["E14"],
)

EXPL["retnet_hf/data.py::PackedChatDataset.__len__"] = E(
    "返回行数。HF 的 RandomSampler 在每个 epoch 开始时读取这个值来决定采样范围，"
    "因此训练中途切换数据集时，旧 sampler 手上可能还攥着上一段的索引。"
    "这也是多阶段数据集必须在 __getitem__ 里做取模保护的原因。",
    ["E19"],
)

EXPL["retnet_hf/data.py::PackedChatDataset.__getitem__"] = E(
    "取一行并切成三段：input_ids 是 row[:-1]，labels 是 row[1:]，wcodes 取 codes 的对应位移。"
    "三者都转成 int64 以适配 embedding 与交叉熵。注意 wcode 也做了错位，"
    "因为权重码描述的是“预测这个 label 时用多大权重”，必须与 label 对齐而不是与输入对齐。",
    [],
)

EXPL["retnet_hf/data.py::ChatDataCollator.__call__"] = E(
    "把一个 batch 的字典堆叠成张量，供 Trainer 直接使用。它不做 padding，"
    "因为所有行都已经是固定宽度，拼接即可。保持无 padding 意味着不需要 attention mask，"
    "这也正是 RetNet 的 chunkwise 前向不必处理变长序列的原因之一。",
    [],
)

EXPL["retnet_hf/data.py::_PackedStage.__init__"] = E(
    "打开一段 packed 语料：mmap 读取 train.ids.npy 与 train.wcode.npy，"
    "断言 ids 是二维且宽度为 seq_len+1，并断言 ids 与 codes 形状一致。"
    "weight 参数用于给整段数据一个固定权重（此时不读 wcode 文件）。"
    "断言失败会直接抛错，这是 v22 第二次失败的现场。",
    ["E14"],
)

EXPL["retnet_hf/data.py::_PackedStage.__len__"] = E(
    "返回本阶段的行数，供多阶段数据集转发与 sampler 使用。阶段切换后这个值会变，"
    "而 HF 的 RandomSampler 只在每个 epoch 开始时缓存一次长度，因此旧 sampler 手上的索引"
    "可能落到新阶段的范围之外；外层的 __getitem__ 必须对行号取模，才不会在切段瞬间抛 IndexError。",
    ["E19"],
)

EXPL["retnet_hf/data.py::_PackedStage.files"] = E(
    "返回本阶段占用的磁盘文件列表，供 GC 使用。只返回自己登记的 ids 与 wcode 两个路径，"
    "不做目录级删除，这是为了让 GC 的破坏面尽可能小。但“最小破坏面”不等于“安全”："
    "如果这是唯一副本，删了仍然不可恢复（见 E13）。",
    ["E13"],
)

EXPL["retnet_hf/data.py::_PackedStage.row"] = E(
    "取第 idx 行，返回 input、label 与权重码。packed 数据的行是预先按 seq_len+1 切好的，"
    "切点由建包脚本决定，因此这里的边界质量完全取决于建包时有没有按对话/文档切分。"
    "v13 的展平重切就是把这一步的边界破坏掉了。",
    ["E12"],
)

EXPL["retnet_hf/data.py::_FlatStage.__init__"] = E(
    "打开一段 flat 语料：记录目录、seq_len、固定权重与排除前缀，调用 _scan 得到分片列表、"
    "每片起始行号、总行数，以及扫描时刻的分片签名（文件名/mtime_ns/大小）。"
    "require_eos 默认 True：打开时就调用 _check_boundaries 抽查是否有 eos，"
    "没有则拒绝——这是 E12 的守卫。签名在这里快照保存，供 refresh 比较，"
    "不能在比较时重新 stat，否则新旧两边永远相同（E19）。",
    ["E12", "E19"],
)

EXPL["retnet_hf/data.py::_FlatStage._scan"] = E(
    "扫描目录下所有 npy，跳过文件名含 val 的分片（验证集绝不能进训练），"
    "按排除前缀过滤，逐个 mmap 并计算可切出的行数，累加得到全局行号起点。"
    "打不开的分片会被跳过而不是崩溃，因为数据可能正在从外部传输进来，"
    "下一轮 refresh 会把它收进来——这是“边传边训”的基础。",
    ["E12"],
)

EXPL["retnet_hf/data.py::_FlatStage.shard_count"] = E(
    "返回当前已扫描到的分片数，供计划回调的 min_shards 守卫判断阶段是否就绪。"
    "上一轮曾把 min_shards 配成 10 而实际只产出 1 个分片，导致阶段永远不就绪、"
    "训练停在上一个阶段每 200 步重试。这个属性就是那个守卫的输入。",
    ["E13"],
)

EXPL["retnet_hf/data.py::_FlatStage.refresh"] = E(
    "重新扫描目录，分片集合或任一分片发生变化就接过并返回 True（E19）。"
    "旧实现只比 len(shards) 与总行数，同尺寸替换检测不到，训练继续读旧 mmap。"
    "新实现比较扫描时快照下来的签名 self._signature 与本次 _scan 返回的签名，"
    "三元组是 (文件名, mtime_ns, 大小)，任何一处变化都会触发重扫。"
    "注意签名必须在扫描时保存：第一版修复在比较时对旧列表重新 stat，"
    "新旧两边读的都是文件当前状态，永远相等，测试当场抓到了这个错误。",
    ["E19"],
)

EXPL["retnet_hf/data.py::_FlatStage.files"] = E(
    "返回本阶段所有分片的路径，供 GC 删除。flat 模式下这些分片就是训练数据的唯一副本，"
    "删掉不可逆；删除前的“可重建”证明由 gc_stage 负责（目录里必须有 meta.json，并能给出 "
    "rebuild_cmd / source_ids / 旧产线的 sources+text_fields）。这个函数只负责列文件，"
    "不做任何安全判断，也不检查源数据是否仍在——那不是它能看到的信息。",
    ["E13"],
)

EXPL["retnet_hf/data.py::_FlatStage.close"] = E(
    "在删除或替换文件之前显式关闭 mmap 句柄。Windows 上 mmap 会占用文件句柄，"
    "不先关就 unlink/覆盖会报 WinError 32 或 OSError 22；Linux 上能删，但显式关闭更干净。"
    "关闭后把分片列表、行数与签名一起清零，避免后续再按旧索引取数据、"
    "也避免 refresh 拿一份已经不存在的签名做比较（E19）。",
    ["E13", "E19"],
)

EXPL["retnet_hf/data.py::_FlatStage.__len__"] = E(
    "返回可切出的总行数，等于所有分片长度除以 (seq_len+1) 之和，分片末尾不足一行的余数被丢弃。"
    "丢弃余数本身是安全的，因为余数凑不满一个训练样本；但它也意味着每个分片末尾的少量 token "
    "永远不会被训练到，在分片很多、每片都很小的时候这部分浪费需要单独统计。",
    ["E12"],
)

EXPL["retnet_hf/data.py::_FlatStage.row"] = E(
    "把全局行号折成（分片，片内偏移）后切出一段固定宽度的 token：input 是 chunk[:-1]，"
    "label 是 chunk[1:]，权重码是整段常量。E12 指出的核心问题就在这里："
    "切点只按行宽对齐，与文档边界无关；权重码恒为 1，所有 token 等权。"
    "模型因此学到的是任意 2048 token 的连续切片，跨文档的噪声让它学会忽略远距离。",
    ["E12"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.__init__"] = E(
    "多阶段数据集的构造：保存 seq_len、清空 spec 与阶段缓存、把活动阶段设为 0，"
    "然后调用 update_stages 建立阶段列表。阶段是惰性打开的，没轮到之前不会 mmap，"
    "因此可以在训练中途才上传下一段数据。",
    ["E12"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.update_stages"] = E(
    "替换阶段列表（计划热更新时调用），并按 (name, data_dir) 复用已经打开的阶段对象，"
    "避免热更新时重复 mmap。活动阶段索引会被夹到新列表范围内。"
    "它只改数据源，不动 DataLoader 与优化器，这是“不重启换语料”的实现位置。",
    ["E19"],
)

EXPL["retnet_hf/data.py::StagePackedDataset._stage"] = E(
    "按需打开第 i 段：根据 mode 选择 packed 或 flat（auto 时看目录里有没有 train.ids.npy），"
    "构造对应的阶段对象并缓存，同时打印阶段名与行数。任何构造失败都会在这里响亮报错，"
    "而不是等到采样时才发现。",
    ["E12"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.set_stage"] = E(
    "切换活动阶段。先调用 _stage 提前打开目标阶段（失败就在这里抛错），"
    "再更新活动索引。提前打开是关键：如果等到第一次取数据才打开，"
    "失败会发生在训练循环深处，现场只剩一个语焉不详的 IndexError。",
    ["E12"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.stage_ready"] = E(
    "判断第 i 段现在能不能用，不抛异常，只返回 (bool, 原因)，供计划回调决定“等还是切”。"
    "它检查索引范围、能否打开、以及分片数是否达到 min_shards。"
    "这就是“数据没传完时继续用当前阶段、每 N 步重试”的守卫逻辑。",
    ["E13"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.refresh_stage"] = E(
    "重新扫描第 i 段目录，接过新到的分片；没有变化则返回 False。"
    "它只对实现了 refresh 的阶段（flat）生效，packed 阶段固定不变。"
    "与 E19 相同，refresh 本身只比较数量，同尺寸替换检测不到。",
    ["E19"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.gc_stage"] = E(
    "删除第 i 段的数据文件并返回释放字节数。四条安全约束全部在**函数体里**执行，"
    "不是文档承诺：只删本阶段登记的文件；目录被别的阶段引用时跳过；"
    "目录被 protect() 登记为验证集时跳过；目录里必须有 meta.json 且能证明可重建。"
    "可重建的证明认三种：rebuild_cmd（完整 argv）、source_ids，或旧产线的 "
    "sources+text_fields。没有“两小时内可重建”的时间约束——那只是以前生成器里的"
    "文字，代码从来没有能力验证它。",
    ["E13"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.active_index"] = E(
    "返回当前活动阶段的索引。计划回调、日志与诊断都靠它确认“现在到底在训哪一段”，"
    "它本身只读不写；真正的切换必须走 set_stage，以保证“提前打开目标阶段”与“更新索引”"
    "这两步是原子发生的，不会出现索引已经改了、数据却还打不开的半吊子状态。",
    ["E12"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.files"] = E(
    "返回当前活动阶段占用的文件列表，仅供诊断与日志；真正的删除必须走 gc_stage。"
    "把“查看”和“删除”拆成两个入口，是为了避免某处顺手调用 files() 之后直接 unlink，"
    "绕过 gc_stage 里的同目录检查与验证集检查——上一轮删光语料，缺的正是这道统一入口。",
    ["E13"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.active_name"] = E(
    "返回当前活动阶段的名字，用于日志、计划回调的打印以及 checkpoint 的元信息。"
    "按名字而不是按索引记录，是因为计划文件允许在训练中途插入新阶段——一旦插入，"
    "同一段语料的索引就会变，只有名字还能与历史日志对上，事后回溯时才不至于张冠李戴。",
    [],
)

EXPL["retnet_hf/data.py::StagePackedDataset.__len__"] = E(
    "返回当前活动阶段的行数，转发给 _stage(len) 的结果。"
    "注意这个值会随阶段切换而变化，HF 的 sampler 在 epoch 开始时缓存它，"
    "因此切段瞬间的索引可能越界，__getitem__ 必须取模。",
    ["E19"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.__getitem__"] = E(
    "取当前阶段的一行，并对行号取模以避免切段后旧索引越界。返回 input_ids、labels、wcodes 三个张量。"
    "取模是一种静默兜底：它让训练不崩，但也掩盖了“索引其实已经失效”这件事；"
    "阶段为空时抛 IndexError，这是唯一会被看见的异常。",
    ["E19"],
)

EXPL["retnet_hf/fused_ce.py::_fused_forward"] = E(
    "融合交叉熵的前向：按 LOGIT_CHUNK 分块计算 logits，只保存每行的 logsumexp 与目标 token 的 logit，"
    "不保存 [N,V] 的完整矩阵。这样峰值显存从 N×V×4 字节降到 N×4。"
    "函数被 torch._dynamo.disable 标记，因为分块动态切片会让 Dynamo 图断裂并退回 eager。",
    ["E22"],
)

EXPL["retnet_hf/fused_ce.py::_fused_backward"] = E(
    "融合交叉熵的反向：按同样的分块重算 logits 与 softmax，减去 onehot 后乘权重码与上游梯度，"
    "分别累加出对隐状态与输出头权重的梯度。代价是反向多一遍 LM head 的 GEMM，"
    "换来的是完全不需要 [N,V] 常驻显存。权重码语义与 modeling 里的损失完全一致。",
    ["E22"],
)

EXPL["retnet_hf/fused_ce.py::FusedLinearCrossEntropy.forward"] = E(
    "自定义 autograd 函数的前向：调用 _fused_forward 得到 lse 与目标 logit，"
    "按权重码求加权平均得到 loss，并把参与反向的张量存进 ctx。"
    "它只返回标量 loss，不返回 logits，因此调用方不能再拿 logits 做别的计算。",
    ["E22"],
)

EXPL["retnet_hf/fused_ce.py::FusedLinearCrossEntropy.backward"] = E(
    "自定义 autograd 函数的反向：调用 _fused_backward 计算对隐状态与权重的梯度，"
    "并按输入顺序返回。target 与 w 不是需要梯度的张量，因此返回 None。"
    "这一步的正确性完全依赖 forward 里保存的 wsum 与权重码，改语义时必须两边同时改。",
    ["E22"],
)

EXPL["retnet_hf/fused_ce.py::fused_linear_cross_entropy"] = E(
    "对外的函数式入口，内部调用 FusedLinearCrossEntropy.apply。"
    "它把自定义 autograd 函数包装成普通函数，调用方不需要知道 ctx 的存在。"
    "hidden/weight/target/w 的形状约定是 [N,D]、[V,D]、[N]、[N]。",
    ["E22"],
)

EXPL["run_task.py::load_config"] = E(
    "读任务 JSON，按是否带 plan 选择必填字段，校验 plan 文件存在，再用 DEFAULTS 补齐缺省值。"
    "DEFAULTS 集中管理的是**launcher 路径**的默认值，不是“全仓库唯一默认值”："
    "`train.py` 作为底层入口仍保留自己的一套（mode=parallel、precision=bf16、batch=10、"
    "max_steps=45000、warmup=1000、grad_checkpoint=False、compile=True），与 launcher "
    "默认值（chunkwise / fp32 / 6 / 65000 / 500 / True / False）不同。这种差异是**入口定位**"
    "造成的，不是缺陷；但前提是 launcher 每次都把字段显式传下去，否则“配置写了”就会"
    "变成“底层默认值生效”。E17/E20 记录的正是这个前提被破坏后的两类事故。"
    "需要跨入口一致的两个字段已经抽到 `retnet_hf/defaults.py`："
    "save_total_limit=4 与 no_decay_1d=True，两处 import 同一个常量。",
    ["E17", "E20"],
)

EXPL["run_task.py::build_cmd"] = E(
    "把配置翻译成 train.py 的命令行。带 plan 时走计划模式，只传模型/输出/计划/批大小/精度等，"
    "把数据源与总步数交给计划文件。2026-10-20 补上了 plan 分支此前漏传的 eval_batch_mult"
    "（E20）：它一直回落到 train.py 的默认 4，评测显存因此偏高。save_steps 与 eval_steps "
    "由计划回调在运行时写入 args，命令行的默认值只在回调生效之前短暂存在。",
    ["E20"],
)

EXPL["run_task.py::build_cmd_full"] = E(
    "在 build_cmd 之后追加条件开关：梯度检查点、关闭编译、编译模式、梯度累积、优化器、"
    "no_decay_1d、DDP 未使用参数、续训，最后在多卡时用 torch.distributed.run 包一层。"
    "E06 的修复让 no_decay_1d 默认开启，这里仍然只在配置为真时追加——"
    "所以默认值必须在 train.py 里也保持开启，两边不能只改一边。",
    ["E06", "E20"],
)

EXPL["run_task.py::last_loss"] = E(
    "从最新的 checkpoint 目录里读 trainer_state.json，取最近 n 条带 loss 或 eval_loss 的记录。"
    "它用于 --status 与 --monitor，不参与训练。注意它只读最新 checkpoint，"
    "如果最佳 checkpoint 被轮换删除，这个接口也看不到它的历史。",
    ["E15"],
)

EXPL["run_task.py::main"] = E(
    "任务入口：解析 --config/--status/--dry/--monitor/--ctrl，检查粘性停止旗标，"
    "加载配置并打印摘要，设置 fast_chunkwise 与 compile 的环境变量，"
    "然后按模式分派：dry 打印命令、status 打印最近 loss、monitor 轮询、否则循环启动训练子进程。"
    "自动重启带次数上限，并在检测到停止旗标时不再重启；子进程退出码会原样传播，"
    "不会再把“崩了”记成“训完了”。",
    ["E17", "E20"],
)

EXPL["scripts/grow_500m.py::main"] = E(
    "宽度生长主流程：校验源模型形状、继承原始归一化切点与非权重配置、构造新模型、"
    "按 fill/grow_norm 完成权重映射、校验参数量、做函数保真与可训性两道校验、最后落盘。"
    "E01–E04 全部发生在这个函数里；修复后新行随机、新列置零、中间层门控为 1、"
    "final_norm 门控为 0，并新增了“跑 5 步反向、逐参数检查梯度”的校验。",
    ["E01", "E02", "E03", "E04"],
)

EXPL["scripts/grow_500m.py::main.fill"] = E(
    "把旧权重放进新矩阵左上角，并决定新增行/列的初值。新列必须置零：否则它会立刻改变"
    "旧维度的输出，函数不再保真，而且这种扰动无法用任何门控挡住。新行必须随机："
    "全零会让新单元输出恒零，读它的下游列拿到的输入是零，梯度也就恒零——"
    "这正是上一轮 3.38 亿个死参数的来源。随机新行让新单元先有输出，"
    "下游列在第 1 步拿到梯度，新列变非零后新行在第 2 步拿到梯度，形成自愈级联。",
    ["E01", "E04"],
)

EXPL["scripts/grow_500m.py::main.grow_norm"] = E(
    "重设两段式 norm 的新段门控。中间层必须为 1：新维度要能流进 q/k/v 的新列，"
    "否则新列输入恒零、梯度恒零。final_norm 必须为 0：它是新维度通往 logits 的唯一出口，"
    "置零才能保证生长后 logits 不变；而它的梯度来自上游 logits 乘以新维度，"
    "上游经 tied lm_head 的新列（随机非零）回传，所以第一步就能学到非零。"
    "旧段 ln_old 的权重与偏置原样继承，保证旧维度的归一化统计完全不变。",
    ["E02", "E05"],
)


# Function -> paper id (Papers/ prefix). Only functions that genuinely rely on a
# paper get a citation; no "attach a list to every paragraph" formalism.
CITE: dict[str, list[str]] = {
    "retnet_hf/modeling_retnet.py::RetNetHFConfig.__init__": ["01"],
    "retnet_hf/modeling_retnet.py::RetNetHFConfig.key_dim": ["01"],
    "retnet_hf/modeling_retnet.py::rotate_every_two": ["02"],
    "retnet_hf/modeling_retnet.py::theta_shift": ["02"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention.__init__": ["01", "03"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention._init_weights": ["01", "03"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention._rotations": ["02"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention._project_and_rotate": ["01", "02"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention._head_group_norm": ["05"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention._decay_mask": ["01"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention._parallel_retention": ["01", "04"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention._parallel_final_state": ["01", "04"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention._recurrent_retention": ["01", "04"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention._chunkwise_retention": ["01", "04"],
    "retnet_hf/modeling_retnet.py::MultiScaleRetention.forward": ["01"],
    "retnet_hf/modeling_retnet.py::GrowthLayerNorm.__init__": ["07", "08", "09", "10", "11", "13"],
    "retnet_hf/modeling_retnet.py::GrowthLayerNorm.forward": ["08", "13"],
    "retnet_hf/modeling_retnet.py::RetNetBlock._norm": ["07", "09"],
    "retnet_hf/modeling_retnet.py::RetNetBlock.__init__": ["01", "06"],
    "retnet_hf/modeling_retnet.py::RetNetBlock._block_forward": ["01"],
    "retnet_hf/modeling_retnet.py::RetNetForCausalLM.__init__": ["01"],
    "retnet_hf/modeling_retnet.py::RetNetForCausalLM._init_weights_local": ["07", "09"],
    "retnet_hf/modeling_retnet.py::RetNetForCausalLM._forward_hidden": ["01"],
    "retnet_hf/modeling_retnet.py::RetNetForCausalLM.generate": ["01", "04"],
    "retnet_hf/retention_fast.py::_index_tables": ["01", "04"],
    "retnet_hf/retention_fast.py::_decay_tables": ["01"],
    "retnet_hf/retention_fast.py::chunkwise_batched": ["01", "04"],
    "retnet_hf/train.py::CompileSafeTrainer.get_decay_parameter_names": ["03"],
    "retnet_hf/train.py::_preflight": ["01", "03"],
    "retnet_hf/train.py::live_param_ratio": ["10"],
    "scripts/grow_500m.py::main": ["07", "08", "09", "10", "11", "12", "13", "14"],
    "scripts/grow_500m.py::main.fill": ["07", "08", "09"],
    "scripts/grow_500m.py::main.grow_norm": ["08", "09", "13"],
}

# --------------------------------------------------------------------- 追加解释
# 2026-10-24 修复时新增的函数（E07 的梯度覆盖、E08 的基线、E12 的 spec 构造、
# E22 的字节预算、E13 的可重建证明、P2-1 的评测上限）。旧提交里这些要么不存在，
# 要么只是一句注释；现在它们是真代码，因此必须逐条进入 01。

EXPL["retnet_hf/train.py::read_live_baseline"] = E(
    "读取历史 run 写在 live_params.json 里的活参数比例（E08）。按传入目录顺序找，"
    "第一个可解析且落在 (0,1] 的值胜出；优先读 ratio_min（历史最低），"
    "这样每轮悄悄掉 1%、又被重设基线抹掉的缓慢侵蚀不会漏过守卫。文件不存在或损坏都返回 None。"
    "它的用途是让“从 33% 活参数的剪枝检查点继续训练”不必每次都手动加开关："
    "只要上一次成功开训记录过基线，比例没有明显变差就自动放行。",
    ["E08"],
)

EXPL["retnet_hf/train.py::write_live_baseline"] = E(
    "把本次开训时的活参数数、总数、本次比例、历史最低比例（ratio_min）与时间写进 output_dir/live_params.json（E08）。"
    "只在 rank 0、训练成功结束后调用；写入失败只告警，不会让已经跑完的训练作废。"
    "下一次续训从 model_dir 或 output_dir 读到它，就能区分“已知剪枝状态”和"
    "“这一轮又把参数弄死了”——后者比例会低于基线，守卫仍然拒绝。",
    ["E08"],
)

EXPL["retnet_hf/train.py::check_live_ratio"] = E(
    "E08 的判定核心，纯函数，四个边界都可直接测试：比例达标放行；显式 --allow_dead_params "
    "放行；有历史基线且比例不低于 baseline-tol 放行；其余拒绝，并在错误信息里同时给出"
    "“新模型先修生长”和“剪枝检查点加开关”两条出路。旧实现只有一条 ratio<0.95 的硬拒绝，"
    "把 docs/02 的 rescue 计划第一步就堵死（它的起点是 33% 活参数）。",
    ["E08"],
)

EXPL["retnet_hf/train.py::classify_dead_grad_params"] = E(
    "把“从没收到梯度”和“梯度恒为零”归为同一类死参数：seen_max 里没有这个键，"
    "或记录的最大绝对值是 0，就判死。返回顺序与传入参数名顺序一致，"
    "这样日志里先看到的是 embedding/前几层，便于定位。纯函数，测试直接喂字典。",
    ["E07"],
)

EXPL["retnet_hf/train.py::GradCoverageCallback.__init__"] = E(
    "给每个 requires_grad 的参数注册反向钩子，并记录每个参数的元素数（E07）。"
    "不能给 requires_grad=False 的参数注册——PyTorch 会直接抛 RuntimeError；"
    "这类参数本来就不会有梯度，保留在 _numel 里即可被判为死参数。"
    "钩子必须在构造时注册，不能等到 on_step_end：HF 在 optimizer.step() 之后会 zero_grad，"
    "回调触发时 p.grad 已经是 None。",
    ["E07"],
)

EXPL["retnet_hf/train.py::GradCoverageCallback._make_hook"] = E(
    "生成单个参数的钩子：反向时取该参数梯度绝对值的最大值，与历史值取 max 后记入 _seen。"
    "钩子原样返回梯度，不改变任何数值；用 max 而不是最后一次，是为了覆盖梯度累积的多个微批"
    "（只要有一个微批拿到非零梯度，这个参数就不是死的）。",
    ["E07"],
)

EXPL["retnet_hf/train.py::GradCoverageCallback._make_hook._hook"] = E(
    "钩子本体。grad 为 None（未参与图）时直接返回；否则把 abs().max() 记进 _seen。"
    "整个函数只读梯度、不修改梯度，因此对训练数值零影响；它的全部价值是把"
    "“参数有没有信号”这件事从不可见变成可断言。",
    ["E07"],
)

EXPL["retnet_hf/train.py::GradCoverageCallback._remove_hooks"] = E(
    "检查完成后摘掉所有反向钩子，避免每一步都白付一次 Python 回调开销（E07）。"
    "逐个 remove 并吞掉异常：钩子可能已被 PyTorch 因张量释放自动清理，"
    "这里的目标是幂等，不是报告清理失败。",
    ["E07"],
)

EXPL["retnet_hf/train.py::GradCoverageCallback.check_now"] = E(
    "立即汇总并返回死参数列表，同时摘钩子、置 _done。单独暴露这个方法，"
    "使测试可以构造模型 + 一次 backward 后直接断言分类结果，不必启动 Trainer；"
    "on_step_end 只是它的调度外壳。",
    ["E07"],
)

EXPL["retnet_hf/train.py::GradCoverageCallback.on_step_end"] = E(
    "在 global_step 达到 check_step（默认 1）时做一次梯度覆盖检查：打印死参数个数、"
    "元素占比与前 12 个名字；占比超过 tol 就 raise SystemExit 停机。"
    "E07 的原缺陷是注释声称“另加了一次梯度覆盖检查”而代码里没有；"
    "现在这句话对应的是这个回调，删掉它 test_grad_coverage 会红。",
    ["E07"],
)

EXPL["retnet_hf/train.py::decay_group_1d_violations"] = E(
    "检查优化器参数组：返回所有出现在 weight_decay>0 组里的 1 维参数名（E06）。"
    "这是比“训练中 gamma 有没有变小”更直接的判据——gamma 本来就会随学习变化，"
    "而“1 维参数不该出现在衰减组”是结构性的硬约束。按 id() 匹配参数对象，"
    "不依赖对参数命名规则的重实现。",
    ["E06"],
)

EXPL["retnet_hf/train.py::DecayGuardCallback.__init__"] = E(
    "保存 trainer 引用与开关状态。取 trainer 而不是依赖回调 kw："
    "transformers 各版本给 on_train_begin 的关键字不保证一致，"
    "显式绑定与 plan 回调同一策略，避免“看着挂了、其实没生效”。",
    ["E06"],
)

EXPL["retnet_hf/train.py::DecayGuardCallback.on_train_begin"] = E(
    "训练开始时（此刻优化器已建好）断言：开 --no_decay_1d 后没有任何 1 维参数落在"
    "weight_decay>0 的参数组里；有就 SystemExit 并列出前 12 个名字。"
    "这是 E06 的运行时守卫：旧测试手工给 trainer 赋 _no_decay_1d，"
    "删掉接线代码也不会红；这个回调查的是最终优化器的真实分组。",
    ["E06"],
)

EXPL["retnet_hf/train.py::cap_eval_rows"] = E(
    "把 --eval_batches 翻译成验证集行数上限：eval_batches × 评测 batch，0/None 表示不限制。"
    "旧实现定义了这个参数、run_task 两个分支也都传 40，但代码里没有任何地方读它——"
    "典型的死字段。现在它真的限制评测规模，并在日志里打印“只评测前 N 行”。",
    [],
)

EXPL["retnet_hf/train.py::plan_stage_specs"] = E(
    "把 Plan.stages 转成 StagePackedDataset 需要的 dict 列表，是 plan JSON -> parse_plan -> "
    "数据集这条链路的唯一构造点。E12 的 require_eos 与 P2-2 的 weight 都必须经过它；"
    "旧实现把这段内联在 main() 里且漏掉这两个键，测试只直接构造 _FlatStage，"
    "于是“计划里写 require_eos: false 不生效”这个缺陷永远测不出来。",
    ["E12"],
)

EXPL["retnet_hf/train.py::build_parser"] = E(
    "把 argparse 抽成纯函数（E06/P1-3）。这样测试可以只解析 argv 断言 "
    "--decay_1d -> no_decay_1d=False、--allow_dead_params 等开关，"
    "不必把 Trainer 跑起来；main() 只负责把解析结果接到实际对象上。"
    "旧实现把 parser 内联在 main 里，开关的解析与接线都没有任何测试覆盖。",
    ["E06", "E08"],
)

EXPL["retnet_hf/modeling_retnet.py::ce_chunk_tokens"] = E(
    "按字节预算反推交叉熵分块能放多少 token：chunk_bytes // (4*V)，至少 1（E22）。"
    "V=20000、预算 64 MB 时是 838 token/块，生产形状 B=16/L=2048 因此切出 40 块。"
    "旧实现把 chunk 固定成 262144 个 token，与 V 无关、且大于生产 token 总数，"
    "于是只有一块、2.6 GB 的 fp32 拷贝原封不动。单独成函数是为了让这条算术可被直接断言。",
    ["E22"],
)

EXPL["retnet_hf/data.py::_proves_rebuildable"] = E(
    "判断 meta.json 是否足以证明这份数据可重建（E13）：rebuild_cmd 或 source_ids 直接认；"
    "旧产线只有 sources+text_fields 也认——字段名与通配符都在，足以按同一口径重跑。"
    "旧实现只认前两个键，而当时的产线一个都不写，于是 GC 永远拒绝删除、"
    "plan 里 6 个 gc:true 的阶段释放 0 字节。",
    ["E13"],
)

EXPL["retnet_hf/data.py::StagePackedDataset.protect"] = E(
    "登记“绝不允许 GC 删除”的目录（验证集），用 provenance.norm_key 规范化后比较。"
    "训练入口在算出 val_path 之后调用它；gc_stage 删除前先查这个集合。"
    "旧 docstring 把“验证集目录跳过”写成安全约束，但函数体里根本没有这条检查，"
    "计划里某段一旦 gc=true 且指向验证集，验证集会被直接删掉。",
    ["E13", "E14"],
)

EXPL["scripts/grow_500m.py::main._check_norm_layout"] = E(
    "生长后的分段布局断言（P1-8）：gamma_new/beta_new 长度必须等于 NEW_D-切点；"
    "第二段必须整段等于预期门控（中间层 BLOCK_GATE、final FINAL_GATE），beta 必须全 0。"
    "它在 5 步可训性反向之前执行，保证“文档写的映射”就是落盘的映射；"
    "旧版本没有这条断言，而 header 写的映射与代码不一致，只能靠事后 --verify 兜。",
    ["E02", "E05"],
)


# English prose overrides every EXPL entry. Placed after the last EXPL definition
# so the coverage check sees the complete map; a missing/extra key aborts instead
# of emitting a half-Chinese document.
if I18N.exists():
    _en = json.loads(I18N.read_text(encoding="utf-8"))
    _missing = sorted(set(EXPL) - set(_en))
    _extra = sorted(set(_en) - set(EXPL))
    if _missing or _extra:
        raise SystemExit(f"doc_i18n/expl_en.json coverage mismatch: "
                         f"missing={_missing[:5]} extra={_extra[:5]}")
    for _k, _v in _en.items():
        EXPL[_k] = E(_v["text"], EXPL[_k][1])


def collect(path: Path):
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    out = []

    def rec(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + child.name
                out.append((q, child.lineno, child.end_lineno))
                rec(child, q + ".")
            elif isinstance(child, ast.ClassDef):
                rec(child, prefix + child.name + ".")

    rec(tree, "")
    return out, lines


def main() -> int:
    missing, short = [], []
    sections = []
    total = 0
    for rel in FILES:
        path = ROOT / rel
        funcs, lines = collect(path)
        sections.append((rel, funcs, lines))
        total += len(funcs)
        for q, a, b in funcs:
            key = f"{rel}::{q}"
            if key not in EXPL:
                missing.append(key)
                continue
            text, refs = EXPL[key]
            if len(text) < 100:
                short.append((key, len(text)))

    if missing or short:
        print(f"missing explanations: {len(missing)}, too short: {len(short)}")
        for m in missing:
            print("  MISSING", m)
        for s in short:
            print("  SHORT", s)
        return 1

    out = [
        "# 01 Per-function reference (rewritten)",
        "",
        "> Generated by `scripts/_gen_code_doc.py`. **Code blocks are extracted from the",
        "> source with AST**, not copied by hand, so they are byte-identical to the tree.",
        "> Each section names the `00_error_table.md` ids it relates to, for cross-checking.",
        f"> Covers {len(FILES)} files and {total} functions/methods; a missing entry or an",
        "> explanation shorter than 100 characters aborts generation.",
        "",
        "## Fix status (against 00; generated from `scripts/fix_status.py`)",
        "",
        "| 00 id | Status | Where |",
        "|---|---|---|",
        *[f"| {k} | {FIX_STATUS[k]} | {FIX_NOTE[k]} |" for k in sorted(FIX_STATUS)],
        "",
        "---",
        "",
    ]
    for rel, funcs, lines in sections:
        out.append(f"## `{rel}`")
        out.append("")
        for q, a, b in funcs:
            text, refs = EXPL[f"{rel}::{q}"]
            src = "\n".join(lines[a - 1:b])
            ref_s = ", ".join(refs) if refs else "none"
            out.append(f"### `{q}` (L{a}–L{b})")
            out.append("")
            out.append(f"**00 links**: {ref_s}")
            out.append("")
            out.append("```python")
            out.append(src)
            out.append("```")
            out.append("")
            ids = CITE.get(f"{rel}::{q}", [])
            out.append(text + (("\n\n" + cite(*ids)) if ids else ""))
            out.append("")
        out.append("---")
        out.append("")

    out.append(refs_md())

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}: {total} functions, {len(OUT.read_text(encoding='utf-8'))} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
