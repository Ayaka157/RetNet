"""Generate docs/00F1_fix_matrix.md.

Each entry corresponds to one id in docs/00, with a fixed structure:
  ## <id> <title>
  **Status**: ...
  ```python  <-- the fixed function/branch/passage, extracted by AST/anchor
  <explanation, >=200 chars>

Checks (any failure aborts):
  * every id in 00 (E01..E23, N01..N04) must have an entry;
  * every code block must be byte-identical to the source range;
  * explanations must be >=200 chars and contain no "omitted/same as above";
  * every referenced file must exist;
  * English prose in ``doc_i18n/fix_en.json`` must cover all entries.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

from _refs import cite, refs_md
from fix_status import FIX_STATUS

#: Default target is the main repository; point the snapshot at its own tree
#: (code lives under <root>/code):
#:   RETNET_GEN_ROOT=<snapshot>/code RETNET_GEN_OUT=<snapshot>/docs
ROOT = Path(os.environ.get("RETNET_GEN_ROOT", r"E:\Workspaces\RetNet"))
OUT_DIR = Path(os.environ.get("RETNET_GEN_OUT", str(ROOT / "docs")))
DOC00 = OUT_DIR / "00_error_table.md"
OUT = OUT_DIR / "00F1_fix_matrix.md"
I18N = Path(__file__).resolve().parent / "doc_i18n" / "fix_en.json"

ENTRIES: list[dict] = []


def entry(eid: str, title: str, status: str, refs: list, text: str) -> None:
    # Status comes **only** from fix_status.py: the old submission had three
    # hand-kept status sources (README / docs/01 / docs/00F1) that disagreed
    # (P0-5). Ignoring the caller's status here prevents a new fork.
    ENTRIES.append(dict(id=eid, title=title, status=FIX_STATUS.get(eid, status),
                        refs=refs, text=text))


def fn(path: str, qual: str):
    return ("fn", path, qual)


def span(path: str, start: str, end: str):
    return ("span", path, start, end)


def find_function(path: Path, qual: str, lines: list[str]) -> tuple[int, int] | None:
    tree = ast.parse("\n".join(lines))

    def rec(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + child.name
                if q == qual:
                    return child.lineno, child.end_lineno
                got = rec(child, q + ".")
                if got:
                    return got
            elif isinstance(child, ast.ClassDef):
                got = rec(child, prefix + child.name + ".")
                if got:
                    return got
        return None

    return rec(tree, "")


def find_span(lines: list[str], start: str, end: str) -> tuple[int, int] | None:
    a = b = None
    for i, ln in enumerate(lines):
        if a is None and start in ln:
            a = i
        elif a is not None and end in ln:
            b = i
            break
    if a is None or b is None:
        return None
    return a + 1, b + 1


# ======================================================================
# 条目
# ======================================================================

entry(
    "E01", "生长时新行必须随机、新列才置零", "已修",
    [fn("scripts/grow_500m.py", "main.fill")],
    "旧实现先用 torch.zeros 建整个新矩阵，再把旧权重贴到左上角，于是新增的行与列**同时**为 0。"
    "这个组合会形成梯度死锁：新列是 0，则新行输出的梯度乘到 0；新行是 0，则新列收到的上游梯度也乘到 0，"
    "两个 0 互为对方梯度的因子，从第 0 步起就精确不动。修复只改一侧：新列保持 0，"
    "因为新列若随机会立刻改变旧维度的输出、破坏函数保真，而且这种扰动没有任何门控能挡住；"
    "新行改成 N(0, NEW_ROW_INIT_STD)。新行随机让新单元先有非零输出，下游“读新单元”的列"
    "在第 1 步就拿到非零输入并开始学习；这些列变非零之后，新行自己在第 2 步拿到梯度。"
    "这是一条自愈级联，而不是死锁。函数保真仍然成立：新列是 0，所以新维度不会影响旧维度的输出。"
    "该行为由 tests/test_growth_trainable.py 的反向用例守护：把 NEW_ROW_INIT_STD 改回 0，"
    "生长脚本必须拒绝落盘。",
)

entry(
    "E02", "中间层门控必须为 1，final_norm 门控为 0", "已修",
    [fn("scripts/grow_500m.py", "main.grow_norm")],
    "旧实现把新段的 gamma_new 与 beta_new 全部置 0，并且把源模型已有的零段继续搬过来，"
    "注释里称之为“零旁支”。问题在于这个门是乘法门：gamma_new=0 时，新维度不仅前向恒 0，"
    "它自己的梯度也恒 0，因为它等于“下游回传梯度 × 新维度输入”，而下游那些读新维度的列初始同样是 0。"
    "修复把门控按位置分开：中间层 norm1/norm2 设为 1.0，让新维度能流进 q/k/v 的新列，"
    "新列即使初值为 0 也能因为输入非零而拿到梯度；final_norm 设为 0.0，它是新维度通往 logits 的"
    "唯一出口，置零保证生长瞬间 logits 逐值不变。final_norm 的门自己会学：它的梯度等于"
    "“上游 logits 梯度 × 新维度”，上游经 tied lm_head 的新列（随机非零）回传，所以第一步就非零。"
    "两者合起来实现“输出端严格保真、新参数从第 0/1 步起可学”，"
    "Tests/test_growth_trainable.py 的旧行为用例会验证门控归零时脚本拒绝保存。",
)

entry(
    "E03", "embedding 的死列与新列必须重新随机", "已修",
    [span("scripts/grow_500m.py", "嵌入：旧列照搬", "new_sd[\"embedding.weight\"] = emb")],
    "旧实现把 embedding 的新列留成 0。这一列同时承担两个角色：输入侧的 token 向量，"
    "以及（因为 tie_weights）输出侧的分类器权重。留 0 的直接后果是第 0 层新维度拿不到任何信号，"
    "整条新子网没有入口。修复把 [grow_old_dim:] 全部重新随机（N(0,0.02)），"
    "注意起点不是 OLD_D 而是 grow_old_dim：源模型上一次生长留下的 256 列同样是死的，"
    "只复活本次新增的 384 列没有意义，必须把 640 列一起复活。"
    "函数保真仍然成立，因为 final_norm 的 gamma_new=0 把新维度挡在 logits 之外；"
    "而这 640 列的随机值正是整条新维度唯一的非零信号源。"
    "这条与 E01、E02 是一组：没有非零输入，新列的梯度仍然为 0；没有新行随机，新单元仍然没有输出；"
    "三者缺一，67% 死参数就会复现。",
)

entry(
    "E04", "生长脚本必须校验可训性，而不只是函数保真", "已修",
    [span("scripts/grow_500m.py", "print(f\"[trainable] 跑 5 步反向", "return 2")],
    "旧脚本的校验只有一条：max|Δlogits| 是否小于容差。这条检查在设计上就不可能失败，"
    "因为一个恒零旁支恰恰是最完美的函数保真——它什么都不影响，所以它必然通过。"
    "修复增加第二道校验：加载生长后的模型，跑 5 步真实反向（AdamW，lr=1e-4），"
    "然后逐参数检查 grad.abs().max() 是否大于 0。用 5 步而不是 1 步，是因为新维度是级联自愈的："
    "第 0 步只有 embedding 新列与读新输入的列有梯度；第 1 步新列变非零，读它们的下游列开始有梯度；"
    "第 2 步新单元的输出投影拿到梯度。5 步足够走完这条链，仍为零的参数就是真死。"
    "两道校验都通过才落盘；任何一道失败返回 2 并拒绝保存。"
    "tests/test_growth_trainable.py 用 monkeypatch 把新行随机改成 0，断言脚本必须返回 2 且不落盘，"
    "从而证明这道守卫确实会失败，而不是一个永远为真的装饰。",
)

entry(
    "E05", "GrowthLayerNorm 的新维度门控默认必须是 1", "已修（默认值 + 测试）",
    [fn("retnet_hf/modeling_retnet.py", "GrowthLayerNorm.__init__")],
    "这个类的构造函数默认 gamma_new=1.0、beta_new=0，本身是正确的；问题出在生长脚本显式把它覆盖成 0，"
    "并且旧代码没有任何地方守住这个默认值。E05 的修复是把它变成可测的契约："
    "构造出的 GrowthLayerNorm 必须满足 gamma_new 全 1、beta_new 全 0；"
    "tests/test_regressions_extra.py 直接断言这一点。"
    "为什么默认值必须是 1 而不是 0：新维度不做归一化，gamma_new 是它唯一的缩放门；"
    "门为 0 时前向恒 0，而它自己的梯度等于“下游回传 × 新维度输入”，下游读新维度的权重同样是 0，"
    "于是永久死锁。默认 1 让新维度在构造时就可达，零旁支的效果改由 final_norm 的输出门控实现——"
    "那里置 0 只挡输出、不挡梯度，因为上游 logits 的梯度经 tied embedding 的新列仍能回传。",
)

entry(
    "E06", "1 维参数默认不做 weight decay", "已修",
    [span("retnet_hf/train.py", "ap.add_argument(\"--no_decay_1d\"", "用之前先想清楚 gamma 会不会被压塌")],
    "旧实现的开关 --no_decay_1d 默认关闭，而 v21 的配置里根本没有这个字段，"
    "于是 msr.gamma 作为裸 nn.Parameter 吃满 wd=0.1。后果是实测可知的："
    "440 个 gamma 的中位数 0.5656，换算半衰期 1.22 token，96.1% 的头短于 128 token；"
    "决定性的证据是 h16–h21 的 gamma 在 20 个不同层里逐位相同，只有“零梯度 + 纯乘性衰减”能产生这种签名。"
    "修复把默认值反过来：不衰减是默认行为，新增 --decay_1d 作为反向开关，只有复现旧 run 时才用。"
    "同时 train.py 的 _preflight 会在开训前打印 gamma 分布并拒绝 <=0 的值。"
    "tests/test_no_decay_1d.py 同时测两个方向：开启时任何 1 维参数（含 gamma）都不得出现在 decay 集合；"
    "关闭时 gamma 必须出现，否则测试本身没在测东西。",
)

entry(
    "E07", "“梯度存在只是为 0”不是正常的理由", "已修",
    [fn("retnet_hf/train.py", "GradCoverageCallback.on_step_end"),
     fn("retnet_hf/train.py", "classify_dead_grad_params"),
     span("retnet_hf/train.py", "trainer.add_callback(GradCoverageCallback(",
          "tol=float(args.dead_grad_tol)))")],
    "旧注释写着“growth 里 gamma=0 的张量仍参与前向，梯度存在只是为 0”，并据此关掉 DDP 的"
    "find_unused_parameters，结论是“本模型确实没有未使用参数”。这段推理在事实层面是对的——"
    "全零梯度确实不是 None——但它恰好掩盖了 bug：它确认了“有梯度”，却没问“梯度是不是恒等于 0”。"
    "上一轮的“修复”只改了叙述，还写了一句“另加一次梯度覆盖检查”，而代码里根本没有那东西；"
    "docs/01 自己都写着“尚未落地”。同一类失败模式升级了一级：从“用注释合理化缺失的检查”"
    "变成“用注释宣称一个不存在的检查”。"
    "现在补的是真检查：GradCoverageCallback 在构造时给每个 requires_grad 的参数注册反向钩子，"
    "记录 grad.abs().max() 的最大值（不能等到 on_step_end 再读 p.grad：HF 在 "
    "optimizer.step() 之后会 zero_grad）；第 1 步结束时汇总“没有梯度或梯度恒零”的参数，"
    "打印个数、元素占比与前 12 个名字，占比超过 --dead_grad_tol（默认 2%）直接 SystemExit。"
    "用钩子而不是事后读 grad 是这条修复的关键细节：后者在真实 Trainer 里永远读到 None，"
    "检查会看起来通过、实际什么都没查。"
    "tests/test_grad_coverage.py 证明三件事：健康模型零死参数；把 blocks.1.msr.q_proj.weight "
    "的 requires_grad 关掉后它被判死；阈值 0 时 on_step_end 抛 SystemExit。"
    "第三个用例是“守卫必须能被证明会失败”的又一次落实。",
)

entry(
    "E08", "参数量必须按非零元素统计", "已修",
    [fn("retnet_hf/train.py", "live_param_ratio"),
     fn("retnet_hf/train.py", "check_live_ratio"),
     fn("retnet_hf/train.py", "read_live_baseline")],
    "旧代码只打印 sum(p.numel())，而 numel() 完全不看数值：哪怕 67% 精确为 0，它照样输出 504.18M。"
    "修复拆成两层：live_param_ratio 返回（非零元素数、总元素数、比例）；"
    "_preflight 在模型加载后立即调用它，并把“是否放行”交给纯函数 check_live_ratio。"
    "为什么要拆出判定函数：上一版只有一条“比例 < 95% 就 SystemExit”，把两件事混成一件——"
    "新模型初始化坏了（必须拒绝），和从一个有意剪枝/生长过的检查点继续训练（33% 活参数是已知状态）。"
    "docs/02 的 rescue 计划起点正是 33% 活参数的 retnet_500m_v21_export，旧守卫让计划第一步就中止。"
    "现在的规则：达标放行；显式 --allow_dead_params 放行；有历史基线（live_params.json）且"
    "比例不低于 baseline-2% 自动放行；其余拒绝并同时给出两条出路。"
    "基线由 write_live_baseline 在每次成功开训后写到输出目录，所以“续训自己导出的检查点”"
    "不需要每次手动加开关，而“这一轮又把参数弄死了”仍会被基线比对抓住。"
    "gamma 检查同时遍历 named_parameters 与 named_buffers：use_gla=False 时 gamma 是 buffer，"
    "旧实现只查参数，那种模型上这条检查是死的。"
    "tests/test_live_params.py 要求零参数统计逐元素精确；tests/test_preflight_live.py 覆盖"
    "纯函数四个边界、基线读写、坏文件容错，以及“33% 活参数 + 开关可开训 / 不加开关拒绝”两条真实路径。",
)

entry(
    "E09", "建包默认必须写文档边界", "已修",
    [span("scripts/build_flat_corpus.py", "ap.add_argument(\"--add-eos\"", "关之前先读 docs/00 E09")],
    "旧实现的 --add-eos 是 store_true，默认 False。所有 hf_pack_*（代码、网页、教材、数学、中文编程）"
    "都是用它建的，因此全部没有文档边界：一篇代码的末行直接接下一篇的 prompt。"
    "后果不是“效果差一点”，而是模型学会忽略远距离内容——2048 窗口里跨文档的部分本来就是噪声。"
    "修复把默认值改成 True，并新增 --no-add-eos 作为显式反向开关；"
    "关掉它需要在 help 里被提醒先读 E09。这个改动让“忘记传参”从错误变成安全，"
    "而不是靠每个调用点记得加一个 flag。build_flat_corpus 的 meta.json 里仍记录 add_eos 的实际值，"
    "供下游（例如 build_ctx_mix 的 E11 守卫）核对。",
)

entry(
    "E10", "建包必须显式指定正文字段，且字段错了要立刻失败", "已修",
    [span("scripts/_data_supervisor.py", "dict(name=\"code\"", "text_field=\"text\", skip=\"stats.json\")"),
     span("scripts/build_flat_corpus.py", "_first = next(iter_records", "请显式指定正文列")],
    "旧实现用 --text-field auto，靠字段名黑名单猜正文；prog_zh 的 10 个数值元数据列"
    "（alnum_ratio、perplexity、industry_type 等）因此被当成正文拼进训练流，"
    "这类错误不报错、只降低数据质量，只能靠抽检解码发现。"
    "修复分两层：调用方（_data_supervisor、_prep_ext）必须显式传 text_field，"
    "每份语料的值写在 SPECS 里并注明“若字段名不对，脚本会在第一条记录处失败”；"
    "build_flat_corpus 增加早失败检查——读第一条记录，若给定的字段一个都不存在，"
    "立刻退出并打印真实可用字段。旧行为是逐条记为 error，跑几小时分词后才在结尾以 >1% 错误率失败，"
    "或者更糟：字段只在部分记录缺失时静默丢掉那部分数据。早失败把“几小时后才知道”变成“几秒内知道”。",
)

entry(
    "E11", "明知无边界还继续造 16K 包，必须拒绝", "已修",
    [span("scripts/build_ctx_mix.py", "2026-10-20（docs/00 E11）", "或显式加 --allow-boundaryless")],
    "这个脚本的 docstring 自己写着“flat 分片的原始边界在分词时就丢了，无法恢复”，然后继续用它造 16K 包。"
    "知道缺陷还把缺陷固化进下一轮，比不知道更糟。修复把这件事变成默认拒绝："
    "每个 flat 来源必须有 meta.json，且 add_eos 为真；缺文件或 add_eos=false 都直接 SystemExit，"
    "并提示用带 --add-eos 的 build_flat_corpus.py 重建。确实要接受无边界流的，"
    "必须显式加 --allow-boundaryless，日志会打印一行警告，这条也会进实验记录。"
    "这不是“多加一个开关”，而是把静默的质量损失变成一次明确的、需要署名的决定。",
)

entry(
    "E12", "flat 阶段打开时必须确认存在文档边界", "已修",
    [fn("retnet_hf/data.py", "_FlatStage._check_boundaries"),
     fn("retnet_hf/plan.py", "Stage.signature"),
     fn("retnet_hf/train.py", "plan_stage_specs")],
    "flat 流按固定宽度切片，切点由行号决定，与文档边界无关；如果分词时没加 eos，"
    "模型看到的就是“上一篇末行 + 下一篇开头”。旧的 _FlatStage 对此完全沉默。"
    "第一版修复在打开阶段时抽查前 2M token，没有 eos 就抛 ValueError，"
    "并给出两条出路：用带 --add-eos 的脚本重建，或在计划里给该阶段显式写 require_eos: false。"
    "但**第二条出路其实不存在**：plan.Stage 没有 require_eos 字段，parse_plan 把它丢掉，"
    "train.py 构造 stage dict 时也不传，于是错误信息教人写的键永远不生效——"
    "守卫报错、操作者照做、仍然报同样的错。"
    "现在 require_eos 是 Stage 的正式字段，parse_plan 解析它，plan_stage_specs 把它带进"
    "StagePackedDataset，热更新（Stage.signature 包含它）也不会丢。"
    "默认 require_eos=True 是刻意的：让忘记加 eos 的旧包在使用时立刻暴露，"
    "而不是训练到一半才发现模型学不会长距离。"
    "验证走完整链路而不是直接构造 _FlatStage：tests/test_plan_eos.py 从 plan JSON 出发，"
    "经 parse_plan、plan_stage_specs 到 stage_ready，证明 require_eos: false 真的能打开、"
    "没写时仍然拒绝、热更新也不会把键丢掉。",
)

entry(
    "E13", "GC 删除前必须证明数据可重建", "已修",
    [fn("retnet_hf/data.py", "StagePackedDataset.gc_stage"),
     fn("retnet_hf/data.py", "_proves_rebuildable"),
     span("scripts/build_flat_corpus.py", "\"rebuild_cmd\":",
          "(out / \"meta.json\").write_text")],
    "旧实现只查两件事：只删本阶段登记的文件、目录被别的阶段引用时跳过。"
    "它缺了最关键的一条——这份数据能不能从源重建。上一轮就是在这里把 hf_pack_* 与 pretrain_v2"
    "的分片删光的：分词脚本在成功后 rmtree 原始 parquet，gc 再删分片，源与产物同时消失。"
    "第一版修复要求 meta.json 里写 source_ids 或 rebuild_cmd，缺一个就拒绝删除。"
    "问题是**产线从来不写这两个键**：build_flat_corpus 的 meta 只有 sources/text_fields，"
    "而唯一通过测试的用例是手写了一份带 rebuild_cmd 的 meta——测试夹具不是产线产物。"
    "结果是从“删错”变成“永远不删”：plan 里 6 个 gc:true 的阶段每次打印拒绝、释放 0 字节，"
    "磁盘回收这个计划前提完全落空。"
    "现在两头都改了：产线把 argv 与 rebuild_cmd 写进 meta，source_ids 由输入通配符生成"
    "（E14 的声明层要用）；GC 侧的可重建证明认三种——rebuild_cmd、source_ids、"
    "旧产线的 sources+text_fields。另外补上“验证集目录跳过”的真实实现（protect() + 规范化路径），"
    "因为旧 docstring 把这条写成安全约束、函数体里却没有。"
    "验证换成**真实产线闭环**：tests/test_producer_gc.py 现场训练一个小 tokenizer、跑 "
    "build_flat_corpus 产出分片与 meta，再用这份 meta 调 gc_stage，断言 freed>0 且分片被删；"
    "同时测旧 meta（只剩 sources+text_fields）仍可删、无证明仍拒绝、protect 过的目录拒绝。",
)

entry(
    "E14", "验证集与训练阶段同目录必须拒绝开训", "已修",
    [fn("retnet_hf/train.py", "assert_val_disjoint"),
     fn("retnet_hf/provenance.py", "overlap_report"),
     fn("retnet_hf/provenance.py", "dir_row_hashes")],
    "上一轮的 val_dir 指向 chat_pack_v13，而最后 14,000 步的 sft_mix 里 40% 就是 chat_pack_v13。"
    "验证集进了训练，训练期 eval_loss 在末段从 4.55 掉到 2.4450 根本不是模型变好，"
    "而是它看到了验证集；load_best_model_at_end 又会拿这个被污染的数去选“最佳”。"
    "第一版修复只比较 Path(...).resolve() 字符串：注释声称覆盖“相对路径、大小写、尾斜杠”，"
    "而测试里的“另一种写法”就是同一个绝对路径——它守的是**路径相等**，事故却是**数据相等**。"
    "现在拆成三层（retnet_hf/provenance.py）："
    "（1）规范化路径，resolve + normcase，真正覆盖相对路径/尾斜杠/大小写；"
    "（2）meta.json 声明来源，source_ids 或旧产线的 sources+text_fields，两边有交集就拒绝；"
    "（3）按行内容指纹，64 位 FNV 逐行哈希，两个目录里存在完全相同的行就拒绝——"
    "它不看文件名，因此“同一批记录重新打包到另一个目录”也能发现。"
    "内容层对大语料有 200k 行上限（约 0.8 GB），截断时在日志里如实打印，不假装扫全了。"
    "验证侧的重点是让旧测试不能再空过：tests/test_val_disjoint.py 用 os.path.relpath、"
    "尾斜杠、大小写三种**真的不同**的写法；用 meta.json 声明同源（名字不同）触发拒绝；"
    "用训练集里复制进验证集两行触发内容命中；并断言验证集侧读到的是 val.ids.npy 而不是空集。",
)

entry(
    "E15", "不能拿被污染的 eval_loss 选最佳模型", "已修",
    [span("retnet_hf/train.py", "2026-10-20（docs/00 E15）", "metric_for_best_model=\"eval_loss\"")],
    "旧配置写的是 load_best_model_at_end=True、metric_for_best_model=eval_loss。"
    "两个问题叠在一起：第一，训练期 eval 用的是 val_dir，而 val_dir 可能被训练过（E14）；"
    "第二，就算 val 干净，它也只是对话分布上的损失，不代表域内预训练能力。"
    "上一轮最终模型在域内 held-out 上比 step 17,500 差 0.655 nat，却因为 eval_loss 的末段下降"
    "而被当成“收敛了”。修复把 load_best_model_at_end 关掉：训练中只按步数存 checkpoint，"
    "跑完后用独立域内 held-out（data_cache/pretrain_v2/*_val.npy，绝不进训练）离线比较再选。"
    "离线选择现在有可执行的工具：scripts/eval_holdout.py 支持 --model 指向任意 checkpoint 目录，"
    "并对每个候选输出同一口径的 loss（--by-position/--ctx-ablation 是附加诊断）。"
    "metric_for_best_model 字段保留但不再驱动加载，避免旧配置直接报错。"
    "上一版文档在这里自相矛盾：docs/02 把“保留 load_best_model_at_end”写成硬前提，"
    "docs/01 说“未修”，代码却是关闭的。现在三处统一为**关闭 + 离线选优**，docs/02 §2 E15 已重写。",
)

entry(
    "E16", "parallel 路径不能有 chunkwise 没有的归一化", "已修",
    [fn("retnet_hf/modeling_retnet.py", "MultiScaleRetention._decay_mask"),
     fn("retnet_hf/modeling_retnet.py", "MultiScaleRetention._parallel_retention")],
    "旧代码在 _decay_mask 里把掩码除以 sqrt(Σmask)，又在 _parallel_retention 里把分数除以"
    "Σ|scores|；chunkwise 与 recurrent 都没有这两步。于是训练（chunkwise）、prefill（parallel）、"
    "解码（recurrent）是三个不同的函数，实测 mean|Δlogit|=0.0158、argmax 一致率 99.66%。"
    "更严重的是评测脚本 eval_holdout.py 强制 parallel，所以上一轮所有 held-out 数字"
    "（3.6287 / 2.8471 / 3.5023）测的都不是训练的那个函数。"
    "修复删掉这两处归一化，让三条路径只差分块与循环的实现方式、不差数学。"
    "tests/test_parallel_chunkwise_equiv.py 用 tiny 模型比较 parallel vs chunkwise、"
    "parallel vs recurrent，容差按 logits 尺度设相对阈值，防止以后有人再往其中一条路径加私有操作。",
)

entry(
    "E17", "save_total_limit 的默认值必须与模板一致", "已修",
    [span("retnet_hf/train.py", "ap.add_argument(\"--save_total_limit\"", "静默轮转删除）"),
     fn("run_task.py", "build_cmd_full"),
     fn("run_task.py", "load_config")],
    "旧实现里 train.py 的默认是 3，而任务模板写的是 4。历史上 run_task 的 plan 分支又不传这个参数，"
    "于是“配置写 4、实际跑 3”，多出来的那个 checkpoint 会被 HF 按 mtime 轮转删掉，"
    "而删除日志是 logger.info，transformers 默认 WARNING 级别，什么都不打印。"
    "这类“两套默认值”是最难查的错误：配置看起来没错、代码看起来没错，只有对着两边数才能发现。"
    "第一版修复只把 train.py 的默认从 3 改成 4，而 run_task.py 的 DEFAULTS 里**仍然是 3**，"
    "且 plan 分支不传这个参数——所以“配置写 4、实际跑 3”在 launcher 路径上原样存在，"
    "而当时的测试只正则 train.py，全绿。"
    "现在默认值是 retnet_hf/defaults.py 里的一个常量，train.py 与 run_task.py 都 import 它；"
    "run_task 的 plan 与非 plan 两个分支都显式传同一个值。"
    "验证不再查字面量，而是走行为：test_save_total_limit_is_one_shared_constant 断言两处来自同一常量；"
    "test_run_task_passes_save_total_limit_in_both_branches 对两个分支、两个取值（默认与 6）"
    "都断言 argv 里的数字等于配置值——旧代码在 plan 分支连参数都不传，这条会直接红。",
)

entry(
    "E18", "热改总步数把 LR 抬上去，必须默认拒绝", "已修",
    [span("retnet_hf/plan.py", "if new.total_steps != old.total_steps:", "state.max_steps = new.total_steps")],
    "计划是可热加载的：改 stages 或 steps 会改变 total_steps，而 LR 余弦是按 total_steps 铺的，"
    "于是当前步的目标 LR 可能突然上跳。上一轮一天热改 6 次，日志里只留下一行“余弦随之重铺”，"
    "实际效果是一次没人批准的受控热重启。修复引入 allow_reschedule："
    "当新计划会让当前 LR 上升超过 2% 且该字段不为 true 时，拒绝整次变更、保留旧计划，"
    "并打印两个 LR 与需要写的那行配置。往下掉是安全的（学习率变小），放行。"
    "为什么不是直接禁止改总步数：缩短预算是常见的止损操作，它只会让 LR 变小；"
    "真正危险的是延长预算把 LR 抬回去——那会覆盖已经学到的低 LR 阶段。",
)

entry(
    "E19", "refresh 必须比较扫描时快照的签名", "已修",
    [fn("retnet_hf/data.py", "_FlatStage.refresh"),
     fn("retnet_hf/data.py", "_FlatStage._scan")],
    "旧 refresh 只比较分片数量与总行数。一个同尺寸的分片被替换时两个值都不变，"
    "函数返回 False，训练继续用旧 mmap 读到过期内容。"
    "第一版修复把签名写成“比较时对旧列表重新 stat”，结果新旧两边读的都是文件当前状态，永远相等——"
    "测试当场抓到了这个错误，也说明“看起来对”的修复必须被能失败的测试验证。"
    "正确做法是 _scan 返回 (shards, starts, total, signature)，签名在扫描那一刻取"
    "（文件名, mtime_ns, 大小）并保存到 self._signature；refresh 只比较保存值与新扫描值。"
    "close() 时把签名一并清空，避免拿一份已不存在的签名做比较。"
    "tests/test_regressions_extra.py 先把 mmap 关闭、原地写同尺寸新内容、显式把 mtime 推到未来，"
    "再断言 refresh 返回 True 且第二次调用返回 False。",
)

entry(
    "E20", "plan 模式必须把 eval_batch_mult 传下去", "已修",
    [fn("run_task.py", "build_cmd")],
    "旧 build_cmd 在带 plan 时走另一条分支，那条分支漏传 eval_batch_mult、save_steps、eval_steps 等参数，"
    "于是计划模式一直回落到 train.py 的默认 eval_batch_mult=4，评测显存偏高；"
    "而 save_steps/eval_steps 虽然由计划回调在运行时写入 args，但在回调第一次生效之前，"
    "命令行默认值仍然存在。这类“两个分支各拼一半参数”是配置系统最常见的腐烂方式。"
    "修复在 plan 分支补上 --eval_batch_mult，默认从配置读、缺省 1；"
    "save/eval steps 继续由计划回调负责（它必须同时改 args 与 state，否则 DefaultFlowCallback 不认）。"
    "tests/test_regressions_extra.py 构造一份最小 plan 配置，断言 build_cmd 的输出里"
    "包含 --eval_batch_mult 且取值等于配置里的 2。",
)

entry(
    "E21", "recurrent 单步必须走快路径", "已修",
    [fn("retnet_hf/modeling_retnet.py", "MultiScaleRetention._recurrent_retention")],
    "recurrent 实现是逐 token 的 Python 循环：状态按 gamma 衰减、加上当前 token 的 k⊗v、"
    "再与 q 求内积。训练不用它，但解码每个 token 都会走一次；长序列 prefill 若误用 recurrent，"
    "循环次数就是序列长度。修复增加 tgt_len==1 的快路径：解码时每次只有一步，"
    "直接做一次状态更新与一次内积，省掉 range 循环、list 追加与 torch.stack 的物化。"
    "数学与循环版逐位相同，因此不需要新的容差。"
    "tests/test_regressions_extra.py 用随机状态手写一步递推，与快路径的输出、新状态逐值比较，"
    "确保“优化”没有改语义；如果哪天有人把快路径写错（例如漏乘 gamma 或错位 q/k/v），这个测试会红。",
)

entry(
    "E22", "交叉熵必须分块，不能整段转 fp32", "已修",
    [fn("retnet_hf/modeling_retnet.py", "_chunked_weighted_ce"),
     fn("retnet_hf/modeling_retnet.py", "ce_chunk_tokens"),
     span("retnet_hf/modeling_retnet.py", "docs/00 E22：不再对整段 logits",
          "DEFAULT_CE_CHUNK_BYTES))")],
    "旧 forward 在非融合路径上先 logits_f = logits.float()，再 reshape 求交叉熵。"
    "对本项目（B=16, L=2048, V=20000）这一步会物化约 2.6 GB 的 fp32 副本，"
    "而 logits 本身（bf16）已经占 1.3 GB，是显存峰值的主要来源。"
    "第一版修复把损失计算抽成 _chunked_weighted_ce，但把 chunk 写死成 262144 个 token："
    "这个数与词表大小无关、且大于生产 token 总数（32768），于是循环只跑一遍，"
    "2.6 GB 的整段拷贝原封不动——分块参数是“装饰”。"
    "现在的 chunk 由**字节预算**反推：ce_chunk_tokens(V, 64MB) = 64MB/(4V)，"
    "V=20000 时 838 token/块，生产形状切出 40 块；每个分块只把 [chunk,V] 转 fp32。"
    "权重码语义与旧实现完全一致（pad 不计、reply/plain=1、prompt=prompt_loss_weight，"
    "最后除以权重和），没有 wcodes 时退化为普通均值。"
    "验证分两层：数值上，test_chunked_ce_matches_naive_weighted 要求与整段 fp32 的"
    "朴素实现 allclose（atol=1e-5）；结构上，"
    "test_chunked_ce_splits_at_production_token_count 在 n=32768 的形状上记录 "
    "F.cross_entropy 实际收到的切片，断言块数 >= 2、每块不超过预算、且总 token 数覆盖完整。"
    "旧测试传 chunk_tokens=7，即使参数被完全无视也会通过——这就是“只测数值不测结构”的漏洞。",
)

entry(
    "E23", "每个修复都要有一个“改回去就会红”的测试", "已修",
    [fn("tests/test_growth_trainable.py", "test_growth_guard_fails_when_reverted_to_zeros")],
    "这一条不是代码错误，是流程缺口：仓库原有 11 个测试，没有一个检查“参数是否有梯度”，"
    "所以 338M 死参数能一路通过所有检查。修复新增 6 个测试文件，每个都对应 00 里的一个编号，"
    "并且都设计成“把修复改回旧行为就会失败”：test_growth_trainable 用 monkeypatch 把新行随机改成 0，"
    "断言生长脚本拒绝落盘；test_no_decay_1d 同时测开启与关闭两个方向；test_val_disjoint 测三种路径写法；"
    "test_eos_present 测无边界/有边界/显式放行；test_parallel_chunkwise_equiv 测两条路径的数值一致；"
    "test_live_params 要求零参数统计逐元素精确。"
    "这段引用的是其中最关键的一个：它证明 E04 的守卫真的会失败，而不是一个永远为真的装饰。",
)

entry(
    "N01", "reinit_gamma 必须按头，而不是按层", "已修",
    [fn("scripts/reinit_gamma.py", "build_gamma"),
     span("scripts/reinit_gamma.py", "keys = sorted(k for k in sd", "拒绝写入")],
    "旧实现取 20 个 blocks.*.msr.gamma 张量，把张量个数当成头数，循环 h=0..19 生成 1-2^(-5-h)，"
    "再用 torch.full_like 灌满每层 22 个头。结果不是“按头设置”，而是“按层设置”，"
    "并且同一层 22 个头完全相同，多尺度衰减被抹成一个尺度；它打印的“旧值 0.399..0.586”"
    "还是每层 22 头的均值，把真实的 0.106..0.999 抹平了，连诊断都是错的。"
    "修复把核心抽成 build_gamma(num_heads, formula, tau_min, tau_max)：按头索引生成半衰期在对数空间"
    "均匀分布的 gamma，默认 128..8192 token（下限对齐 chunk=128，上限对齐 8K 窗口）。"
    "写入前断言同一层内的取值互不相同——一旦重复就说明又退化成按层设置，拒绝落盘。"
    "本地 dry-run 实测：440 个 gamma、每层 22 个不同值、旧半衰期 p50=1.22、新范围 128–8192。",
)

entry(
    "N02", "gamma 的“中位数”不能用死头混算", "文档订正（代码见 N01）",
    [span("scripts/reinit_gamma.py", "旧实现把 20 个", "所以连诊断都是错的")],
    "00 里写的“gamma 中位数 0.5656”是把 22 个头混在一起算的，而其中 10 个头是死的"
    "（值固定在 0.213622，纯 wd 衰减、零梯度）。用这个数描述“模型的记忆有多长”会被死头带偏；"
    "按活头算，ep3 的中位是 0.848（半衰期 4.21 token），v21 是 0.693（1.89 token）。"
    "这不是计算错误，是口径错误：混算把两种完全不同的对象（学过的、从没学过的）平均成一个数。"
    "修复不改数值逻辑，而是把 reinit_gamma 的文档写清楚：判断“记忆被 wd 吃掉多少”必须只看活头，"
    "判断“整体分布有多碎”才看全部 440 个。今后任何引用 gamma 中位数的报告，"
    "都必须同时给出“全部/活头”两个口径，否则结论不可比。",
)

entry(
    "N03", "层 18/19 是 d1024，不是 d1408", "文档订正（无代码）",
    [],
    "恢复计划里写“层 18/19 全宽活（d1408）”，据此推断“缩核不可行”。"
    "逐张量统计的实测结果是：q_proj 的输入活列 1024、输出活行 1024，v_proj 输入 1024，"
    "out_proj 输入 2048；输入列 1024–1407 的 absmax 精确为 0。也就是说层 18/19 是 d1024，"
    "不是 d1408；报告里看到的 q[768:] 非零来自 768–1023 这一段。"
    "这条只改依据、不改结论：模型仍是 18×d768 + 2×d1024 的混合宽度，"
    "单一 RetNetHFConfig 同样表达不了，所以“缩核不可行”依然成立。"
    "但它影响上机第一步的核对：如果按“活参数应高于 166M”去对账，会发现对不上，"
    "反而可能误判成“切片解读反了”。今后引用层宽必须按张量统计，不能按 q[768:] 非零推断。",
)

entry(
    "N04", "评测必须走训练用的那条路径", "已修",
    [span("scripts/eval_holdout.py", "2026-10-20（docs/00 N04/E16）", "flush=True)")],
    "旧 eval_holdout 强制 model.config.mode=\"parallel\"，理由写在注释里："
    "“parallel 与 chunkwise 在整段前向上数学等价”。但 E16 证明当时 parallel 多做两次归一化，"
    "两者并不等价。于是上一轮所有 held-out 数字（3.6287 / 2.8471 / 3.5023）"
    "测的都是一个和训练不同的函数；用它当判据，等于拿别人的成绩给自己打分。"
    "修复把评测切到 chunkwise，并把 chunk_size 与训练值对齐（新增 --chunk-size，默认 128）。"
    "注释里提到 chunkwise 在 bf16 下有 dtype 冲突——如果真实存在，应当修内核，"
    "而不是换一条不等价的路径来测。E16 删掉那两处归一化之后，两条路径只该差浮点噪声，"
    "tests/test_parallel_chunkwise_equiv.py 会持续守住这一点。",
)


# ======================================================================
# 每条再补一段"怎么证明修对了"——不是凑字数，是审核时真正要对着跑的东西
# ======================================================================
EVIDENCE = {
    "E01": "验证方式：`tests/test_growth_trainable.py::test_growth_guard_fails_when_reverted_to_zeros` "
           "把 `NEW_ROW_INIT_STD` 改回 0，断言生长脚本返回 2 且不落盘；正常用例要求 rc∈{0,1} 且 "
           "`model.safetensors` 存在。旧权重的实测基线是 338,120,704 个元素精确为 0，占 67.06%。",
    "E02": "验证方式：同一测试把 `BLOCK_GATE` 改成 0，覆盖“中间层门控归零”的旧行为；正常路径要求 "
           "final_norm 的 `gamma_new` 全 0、中间层全 1。旧模型实测 41 个 `gamma_new` 张量全部为 0，"
           "这正是“宽度生长从未生效”的直接证据。",
    "E03": "验证方式：生长后统计 embedding 的列，前 `grow_old_dim` 列必须非零，其后 640 列不得全零。"
           "旧模型实测 embedding 形状 [20000,1408]，后 640 列 absmax 精确为 0，"
           "活跃列只有 0–767，与“新维度没有信号源”完全吻合。",
    "E04": "验证方式：`test_growth_trainable.py` 的两个用例一正一反，正向要求校验通过并落盘，"
           "反向要求校验失败并拒绝落盘。旧脚本只有 `max|Δlogits|` 一条，任何恒零旁支都能通过，"
           "所以这条反例本身就是“旧校验不可能失败”的证明。",
    "E05": "验证方式：`tests/test_regressions_extra.py::test_growth_ln_new_gate_defaults_to_one` 直接断言"
           "构造出的 `GrowthLayerNorm.gamma_new` 全 1、`beta_new` 全 0。旧模型里这 41 个张量全零，"
           "而类默认值本身是 1——说明破坏发生在生长脚本覆写的那一步，而不是类设计。",
    "E06": "验证方式：`tests/test_no_decay_1d.py` 测两个方向——开启时任何 1 维参数都不得出现在 decay 集合，"
           "关闭时 `msr.gamma` 必须出现。旧权重实测 gamma 中位 0.5656、半衰期 1.22 token、"
           "96.1% 的头短于 128 token，这三组数就是“开关没开”的后果。",
    "E07": "验证方式：`tests/test_grad_coverage.py` 四个用例——健康模型零死参数；把 "
           "`blocks.1.msr.q_proj.weight` 的 requires_grad 关掉后它被判死；阈值 0 时 "
           "`on_step_end` 抛 SystemExit；检查后钩子被摘掉。`on_step_end` 是真实注册到 "
           "Trainer 的回调，删掉 `trainer.add_callback(GradCoverageCallback(...))` 或删掉回调类，"
           "这组测试会红。旧状态是：注释声称“另加一次梯度覆盖检查”，`grep grad.abs` 只在 "
           "`scripts/grow_500m.py` 找到（那是生长脚本的 5 步探针，不是训练期检查）。",
    "E08": "验证方式：`tests/test_live_params.py` 三个用例——新模型比例 >99%、整块清零后比例下降、"
           "已知张量清零一半后 live 数精确减少对应元素数。"
           "`tests/test_preflight_live.py` 再覆盖判定边界与真实路径：比例 0.33 无开关时拒绝、"
           "带 --allow_dead_params 放行、与基线一致放行、比基线更死拒绝；use_gla=False（gamma 是 buffer）"
           "时 gamma<=0 仍被拒绝。旧代码只打印 sum(p.numel())，并在 67% 为零时仍输出 504,179,896。",
    "E09": "验证方式：`tests/test_build_scripts.py::test_add_eos_is_default` 直接跑命令行脚本，"
           "不传 `--add-eos`，然后断言产出分片里 eos(2) 的数量不少于记录数、且 `meta.json` 的 "
           "`add_eos` 为 true。旧包全部是 false，v21 只有第 0 段（由 `prepare_pretrain_v2.py` 显式加 eos）"
           "有边界，后 124,652 步（78%）都在无边界流上训练。",
    "E10": "验证方式：`tests/test_build_scripts.py::test_wrong_text_field_fails_fast` 传一个不存在的字段名，"
            "要求进程非零退出且输出里出现“一个都不存在/可用字段”；旧行为要等分词跑完才以 >1% 错误率失败。"
            "实证据是 `prog_zh` 的 10 个数值列被 `auto` 选中，这类污染不报错、只降低质量。",
    "E11": "验证方式：`tests/test_build_scripts.py` 两个用例——没有 `meta.json` 的目录必须非零退出且"
            "出现“无法确认文档边界”；加 `--allow-boundaryless` 时不得出现这类拒绝对话、且要打印显式警告。"
            "旧脚本在 docstring 里承认边界丢失，却仍然用它造 16K 包，这是“知道缺陷还固化缺陷”的典型。"
            "该开关会写进 README，任何一次使用都必须在实验记录里注明理由与影响范围，"
            "并同时报告该包占总训练 token 的比例，方便事后判断影响有多大。",
    "E12": "验证方式：`tests/test_plan_eos.py` 从 plan JSON 出发走完整链路——`require_eos` 默认 True、"
           "写 false 后 `StagePackedDataset.stage_ready(0)` 真的就绪、没写时仍然拒绝且错误信息里的键名"
           "就是 `require_eos: false`、热更新不会丢这个键；`tests/test_eos_present.py` 另测 "
           "`_FlatStage` 自身的三态。旧实现里 Stage 没有这个字段，parse_plan 直接丢掉它，"
           "任何测试都摸不到这条链路。",
    "E13": "验证方式：`tests/test_producer_gc.py` 跑**真实产线**——现场训练一个小 tokenizer、执行 "
           "build_flat_corpus、读它写出的 meta、调 gc_stage，断言 freed>0 且分片被删；"
           "另测旧 meta（只有 sources+text_fields）仍可删、无任何证明仍拒绝、protect 过的目录拒绝。"
           "旧测试（test_plan_gc 4b）手写了一个真实的产线从不生成的 meta，"
           "所以“可重建”这件事在真实数据上从未成立过。",
    "E14": "验证方式：`tests/test_val_disjoint.py` 用真的不同写法（os.path.relpath、尾斜杠、大小写）"
           "逐一触发拒绝；用两份 meta.json 声明同源（目录名不同）触发声明层；"
           "把验证集两行复制进训练集触发内容层；并断言验证集侧读到的是 val.ids.npy。"
           "旧测试的“另一种写法”只是同一个绝对路径，等于没测。",
    "E15": "验证方式：`tests/test_train_config_guards.py::test_best_model_selection_is_disabled` 断言源码里"
            "`load_best_model_at_end=False` 存在、且不得再出现 `=True`；离线选模型改用 `eval_holdout.py` "
            "在 `pretrain_v2/*_val.npy` 上跑。旧行为对应的事实是：最终模型域内 held-out 3.5023，"
            "比 step 17,500 的 2.8471 差 0.655 nat。",
    "E16": "验证方式：`tests/test_parallel_chunkwise_equiv.py` 比较 parallel vs chunkwise、"
            "parallel vs recurrent，容差按 logits 尺度设相对阈值。旧实现实测 "
            "mean|Δlogit|=0.0158、argmax 一致率 99.66%，而评测脚本强制 parallel，"
            "所以那三组 held-out 数字测的不是训练的函数。",
    "E17": "验证方式：`tests/test_train_config_guards.py` 走行为而不是字面量——"
            "`test_save_total_limit_is_one_shared_constant` 断言 train.py 的 parser 默认等于 "
            "`retnet_hf/defaults.py` 的常量；"
            "`test_run_task_passes_save_total_limit_in_both_branches` 对 plan / 非 plan 两个分支、"
            "默认值 / 6 两个取值都断言 argv 里的数字等于配置值。"
            "旧代码在 plan 分支根本不传这个参数，非 plan 分支兜底成 3，而旧测试只正则 train.py，"
            "所以 E17 的原始缺陷（配置写 4、实际跑 3）原样存在却全绿。",
    "E18": "验证方式：`tests/test_plan_reschedule.py` 四个用例——延长计划（LR 上跳）必须拒绝；"
            "加 `allow_reschedule: true` 后放行；缩短计划（LR 只变小）放行；总步数不变放行。"
            "判定被抽成纯函数 `should_accept_reschedule`，测试直接构造两个 Plan，不需要启动 Trainer。"
            "旧行为对应的是 `plan_ddp2.json` 里 8 次 `_readme` 热改记录与 `lr` 列的跳变。"
            "这条规则也会写进计划 schema 的示例与 README，避免下次又靠日志里的“余弦重铺”去发现。",
    "E19": "验证方式：`tests/test_regressions_extra.py::test_refresh_detects_same_size_replacement` "
            "关闭 mmap、原地写同尺寸新内容、把 mtime 推到未来，断言 refresh 返回 True、第二次返回 False。"
            "第一版修复在比较时重新 stat，新旧两边永远相等，测试当场抓到了这个错误。",
    "E20": "验证方式：`tests/test_regressions_extra.py::test_build_cmd_plan_passes_eval_batch_mult` "
            "构造最小 plan 配置，断言 `build_cmd` 输出含 `--eval_batch_mult` 且等于配置值 2。"
            "旧 plan 分支漏传它，实际一直回落 `train.py` 的默认 4，评测显存因此偏高。",
    "E21": "验证方式：`tests/test_regressions_extra.py::test_recurrent_single_step_matches_manual` "
            "用随机状态手写一步递推，与快路径的输出、新状态逐值比较（atol=1e-5）。"
            "解码时 `tgt_len` 恒为 1，旧实现每个 token 都走一次 Python 循环与 `torch.stack`。",
    "E22": "验证方式：数值层——`test_chunked_ce_matches_naive_weighted` 与 "
            "`test_chunked_ce_without_weights_is_plain_mean` 要求与整段 fp32 的朴素实现 allclose"
            "（atol=1e-5）；结构层——`test_ce_chunk_size_is_derived_from_a_byte_budget` 断言 "
            "V=20000 时每块 838 token、V 越大块越小；"
            "`test_chunked_ce_splits_at_production_token_count` 在 n=32768 上记录 F.cross_entropy "
            "实际收到的切片，断言块数 >= 2、每块不超过预算、总量覆盖完整。"
            "只有数值层时，把 chunk 写死成 262144（即生产形状只切一块）也能通过。",
    "E23": "验证方式：`cd code && python -m pytest tests -q` → **82 passed, 4 skipped**，"
            "两个脚本式测试 `python tests/test_plan_gc.py` / `test_plan_pipeline.py` 单独跑也全过。"
            "4 个 skip 是 test_build_scripts 在缺 `data_cache/chat_pack_v13/bpe_tokenizer.json` 时的"
            "显式 skip，不是假通过。更关键的是反向验证："
            "`test_growth_guard_fails_when_reverted_to_zeros` 把初始化改回全零时必须 rc=2；"
            "`test_grad_coverage` 冻结参数时必须停机；`test_plan_eos` 去掉 require_eos 透传时必须红。"
            "计数以本快照实测为准，不再引用远端环境的历史数字。",
    "N01": "验证方式：`tests/test_gamma_reinit.py` 三个用例——22 个头的取值必须互不相同（重复即退化成按层）、"
            "半衰期单调且落在 128–8192 区间、非法 tau 范围必须抛 ValueError；"
            "再用 `reinit_gamma.py --dry-run` 在真实 checkpoint 上输出“20 层 × 22 头 = 440 个 gamma”与新旧分布。"
            "旧实现输出的是每层均值 0.399..0.586，真实值域 0.106..0.999 被抹掉。",
    "N02": "验证方式：分别按全部 440 个与只按活头统计 gamma 中位数，两个数必须同时出现在报告里。"
            "实测 ep3 活头中位 0.848（半衰期 4.21 token）、v21 0.693（1.89 token），"
            "而死头固定 0.213622——把两者平均得到的 0.5656 不能用来描述记忆长度。"
            "报告模板里已经把“全部/活头”两个口径列为必填项，缺一个就算数据不完整。",
    "N03": "验证方式：对 `checkpoints/retnet_500m_v21_export` 逐张量统计活行/活列。"
            "实测 L18/L19 的 q_proj 输入活列 1024、输出活行 1024，v_proj 输入 1024，out_proj 输入 2048；"
            "输入列 1024–1407 的 absmax 精确为 0，故层宽是 1024 而不是 1408。",
    "N04": "验证方式：`tests/test_train_config_guards.py::test_eval_uses_chunkwise` 断言源码里 "
            "`model.config.mode = \"chunkwise\"` 存在、`\"parallel\"` 不存在、且带 `--chunk-size`；"
            "`test_eval_has_both_probes` 断言两个探针开关存在；"
            "`tests/test_eval_probes.py` 用 tiny 模型端到端跑一次，检查 by_position 的 8 个桶、"
            "ctx_ablation 的 Δ_ctx 与 token 计数，以及 ctx 不是 chunk 倍数时提前拒绝。"
            "旧行为下 parallel 多两次归一化，所有 held-out 数字与训练函数错位，判据必须转 delta 才有意义。",
}

for _e in ENTRIES:
    _extra = EVIDENCE.get(_e["id"], "")
    if _extra:
        _e["text"] = _e["text"] + "\n\n" + _extra

# English prose overrides every entry's title and explanation. If the file is
# present but incomplete, fail instead of emitting a half-Chinese document.
if I18N.exists():
    _en = {e["id"]: e for e in json.loads(I18N.read_text(encoding="utf-8"))}
    _missing = sorted({e["id"] for e in ENTRIES} - set(_en))
    if _missing:
        raise SystemExit(f"doc_i18n/fix_en.json is missing entries: {_missing}")
    for _e in ENTRIES:
        _e["title"] = _en[_e["id"]]["title"]
        _e["text"] = _en[_e["id"]]["text"]


# ======================================================================
# 生成
# ======================================================================

# 错误编号 -> 相关论文（编号对应 Papers/ 文件前缀）。只标真正构成依据的。
CITE_E: dict[str, list[str]] = {
    "E01": ["07", "08", "09", "10"],
    "E02": ["08", "09", "13"],
    "E03": ["07", "09"],
    "E04": ["10", "11"],
    "E05": ["08", "13"],
    "E06": ["03"],
    "E09": ["16"],
    "E10": ["16"],
    "E11": ["16"],
    "E12": ["16"],
    "E16": ["01", "04"],
    "E21": ["04"],
    "E23": ["10", "11"],
    "N01": ["03"],
    "N02": ["03"],
    "N04": ["01", "04"],
}


def resolve(ref):
    kind = ref[0]
    path = ROOT / ref[1]
    if not path.exists():
        raise SystemExit(f"[ref] 文件不存在: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if kind == "fn":
        got = find_function(path, ref[2], lines)
        if not got:
            raise SystemExit(f"[ref] 找不到函数 {ref[1]}::{ref[2]}")
        return path, lines, got[0], got[1]
    got = find_span(lines, ref[2], ref[3])
    if not got:
        raise SystemExit(f"[ref] 找不到锚点 {ref[1]}::{ref[2][:30]}..{ref[3][:30]}")
    return path, lines, got[0], got[1]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    doc00 = DOC00.read_text(encoding="utf-8")
    wanted = set(re.findall(r"\b(E\d{2}|N\d{2})\b", doc00))
    have = {e["id"] for e in ENTRIES}
    missing = sorted(wanted - have)
    extra = sorted(have - wanted)
    if missing or extra:
        print(f"coverage mismatch: missing={missing}, extra={extra}")
        return 1

    fails = []
    for e in ENTRIES:
        if len(e["text"]) < 200:
            fails.append(f"{e['id']} explanation is only {len(e['text'])} chars")
        for bad in ("省略", "同上", "omitted", "same as above"):
            if bad in e["text"]:
                fails.append(f"{e['id']} contains placeholder text: {bad!r}")
        if e["status"].startswith("Fixed") and not e["refs"]:
            fails.append(f"{e['id']} is Fixed but has no code reference")
    if fails:
        print("entry validation failed:")
        for f in fails:
            print("   ", f)
        return 1

    out = [
        "# 00F1 Fix matrix (per item: fixed code + explanation)",
        "",
        "> One entry per id in `00_error_table.md`. Code blocks are extracted from the",
        "> source by `scripts/_gen_fix_doc.py` with AST/anchors, never copied by hand;",
        "> each explanation is >=200 chars. `verify.py` re-checks the code blocks and",
        "> that the status table agrees in all four places.",
        "",
        "---",
        "",
    ]
    for e in ENTRIES:
        out.append(f"## {e['id']} {e['title']}")
        out.append("")
        out.append(f"**Status**: {e['status']}")
        out.append("")
        for ref in e["refs"]:
            path, lines, a, b = resolve(ref)
            rel = path.relative_to(ROOT).as_posix()
            out.append(f"`{rel}` L{a}–L{b}")
            out.append("")
            out.append("```python")
            out.append("\n".join(lines[a - 1:b]))
            out.append("```")
            out.append("")
        out.append(e["text"])
        ids = CITE_E.get(e["id"], [])
        if ids:
            out.append("")
            out.append(cite(*ids))
        out.append("")
        out.append("---")
        out.append("")

    out.append(refs_md())
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}: {len(ENTRIES)} entries, "
          f"{len(OUT.read_text(encoding='utf-8'))} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
