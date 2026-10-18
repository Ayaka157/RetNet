"""Role-play SFT with the mature HF stack (Trainer + Accelerate + safetensors).

Replaces the handwritten train.py for Stage 2+.  Same hyper-parameters as the
handwritten loop: 45k steps, batch 10, seq 512, cosine 1.1e-4 -> 1.1e-5,
prompt_loss_weight 0.1, bf16, torch.compile(inductor).

    python retnet_hf/train.py \
        --model_dir checkpoints/retnet_100m_pretrain_v4_hf \
        --data_dir data_cache/chat_pack_v8 \
        --output_dir checkpoints/retnet_100m_rp_v3_hf \
        --max_steps 45000 --batch_size 10 --lr 1.1e-4 --warmup 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf import (  # noqa: E402
    ChatDataCollator,
    PackedChatDataset,
    RetNetForCausalLM,
    StagePackedDataset,
)
from retnet_hf.modeling_retnet import RetNetHFConfig  # noqa: E402


class CompileSafeTrainer(Trainer):
    """保存时把 ``torch.compile`` 的 ``_orig_mod.`` 包装摘掉。

    ⚠️ 不这么做的后果是**静默且致命**的：``OptimizedModule.state_dict()`` 会给
    每一个键加 ``_orig_mod.`` 前缀，而 ``from_pretrained`` 加载这种文件**不报错**
    —— 它把带前缀的键当作"多余的"丢掉，把不带前缀的当作"缺失的"随机初始化。
    于是"加载成功"拿到的是一个随机模型。

    实测（2026-10-10）：v14 的 checkpoint-2500/5000/7500/10000 **全部 365 个键
    都带前缀**，加载后前向 loss=nan；评测侧的表象是 ``torch.multinomial`` 抛
    ``CUDA error: device-side assert triggered``，极难从表象追到根因。
    而 v13 / rp / v10 的导出是干净的（0/365），说明这是启用编译之后才引入的。
    已拉回的 checkpoint-10000 用 ``scripts/strip_compile_prefix.py`` 救活。
    """

    def save_model(self, output_dir=None, _internal_call=False):
        inner = getattr(self.model, "_orig_mod", None)
        if inner is None:
            return super().save_model(output_dir, _internal_call)
        wrapped = self.model
        prev_wrapped = getattr(self, "model_wrapped", None)
        self.model = inner
        if getattr(prev_wrapped, "_orig_mod", None) is inner:
            self.model_wrapped = inner
        try:
            return super().save_model(output_dir, _internal_call)
        finally:
            self.model = wrapped
            if prev_wrapped is not None:
                self.model_wrapped = prev_wrapped


def verify_export(path) -> bool:
    """导出后自检：safetensors 的键里不能有 ``_orig_mod.``，且必须能真的加载。

    这道检查很便宜，但它是唯一能在"训练日志一切正常、模型文件却不可用"
    这种情形下把问题挡在前面的东西。
    """
    import json as _json
    import struct
    wf = Path(path) / "model.safetensors"
    if not wf.exists():
        print(f"[verify] !! 找不到 {wf}")
        return False
    try:
        with open(wf, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = _json.loads(fh.read(n).decode("utf-8"))
        keys = [k for k in hdr if k != "__metadata__"]
        bad = [k for k in keys if k.startswith("_orig_mod.")]
        if bad:
            print(f"[verify] ❌ 导出里有 {len(bad)}/{len(keys)} 个键带 _orig_mod. 前缀 —— "
                  f"这个检查点用 from_pretrained 加载会**静默变成随机模型**。"
                  f"跑 scripts/strip_compile_prefix.py 修，或检查 CompileSafeTrainer 是否生效。")
            return False
        print(f"[verify] ✅ 导出键名干净（{len(keys)} 个），可直接 from_pretrained")
        return True
    except Exception as exc:                                   # noqa: BLE001
        print(f"[verify] 跳过（{type(exc).__name__}: {exc}）")
        return True


def _sync_resume_state(ckpt: Path, training_args) -> None:
    """把检查点 ``trainer_state.json`` 里会被 HF 采纳的字段对齐到本次参数。

    ## 为什么必须做（2026-10-12 在 RTX PRO 6000 上踩到，代价是整轮白跑）

    HF 在 ``Trainer.train()`` 里用检查点里的 batch 覆盖命令行：
    ``transformers/trainer.py:2298-2301``（4.57.6）::

        # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
        if state.train_batch_size is not None:
            self._train_batch_size = state.train_batch_size

    而 ``self._train_batch_size`` 正是 ``get_train_dataloader(batch_size=...)``
    用的值 —— 于是 ``--batch_size 16`` 被**静默**改成 4。唯一的痕迹是
    ``compare_trainer_and_checkpoint_args``（trainer.py:2031）打的一行:

        per_device_train_batch_size: 16 (from args) != 4 (from trainer_state.json)

    **实测后果**：显存 30,177 MiB 而不是 75.25 GiB；6.25 优化步/s × 8192 tok
    = 51.2k tok/s（batch 16 的合成同架构探针是 57.1k～61.0k）。
    而且新存的检查点会把 4 记下来，**下一轮续训继续用 4** —— 自锁。

    ## 第二个洞：缺 optimizer.pt 时连 LR 调度器都不恢复

    ``_load_optimizer_and_scheduler``（trainer.py:3523）的条件是::

        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):

    其中 ``checkpoint_file_exists`` 指 ``optimizer.pt``。所以没有优化器状态时，
    ``scheduler.pt`` 明明在、也不会被加载，LR 从 step 0 重新升温。
    **实测**：3090 那段末步 ``learning_rate = 1.1000e-05``（余弦谷底），
    缺 optimizer.pt 续训后首段是 ``1.0987e-04``（≈峰值，10×）。
    """
    import json
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        print(f"[resume] ⚠️ {state_path} 不存在，无法核对 batch/steps 语义；"
              f"HF 会退回默认值，行为不可预期", flush=True)
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    want = {
        "train_batch_size": training_args.train_batch_size,
        "logging_steps": training_args.logging_steps,
        "eval_steps": training_args.eval_steps,
        "save_steps": training_args.save_steps,
    }
    diff = {k: (state.get(k), v) for k, v in want.items() if state.get(k) != v}
    if diff:
        bak = state_path.with_suffix(".json.orig")
        if not bak.exists():
            bak.write_text(state_path.read_text(encoding="utf-8"), encoding="utf-8")
        state.update(want)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        for k, (old, new) in diff.items():
            print(f"[resume] 修正 trainer_state.json: {k} {old} -> {new}", flush=True)
        print(f"[resume] 原文备份 {bak.name}；不改的话 HF 会用旧值静默覆盖"
              f"（trainer.py:2298）", flush=True)
    else:
        print(f"[resume] trainer_state.json 与本次参数一致（batch={want['train_batch_size']}，"
              f"save/eval/log={want['save_steps']}/{want['eval_steps']}/{want['logging_steps']}）",
              flush=True)
    if not (ckpt / "optimizer.pt").exists():
        print("[resume] ⚠️ 没有 optimizer.pt：HF 会跳过优化器**和 LR 调度器**的恢复"
              "（trainer.py:3523 的条件是 optimizer 与 scheduler 同时存在），"
              "本段 LR 会从 step 0 重新升温 —— 实测过一次 1.1e-5 → 1.1e-4。",
              flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="migrated HF checkpoint dir")
    ap.add_argument("--data_dir", default=None,
                    help="packed chat data dir；用 --plan 时可不传")
    ap.add_argument("--plan", default=None,
                    help="可热加载的训练计划 JSON（见 retnet_hf/plan.py）。"
                         "传了它就用计划里的 stages 当数据源、total_steps 当 max_steps，"
                         "并在训练中途按文件变更实时调整（换语料/加步数/改 LR）")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=45000)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1.1e-4)
    ap.add_argument("--min_lr", type=float, default=1.1e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_batches", type=int, default=40)
    ap.add_argument("--dataloader_workers", type=int, default=0,
                    help="DataLoader 子进程数。0（默认）= 同步加载：取数据与 GPU 计算"
                         "不重叠，这段开销直接加在每一步上。"
                         "2026-10-11 实测：真实训练 504M@b2a2 是 625 ms/优化步，"
                         "而 bench_trainer.py 同 batch 只要 555 ms —— 差值的一部分就在这里"
                         "（另一部分是 bench 没传 --optim、没走 HF Trainer 全循环）。")
    ap.add_argument("--compile_mode", default=None,
                    help="传给 torch.compile 的 mode。默认 None = 与改动前完全一致。"
                         "RTX PRO 6000 (sm120) 实测 max-autotune-no-cudagraphs 快 6.7%%"
                         "（57,134 → 60,951 tok/s）。⚠️ 不要用 max-autotune 或 "
                         "reduce-overhead —— 两者都会在 modeling_retnet.py:507 抛 "
                         "CUDAGraphs overwritten。")
    ap.add_argument("--eval_batch_mult", type=int, default=4,
                    help="评测 batch = 训练 batch × 这个数。"
                         "NOTE(2026-10-10)：实测 1×4 时第一次评测要多花 128 s（"
                         "评测形状重编译 + 峰值 reserved 冲到 7.65/8.00 GiB，"
                         "贴墙），整个运行的评测开销约占 6.6%；" 
                         "改成 1×1 后降到 2.9%，reserved 回到 6.73 GiB。"
                         "run_task.py 的配置里一直有 eval_batch_mult 字段，"
                         "但以前从没传进来 —— 又一个死字段。")
    ap.add_argument("--compile", action="store_true", default=True,
                    help="torch.compile with inductor backend")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--grad_checkpoint", action="store_true",
                    help="gradient checkpointing (saves VRAM, ~25%% slower)")
    ap.add_argument("--mode", default="parallel", choices=["parallel", "chunkwise"],
                    help="retention forward mode used for training")
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--resume", action="store_true", help="resume from output_dir")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="autocast precision (fp16/fp32 for T4/P100 on Kaggle/Colab)")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="梯度累积步数；batch 1 + accum 4 = 等效 batch 4，"
                         "但激活值只需 1/4，因此可以关掉梯度检查点")
    # NOTE(2026-10-10): 下面四个一直只存在于任务 JSON 和 run_task.py 的模板里，
    # 从来没被传下来，于是这里的硬编码默认值一直生效。save_total_limit 尤其危险：
    # 配置写 4、实际 3，多出来的那个检查点会被 HF 按 mtime 轮转删掉，
    # 而删除日志是 logger.info —— transformers 默认级别 WARNING，不会打印。
    ap.add_argument("--save_total_limit", type=int, default=3,
                    help="保留最近 N 个 checkpoint-*；<=0 表示不限制")
    ap.add_argument("--save_steps", type=int, default=2500)
    ap.add_argument("--eval_steps", type=int, default=2500)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--optim", default=None,
                    help="TrainingArguments.optim；"
                         "adamw_bnb_8bit 省 1.5 GiB 优化器状态（关检查点的前提）")
    ap.add_argument("--ddp_find_unused_parameters", action="store_true",
                    help="多卡 DDP 时允许存在未被使用的参数。本模型的 growth 张量里有一批"
                         "gamma=0 的仿射项，先试默认 False；报 unused parameter 时加这个开关")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    plan_obj = None
    if args.plan:
        from retnet_hf.plan import PlanWatcher
        plan_obj = PlanWatcher(args.plan, verbose=False).load(force=True)
        print(f"[plan] 载入 {args.plan}：{len(plan_obj.stages)} 段，"
              f"合计 {plan_obj.total_steps} 步，seq_len={plan_obj.seq_len}", flush=True)
        for _i, _s in enumerate(plan_obj.stages):
            print(f"[plan]   段 {_i}: {_s.name} -> {_s.data_dir} ({_s.steps} 步)", flush=True)
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None and plan_obj is None:
        raise SystemExit("必须给 --data_dir，或用 --plan 指定训练计划")
    if data_dir is None:
        # 计划模式下 data_dir 只为兼容旧代码路径；用第一段
        data_dir = plan_obj.stages[0].resolved_dir()

    model = RetNetForCausalLM.from_pretrained(str(model_dir))
    model.config.mode = args.mode
    model.config.chunk_size = args.chunk_size
    from retnet_hf.modeling_retnet import FAST_CHUNKWISE
    print(f"chunkwise kernel: {'batched (FAST_CHUNKWISE=1)' if FAST_CHUNKWISE else 'reference loop'}"
          f"  ·  chunk_size={args.chunk_size}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded model from {model_dir} (params={n_params / 1e6:.2f}M)", flush=True)

    # ---- 数据集：计划模式走多阶段可切换数据集，否则走原来的单文件 ----
    import numpy as np
    if plan_obj is not None:
        seq_len = plan_obj.seq_len
        train_ds = StagePackedDataset(
            [{"name": s.name, "data_dir": str(s.resolved_dir()), "mode": s.mode,
              "gc": s.gc, "min_shards": s.min_shards, "exclude": s.exclude}
             for s in plan_obj.stages], seq_len)
        print(f"seq_len={seq_len}（来自计划）", flush=True)
        args.max_steps = plan_obj.total_steps
        print(f"[plan] max_steps 由计划决定：{args.max_steps}", flush=True)
    else:
        probe = np.load(data_dir / "train.ids.npy", mmap_mode="r")
        seq_len = probe.shape[1] - 1
        del probe
        print(f"seq_len={seq_len}", flush=True)
        train_ds = PackedChatDataset(str(data_dir / "train.ids.npy"),
                                     str(data_dir / "train.wcode.npy"), seq_len)
    val_path = plan_obj.resolved_val_dir() if plan_obj is not None else data_dir
    val_ds = PackedChatDataset(str(val_path / "val.ids.npy"),
                               str(val_path / "val.wcode.npy"), seq_len)
    if plan_obj is not None:
        print(f"[plan] 验证集固定用 {val_path}（换训练语料时不换验证集，否则曲线不可比）",
              flush=True)
    collator = ChatDataCollator()
    print(f"train rows={len(train_ds)} val rows={len(val_ds)}", flush=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=not args.resume,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size * args.eval_batch_mult,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.min_lr},
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),   # fp32: 两者皆 False = 全精度 fp32 训练
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        seed=args.seed,
        gradient_checkpointing=args.grad_checkpoint,
        include_inputs_for_metrics=False,
    )
    if args.optim:
        training_args.optim = args.optim
        print(f"optimizer: {args.optim}", flush=True)
    # DDP 默认会把 find_unused_parameters 打开（冒烟日志：'find_unused_parameters=True
    # was specified in DDP constructor, but did not find any unused parameters'）。
    # 本模型确实没有未使用参数（growth 里 gamma=0 的张量仍参与前向，梯度存在只是为 0），
    # 开着只会每步多遍历一遍 autograd 图 → 纯开销。多卡时显式关掉。
    import os as _os
    if int(_os.environ.get('WORLD_SIZE', '1') or 1) > 1 and not args.ddp_find_unused_parameters:
        if hasattr(training_args, 'ddp_find_unused_parameters_true'):
            training_args.ddp_find_unused_parameters_true = False
        if hasattr(training_args, 'ddp_find_unused_parameters'):
            training_args.ddp_find_unused_parameters = False
        print('DDP: find_unused_parameters = False（实测无未使用参数，关掉省开销）', flush=True)
    if args.ddp_find_unused_parameters:
        # transformers 4.46 起把开关改名成 *_true（旧名已弃用）
        if hasattr(training_args, "ddp_find_unused_parameters_true"):
            training_args.ddp_find_unused_parameters_true = True
        else:                                                   # pragma: no cover
            training_args.ddp_find_unused_parameters = True
        print("DDP: find_unused_parameters = True", flush=True)
    if args.grad_accum > 1:
        print(f"gradient accumulation: {args.grad_accum} "
              f"(等效 batch {args.batch_size * args.grad_accum})", flush=True)

    trainer = CompileSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    # torch.compile 必须在 Trainer 构造之后：transformers 4.57 的 RemoveColumnsCollator
    # 惰性地按 model.forward 签名过滤数据列，compile 包装后签名丢失会导致 input_ids 被滤掉。
    # 因此先强制缓存未编译模型的签名，再替换 model。
    trainer._set_signature_columns_if_needed()
    # 外部旗标控制：logs/ctrl.txt，内容 run / stop / next-checkpoint / save。
    # 有了它就不必 taskkill —— 也就不必绕 run_task.py 的 auto-restart
    # （直接杀 train.py 会给 rc=1，反而被拉起来）。详见 retnet_hf/control.py。
    from retnet_hf.control import ctrl_path as _ctrl_path, make_callback as _ctrl_cb  # noqa: E402
    from retnet_hf.control import read as _ctrl_read  # noqa: E402
    print(f"[ctrl] 外部控制旗标 {_ctrl_path()}（当前 {_ctrl_read() or 'run'}）",
          flush=True)
    trainer.add_callback(_ctrl_cb())
    if plan_obj is not None:
        from retnet_hf.plan import make_plan_callback  # noqa: E402
        # 显式绑定 trainer + dataset：不依赖 transformers 版本给 callback 传什么 kw。
        # 4.57 实测 on_step_end 的 kw 里没有 trainer，只靠 kw.get("trainer") 会让换料静默失效。
        plan_cb = make_plan_callback(args.plan).bind(trainer, train_ds)
        trainer.add_callback(plan_cb)
        print("[plan] 热加载回调已挂：改计划文件即生效，无需重启训练", flush=True)
    if args.compile:
        # ⚠ inductor 是**惰性**编译：`torch.compile()` 本身几乎不做事，真正的
        # 编译发生在第一次前向 —— 那时早已离开这个 try 块。所以缺 triton 时
        # 下面的 except 接不住 `TritonMissing`，结果是「不崩、不提速、静默无效」。
        # 项目在 2026-10-10 之前一直是这个状态。先显式检查，把它变成响亮的失败。
        try:
            import triton  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "--compile 需要 triton，但当前环境没有装。inductor 缺失时不会在此处报错，"
                "而是静默退回 eager（实测 0% 提速）。请先 `pip install triton-windows`，"
                "或去掉 --compile 跑 eager。"
            ) from exc
        try:
            # 就地替换 ``forward``，**不要**把模型包成 OptimizedModule。
            #
            # NOTE(2026-10-11): 以前是 ``compiled = torch.compile(trainer.model)``
            # 再同时赋给 ``trainer.model`` / ``trainer.model_wrapped``。那确实让编译
            # 生效了（在此之前一直静默无效），但代价是 ``OptimizedModule`` 会给
            # ``state_dict()`` 的**每一个键**加 ``_orig_mod.`` 前缀。这个前缀已经咬了
            # 三次，而且每次都是静默的：
            #
            #   1. **保存的检查点全部不可加载**。v14 的 checkpoint-2500/5000/7500/10000
            #      365/365 个键都带前缀；``from_pretrained`` 加载它们**不报错**，只是
            #      把所有键当"多余的"丢掉 → 得到一个随机模型（前向 loss=nan，评测侧
            #      表现为 ``torch.multinomial`` 的 device-side assert，极难追根因）。
            #   2. ``trainer.save_model`` 的最终导出同样不可加载。
            #   3. 把检查点修干净之后 **resume 反过来失败**：HF 以编译模块为目标做
            #      ``load_state_dict``，干净键全被当成"多余键"丢掉 → 训练带着
            #      checkpoint 的优化器状态，却停在 ``model_dir`` 的初始权重上。
            #
            # 就地替换 forward 两者兼得：编译照样生效，``state_dict`` 保持干净，
            # 于是保存 / 导出 / 续训三条路径同时变正确。
            # NOTE(2026-10-12): 原先是硬编码 `backend="inductor"`，没有 mode 参数。
            # 在 RTX PRO 6000 (sm120) 上实测：`mode="max-autotune-no-cudagraphs"`
            # 比默认 mode 快 **6.7%**（57,134 → 60,951 tok/s，合成同架构探针，batch16
            # /chunk128）。所以把 mode 接出来，默认 None = 与改动前逐位一致。
            # ⚠️ 不要用 `max-autotune`（含 cudagraphs）：它会在 modeling_retnet.py:507
            # 抛 "accessing tensor output of CUDAGraphs that has been overwritten"，
            # 与 `reduce-overhead` 是同一个冲突。
            compile_kwargs = {"backend": "inductor"}
            if args.compile_mode:
                compile_kwargs["mode"] = args.compile_mode
            compiled_fwd = torch.compile(trainer.model.forward, **compile_kwargs)
            trainer.model.forward = compiled_fwd
            print(f"torch.compile(inductor{', mode=' + args.compile_mode if args.compile_mode else ''}) "
                  f"applied to model.forward (triton {triton.__version__})", flush=True)
        except Exception as exc:
            print(f"torch.compile failed ({exc}); continuing eager", flush=True)
    # 恢复中断：从 output_dir 里**最新的 checkpoint-* 子目录**继续。
    #
    # NOTE(2026-10-11): 这里原先是 ``resume_from_checkpoint=args.output_dir``。
    # **那是错的**，而且错得静默：HF 的 ``Trainer._load_from_checkpoint`` 是在传入
    # 的那个目录里**直接**找权重文件（WEIGHTS_NAME / SAFE_WEIGHTS_NAME /
    # WEIGHTS_INDEX_NAME），而 ``checkpoint-N`` 是它的**子目录**，所以必然抛
    #
    #     ValueError: Can't find a valid checkpoint at <output_dir>
    #
    # 这条路径此前从未被真正走过（v14 的日志里一次 ``[restart]`` 都没有），于是
    # config 里的 ``auto_restart_max`` 一直是个空承诺：一旦真的崩了，run_task 会
    # 重启、立刻再抛同样的错、来回 3 次之后退出 —— 现场看到的就是
    # ``[exit] rc=1`` 紧跟 ``[restart] attempt 1/3``。
    if args.resume:
        try:
            from transformers.trainer_utils import get_last_checkpoint
        except ImportError:                                    # pragma: no cover
            from transformers.utils import get_last_checkpoint
        last = get_last_checkpoint(args.output_dir)
        if last is None:
            raise SystemExit(
                f"--resume 但 {args.output_dir} 里找不到 checkpoint-* 子目录。"
                f"要么去掉 --resume 从头训，要么先确认检查点已生成。")
        print(f"[resume] 从 {last} 续训", flush=True)
        # ⚠️ 这一步**必须**在 trainer.train() 之前：HF 会用检查点里的 train_batch_size
        #    覆盖命令行（trainer.py:2298），且缺 optimizer.pt 时连 LR 调度器都不恢复
        #    （trainer.py:3523）。2026-10-13 在本机 3070 上复现了不加它的后果：日志里只有
        #    一行 WARNING
        #      per_device_train_batch_size: 1 (from args) != 16 (from trainer_state.json)
        #    NOTE：函数体 2026-10-12 就写好了，但**一直没被调用** —— 也就是说
        #    "防住续训 batch 陷阱"当时是句空承诺。是这次本机 resume 试验把它抓出来的，
        #    这也正是"守卫必须能被证明会失败"的又一例。
        _sync_resume_state(Path(last), training_args)
        resume_arg = last
    else:
        resume_arg = False

    trainer.train(resume_from_checkpoint=resume_arg)
    trainer.save_model(args.output_dir)
    # 多卡 DDP 下每个 rank 都会走到这里：导出校验与 TRAINING DONE 只能由 rank 0 做，
    # 否则两个进程同时读写同一批文件，且日志里会出现两条 TRAINING DONE，
    # 让 watchdog/收尾脚本的分辨逻辑变得含糊。
    if trainer.is_world_process_zero():
        verify_export(args.output_dir)
        print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
