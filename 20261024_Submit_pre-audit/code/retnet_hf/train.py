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
from transformers import TrainingArguments, Trainer, TrainerCallback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet_hf import (  # noqa: E402
    ChatDataCollator,
    PackedChatDataset,
    RetNetForCausalLM,
    StagePackedDataset,
)
from retnet_hf.modeling_retnet import RetNetHFConfig  # noqa: E402
from retnet_hf.defaults import DEFAULT_NO_DECAY_1D, DEFAULT_SAVE_TOTAL_LIMIT  # noqa: E402


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

    def get_decay_parameter_names(self, model) -> list[str]:
        """哪些参数要施加 weight decay。默认规则只排除 bias 与 LayerNorm.weight，
        而 `msr.gamma` 是**裸 nn.Parameter**，两头都不沾，于是被 wd=0.1 一路拉小。

        实测代价（2026-10-18，subagent 从 safetensors 里读出来的）：
        gamma 从初值 1-2^(-5-h) 降到 0.667(ep1) → 0.601(ep2) → 0.543(ep3)，
        每 epoch 稳定 ×0.90 —— 这正是 AdamW 解耦权重衰减的固定衰减率，不是学出来的；
        10 个头的 gamma 在 20 层里**逐位相同**（0.542977±1e-6）就是铁证。
        后果：半衰期从 22–1400 token 掉到约 1 token，**有效上下文只剩约 500 token**，
        2048 窗口后 3/4 用不上。

        修法：所有 **1 维参数**（各种 norm 的 scale、门控、衰减）都不做衰减。
        用 ``--no_decay_1d`` 开启，默认关闭以保持与历史 run 一致。
        """
        names = super().get_decay_parameter_names(model)
        if not getattr(self, "_no_decay_1d", False):
            return names
        one_d = {n for n, p in model.named_parameters() if p.ndim <= 1}
        kept = [n for n in names if n not in one_d]
        dropped = len(names) - len(kept)
        if dropped and not getattr(self, "_no_decay_logged", False):
            self._no_decay_logged = True
            print(f"[opt] 1 维参数移出 weight decay：{dropped} 个"
                  f"（含 gamma）", flush=True)
        return kept

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


def live_param_ratio(model):
    """返回 (非零参数数, 总参数数, 比例)。参数量本身不说明任何问题（E08）。"""
    total = sum(p.numel() for p in model.parameters())
    live = sum(int((p != 0).sum()) for p in model.parameters())
    return live, total, (live / max(1, total))


#: 活参数比例的记录文件名。每次成功开训后写到 output_dir，续训时读回来当基线。
LIVE_PARAMS_FILE = "live_params.json"
DEFAULT_MIN_LIVE_RATIO = 0.95
LIVE_BASELINE_TOL = 0.02


def read_live_baseline(*dirs) -> float | None:
    """读取某个历史 run 记录下来的活参数比例；没有记录就返回 None。

    优先读 ``ratio_min``（历史最低），而不是最近一次的 ``ratio``：否则每轮
    悄悄掉 1%（低于 2% 容差）都会被接受，然后把基线重设到更低的位置，
    连续几十轮就等于没有守卫。
    """
    import json
    for d in dirs:
        if d is None:
            continue
        p = Path(d) / LIVE_PARAMS_FILE
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ratio = float(data.get("ratio_min", data["ratio"]))
        except Exception:                                   # noqa: BLE001
            continue
        if 0.0 < ratio <= 1.0:
            return ratio
    return None


def write_live_baseline(d, live: int, total: int, ratio: float,
                        previous: float | None = None) -> Path:
    """记录本次开训时的活参数比例，供下一次续训判断“有没有变得更死”。

    ``ratio_min`` 保留历史最低值（含本次），这样缓慢侵蚀不会被“每次都重设
    基线”抹掉；``ratio`` 仍记录本次实测值，便于看趋势。
    """
    import json
    p = Path(d) / LIVE_PARAMS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    prev_min = ratio if previous is None else min(float(previous), ratio)
    p.write_text(json.dumps({"live": int(live), "total": int(total),
                             "ratio": float(ratio),
                             "ratio_min": float(prev_min),
                             "written_at": __import__("time").strftime(
                                 "%Y-%m-%d %H:%M:%S")},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def check_live_ratio(ratio: float, threshold: float = DEFAULT_MIN_LIVE_RATIO,
                     allow_dead: bool = False,
                     baseline: float | None = None,
                     tol: float = LIVE_BASELINE_TOL) -> tuple[bool, str]:
    """E08 的判定核心（纯函数，便于直接测试边界）。

    旧实现只有一条 ``ratio < 0.95 -> SystemExit``，把两件不同的事混成一件：

    * **新模型初始化坏了** —— 大片恒零参数是 bug，必须拒绝；
    * **从一个有意剪枝/生长过的检查点继续训** —— 33% 活参数是**已知状态**，
      拒绝它等于把 rescue 计划的第一步堵死（docs/02 §1）。

    现在的规则（fail-closed，但可显式放行）：

    1. ``ratio >= threshold`` -> 通过；
    2. 显式 ``--allow_dead_params`` -> 通过（调用方打印比例）；
    3. 有历史基线且 ``ratio >= baseline - tol`` -> 通过（续训自己导出的检查点，
       比例本来就不该是 100%；只要没有变得更死就行）；
    4. 其余 -> 拒绝，并给出两条出路。
    """
    if ratio >= threshold:
        return True, f"活参数比例 {ratio * 100:.2f}% >= 阈值 {threshold * 100:.0f}%"
    if allow_dead:
        return True, (f"活参数比例 {ratio * 100:.2f}% < 阈值 "
                      f"{threshold * 100:.0f}%，但给了 --allow_dead_params："
                      f"按已知的死参数继续训练（E08 的显式豁免）")
    if baseline is not None and ratio >= baseline - tol:
        return True, (f"活参数比例 {ratio * 100:.2f}% < 阈值 "
                      f"{threshold * 100:.0f}%，但与历史基线 {baseline * 100:.2f}% "
                      f"一致（容差 {tol * 100:.0f}%）：这是已记录的剪枝/生长状态")
    return False, (
        f"活参数比例 {ratio * 100:.2f}% < 阈值 {threshold * 100:.0f}%。\n"
        f"  如果这是**新模型**：说明生长/初始化坏了（docs/00 E01–E08），先修再训。\n"
        f"  如果这是**有意剪枝/生长过的检查点**：加 --allow_dead_params 显式放行，\n"
        f"  或先跑一次带该开关的训练，把 {LIVE_PARAMS_FILE} 写进输出目录当基线。")


#: 梯度覆盖检查的默认阈值：死参数（无梯度或梯度恒零）占总参数元素的比例。
DEFAULT_DEAD_GRAD_TOL = 0.02


def classify_dead_grad_params(names, seen_max: dict) -> list[str]:
    """返回“从没拿到梯度、或梯度恒为零”的参数名，保持 ``names`` 原序。"""
    return [n for n in names if not seen_max.get(n)]


class GradCoverageCallback(TrainerCallback):
    """第 ``check_step`` 步之后做一次梯度覆盖检查（E07）。

    背景：``ddp_find_unused_parameters=False`` 检测的是“梯度是否为 None”，
    **不是**“梯度是否恒为零”。一个全零梯度的参数在它看来是“用到了”，于是
    v21 那 338M 个死参数一路静默跑完。

    为什么用 ``register_hook`` 而不是在 ``on_step_end`` 里读 ``p.grad``：
    HF 的顺序是 ``optimizer.step()`` -> ``scheduler.step()`` -> ``zero_grad()``，
    而 ``zero_grad(set_to_none=True)`` 会在回调之前把梯度清成 None。
    在反向时记录每个参数 ``grad.abs().max()`` 的最大值，才不受清零影响。
    """

    def __init__(self, model, tol: float = DEFAULT_DEAD_GRAD_TOL,
                 check_step: int = 1):
        self.model = model
        self.tol = float(tol)
        self.check_step = int(check_step)
        self._seen: dict[str, float] = {}
        self._numel: dict[str, int] = {}
        self._handles = []
        self._done = False
        for name, p in model.named_parameters():
            self._numel[name] = int(p.numel())
            # requires_grad=False 的参数注册 hook 会直接抛 RuntimeError；
            # 它们本来就不会有梯度，留在 _numel 里就会被判为死参数。
            if p.requires_grad:
                self._handles.append(p.register_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def _hook(grad):
            if grad is None:
                return grad
            v = float(grad.detach().abs().max()) if grad.numel() else 0.0
            prev = self._seen.get(name)
            self._seen[name] = v if prev is None else max(prev, v)
            return grad
        return _hook

    def _remove_hooks(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:                               # noqa: BLE001
                pass
        self._handles = []

    def check_now(self, global_step: int = 1) -> list[str]:
        """立即汇总并返回死参数（供测试直接调用，不必起 Trainer）。"""
        self._done = True
        self._remove_hooks()
        return classify_dead_grad_params(list(self._numel), self._seen)

    def on_step_end(self, args, state, control, **kw):
        if self._done or int(getattr(state, "global_step", 0)) < self.check_step:
            return control
        dead = self.check_now()
        total = max(1, sum(self._numel.values()))
        dead_numel = sum(self._numel[n] for n in dead)
        frac = dead_numel / total
        if not getattr(state, "is_world_process_zero", True):
            return control
        step = int(getattr(state, "global_step", 0))
        if dead:
            print(f"[grad] step {step}：{len(dead)}/{len(self._numel)} 个参数"
                  f"（{dead_numel:,}/{total:,} 元素 = {frac * 100:.2f}%）"
                  f"没有任何非零梯度：", flush=True)
            for name in dead[:12]:
                print(f"[grad]    {name}", flush=True)
            if len(dead) > 12:
                print(f"[grad]    ... 还有 {len(dead) - 12} 个", flush=True)
        else:
            print(f"[grad] step {step}：{len(self._numel)} 个参数全部拿到非零梯度",
                  flush=True)
        if frac > self.tol:
            raise SystemExit(
                f"[grad] 死参数占比 {frac * 100:.2f}% 超过阈值 {self.tol * 100:.2f}%："
                f"模型里有参数从第 1 步起就没有信号，训练多少步都不会变"
                f"（docs/00 E07/E08）。先修生长/初始化；确认是有意冻结再提高 "
                f"--dead_grad_tol。")
        return control


def decay_group_1d_violations(model, optimizer) -> list[str]:
    """返回优化器里 ``weight_decay>0`` 的参数组中出现的 1 维参数名（E06）。

    这是比“训练中 gamma 有没有变小”更直接的判据：E06 的根因就是
    ``msr.gamma``（以及各种 norm scale）落进了带 weight decay 的组。
    直接查最终优化器分组，不依赖参数命名规则的重实现。
    """
    id2name = {id(p): n for n, p in model.named_parameters()}
    one_d = {id(p) for p in model.parameters() if p.ndim <= 1}
    bad: list[str] = []
    for gi, group in enumerate(getattr(optimizer, "param_groups", []) or []):
        if float(group.get("weight_decay", 0.0) or 0.0) <= 0.0:
            continue
        for p in group.get("params", []):
            if id(p) in one_d:
                bad.append(id2name.get(id(p), f"<unnamed in group {gi}>"))
    return bad


class DecayGuardCallback(TrainerCallback):
    """训练开始时断言：开 ``--no_decay_1d`` 后没有 1 维参数在衰减组里。"""

    def __init__(self, trainer, enabled: bool = True):
        self.trainer = trainer
        self.enabled = bool(enabled)

    def on_train_begin(self, args, state, control, **kw):
        if not self.enabled:
            return control
        opt = getattr(self.trainer, "optimizer", None)
        if opt is None:
            print("[opt] ⚠️ 训练开始时还没有优化器，weight decay 分组断言跳过",
                  flush=True)
            return control
        bad = decay_group_1d_violations(self.trainer.model, opt)
        if bad:
            raise SystemExit(
                "[opt] --no_decay_1d 已开启，但下列 1 维参数仍在 weight_decay>0 "
                "的参数组里：" + ", ".join(bad[:12])
                + "。gamma 会被 AdamW 按固定比例压小（E06），先修 "
                  "get_decay_parameter_names 或优化器分组。")
        print(f"[opt] weight decay 分组断言通过：{len(opt.param_groups)} 组里没有 "
              f"1 维参数", flush=True)
        return control


def cap_eval_rows(n_rows: int, eval_batches: int,
                  eval_batch_size: int) -> int:
    """``--eval_batches`` 的语义：最多评测多少个 batch（0/NULL = 全部）。

    旧实现定义了 ``--eval_batches`` 却**没有任何地方读它**（run_task 两个
    分支都传 40），是典型的死字段。现在把它落成真正的行数上限。
    """
    n = int(n_rows)
    if not eval_batches or int(eval_batches) <= 0 or not eval_batch_size:
        return n
    return min(n, int(eval_batches) * int(eval_batch_size))


def plan_stage_specs(plan_obj) -> list[dict]:
    """计划 -> ``StagePackedDataset`` 的 stage dict（train.py 与测试共用）。

    单独成函数是为了让“计划 JSON -> parse_plan -> 数据集”这条链路可以被整体
    测试。旧实现把这段 dict 内联在 ``main()`` 里、测试只直接构造
    ``_FlatStage(require_eos=...)``，于是 ``require_eos`` 在 Stage / parse_plan
    里被丢掉这件事，任何测试都看不见（docs/00 E12/P0-3）。
    """
    return [{"name": s.name, "data_dir": str(s.resolved_dir()), "mode": s.mode,
             "gc": s.gc, "min_shards": s.min_shards, "exclude": s.exclude,
             # E12：require_eos 必须从计划一路带到数据集。旧实现里
             # Stage / parse_plan / 这个 dict 三处都漏，于是守卫的错误提示
             # “在计划里给该阶段写 require_eos: false”根本不可能生效。
             "weight": s.weight, "require_eos": s.require_eos}
            for s in plan_obj.stages]


def assert_val_disjoint(val_path, train_dirs, seq_len: int = 2048) -> None:
    """验证集目录不能与任何训练阶段同源（E14）。

    上一轮 val_dir 是 chat_pack_v13，而 sft_mix 里 40% 就是它：训练期
    eval_loss 因此不可信、best-model 选择也被污染。

    旧实现只比较 ``Path(...).resolve()`` 字符串——那是**路径相等**，不是
    **数据相等**。现在三层都查（见 ``retnet_hf/provenance.py``）：
    规范化路径、meta.json 声明的来源、以及按行内容指纹。任何一层命中都拒绝。
    """
    from retnet_hf.provenance import describe, overlap_report
    rep = overlap_report(val_path, train_dirs, seq_len)
    hits = rep["path_hits"] + rep["source_hits"] + rep["content_hits"]
    if hits:
        raise SystemExit(
            "[preflight] 验证集与训练阶段同源，拒绝开训：\n  "
            + "\n  ".join(hits)
            + "\n验证集进了训练，eval_loss 与 best-model 选择都不可信（docs/00 E14）。")
    print(f"[preflight] 验证集隔离 OK：{describe(rep)}", flush=True)


def _preflight(model, args, baseline: float | None = None) -> None:
    """开训前的硬检查：活参数比例 + gamma 分布。任一不过就拒绝开训。

    为什么需要它（docs/00 E08）：参数量是从 config 数出来的，
    504,179,896 哪怕 67% 是精确 0 也照样打印 504M。这里直接扫权重数值，
    把“名义参数”和“可学参数”分开。判定规则（含剪枝检查点的豁免）在
    ``check_live_ratio`` 里，是纯函数，便于测试边界。

    gamma 的检查对应 E06：衰减率被 weight decay 压小后，半衰期会从
    22~1400 token 掉到约 1 token，跨 chunk 的信息在数值上等于零。
    这里做下限检查（<=0 直接拒绝）并打印分布；gamma 是否仍在 weight decay
    集合里，由训练开始时的 ``DecayGuardCallback`` 直接查优化器参数组。
    **注意**：`use_gla=False` 时 gamma 是 buffer 而不是 Parameter，
    旧实现只遍历 `named_parameters()`，那种模型上这条检查是死的。
    """
    live, total, ratio = live_param_ratio(model)
    print(f"[preflight] live params {live:,}/{total:,} = {ratio * 100:.2f}%", flush=True)
    threshold = float(getattr(args, "min_live_ratio", DEFAULT_MIN_LIVE_RATIO))
    ok, why = check_live_ratio(ratio, threshold,
                               allow_dead=bool(getattr(args, "allow_dead_params", False)),
                               baseline=baseline)
    print(f"[preflight] 活参数判定：{why}", flush=True)
    if not ok:
        raise SystemExit("[preflight] " + why)

    # gamma 可能是 Parameter（use_gla=True）也可能是 buffer（use_gla=False），
    # 两头都收，否则后者这条检查静默失效。
    gammas = [t.detach().float().flatten()
              for n, t in list(model.named_parameters()) + list(model.named_buffers())
              if n.endswith("msr.gamma")]
    if gammas:
        g = torch.cat(gammas)
        print(f"[preflight] gamma n={g.numel()} min={g.min():.4f} "
              f"p50={g.median():.4f} max={g.max():.4f}", flush=True)
        if float(g.min()) <= 0.0:
            raise SystemExit("[preflight] gamma 里有 <=0 的值：log(0) 会让整层 NaN。")
    else:
        print("[preflight] ⚠️ 没找到任何 msr.gamma（参数或 buffer），gamma 检查跳过",
              flush=True)


def build_parser() -> argparse.ArgumentParser:
    """把 argparse 抽成纯函数。

    测试可以只解析 argv、断言 ``--decay_1d`` -> ``no_decay_1d=False``，
    不必把整个 Trainer 跑起来（E06 的“接线”以前完全没有测试覆盖：
    删掉 main 里那行赋值，所有测试照样绿）。
    """
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
    ap.add_argument("--eval_batches", type=int, default=0,
                    help="评测最多跑多少个 batch；0 = 不限制（默认）。"
                         "旧版本定义了这个参数但没有任何地方读它，是死字段（P2-1）；"
                         "现在它真的限制行数，默认不限制以保持历史 eval_loss 口径可比。")
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
    ap.add_argument("--save_total_limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT,
                    help="保留最近 N 个 checkpoint-*；<=0 表示不限制。"
                         "默认值来自 retnet_hf/defaults.py（与 run_task.py 同一个常量，"
                         "docs/00 E17：以前模板写 4、这里硬编码 3，多出来的那个检查点"
                         "会被 HF 按 mtime 静默轮转删除）。")
    ap.add_argument("--save_steps", type=int, default=2500)
    ap.add_argument("--eval_steps", type=int, default=2500)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--optim", default=None,
                    help="TrainingArguments.optim；"
                         "adamw_bnb_8bit 省 1.5 GiB 优化器状态（关检查点的前提）")
    ap.add_argument("--no_decay_1d", action="store_true", default=DEFAULT_NO_DECAY_1D,
                    help="把 1 维参数（各种 norm scale / 门控 / msr.gamma）移出 weight decay。"
                         "**默认开启**：v21 没开这个开关，gamma 被 wd 从 1-2^-5 压到 0.5656"
                         "（半衰期 1.22 token），96.1%% 的头 <128 token。")
    ap.add_argument("--decay_1d", dest="no_decay_1d", action="store_false",
                    help="反向开关：允许 1 维参数吃 weight decay。只有复现旧 run 时才用；"
                         "用之前先想清楚 gamma 会不会被压塌。")
    ap.add_argument("--min_live_ratio", type=float, default=0.95,
                    help="活参数比例下限（docs/00 E08）。新模型低于它直接拒绝开训；"
                         "从有意剪枝/生长的检查点继续训练时用 --allow_dead_params 放行。")
    ap.add_argument("--allow_dead_params", action="store_true",
                    help="显式承认“这个模型有已知的死参数”，跳过 95%% 活参数下限。"
                         "docs/02 的 rescue 计划从 33%% 活参数的 v21_export 起步，"
                         "必须带这个开关；日志仍会打印比例。")
    ap.add_argument("--dead_grad_tol", type=float, default=0.02,
                    help="第 1 步梯度覆盖检查的阈值：死参数元素占比超过它就停机"
                         "（docs/00 E07）。设为 1.0 可只观察不停机。")
    ap.add_argument("--ddp_find_unused_parameters", action="store_true",
                    help="多卡 DDP 时允许存在未被使用的参数。本模型的 growth 张量里有一批"
                         "gamma=0 的仿射项，先试默认 False；报 unused parameter 时加这个开关")
    return ap


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args()

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
    # 基线来自源检查点目录或输出目录里上一次成功开训写下的 live_params.json。
    # 续训自己导出的、已被剪枝/生长的检查点时，比例天然不是 100%，不能按新模型判。
    baseline = read_live_baseline(model_dir, Path(args.output_dir))
    _preflight(model, args, baseline)

    # ---- 数据集：计划模式走多阶段可切换数据集，否则走原来的单文件 ----
    import numpy as np
    if plan_obj is not None:
        seq_len = plan_obj.seq_len
        train_ds = StagePackedDataset(plan_stage_specs(plan_obj), seq_len)
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
    # 验证集隔离断言（docs/00 E14）：路径、meta.json 声明来源、按行内容指纹三层。
    # 上一轮的 val_dir 是 chat_pack_v13，而最后 14,000 步的 sft_mix 里 40% 就是它，
    # 于是 eval_loss 在末段从 4.55 掉到 2.4450 这件事根本不可信。
    if plan_obj is not None:
        assert_val_disjoint(val_path, [s.resolved_dir() for s in plan_obj.stages],
                            seq_len=seq_len)
    # 验证集目录同时登记为“不许 GC”：计划里某段一旦把 gc 打开、又恰好指向
    # 验证集，gc_stage 必须在删之前就拒绝（旧实现只在文档里写了这条约束）。
    if hasattr(train_ds, "protect"):
        train_ds.protect(val_path)
    val_ds = PackedChatDataset(str(val_path / "val.ids.npy"),
                               str(val_path / "val.wcode.npy"), seq_len)
    if plan_obj is not None:
        print(f"[plan] 验证集固定用 {val_path}（换训练语料时不换验证集，否则曲线不可比）",
              flush=True)
    # P2-1：--eval_batches 以前只定义、没人读。现在它是真正的“最多评测几个 batch”。
    _eval_rows = cap_eval_rows(len(val_ds), args.eval_batches,
                               args.batch_size * args.eval_batch_mult)
    if _eval_rows < len(val_ds):
        print(f"[eval] --eval_batches {args.eval_batches} × batch "
              f"{args.batch_size * args.eval_batch_mult} = {_eval_rows} 行；"
              f"验证集共 {len(val_ds)} 行，只评测前 {_eval_rows} 行", flush=True)
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, range(_eval_rows))
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
        # 2026-10-20（docs/00 E15）：关闭“按 eval_loss 选最佳”。
        # 训练期 eval 用的是 val_dir（上一轮是 chat_pack_v13），而最后 14,000 步的
        # sft_mix 里 40% 就是它——这个指标被污染，拿它选最佳等于拿训练集选模型。
        # 正确做法：训练中只按步数存 checkpoint，跑完后用独立域内 held-out
        # （data_cache/pretrain_v2/*_val.npy，绝不进训练）在离线脚本里选。
        load_best_model_at_end=False,
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
    # DDP 的 find_unused_parameters 检测的是"梯度是否为 None"，**不是"梯度是否恒为零"**。
    # 一个全零梯度的参数在这个开关看来是"用到了"，所以关掉它不会报错，
    # v21 那 338M 个死参数就是这样一路静默跑完的（见 docs/00 E07）。
    # 这里保留关闭（省一次 autograd 图遍历）；梯度覆盖检查由下面真正注册的
    # GradCoverageCallback 负责 —— 以前这段注释写着"另加了一次检查"，但代码里
    # 根本没有那东西（docs/01 自己都写着"尚未落地"），注释承诺 ≠ 实现。
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
    # 1 维参数（含 msr.gamma）是否免 weight decay —— 见 get_decay_parameter_names 的注释
    trainer._no_decay_1d = bool(getattr(args, "no_decay_1d", False))
    if trainer._no_decay_1d:
        print("[opt] --no_decay_1d 已开启：所有 1 维参数不做 weight decay", flush=True)
    # E06 的运行时守卫：优化器建好后直接查参数组，1 维参数不得出现在
    # weight_decay>0 的组里。比“训练中 gamma 有没有变小”更直接（那是学习也会
    # 引起的），也不依赖对参数命名规则的重实现。
    trainer.add_callback(DecayGuardCallback(trainer, enabled=trainer._no_decay_1d))
    # E07 的梯度覆盖检查：第 1 步之后统计“从没拿到非零梯度”的参数。
    # 注册在参数上（反向时记录），因为 on_step_end 时 HF 已经 zero_grad 了。
    trainer.add_callback(GradCoverageCallback(
        trainer.model, tol=float(args.dead_grad_tol)))
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
        # 记录本次 run 的活参数比例，供下一次续训判断“有没有变得更死”（E08）。
        # 只在 rank 0 写；写入失败不该让训练成果作废，因此只告警。
        try:
            _live, _total, _ratio = live_param_ratio(model)
            _p = write_live_baseline(args.output_dir, _live, _total, _ratio, baseline)
            print(f"[preflight] 活参数基线已写入 {_p}（{_ratio * 100:.2f}%）", flush=True)
        except Exception as _exc:                            # noqa: BLE001
            print(f"[preflight] ⚠️ 活参数基线写入失败：{_exc}", flush=True)
        print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
