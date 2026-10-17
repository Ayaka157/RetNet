"""可热加载的训练计划：阶段化数据 + 动态 max_steps / LR / 存档节奏。

## 为什么要有这个

在此之前，训练计划是**写死在任务 JSON 里、进程启动后不可变**的：

- `max_steps` 固定 → 想多练只能杀进程重启（白丢一个 cosine，还要重付热重启代价）
- `data_dir` 固定 → **只能在同一份语料上加轮次**，换不了料
- LR 曲线在启动时按固定步数铺死 → 改步数就等于换了一条曲线

结果是「每次加钱 = 把同一份数据再读一遍」，唯一 token 数从头到尾没动过。
这个模块把这三件事都变成**运行时可改**的。

## 计划文件（默认 `<repo>/tasks/plan.json`，可用 `RETNET_PLAN_FILE` 覆盖）

```json
{
  "plan_version": 3,
  "seq_len": 2048,
  "stages": [
    {"name": "chat_v13",   "data_dir": "data_cache/chat_pack_v13",   "steps": 16848},
    {"name": "fineweb_v2", "data_dir": "data_cache/pretrain_v2_flat", "steps": 40000}
  ],
  "lr": {"peak": 1.1e-4, "min": 1.1e-5, "warmup": 500},
  "save_steps": 2400,
  "eval_steps": 2400,
  "notes": "给人看的话，随便写"
}
```

- `total_steps = sum(stages[].steps)`，LR 余弦铺满整个 `total_steps`。
- **改 `steps` 或往 `stages` 里追加一段 → 总步数变长 → 余弦自动拉长**。
  代价是当前 LR 会按新曲线跳上去（相当于一次受控的热重启，但**优化器状态不丢**）。
  回调会把「旧 LR → 新 LR」打印出来，不做隐式行为。
- **跑到当前 stage 的步数预算用尽 → 自动切到下一段的语料**，进程不退出、
  DataLoader 不重建、优化器状态不丢。这就是「在当前材料吃完前替换上新材料」。
- 计划文件按 mtime 轮询（默认每 200 步查一次），改了就在下个步边界生效。

## 数据目录的两种形态

`data_dir` 里放什么，由下面自动判断：

1. **packed**（chat pack）：`train.ids.npy` + `train.wcode.npy`，二维 `(rows, seq_len+1)`
2. **flat**（预训练语料）：目录下若干 `*.npy` 一维 token 流（int16/int32），
   `wcode` 全为 1；行数 = `总token // (seq_len+1)`，跨分片的余数丢弃

flat 形态是给 `data_cache/pretrain_v2/*.npy` 这种「已经 tokenize 过、但没打包」的
语料用的 —— 不必再花几个小时打包出一份重复的 5 GB 文件。

## 兼容性

不传 `--plan` 时训练行为与以前**完全一致**（走 `PackedChatDataset` + `--max_steps`）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------- 计划结构

@dataclass
class Stage:
    name: str
    data_dir: str
    steps: int
    mode: str = "auto"          # auto / packed / flat

    def resolved_dir(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else ROOT / p


@dataclass
class Plan:
    plan_version: int
    seq_len: int
    stages: list[Stage]
    lr_peak: float
    lr_min: float
    warmup: int
    save_steps: int
    eval_steps: int
    notes: str = ""
    source: Path | None = None
    raw: dict = field(default_factory=dict)
    val_dir: str = ""

    def resolved_val_dir(self) -> Path:
        """验证集目录：优先 ``val_dir``，否则用第一段的目录。

        ⚠️ 换训练语料时**不要把验证集也换掉** —— 否则 loss 曲线前后不可比，
        看上去"降了"可能只是换了个更简单的验证集。
        """
        p = Path(self.val_dir) if self.val_dir else self.stages[0].resolved_dir()
        return p if p.is_absolute() else ROOT / p

    @property
    def total_steps(self) -> int:
        return sum(s.steps for s in self.stages)

    def cumulative(self) -> list[int]:
        """每段结束时的累计步数，如 [16848, 56848]。"""
        out, acc = [], 0
        for s in self.stages:
            acc += s.steps
            out.append(acc)
        return out

    def stage_index_at(self, step: int) -> int:
        """该步数应该读第几段（0 基）。越界时返回最后一段。"""
        cum = self.cumulative()
        for i, c in enumerate(cum):
            if step < c:
                return i
        return len(self.stages) - 1

    def lr_at(self, step: int) -> float:
        """按当前计划的余弦算该步的目标 LR。

        与 HF 的 ``cosine_with_min_lr`` 同形：warmup 线性升到 peak，
        之后 cos 从 peak 降到 min。
        """
        t = max(0, min(int(step), self.total_steps))
        if self.warmup > 0 and t < self.warmup:
            return self.lr_peak * (t + 1) / self.warmup
        span = max(1, self.total_steps - self.warmup)
        prog = min(1.0, max(0.0, (t - self.warmup) / span))
        import math
        return self.lr_min + (self.lr_peak - self.lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))


def _need(d: dict, key: str, where: str):
    if key not in d:
        raise ValueError(f"计划文件 {where} 缺少字段 {key!r}")
    return d[key]


def parse_plan(data: dict, source: Path | None = None) -> Plan:
    where = str(source or "<dict>")
    stages_raw = _need(data, "stages", where)
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError(f"计划文件 {where} 的 stages 必须是非空数组")
    stages = []
    for i, s in enumerate(stages_raw):
        steps = int(_need(s, "steps", f"{where}:stages[{i}]"))
        if steps < 0:
            raise ValueError(f"计划文件 {where} 的 stages[{i}].steps 不能为负，收到 {steps}")
        stages.append(Stage(
            name=str(_need(s, "name", f"{where}:stages[{i}]")),
            data_dir=str(_need(s, "data_dir", f"{where}:stages[{i}]")),
            steps=steps,
            mode=str(s.get("mode", "auto")),
        ))
    lr = data.get("lr", {}) or {}
    plan = Plan(
        plan_version=int(data.get("plan_version", 0)),
        seq_len=int(data.get("seq_len", 2048)),
        stages=stages,
        lr_peak=float(lr.get("peak", 1.1e-4)),
        lr_min=float(lr.get("min", 1.1e-5)),
        warmup=int(lr.get("warmup", 500)),
        save_steps=int(data.get("save_steps", 2400)),
        eval_steps=int(data.get("eval_steps", 2400)),
        notes=str(data.get("notes", "")),
        source=source,
        raw=data,
        val_dir=str(data.get("val_dir", "")),
    )
    if plan.lr_min > plan.lr_peak:
        raise ValueError(f"计划文件 {where}：lr.min 大于 lr.peak")
    if plan.total_steps <= 0:
        raise ValueError(f"计划文件 {where}：所有 stages 的 steps 都是 0，没有可训练的步数")
    if plan.warmup >= plan.total_steps:
        raise ValueError(f"计划文件 {where}：warmup({plan.warmup}) 不小于 total_steps({plan.total_steps})")
    return plan


def plan_path() -> Path:
    env = os.environ.get("RETNET_PLAN_FILE")
    if env:
        p = Path(env)
        return p if p.is_absolute() else ROOT / p
    return ROOT / "tasks" / "plan.json"


# --------------------------------------------------------------------- 热加载

class PlanWatcher:
    """按 mtime 判断计划文件是否变过；读坏了**保留上一份可用的计划**，绝不崩训练。"""

    def __init__(self, path: str | os.PathLike | None = None, verbose: bool = True):
        self.path = Path(path) if path else plan_path()
        self.verbose = verbose
        self._mtime: float | None = None
        self._last_error: str | None = None
        self.plan: Plan | None = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def load(self, force: bool = False) -> Plan | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            if force:
                raise SystemExit(f"找不到训练计划 {self.path}")
            return None
        if not force and self._mtime is not None and st.st_mtime == self._mtime:
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            plan = parse_plan(data, source=self.path)
        except Exception as exc:                                # noqa: BLE001
            msg = f"[plan] 计划文件解析失败，**沿用上一份计划**：{type(exc).__name__}: {exc}"
            if msg != self._last_error:
                self._log(msg)
                self._last_error = msg
            self._mtime = st.st_mtime      # 记住这个坏的版本，别每步重复报错
            return None
        self._mtime = st.st_mtime
        self._last_error = None
        self.plan = plan
        return plan

    def maybe_reload(self) -> Plan | None:
        return self.load(force=False)


# --------------------------------------------------------------------- 回调

def make_plan_callback(path=None, verbose: bool = True, check_every: int = 200):
    """构造 ``TrainerCallback``：在步边界应用计划变更 + 切语料 + 覆写 LR。

    为什么自己覆写 LR 而不是重建 HF 的 scheduler：
    HF 的 scheduler 在构造时就绑定了一个固定的 ``num_training_steps``，
    中途换曲线只能重建，而重建会丢掉 ``last_epoch``、并且和 optimizer 的
    绑定关系要重做。直接在 ``on_step_begin`` 里按计划算 LR 写进 ``param_group``
    更简单：HF 的 scheduler 仍在跑，但**每一步都被我们覆盖**，有效 LR 就是计划的。
    日志里看到的 ``learning_rate`` 是 HF scheduler 的值，会与真实 LR 有偏差，
    所以本回调在 LR 变化时自己打印一行真实值。
    """
    from transformers import TrainerCallback

    class _PlanCallback(TrainerCallback):
        def __init__(self):
            self.watcher = PlanWatcher(path, verbose=verbose)
            self.plan: Plan | None = None
            self._applied_save = None
            self._applied_eval = None
            self._last_lr_logged = None
            self._stage_logged = None
            self._steps_since_check = 0
            self._check_every = max(1, int(check_every))
            # 显式绑定：不同 transformers 版本给 callback 的 kw 不一样，
            # 实测 4.57 里 on_step_end 的 kw 并**不**保证带 trainer，
            # 靠 kw.get("trainer") 会让换料静默失效。由 train.py 直接 bind。
            self._trainer = None
            self._dataset_obj = None
            self._optimizer = None

        def bind(self, trainer=None, dataset=None) -> "_PlanCallback":
            """由 train.py 调用，把训练器 / 数据集 / 优化器显式交给回调。"""
            if trainer is not None:
                self._trainer = trainer
                self._optimizer = getattr(trainer, "optimizer", None)
                if dataset is None:
                    dataset = getattr(trainer, "train_dataset", None)
            if dataset is not None:
                self._dataset_obj = dataset
            return self

        # -- 工具 ---------------------------------------------------
        def _dataset(self, trainer=None):
            ds = self._dataset_obj
            if ds is None:
                cand = trainer if trainer is not None else self._trainer
                if cand is not None:
                    ds = getattr(cand, "train_dataset", None)
                if ds is None:
                    ds = self._dataset_obj
            return ds if hasattr(ds, "set_stage") else None

        def _optimizer_obj(self, trainer=None):
            if self._optimizer is not None:
                return self._optimizer
            cand = trainer if trainer is not None else self._trainer
            self._optimizer = getattr(cand, "optimizer", None) if cand is not None else None
            return self._optimizer

        def _log(self, msg: str) -> None:
            if verbose:
                print(msg, flush=True)

        def _apply_intervals(self, args) -> None:
            if self.plan is None:
                return
            if self.plan.save_steps != self._applied_save:
                args.save_steps = self.plan.save_steps
                self._applied_save = self.plan.save_steps
                self._log(f"[plan] save_steps -> {args.save_steps}")
            if self.plan.eval_steps != self._applied_eval:
                args.eval_steps = self.plan.eval_steps
                self._applied_eval = self.plan.eval_steps
                self._log(f"[plan] eval_steps -> {args.eval_steps}")

        def _sync_stage(self, trainer, state, force: bool = False) -> None:
            ds = self._dataset(trainer)
            if ds is None or self.plan is None:
                return
            want = self.plan.stage_index_at(state.global_step)
            have = getattr(ds, "active_index", 0)
            if force or want != have:
                ds.set_stage(want)
                name = self.plan.stages[want].name
                if self._stage_logged != name:
                    self._log(
                        f"[plan] step {state.global_step} 切到第 {want} 段语料 "
                        f"「{name}」（{self.plan.stages[want].data_dir}，"
                        f"预算 {self.plan.stages[want].steps} 步）"
                    )
                    self._stage_logged = name

        def _set_lr(self, trainer, state) -> None:
            if self.plan is None:
                return
            opt = self._optimizer_obj(trainer)
            if opt is None:
                return
            lr = self.plan.lr_at(state.global_step)
            for g in opt.param_groups:
                g["lr"] = lr
            if self._last_lr_logged is None or abs(lr - self._last_lr_logged) > 1e-9:
                step = state.global_step
                if step == 0 or step % 500 == 0 or step == self.plan.warmup:
                    self._log(f"[plan] step {step} 真实 LR = {lr:.3e} "
                              f"(peak {self.plan.lr_peak:.1e}, total {self.plan.total_steps} 步)")
                self._last_lr_logged = lr

        # -- 生命周期 -----------------------------------------------
        def on_train_begin(self, args, state, control, **kw):
            plan = self.watcher.load(force=True)
            self.plan = plan
            old_max = state.max_steps
            state.max_steps = plan.total_steps
            if old_max != state.max_steps:
                self._log(f"[plan] max_steps {old_max} -> {state.max_steps}（按计划覆盖）")
            self._apply_intervals(args)
            self._log(f"[plan] 计划 {self.watcher.path}：{len(plan.stages)} 段，"
                      f"共 {plan.total_steps} 步")
            for i, s in enumerate(plan.stages):
                self._log(f"[plan]   {i}. {s.name:<18} {s.steps:>7} 步  {s.data_dir}")
            self._sync_stage(kw.get("trainer"), state, force=True)
            return control

        def on_step_begin(self, args, state, control, **kw):
            self._set_lr(kw.get("trainer"), state)
            return control

        def on_step_end(self, args, state, control, **kw):
            trainer = kw.get("trainer")
            new = self.watcher.maybe_reload()
            if new is not None:
                self._apply_plan_change(trainer, args, state)
            self._sync_stage(trainer, state)
            self._steps_since_check += 1
            if self._steps_since_check >= self._check_every:
                self._steps_since_check = 0
            return control

        def _apply_plan_change(self, trainer, args, state) -> None:
            old = self.plan
            new = self.watcher.plan
            if new is None:
                return
            self.plan = new
            self._log(f"[plan] 检测到计划变更（plan_version {getattr(old,'plan_version',None)}"
                      f" -> {new.plan_version}）")
            if new.total_steps != old.total_steps:
                lr_now_old = old.lr_at(state.global_step)
                lr_now_new = new.lr_at(state.global_step)
                state.max_steps = new.total_steps
                self._log(
                    f"[plan] total_steps {old.total_steps} -> {new.total_steps}；"
                    f"余弦随之重铺，当前步 LR {lr_now_old:.3e} -> {lr_now_new:.3e}"
                    + ("（上跳，相当于一次受控热重启，但优化器状态保留）"
                       if lr_now_new > lr_now_old * 1.02 else "")
                )
            if [ (s.name, s.data_dir, s.steps) for s in new.stages ] != \
               [ (s.name, s.data_dir, s.steps) for s in old.stages ]:
                ds = self._dataset(trainer)
                if ds is not None:
                    ds.update_stages([{"name": s.name, "data_dir": str(s.resolved_dir()),
                                       "mode": s.mode} for s in new.stages])
                    self._log(f"[plan] 语料阶段已更新为 {len(new.stages)} 段")
                    self._stage_logged = None
            self._apply_intervals(args)

    return _PlanCallback()
