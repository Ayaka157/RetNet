"""外部旗标控制训练进程：继续 / 立刻停 / 下一个检查点后停 / 立刻存一次。

## 为什么需要这个

原本要停一个正在跑的训练，只能 ``taskkill``，而且**顺序还必须对**：

    cmd(18804) -> python run_task.py(5428) -> python retnet_hf/train.py(24012)

``run_task.py`` 的循环是 ``if rc != 0 and attempts < auto_restart_max: 重启``。
直接杀 ``train.py`` 会拿到 rc=1，于是被自动拉起来；**必须先杀 run_task.py**。
这种操作既危险又不可复现。

有了旗标，训练在**步边界优雅退出**：``on_step_end`` 里置
``control.should_training_stop``，HF 先执行本步的
``_maybe_log_save_evaluate``（存检查点 + 评测），再到下一行 break。
``trainer.train()`` 正常返回 -> ``trainer.save_model(output_dir)`` 照常导出
-> 进程以 **rc=0** 退出 -> ``auto_restart`` 不会触发。

> 顺序依据：``transformers/trainer.py`` 的 ``_inner_training_loop``
> 2753 行 ``global_step += 1`` -> 2755 行 ``on_step_end`` ->
> 2756 行 ``_maybe_log_save_evaluate`` -> 2772 行 ``if should_training_stop: break``。
> 钩子在存检查点**之前**，这就是「下一个检查点后停」能写对的原因。

## 旗标文件

默认 ``<repo>/logs/ctrl.txt``，可用环境变量 ``RETNET_CTRL_FILE`` 覆盖。
取第一行非空、且不以 ``#`` 开头的词，大小写不敏感：

===============  ==========================================================
``run``          继续（默认；也用来清除已有指令）。**删掉文件等价于 run**
``stop``         当前优化步做完就停，不等检查点
``next-checkpoint``  下一次写检查点之后停 —— 检查点先落盘，再退出
``save``         立刻存一次检查点，然后继续训练（一次性）
===============  ==========================================================

``stop`` 与 ``save`` 是一次性的：执行后回调会把文件改写成 ``run``，
免得每一步都重复触发。``next-checkpoint`` 在命中保存点之前一直有效。

## 用法

    python -m retnet_hf.control next-checkpoint
    python -m retnet_hf.control status
    python -m retnet_hf.control run          # 清除指令

也可以走 ``run_task.py --ctrl <命令>``。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUN = "run"
STOP = "stop"
NEXT_CHECKPOINT = "next-checkpoint"
SAVE = "save"

#: 允许的别名 -> 规范名
ALIASES = {
    "run": RUN, "continue": RUN, "go": RUN, "resume": RUN, "none": RUN,
    "stop": STOP, "halt": STOP, "now": STOP,
    "next-checkpoint": NEXT_CHECKPOINT, "checkpoint": NEXT_CHECKPOINT,
    "next": NEXT_CHECKPOINT, "stop-after-checkpoint": NEXT_CHECKPOINT,
    "save": SAVE, "checkpoint-now": SAVE,
}

VALID = (RUN, STOP, NEXT_CHECKPOINT, SAVE)

HELP = """指令：
  run               继续（默认；也用来清除已有指令）
  stop              当前优化步做完就停，不等检查点
  next-checkpoint   下一次写检查点之后停（检查点先落盘再退出）
  save              立刻存一次检查点，然后继续训练
  status            只打印当前指令，不写文件
"""


def ctrl_path() -> Path:
    """旗标文件路径。``RETNET_CTRL_FILE`` 可覆盖（绝对或相对仓库根）。"""
    env = os.environ.get("RETNET_CTRL_FILE")
    if env:
        p = Path(env)
        return p if p.is_absolute() else ROOT / p
    return ROOT / "logs" / "ctrl.txt"


def normalize(word: str | None) -> str | None:
    """把一行文本归一成规范指令；认不出来返回 None。"""
    if not word:
        return None
    return ALIASES.get(word.strip().lower())


def read() -> str | None:
    """读当前指令。文件不存在 / 空 / 全是注释 -> ``None``（等同于 run）。

    ⚠️ 任何读取异常都吞掉并返回 None：这是一个控制通道，**绝不能因为旗标文件
    写坏了就把训练搞崩**。宁可忽略指令。
    """
    try:
        text = ctrl_path().read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except Exception:                                          # noqa: BLE001
        return None
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return normalize(line.split()[0])
    return None


def write(cmd: str) -> Path:
    """写指令。返回写入的路径。"""
    norm = normalize(cmd)
    if norm is None:
        raise SystemExit(f"未知指令 {cmd!r}。\n{HELP}")
    p = ctrl_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = ""
    try:
        import datetime as _dt
        stamp = f"# 写入于 {_dt.datetime.now():%Y-%m-%d %H:%M:%S}\n"
    except Exception:                                          # noqa: BLE001
        pass
    p.write_text(f"{stamp}{norm}\n", encoding="utf-8")
    return p


def clear() -> Path:
    """清掉指令（写 ``run``）。"""
    return write(RUN)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = ctrl_path()
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP)
        print(f"旗标文件：{p}")
        print(f"当前指令：{read() or '(无 = run)'}")
        return 0

    word = argv[0].strip().lower()
    if word == "status":
        print(f"{p}  ->  {read() or '(无 = run)'}")
        return 0
    if word == "path":
        print(p)
        return 0

    try:
        written = write(word)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"已写入 {normalize(word)!r} -> {written}")
    if normalize(word) == NEXT_CHECKPOINT:
        print("训练会在**下一个 save_steps 边界**存完检查点后退出，"
              "trainer.train() 正常返回，最终模型照常导出到 output_dir。")
    return 0


def make_callback(verbose: bool = True):
    """构造 ``transformers.TrainerCallback``，在步边界消费旗标。

    放在这里而不是 train.py，是为了让「读 / 写 / 消费」三种语义共用一份代码，
    不至于两边对不上。
    """
    from transformers import TrainerCallback

    class _ExternalControl(TrainerCallback):
        def __init__(self):
            self._last_seen = None
            self._pending_next_checkpoint = False
            self.stop_reason = None

        def _log(self, msg: str) -> None:
            if verbose:
                print(msg, flush=True)

        # -- 生命周期 --------------------------------------------------
        def on_train_begin(self, args, state, control, **kw):
            cmd = read()
            if cmd == NEXT_CHECKPOINT:
                self._pending_next_checkpoint = True
            if cmd and cmd != RUN:
                self._log(f"[ctrl] 开训前已有指令 {cmd!r}（{ctrl_path()}）")
            return control

        def on_step_end(self, args, state, control, **kw):
            cmd = read()
            if cmd != self._last_seen:
                self._last_seen = cmd
                if cmd and cmd != RUN:
                    self._log(f"[ctrl] step {state.global_step} 读到指令 {cmd!r}")

            if cmd == NEXT_CHECKPOINT:
                self._pending_next_checkpoint = True

            if cmd == SAVE:
                self._log(f"[ctrl] step {state.global_step} 立刻存一次检查点")
                control.should_save = True
                self._consume()

            elif cmd == STOP:
                self._log(f"[ctrl] step {state.global_step} 收到 stop，本步结束后退出")
                self.stop_reason = STOP
                self._consume(sticky=True)
                control.should_training_stop = True

            elif self._pending_next_checkpoint and self._at_save_step(args, state):
                # ⚠️ 这一置位之所以正确：on_step_end 在 _maybe_log_save_evaluate
                # **之前**（trainer.py 2755 vs 2756），所以本步的检查点会先落盘，
                # 然后才走到 2772 行的 break。
                self._log(f"[ctrl] step {state.global_step} 是保存点，"
                          f"检查点落盘后退出")
                self.stop_reason = NEXT_CHECKPOINT
                self._pending_next_checkpoint = False
                self._consume(sticky=True)
                control.should_training_stop = True

            return control

        def on_save(self, args, state, control, **kw):
            # 保险：should_save 若不复位，HF 会**每一步**都存检查点。
            control.should_save = False
            return control

        def on_train_end(self, args, state, control, **kw):
            if self.stop_reason:
                self._log(f"[ctrl] 训练于 step {state.global_step} 按 "
                          f"{self.stop_reason!r} 正常结束；"
                          f"最终模型仍会导出到 output_dir")
            return control

        # -- 工具 ------------------------------------------------------
        def _consume(self, sticky: bool = False) -> None:
            """收尾处理。

            ``stop`` / ``next-checkpoint`` 是**粘性**的：执行后**不清除**旗标。
            这样 ``run_task.py`` 在进程退出后仍能分辨「这是被外部要求停的，不要
            自动重启」，操作者也必须先显式 ``--ctrl run`` 才能再开一轮。
            ``save`` 是一次性的，执行后写回 ``run``，否则会每一步都存检查点。
            """
            if sticky:
                return
            try:
                clear()
            except Exception:                                  # noqa: BLE001
                pass
            self._last_seen = RUN

        @staticmethod
        def _at_save_step(args, state) -> bool:
            strat = getattr(args, "save_strategy", None)
            if strat is None or str(strat).lower().rsplit(".", 1)[-1] != "steps":
                return False
            ss = getattr(args, "save_steps", None)
            if not ss:
                return False
            return state.global_step % int(ss) == 0

    return _ExternalControl()


if __name__ == "__main__":
    raise SystemExit(main())
