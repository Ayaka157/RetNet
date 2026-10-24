"""外部旗标控制的行为回归（`retnet_hf/control.py`）。

不需要真的起训练：回调只用 duck typing 碰 ``args`` / ``state`` / ``control``
三个对象，所以直接喂假的就能把决策逻辑覆盖掉。

重点覆盖**顺序**这个最容易写错的地方：``on_step_end`` 在
``_maybe_log_save_evaluate`` 之前（trainer.py 2755 vs 2756），所以
「next-checkpoint」必须在 ``on_step_end`` 里置位，检查点才会先落盘。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = Path(tempfile.mkdtemp(prefix="ctrl_test_"))
os.environ["RETNET_CTRL_FILE"] = str(_tmp / "ctrl.txt")

import retnet_hf.control as C  # noqa: E402


class Args:
    save_strategy = "steps"
    save_steps = 100


class State:
    def __init__(self, step):
        self.global_step = step


class Ctrl:
    def __init__(self):
        self.should_training_stop = False
        self.should_save = False


def _fresh():
    C.clear()
    return C.make_callback(verbose=False)


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    fails = []

    def check(name, cond, extra=""):
        print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  ' + extra) if extra else ''}")
        if not cond:
            fails.append(name)

    print("== 1. 指令归一化与读写 ==")
    check("alias checkpoint -> next-checkpoint",
          C.normalize("Checkpoint") == C.NEXT_CHECKPOINT)
    check("alias halt -> stop", C.normalize("halt") == C.STOP)
    check("alias resume -> run", C.normalize("resume") == C.RUN)
    check("未知词 -> None", C.normalize("frobnicate") is None)

    C.write("stop")
    check("写入 stop 后读回 stop", C.read() == C.STOP)
    C.write("  NEXT-CHECKPOINT  ")
    check("大小写/空白容错", C.read() == C.NEXT_CHECKPOINT)

    C.ctrl_path().write_text("# 只有注释\n\n", encoding="utf-8")
    check("全是注释 -> None（等同 run）", C.read() is None)
    C.ctrl_path().write_text("\x00\xff 垃圾字节", encoding="utf-8", errors="replace")
    check("垃圾内容不抛异常", C.read() is None)
    C.ctrl_path().unlink()
    check("文件不存在 -> None", C.read() is None)

    print("\n== 2. stop：立刻停，且粘性 ==")
    cb = _fresh()
    C.write(C.STOP)
    ctrl = Ctrl()
    cb.on_step_end(Args(), State(37), ctrl)
    check("置 should_training_stop", ctrl.should_training_stop)
    check("旗标**保留**（粘性，供 run_task.py 判断不重启）", C.read() == C.STOP)

    print("\n== 3. next-checkpoint：非保存步不动，保存步才停 ==")
    cb = _fresh()
    C.write(C.NEXT_CHECKPOINT)
    cb.on_train_begin(Args(), State(0), Ctrl())
    ctrl = Ctrl()
    cb.on_step_end(Args(), State(37), ctrl)          # 37 % 100 != 0
    check("非保存步不停", not ctrl.should_training_stop)
    check("仍未消费", C.read() == C.NEXT_CHECKPOINT)
    ctrl = Ctrl()
    cb.on_step_end(Args(), State(100), ctrl)         # 100 % 100 == 0
    check("保存步停", ctrl.should_training_stop)
    check("旗标保留", C.read() == C.NEXT_CHECKPOINT)

    print("\n== 4. next-checkpoint 在 save_strategy=no 时不应误触发 ==")
    cb = _fresh()
    C.write(C.NEXT_CHECKPOINT)

    class NoSave(Args):
        save_strategy = "no"
        save_steps = None

    ctrl = Ctrl()
    cb.on_step_end(NoSave(), State(100), ctrl)
    check("无保存策略时不置位", not ctrl.should_training_stop)

    print("\n== 5. save：一次性 ==")
    cb = _fresh()
    C.write(C.SAVE)
    ctrl = Ctrl()
    cb.on_step_end(Args(), State(7), ctrl)
    check("置 should_save", ctrl.should_save)
    check("不停止训练", not ctrl.should_training_stop)
    check("旗标被清成 run（否则每步都存）", C.read() == C.RUN)
    ctrl2 = Ctrl()
    cb.on_step_end(Args(), State(8), ctrl2)
    check("下一步不再重复存", not ctrl2.should_save)

    print("\n== 6. on_save 必须复位 should_save（HF 不复位会每步都存）==")
    ctrl = Ctrl()
    ctrl.should_save = True
    cb.on_save(Args(), State(7), ctrl)
    check("on_save 后 should_save=False", not ctrl.should_save)

    print("\n== 7. run / 无旗标：什么都不做 ==")
    cb = _fresh()
    C.write(C.RUN)
    ctrl = Ctrl()
    cb.on_step_end(Args(), State(100), ctrl)
    check("run 不触发任何动作",
          not ctrl.should_training_stop and not ctrl.should_save)

    print()
    if fails:
        print(f"FAIL（{len(fails)} 项）：" + ", ".join(fails))
        return 1
    print("PASS 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
