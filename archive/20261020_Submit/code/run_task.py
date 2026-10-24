"""Config-driven training task runner (deploy target: 192.168.1.100).

Standardizes RetNet SFT launches into ONE config file + ONE command.  The
config defines model, data, precision, optimizer, schedule, checkpoints, and
optional post-training steps.  Run via:

    python run_task.py --config tasks/v10_272m.json           # run (or resume)
    python run_task.py --config tasks/v10_272m.json --status  # show last loss
    python run_task.py --config tasks/v10_272m.json --dry     # validate only
    python run_task.py --config tasks/v10_272m.json --monitor # tail loss live

Config schema (all optional except model_dir/data_dir/output_dir):
{
  "name": "v10_272m",
  "model_dir": "checkpoints/retnet_272m_rp",
  "data_dir": "data_cache/chat_pack_v10",
  "output_dir": "checkpoints/retnet_272m_v10",
  "precision": "fp32",            // bf16 | fp16 | fp32
  "mode": "chunkwise",            // parallel | chunkwise
  "chunk_size": 64,
  "batch_size": 6,
  "eval_batch_mult": 4,
  "max_steps": 65000,
  "lr": 1.1e-4,
  "min_lr": 1.1e-5,
  "warmup": 500,
  "weight_decay": 0.1,
  "max_grad_norm": 1.0,
  "save_steps": 2500,
  "eval_steps": 2500,
  "save_total_limit": 3,
  "grad_checkpoint": true,
  "compile": false,
  "seed": 1234,
  "resume": false,                // resume from output_dir
  "load_best": true,
  "auto_restart_max": 0,          // >0: auto-restart N times after crash
  "restart_wait_min": 10
}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REQUIRED = ("model_dir", "data_dir", "output_dir")
#: 用 ``plan`` 时数据源与步数都由计划文件决定，data_dir/max_steps 不再必填
REQUIRED_WITH_PLAN = ("model_dir", "output_dir")
DEFAULTS = {
    "name": "task",
    "precision": "fp32",
    "mode": "chunkwise",
    "chunk_size": 64,
    "batch_size": 6,
    "eval_batch_mult": 4,
    "max_steps": 65000,
    "lr": 1.1e-4,
    "min_lr": 1.1e-5,
    "warmup": 500,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "save_steps": 2500,
    "eval_steps": 2500,
    "save_total_limit": 3,
    "grad_checkpoint": True,
    "compile": False,
    "fast_chunkwise": False,
    "grad_accum": 1,
    "optim": None,
    "dataloader_workers": 0,
    "seed": 1234,
    "resume": False,
    "load_best": True,
    "auto_restart_max": 0,
    "restart_wait_min": 10,
    "nproc": 1,              # >1 时走 torch.distributed.run（多卡 DDP）
    "master_port": 29500,
    "train_script": "retnet_hf/train.py",  # override for forks (e.g. retnet_hf_fp16/train.py)
}


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = REQUIRED_WITH_PLAN if cfg.get("plan") else REQUIRED
    for k in required:
        if not cfg.get(k):
            raise ValueError(f"missing required key: {k}")
    if cfg.get("plan"):
        if not Path(str(cfg["plan"])).exists():
            raise ValueError(f"plan 文件不存在: {cfg['plan']}")
        cfg.setdefault("data_dir", "")
        cfg.setdefault("max_steps", 0)
    merged = {**DEFAULTS, **cfg}
    if merged["compile"]:
        merged["compile"] = True
    return merged


def build_cmd(cfg):
    """Translate config -> train.py argv list.

    ``plan`` 存在时改走计划模式：数据源、总步数、LR 曲线全部由计划文件决定，
    并且训练进程会**热加载**该文件（见 retnet_hf/plan.py）。此时不传
    ``--data_dir`` / ``--max_steps``，否则会和计划里的值打架。
    配置里仍可写 ``data_dir``，仅作为日志/兼容用途。
    """
    if cfg.get("plan"):
        cmd = [
            sys.executable, cfg.get("train_script", "retnet_hf/train.py"),
            "--model_dir", str(cfg["model_dir"]),
            "--output_dir", str(cfg["output_dir"]),
            "--plan", str(cfg["plan"]),
            "--batch_size", str(cfg["batch_size"]),
            "--mode", str(cfg["mode"]),
            "--chunk_size", str(cfg["chunk_size"]),
            "--precision", str(cfg["precision"]),
            "--seed", str(cfg["seed"]),
            "--eval_batches", "40",
            # 2026-10-20: plan 分支此前漏了这一项，实际一直用 train.py 的默认 4。
            # 见 docs/00 E20。
            "--eval_batch_mult", str(cfg.get("eval_batch_mult", 1)),
            "--save_total_limit", str(cfg.get("save_total_limit", 4)),
            "--weight_decay", str(cfg.get("weight_decay", 0.1)),
            "--max_grad_norm", str(cfg.get("max_grad_norm", 1.0)),
            "--dataloader_workers", str(cfg.get("dataloader_workers", 0)),
        ]
        return cmd
    return [
        sys.executable, cfg.get("train_script", "retnet_hf/train.py"),
        "--model_dir", str(cfg["model_dir"]),
        "--data_dir", str(cfg["data_dir"]),
        "--output_dir", str(cfg["output_dir"]),
        "--max_steps", str(cfg["max_steps"]),
        "--batch_size", str(cfg["batch_size"]),
        "--lr", str(cfg["lr"]),
        "--min_lr", str(cfg["min_lr"]),
        "--warmup", str(cfg["warmup"]),
        "--mode", str(cfg["mode"]),
        "--chunk_size", str(cfg["chunk_size"]),
        "--precision", str(cfg["precision"]),
        "--seed", str(cfg["seed"]),
        "--eval_batches", "40",
        # NOTE(2026-10-10): 配置模板里一直有 eval_batch_mult（见文件头 schema），
        # 但以前**从没传下去**，于是 retnet_hf/train.py 里硬编码的 *4 一直生效。
        # 实测 1×4 会让第一次评测多花 128 s、峰值 reserved 冲到 7.65/8.00 GiB；
        # 1×1 则评测开销从 6.6% 降到 2.9%、reserved 回到 6.73 GiB。
        "--eval_batch_mult", str(cfg.get("eval_batch_mult", 4)),
        # NOTE(2026-10-10): save_total_limit / save_steps / eval_steps /
        # weight_decay / max_grad_norm 此前**从没拼进来过**，train.py 的硬编码
        # 默认值一直生效。模板里写 save_total_limit=4，实际跑的是 3，
        # 多出来的 checkpoint-10000 被 HF 按 mtime 轮转静默删掉了
        # （删除日志是 logger.info，transformers 默认 WARNING 不打印）。
        "--save_total_limit", str(cfg.get("save_total_limit", 3)),
        "--save_steps", str(cfg.get("save_steps", 2500)),
        "--eval_steps", str(cfg.get("eval_steps", 2500)),
        "--weight_decay", str(cfg.get("weight_decay", 0.1)),
        "--max_grad_norm", str(cfg.get("max_grad_norm", 1.0)),
        # 无条件拼进去（不是条件 flag）：这样 check_task_config.py 能直接验证
        # 「JSON 里写了 dataloader_workers 就真的生效」。默认 0 = 行为与改动前一致。
        "--dataloader_workers", str(cfg.get("dataloader_workers", 0)),
    ]
    # flags appended conditionally below


def build_cmd_full(cfg):
    cmd = build_cmd(cfg)
    if cfg["grad_checkpoint"]:
        cmd.append("--grad_checkpoint")
    if not cfg["compile"]:
        cmd.append("--no-compile")
    # compile_mode 只在真的编译时才传（否则它就是个死字段，check_task_config 会报出来）
    if cfg.get("compile") and cfg.get("compile_mode"):
        cmd += ["--compile_mode", str(cfg["compile_mode"])]
    # 无条件拼进去（不是条件 flag）：以前是 `if grad_accum > 1` 才拼，于是
    # `grad_accum: 1` 的配置在 check_task_config.py 里被报成「死字段」。
    # train.py 的默认值也是 1，所以无条件拼不改变任何行为，但让守卫的规则保持
    # 「JSON 写了就必须在命令行里出现」这一条，不必为它开特例。
    cmd += ["--grad_accum", str(cfg.get("grad_accum", 1))]
    if cfg.get("optim"):
        cmd += ["--optim", str(cfg["optim"])]
    if cfg.get("no_decay_1d"):
        cmd.append("--no_decay_1d")
    if cfg.get("ddp_find_unused_parameters"):
        cmd.append("--ddp_find_unused_parameters")
    if cfg["resume"]:
        cmd.append("--resume")
    # 多卡：用 torch.distributed.run 包一层，train.py 本身不用改入口。
    # ⚠️ batch_size 是**每张卡**的：2 卡 × batch 8 = 单卡 batch 16 的等效批量，
    #    所以切到多卡时要把 batch_size 减半、steps 不变（LR/评测节奏也不变）。
    nproc = int(cfg.get("nproc", 1) or 1)
    if nproc > 1:
        script, rest = cmd[1], cmd[2:]
        cmd = [sys.executable, "-m", "torch.distributed.run",
               "--nproc_per_node", str(nproc),
               "--master_port", str(cfg.get("master_port", 29500)),
               script] + rest
    return cmd


def last_loss(output_dir: Path, n: int = 5) -> list:
    """Read latest loss log entries from trainer_state.json of newest ckpt."""
    ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        return []
    st = ckpts[-1] / "trainer_state.json"
    if not st.exists():
        return []
    state = json.loads(st.read_text(encoding="utf-8"))
    hist = [e for e in state.get("log_history", []) if "loss" in e or "eval_loss" in e]
    return hist[-n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=False,
                    help="任务配置 JSON；用 --ctrl 发指令时可以不填")
    ap.add_argument("--status", action="store_true", help="print latest loss only")
    ap.add_argument("--dry", action="store_true", help="validate config + print cmd")
    ap.add_argument("--monitor", action="store_true", help="watch trainer_state loss")
    ap.add_argument("--resume", action="store_true", help="force resume from output_dir")
    ap.add_argument("--ctrl", default=None,
                    help="给正在跑的训练发指令后退出：run / stop / next-checkpoint / "
                         "save / status。写 logs/ctrl.txt，训练在**步边界**消费它"
                         "（见 retnet_hf/control.py）。用这个就不必 taskkill。")
    args = ap.parse_args()

    if args.ctrl is not None:
        from retnet_hf.control import main as _ctrl_main
        return _ctrl_main([args.ctrl])

    if not args.config:
        ap.error("需要 --config（除非用 --ctrl 发指令）")

    # 粘性停止守卫：旗标里还留着 stop / next-checkpoint 时**不要开训**，
    # 否则新进程会在第一步就立刻退出，看起来像「启动失败」。
    from retnet_hf.control import (read as _ctrl_read, ctrl_path as _ctrl_path,
                                   STOP as _C_STOP, NEXT_CHECKPOINT as _C_NEXT)
    _pending = _ctrl_read()
    if _pending in (_C_STOP, _C_NEXT):
        print(f"[ctrl] 旗标 {_ctrl_path()} 里还有未清除的指令 {_pending!r} —— 不会开训。\n"
              f"       先跑 `python run_task.py --ctrl run`（或删掉该文件）再启动。")
        raise SystemExit(1)
    print(f"[ctrl] 外部控制旗标 {_ctrl_path()}（当前 {_pending or 'run'}）")

    cfg = load_config(args.config)
    if args.resume:
        cfg["resume"] = True
    print(f"task: {cfg['name']}  precision={cfg['precision']} mode={cfg['mode']} "
          f"max_steps={cfg['max_steps']} batch={cfg['batch_size']} resume={cfg['resume']}")
    print(f"   model: {cfg['model_dir']}")
    print(f"   data : {cfg['data_dir']}")
    print(f"   out  : {cfg['output_dir']}")

    # 优化开关的状态放在 --dry 之前，dry-run 也要能看见
    env = dict(os.environ)
    if cfg.get("fast_chunkwise"):
        # 批量 chunkwise：数学等价、内核发射次数降一个数量级。
        # 由 modeling_retnet 在 import 时读取，所以必须设在子进程环境里。
        env["RETNET_FAST_CHUNKWISE"] = "1"
        print("[opt] RETNET_FAST_CHUNKWISE=1（批量 chunkwise）")
    else:
        print("[opt] fast_chunkwise=false（用参考的逐 chunk 循环）")
    if cfg.get("compile"):
        try:
            import triton  # noqa: F401
            print(f"[opt] torch.compile 可用（triton {triton.__version__}）")
        except ImportError:
            print("[opt] ⚠ 配置要求 compile，但当前环境没有 triton —— inductor 是惰性编译，"
                  "train.py 的 try/except 接不住这个错误，会静默地不提速。"
                  "先 `pip install triton-windows`。")
    else:
        print("[opt] compile=false")
    if cfg.get("grad_accum", 1) > 1 or cfg.get("optim"):
        print(f"[opt] grad_accum={cfg.get('grad_accum', 1)} "
              f"optim={cfg.get('optim') or '默认 adamw_torch'} "
              f"grad_checkpoint={cfg['grad_checkpoint']}")

    if args.dry:
        print("\n[cmd]")
        print(" ".join(build_cmd_full(cfg)))
        return

    out = Path(cfg["output_dir"])
    if args.status:
        hist = last_loss(out)
        if not hist:
            print("no checkpoint yet")
        else:
            for e in hist:
                print(" ", e)
        return

    if args.monitor:
        print("watching (ctrl-C to quit)...")
        while True:
            hist = last_loss(out)
            if hist:
                print(f"\rstep={hist[-1].get('step')} loss={hist[-1].get('loss')} "
                      f"eval={hist[-1].get('eval_loss')}   ", end="")
            time.sleep(30)
        # unreachable

    # run with optional auto-restart
    attempts = 0
    final_rc = 0
    while True:
        cmd = build_cmd_full(cfg)
        print("\n[train] " + " ".join(x for x in cmd if x not in ("retnet_hf/train.py",)))
        rc = subprocess.call(cmd, env=env)
        print(f"[exit] rc={rc}")
        final_rc = rc
        if rc in (0, 1) and attempts >= cfg["auto_restart_max"]:
            break
        if rc != 0 and attempts < cfg["auto_restart_max"]:
            # 外部要求停止时不要自动重启。stop / next-checkpoint 是**粘性**的，
            # 进程退出后旗标仍在，所以这里分辨得出来 —— 这也正是当初为什么
            # 只能靠「先杀 run_task.py 再杀 train.py」的原因。
            _now = _ctrl_read()
            if _now in (_C_STOP, _C_NEXT):
                print(f"[ctrl] 检测到停止指令 {_now!r}，不自动重启")
                break
            attempts += 1
            print(f"[restart] attempt {attempts}/{cfg['auto_restart_max']} "
                  f"wait {cfg['restart_wait_min']}min")
            time.sleep(cfg["restart_wait_min"] * 60)
            cfg["resume"] = True  # restart always resumes
        else:
            break

    # ⚠ 2026-10-11 修复（静默失败）：以前无论子进程怎么死，main() 都正常返回，
    # run_task.py 自己以 0 退出。实测 train.py 因 TrainingArguments 校验抛异常
    # （rc=1）时，日志里明明有 traceback 和 `[exit] rc=1`，run_train.sh 拿到的却是
    # 「训练进程退出码: 0」—— 「崩了」被记成「训完了」，无人值守时还会接着自动关机。
    # 现在把子进程退出码原样传播；rc=1 同时覆盖 `raise SystemExit(...)` 的启动拒绝
    # 和未捕获异常两种失败，二者对操作者都是失败，不再区分。
    if final_rc != 0:
        print(f"[FAIL] 训练进程以 rc={final_rc} 结束 —— 视为失败"
              f"（日志里没有 TRAINING DONE）", flush=True)
    sys.exit(final_rc)


if __name__ == "__main__":
    main()
