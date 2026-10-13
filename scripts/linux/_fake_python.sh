#!/usr/bin/env bash
# 假解释器 —— 只给 _test_run_train.sh 用。
#
# 目的：在**不真的训练**的前提下，验证 run_train.sh 能不能分辨
# 「真跑完」「崩溃」「rc=0 但没跑完」「配置校验不过」四种情况。
# 它替掉的是 /root/miniconda3/bin/python。
#
#   -c / - 开头的调用（取版本号、读写临时配置 JSON）转发给真 python；
#   其余（即 `... run_task.py --config ...`）返回伪造结果，由环境变量控制。
set -uo pipefail
REAL_PY="${REAL_PY:-/root/miniconda3/bin/python}"

case "${1:-}" in
  -c|-) exec "$REAL_PY" "$@" ;;
esac

# 走到这里说明 run_train.sh 在调 run_task.py
for a in "$@"; do
  if [[ "$a" == "--dry" ]]; then
    case "${FAKE_DRY:-ok}" in
      ok)  echo "[cmd]"; echo "fake --dry：配置没问题"; exit 0 ;;
      bad) echo "!! 未知 flag --bogus（伪造）"; exit 3 ;;
      *)   echo "FAKE_DRY 未知: $FAKE_DRY"; exit 9 ;;
    esac
  fi
done

echo "[train] fake run_task.py（没有真的训练）"
case "${FAKE_MODE:-ok}" in
  ok)     echo "TRAINING DONE"; exit 0 ;;
  fail)   echo "Traceback (most recent call last):"
          echo '  File "retnet_hf/train.py", line 175, in main'
          echo "ValueError: 伪造的崩溃（rc=1）"; exit 1 ;;
  silent) echo "（伪造：进程正常退出，但从来没打完成标记）"; exit 0 ;;
  *)      echo "FAKE_MODE 未知: $FAKE_MODE"; exit 9 ;;
esac
