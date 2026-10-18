#!/usr/bin/env bash
# 部署加固后的 _watchdog.sh：校验语法 → 测参数校验 → 安全替换（不动运行中的旧 inode）
set -uo pipefail
cd /root/autodl-tmp/RetNet/scripts/linux

sed -i 's/\r$//' _watchdog.sh.new

echo "=== 1) 语法检查 ==="
if bash -n _watchdog.sh.new; then echo "  OK"; else echo "  FAIL"; exit 1; fi

echo "=== 2) 参数校验：传 9.0 应被拒绝 ==="
bash _watchdog.sh.new 9.0 10
rc=$?
echo "  退出码=$rc（期望 2）"

echo "=== 3) 参数校验：传 9 应通过校验（会进入等待循环，用 timeout 掐断）==="
timeout 3 bash _watchdog.sh.new 9 10
echo "  超时退出码=$?（124 表示它正常进入了等待循环）"

echo "=== 4) 安全替换：运行中的旧进程持有旧 inode，不受影响 ==="
mv _watchdog.sh _watchdog.sh.v1
mv _watchdog.sh.new _watchdog.sh
chmod +x _watchdog.sh
ls -la _watchdog.sh _watchdog.sh.v1
