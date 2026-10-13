#!/usr/bin/env bash
# 实例侧环境探针：由本机用 `Get-Content _probe_instance.sh -Raw | ssh gpu-host "bash -s"` 送入。
# 之所以写成文件而不是一行命令：PowerShell -> ssh 的内联引号极易被吃掉（本会话栽过多次），
# 走 stdin 就完全绕开了这个问题。
echo "=== whoami ==="
whoami

echo "=== PATH（非交互 shell）==="
echo "$PATH"

echo "=== 候选 python 解释器 ==="
for p in /root/miniconda3/bin/python /root/miniconda3/bin/python3 \
         /opt/conda/bin/python /usr/bin/python3 /usr/local/bin/python3; do
  if [ -x "$p" ]; then echo "FOUND $p -> $("$p" -V 2>&1)"; fi
done

echo "=== /etc/profile.d ==="
ls -1 /etc/profile.d/ 2>/dev/null

echo "=== gpu-host.env.sh 内容 ==="
head -25 /etc/profile.d/gpu-host.env.sh 2>/dev/null

echo "=== ~/.bashrc 里的 conda/PATH 行 ==="
grep -nE 'conda|PATH' /root/.bashrc 2>/dev/null | head -10

echo "=== /root/miniconda3/bin 里前 15 项 ==="
ls -1 /root/miniconda3/bin 2>/dev/null | head -15

echo "=== 加上 PATH 之后 ==="
export PATH=/root/miniconda3/bin:$PATH
which python python3 2>/dev/null
python -V 2>&1
python - <<'PY' 2>&1
import sys
print("executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda,
          "| available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0),
              "| cap:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch 导入失败:", type(e).__name__, e)
for m in ("transformers", "bitsandbytes", "accelerate", "numpy", "safetensors"):
    try:
        mod = __import__(m)
        print(f"{m}:", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"{m}: 未安装 ({type(e).__name__})")
PY

echo "=== triton / ninja（torch.compile 需要）==="
python -c "import triton;print('triton',triton.__version__)" 2>&1 | head -2
which ninja gcc g++ 2>/dev/null
