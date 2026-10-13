"""Print GPU usage and the latest training log lines."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="train_retnet_chat.log")
    ap.add_argument("--lines", type=int, default=25)
    args = ap.parse_args()
    print("===== GPU =====")
    print(run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv"))
    print("===== processes =====")
    print(run("powershell.exe -NoProfile -Command \"Get-CimInstance Win32_Process -Filter \\\"Name='python.exe'\\\" | Select-Object ProcessId,CommandLine | Format-List\""))
    print(f"===== tail {args.log} =====")
    p = Path(args.log)
    if p.exists():
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="gbk", errors="replace")
        print("\n".join(text.splitlines()[-args.lines:]))
    else:
        print("log not found")


if __name__ == "__main__":
    main()
