"""延长段的备料：code2 + math 分词，再构建新旧混合的 SFT 包。

为什么不复用 `_data_supervisor.py`：
  那个脚本看到 `hf_pack_code` 已就绪就会**把 data/hf/ling_coder 整个删掉**
  （它会认定这个数据集已经消化完了），而那正是 code2 的原料。这里只做延长段
  需要的事，不碰已有产物。

流程：
  1. 等 ling_coder 的 27 个分片齐全 → 分词到 hf_pack_code2（上限 5B token）
  2. 等 finemath 下载完 → 分词到 hf_pack_math（上限 0.3B）
  3. 调用 build_sft_mix.py 生成 hf_pack_sft
  4. 每步都可重入（产物已存在就跳过）
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/RetNet")
PY = "/root/miniconda3/bin/python"
MANIFEST = ROOT / "logs/_dl_manifest.tsv"
BASE = ROOT / "data/hf"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def manifest() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if not MANIFEST.exists():
        return out
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        p = line.rstrip("\r").split("\t")
        if len(p) < 4:
            continue
        out.setdefault(p[1], {})[p[2]] = int(p[3])
    return out


def complete_bytes(local: str) -> tuple[int, int]:
    want = manifest().get(local, {})
    have_b = want_b = 0
    for fname, w in want.items():
        want_b += w
        f = BASE / local / fname
        if f.exists() and f.stat().st_size == w:
            have_b += w
    return have_b, want_b


def pack_ready(d: Path) -> bool:
    try:
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return int(m.get("total_tokens", m.get("tokens", 0))) > 0 and len(m.get("shards", m.get("rows", 1))) > 0
    except Exception:                                          # noqa: BLE001
        return False


def tokenize(name: str, src: str, out: str, cap: int, patterns: list[str]) -> bool:
    out_dir = ROOT / out
    if pack_ready(out_dir):
        log(f"  {name} 已就绪，跳过")
        return True
    have_b, want_b = complete_bytes(src)
    if have_b < want_b:
        log(f"  {name} 还没下完（{have_b / 1e9:.2f}/{want_b / 1e9:.2f} GB）")
        return False
    files = []
    for pat in patterns:
        files.extend(sorted(str(p) for p in (BASE / src).rglob(pat)))
    if not files:
        log(f"  {name} 找不到输入文件")
        return False
    log(f"  开始分词 {name}：{len(files)} 个文件 -> {out}")
    cmd = [PY, "-u", "scripts/build_flat_corpus.py",
           "--tokenizer", "data_cache/chat_pack_v13/bpe_tokenizer.json",
           "--out", out, "--parquet", *files,
           "--text-field", "auto", "--max-tokens", str(cap),
           "--tokens-per-shard", "250000000"]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    ok = rc == 0 and pack_ready(out_dir)
    log(f"  {name} 分词{'完成' if ok else f'失败 rc={rc}'}")
    return ok


def main() -> int:
    log("延长段备料启动")
    mix_done = False
    for rnd in range(1, 240):
        c2 = tokenize("code2", "ling_coder", "data_cache/hf_pack_code2",
                      5_000_000_000, ["*.parquet"])
        mt = tokenize("math", "math_zh", "data_cache/hf_pack_math",
                      300_000_000, ["*.parquet"])
        # prog_zh：BAAI 中文编程语料（apache-2.0），用来把唯一 token 推过 20/参数。
        # 子代理实测口径：中文约 0.27 token/字节，0.74 GB -> 约 0.3B token。
        pg = tokenize("prog", "prog_zh", "data_cache/hf_pack_prog",
                      350_000_000, ["*.parquet"])
        # ⚠️ 混合包只依赖 code2 + math，**不依赖 prog**。
        # prog 在下载顺序里排最后，如果要求三个都齐，prog 一慢就永远不建混合包，
        # 而 sft_mix 段明天 15:30 就要用（HF 混合比例里本来也没有 prog）。
        if c2 and mt and not mix_done:
            log("code2 + math 就绪，构建 SFT 混合包（不等 prog）")
            rc = subprocess.call(
                [PY, "-u", "scripts/build_sft_mix.py", "--out", "data_cache/hf_pack_sft",
                 "--rows", "300000"], cwd=str(ROOT))
            if rc == 0:
                mix_done = True
                log("SFT 混合包完成")
            else:
                log(f"混合包构建失败 rc={rc}")
        if mix_done and pg:
            log("全部完成（code2 / math / prog / sft_mix）")
            return 0
        time.sleep(300)
    log("达到最大轮数")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
