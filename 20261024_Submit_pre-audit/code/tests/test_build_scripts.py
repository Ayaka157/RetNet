"""E09/E10/E11 的守卫：建包脚本的默认值、早失败与无边界拒绝。

这些测试直接跑真实的命令行脚本（不是 import 后调内部函数），
因为要守的正是“命令行默认值”与“进程退出码”这两件事。

没有本地 tokenizer 时整体跳过：`data_cache/` 被 gitignore，远端 clone 里没有它。
跳过是显式的，不会伪装成通过。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TOK = ROOT / "data_cache" / "chat_pack_v13" / "bpe_tokenizer.json"

pytestmark = pytest.mark.skipif(
    not TOK.exists(), reason="本地没有 data_cache/chat_pack_v13/bpe_tokenizer.json")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=600)


def _jsonl(tmp_path: Path, n: int = 40) -> Path:
    p = tmp_path / "src.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"text": f"第 {i} 段：这是用来测试文档边界的样例文本。" * 4},
                               ensure_ascii=False) + "\n")
    return p


def test_add_eos_is_default(tmp_path: Path):
    """E09：不传 --add-eos 也必须写 eos，并在 meta.json 里记 add_eos=true。"""
    src = _jsonl(tmp_path)
    out = tmp_path / "pack"
    r = _run(["scripts/build_flat_corpus.py", "--tokenizer", str(TOK),
              "--out", str(out), "--jsonl", str(src),
              "--text-field", "text", "--tokens-per-shard", "1000000"])
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    shards = sorted(out.glob("shard_*.npy"))
    assert shards, "没有产出分片"
    arr = np.load(shards[0])
    assert int((arr == 2).sum()) >= 40, "40 条记录应当至少写入 40 个 eos(2)"
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("add_eos") is True


def test_wrong_text_field_fails_fast(tmp_path: Path):
    """E10：字段名写错必须在几秒内失败，而不是跑完分词再报错。"""
    src = _jsonl(tmp_path)
    out = tmp_path / "pack"
    r = _run(["scripts/build_flat_corpus.py", "--tokenizer", str(TOK),
              "--out", str(out), "--jsonl", str(src),
              "--text-field", "no_such_field", "--tokens-per-shard", "1000000"])
    assert r.returncode != 0, "字段不存在却返回了 0"
    both = r.stdout + r.stderr
    assert "一个都不存在" in both or "可用字段" in both, both[-2000:]


def test_ctx_mix_refuses_boundaryless(tmp_path: Path):
    """E11：来源没有 meta.json 时，build_ctx_mix 必须拒绝，而不是照常造包。"""
    out = tmp_path / "ctx"
    r = _run(["scripts/build_ctx_mix.py", "--out", str(out), "--tokens", "1000000",
              "--code2", str(tmp_path / "code2"),
              "--math", str(tmp_path / "math"),
              "--prog", str(tmp_path / "prog"),
              "--sft", str(tmp_path / "sft")])
    assert r.returncode != 0, "无边界来源却返回了 0"
    both = r.stdout + r.stderr
    assert "E11" in both, both[-2000:]


def test_ctx_mix_allows_explicit_opt_out(tmp_path: Path):
    """显式 --allow-boundaryless 时应继续走（随后会因为没数据而失败，但不是被 E11 拦住）。"""
    out = tmp_path / "ctx"
    r = _run(["scripts/build_ctx_mix.py", "--out", str(out), "--tokens", "1000000",
              "--code2", str(tmp_path / "code2"),
              "--math", str(tmp_path / "math"),
              "--prog", str(tmp_path / "prog"),
              "--sft", str(tmp_path / "sft"),
              "--allow-boundaryless"])
    both = r.stdout + r.stderr
    # 警告行本身会提到 E11，所以不能拿 "E11" 当拒绝的标志；
    # 要看的是"拒绝删除/无法确认边界"这类拒绝对话是否出现。
    assert "接受无文档边界来源" in both, both[-2000:]
    assert "无法确认文档边界" not in both and "add_eos=false" not in both
