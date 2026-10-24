"""E13 的守卫必须由**真实的产线产物**满足，而不是测试手写的 meta.json。

旧 `test_plan_gc.py` 的 4b 直接写了一个 `{"rebuild_cmd": ...}` 的 meta，
但 `build_flat_corpus.py` 从来不写这个键 —— 于是“能证明可重建”这件事，
在真实数据上永远不成立，plan 里 6 个 `gc:true` 的阶段全部释放 0 字节。

这个测试跑真实的 `build_flat_corpus.py`（用临时训练出来的小 tokenizer），
再拿它产出的 meta 去调 `gc_stage`。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from retnet_hf.data import StagePackedDataset, _proves_rebuildable  # noqa: E402

SEQ = 8


def _tiny_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=64, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    corpus = ["alpha beta gamma delta epsilon zeta eta theta " * 8 for _ in range(50)]
    tok.train_from_iterator(corpus, trainer)
    tok.save(str(path))


def _produce(tmp_path: Path, tokens_per_shard: int = 200) -> Path:
    import build_flat_corpus

    tok = tmp_path / "tok.json"
    _tiny_tokenizer(tok)
    src = tmp_path / "src.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"text": f"hello world number {i} alpha beta gamma"},
                               ensure_ascii=False) + "\n")
    out = tmp_path / "pack"
    rc = build_flat_corpus.main([
        "--tokenizer", str(tok), "--out", str(out), "--jsonl", str(src),
        "--text-field", "text", "--tokens-per-shard", str(tokens_per_shard),
    ])
    assert rc == 0
    return out


def test_producer_writes_a_rebuild_proof(tmp_path: Path):
    out = _produce(tmp_path)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["rebuild_cmd"].startswith("python scripts/build_flat_corpus.py ")
    assert meta["source_ids"], "source_ids 必须写进 meta（E14 的声明层要用）"
    assert meta["sources"] and meta["text_fields"]
    assert _proves_rebuildable(meta)


def test_gc_accepts_real_producer_meta(tmp_path: Path):
    out = _produce(tmp_path)
    before = sorted(p.name for p in out.glob("*.npy"))
    assert before, "产线应该产出至少一个分片"
    ds = StagePackedDataset([{"name": "p", "data_dir": str(out), "mode": "flat"}], SEQ)
    ds.set_stage(0)
    freed = ds.gc_stage(0)
    assert freed > 0, "真实的产线 meta 仍然被 GC 拒绝 —— E13 的修复没有闭环"
    assert not list(out.glob("*.npy")), "数据分片应当被删除"


def test_gc_accepts_legacy_meta_without_the_new_keys(tmp_path: Path):
    """旧产线的 meta 只有 sources+text_fields，也必须认。"""
    out = _produce(tmp_path)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    meta.pop("rebuild_cmd", None)
    meta.pop("source_ids", None)
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    ds = StagePackedDataset([{"name": "p", "data_dir": str(out), "mode": "flat"}], SEQ)
    ds.set_stage(0)
    assert ds.gc_stage(0) > 0


def test_gc_still_refuses_when_nothing_proves_rebuildability(tmp_path: Path):
    out = _produce(tmp_path)
    (out / "meta.json").write_text(json.dumps({"total_tokens": 1}), encoding="utf-8")
    ds = StagePackedDataset([{"name": "p", "data_dir": str(out), "mode": "flat"}], SEQ)
    ds.set_stage(0)
    assert ds.gc_stage(0) == 0
    assert list(out.glob("*.npy")), "证明不足时不许删"


def test_gc_refuses_a_protected_val_dir(tmp_path: Path):
    """旧 docstring 写了“验证集目录跳过”，代码里没有这条检查（P1-9.5）。"""
    out = _produce(tmp_path)
    (out / "val.ids.npy").write_bytes(b"")          # 名字含 val 也不该被 hash
    ds = StagePackedDataset([{"name": "p", "data_dir": str(out), "mode": "flat"}], SEQ)
    ds.protect(out)
    ds.set_stage(0)
    assert ds.gc_stage(0) == 0
    assert list(out.glob("shard_*.npy"))
