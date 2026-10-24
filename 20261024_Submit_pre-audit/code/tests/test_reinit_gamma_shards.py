"""P1-7：`reinit_gamma.py` 必须能处理分片检查点，不能只读第一片。

旧实现 `sorted(glob("*.safetensors"))[0]`，把 `model-00001-of-00002` 当成全部，
写出的模型缺一半键；`from_pretrained` 不报错，缺的键被随机初始化。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from safetensors.torch import load_file, save_file  # noqa: E402

import reinit_gamma  # noqa: E402


def _sharded(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    s1 = {"blocks.0.msr.gamma": torch.tensor([0.5, 0.6]),
          "embedding.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)}
    s2 = {"blocks.1.msr.gamma": torch.tensor([0.7, 0.8]),
          "final_norm.weight": torch.ones(4)}
    n1, n2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    save_file(s1, str(src / n1))
    save_file(s2, str(src / n2))
    wm = {k: n1 for k in s1}
    wm.update({k: n2 for k in s2})
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wm}), encoding="utf-8")
    (src / "config.json").write_text("{}", encoding="utf-8")
    return src


def test_sharded_checkpoint_is_merged(tmp_path: Path):
    src = _sharded(tmp_path)
    dst = tmp_path / "dst"
    assert reinit_gamma.main(["--model", str(src), "--out", str(dst)]) == 0
    out = load_file(str(dst / "model.safetensors"))
    src_meta = json.loads((src / "model.safetensors.index.json").read_text())
    assert set(out) == set(src_meta["weight_map"]), "写出的键集必须等于源索引的键集"
    assert not (dst / "model.safetensors.index.json").exists(), \
        "写出的是单文件，不能把源的分片索引一起搬过去"
    want = reinit_gamma.build_gamma(2).float()
    assert torch.allclose(out["blocks.0.msr.gamma"], want)
    assert torch.allclose(out["blocks.1.msr.gamma"], want), "第二个分片的 gamma 没被替换"
    # 非 gamma 权重原样保留
    assert torch.equal(out["embedding.weight"],
                       torch.arange(16, dtype=torch.float32).reshape(4, 4))


def test_multiple_files_without_index_are_refused(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    save_file({"blocks.0.msr.gamma": torch.tensor([0.5, 0.6]),
               "embedding.weight": torch.zeros(2, 2)}, str(src / "a.safetensors"))
    save_file({"blocks.1.msr.gamma": torch.tensor([0.7, 0.8])},
              str(src / "b.safetensors"))
    with pytest.raises(SystemExit) as exc:
        reinit_gamma.main(["--model", str(src), "--out", str(tmp_path / "dst")])
    assert "index" in str(exc.value)
    assert not (tmp_path / "dst" / "model.safetensors").exists()
