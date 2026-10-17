"""Packed chat dataset + collator for the HF Trainer (retnet_hf)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedChatDataset(Dataset):
    """Reads the packed int16 rows written by scripts/build_chat_mix_v4.py.

    Row layout (seq_len + 1 tokens): [x_0 ... x_{seq-1} x_seq]
    - input_ids = row[:-1], labels = row[1:]
    - wcodes    = per-target weight codes (0 pad / 1 reply / 2 prompt / 3 plain)
    """

    def __init__(self, ids_path, wcode_path, seq_len: int):
        self.ids = np.load(ids_path, mmap_mode="r")
        self.codes = np.load(wcode_path, mmap_mode="r")
        assert self.ids.ndim == 2 and self.ids.shape == self.codes.shape
        assert self.ids.shape[1] == seq_len + 1, \
            f"rows are {self.ids.shape[1]} wide, expected seq_len+1={seq_len + 1}"
        self.seq_len = seq_len

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        row = self.ids[idx]
        return {
            "input_ids": torch.from_numpy(row[:-1].astype(np.int64)),
            "labels": torch.from_numpy(row[1:].astype(np.int64)),
            "wcodes": torch.from_numpy(self.codes[idx][1:].astype(np.int64)),
        }


class ChatDataCollator:
    def __call__(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "wcodes": torch.stack([b["wcodes"] for b in batch]),
        }


# --------------------------------------------------------------- 多阶段数据集

class _PackedStage:
    """一段 packed 语料：二维 ``(rows, seq_len+1)`` 的 ids + wcode。"""

    def __init__(self, name, data_dir, seq_len, weight=None):
        self.name = name
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        ids = self.data_dir / "train.ids.npy"
        wcode = self.data_dir / "train.wcode.npy"
        if not ids.exists():
            raise FileNotFoundError(f"阶段「{name}」找不到 {ids}")
        self.ids = np.load(ids, mmap_mode="r")
        self.codes = None if weight is not None else np.load(wcode, mmap_mode="r")
        assert self.ids.ndim == 2 and self.ids.shape[1] == seq_len + 1, \
            f"阶段「{name}」行宽 {self.ids.shape[1]} != seq_len+1={seq_len + 1}"
        if self.codes is not None:
            assert self.ids.shape == self.codes.shape, f"阶段「{name}」ids/wcode 形状不一致"
        self.weight = weight

    def __len__(self):
        return self.ids.shape[0]

    def row(self, idx):
        row = self.ids[idx]
        if self.codes is not None:
            wc = self.codes[idx][1:].astype(np.int64)
        else:
            wc = np.full(self.seq_len, int(self.weight), dtype=np.int64)
        return row[:-1], row[1:], wc


class _FlatStage:
    """一段 flat 语料：目录下若干一维 token 流（int16/int32），不打包、不复制。

    行数 = 每个分片 ``len // (seq_len+1)`` 之和（分片末尾余数丢弃，占比 <0.05%）。
    这样 ``data_cache/pretrain_v2`` 那种「已 tokenize 但没打包」的语料可以直接用，
    不必先花几小时生成一份 5 GB 的重复文件。
    """

    def __init__(self, name, data_dir, seq_len, weight=1):
        self.name = name
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.weight = int(weight)
        width = seq_len + 1
        # 排除验证集分片（形如 wikipedia_zh_val.npy）
        files = sorted(p for p in self.data_dir.glob("*.npy") if "val" not in p.name.lower())
        if not files:
            raise FileNotFoundError(f"阶段「{name}」在 {self.data_dir} 下没有可用的 *.npy")
        self.shards = []
        self._starts = []
        total = 0
        for p in files:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 1:
                raise ValueError(f"{p} 是 {arr.ndim} 维，flat 模式要求一维 token 流")
            n = arr.shape[0] // width
            if n == 0:
                continue
            self.shards.append((p, arr, n))
            self._starts.append(total)
            total += n
        if total == 0:
            raise ValueError(f"阶段「{name}」的分片长度都不足一行（{width} token）")
        self._rows = total
        self._wc = np.full(seq_len, self.weight, dtype=np.int64)

    def __len__(self):
        return self._rows

    def row(self, idx):
        width = self.seq_len + 1
        # 行号 -> 分片（二分查找）
        import bisect
        si = bisect.bisect_right(self._starts, idx) - 1
        local = idx - self._starts[si]
        _p, arr, _n = self.shards[si]
        off = local * width
        chunk = arr[off:off + width]
        return chunk[:-1], chunk[1:], self._wc


class StagePackedDataset(Dataset):
    """**可在训练中途切换语料阶段**的数据集。

    DataLoader 与优化器都不重建：``set_stage(i)`` 只改内部指针。
    阶段是惰性打开的 —— 某一段没轮到之前不会 mmap 它的文件。

    ``__getitem__`` 对行号取模，避免「切段后旧 sampler 的索引越界」：
    HF Trainer 的 RandomSampler 是按上一次的 ``__len__`` 采样的，切段那一刻
    手上可能还攥着旧范围的行号。
    """

    def __init__(self, stages: list[dict], seq_len: int):
        self._seq_len = int(seq_len)
        self._specs: list[dict] = []
        self._cache: dict[int, object] = {}
        self._active = 0
        self.update_stages(stages)

    # -- 阶段管理 ---------------------------------------------------
    def update_stages(self, stages: list[dict]) -> None:
        """替换阶段列表（计划热更新时调用）。已打开的阶段尽量复用。"""
        new_specs = []
        for s in stages:
            new_specs.append({
                "name": str(s["name"]),
                "data_dir": str(s["data_dir"]),
                "mode": str(s.get("mode", "auto")),
                "weight": s.get("weight"),
            })
        # 按 (name, data_dir) 复用已打开的阶段，避免热更新时重复 mmap
        reuse = {}
        for i, sp in enumerate(self._specs):
            if i in self._cache:
                reuse[(sp["name"], sp["data_dir"])] = self._cache[i]
        self._specs = new_specs
        self._cache = {}
        for i, sp in enumerate(new_specs):
            key = (sp["name"], sp["data_dir"])
            if key in reuse:
                self._cache[i] = reuse[key]
        self._active = min(self._active, max(0, len(self._specs) - 1))

    def _stage(self, i: int):
        st = self._cache.get(i)
        if st is None:
            sp = self._specs[i]
            mode = sp["mode"]
            if mode == "auto":
                mode = "packed" if (Path(sp["data_dir"]) / "train.ids.npy").exists() else "flat"
            if mode == "packed":
                st = _PackedStage(sp["name"], sp["data_dir"], self._seq_len, sp.get("weight"))
            elif mode == "flat":
                st = _FlatStage(sp["name"], sp["data_dir"], self._seq_len)
            else:
                raise ValueError(f"未知的 stage mode {mode!r}（阶段「{sp['name']}」）")
            self._cache[i] = st
            print(f"[stage] 打开阶段 {i}「{sp['name']}」mode={mode} rows={len(st)}",
                  flush=True)
        return st

    def set_stage(self, i: int) -> None:
        if not (0 <= i < len(self._specs)):
            raise IndexError(f"stage {i} 越界（共 {len(self._specs)} 段）")
        self._stage(i)                     # 提前打开，失败就在这里响亮报错
        self._active = int(i)

    @property
    def active_index(self) -> int:
        return self._active

    @property
    def active_name(self) -> str:
        return self._specs[self._active]["name"]

    # -- Dataset ----------------------------------------------------
    def __len__(self):
        return len(self._stage(self._active))

    def __getitem__(self, idx):
        st = self._stage(self._active)
        n = len(st)
        if n <= 0:
            raise IndexError(f"阶段「{st.name}」为空")
        inp, lab, wc = st.row(int(idx) % n)
        return {
            "input_ids": torch.from_numpy(inp.astype(np.int64)),
            "labels": torch.from_numpy(lab.astype(np.int64)),
            "wcodes": torch.from_numpy(wc.astype(np.int64)),
        }

