"""Memory-mapped, sharded token dataset for large-corpus pretraining.

The corpus lives as many ``*.npy`` shards (int16 written by
``scripts/prepare_pretrain_v2.py``, or legacy int64 files).  Nothing is loaded
into RAM: each ``__getitem__`` slices ``seq_len + 1`` tokens straight out of a
memory map, so a 3B-token corpus costs 6 GB of disk and ~0 GB of RAM.

Why not ``make_blocks``: it materializes the whole corpus as int64 in RAM
(3B tokens -> 24 GB), which is exactly what we must avoid here.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def find_shards(dirs: Sequence[str | Path], include_val: bool = False) -> List[Path]:
    """Collect shard files from directories (val shards excluded by default)."""
    out: List[Path] = []
    for d in dirs:
        p = Path(d)
        if p.is_file():
            out.append(p)
            continue
        if not p.exists():
            continue
        for f in sorted(p.glob("*.npy")):
            is_val = "_val" in f.name or f.name.startswith("val")
            if is_val != include_val:
                continue
            if f.name.endswith(".tmp.npy"):
                continue
            out.append(f)
        legacy = p / ("val.ids.npy" if include_val else "train.ids.npy")
        if legacy.exists() and legacy not in out:
            out.append(legacy)
    return out


class ShardedTokenBlocks(Dataset):
    """Next-token blocks over a list of flat token shards."""

    def __init__(self, paths: Sequence[str | Path], seq_len: int):
        self.paths = [str(p) for p in paths]
        if not self.paths:
            raise SystemExit("ShardedTokenBlocks: no shards found")
        self.seq_len = int(seq_len)
        self.counts: List[int] = []
        for p in self.paths:
            arr = np.load(p, mmap_mode="r")
            self.counts.append(max(0, (arr.shape[0] - 1) // self.seq_len))
            del arr
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64)
        self._cache: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- helpers
    def _array(self, shard: int) -> np.ndarray:
        arr = self._cache.get(shard)
        if arr is None:
            arr = np.load(self.paths[shard], mmap_mode="r")
            self._cache[shard] = arr
        return arr

    @property
    def total_tokens(self) -> int:
        return int(sum(self.counts) * self.seq_len)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, idx: int):
        shard = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        local = idx - int(self.offsets[shard])
        arr = self._array(shard)
        start = local * self.seq_len
        block = np.asarray(arr[start:start + self.seq_len + 1], dtype=np.int32)
        if block.shape[0] < self.seq_len + 1:                    # last partial block
            pad = np.zeros(self.seq_len + 1 - block.shape[0], dtype=np.int32)
            block = np.concatenate([block, pad])
        return torch.from_numpy(block[:-1]), torch.from_numpy(block[1:])

    # pickling for DataLoader workers: drop the memmap cache
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state
