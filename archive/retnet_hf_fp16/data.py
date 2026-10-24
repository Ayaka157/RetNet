"""Packed chat dataset + collator for the HF Trainer (retnet_hf)."""
from __future__ import annotations

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
