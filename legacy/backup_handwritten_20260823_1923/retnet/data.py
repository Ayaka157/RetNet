"""Data loading and tokenization for the RetNet language model.

Uses the HuggingFace ``datasets`` library to pull ``wikitext-2-raw-v1`` and
prepares a local integer-token cache.  The cache is only created once.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .tokenizer import BPETokenizer

logger = logging.getLogger("retnet.data")


def _nonempty_texts(ds):
    for text in ds["text"]:
        if text and text.strip():
            yield text


def prepare_wikitext(seq_len: int = 256, vocab_size: int = 8192,
                     cache_dir: str | Path = "data_cache/wikitext2",
                     tokenizer_path: Optional[str | Path] = None,
                     force_rebuild: bool = False,
                     dataset_config: str = "wikitext-2-raw-v1") -> Dict[str, np.ndarray]:
    """Download/prepare a raw Wikitext config and return integer token arrays.

    Supported HuggingFace configs include ``wikitext-2-raw-v1`` (small) and
    ``wikitext-103-raw-v1`` (about 100M words, used for the larger model).

    Returns:
        dict with keys ``train`` and ``validation``.  Each value is an int64
        NumPy array of BPE token ids.  A BOS token is placed before every
        non-empty text block and an EOS token after it.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok_path = Path(tokenizer_path) if tokenizer_path else cache_dir / "bpe_tokenizer.json"
    arrays = {split: cache_dir / f"{split}.ids.npy" for split in ("train", "validation")}

    if (not force_rebuild and tok_path.exists()
            and all(p.exists() for p in arrays.values())):
        tokenizer = BPETokenizer.load(tok_path)
        return {s: np.load(str(arrays[s]), mmap_mode="r") for s in arrays}

    from datasets import load_dataset

    logger.info("Loading %s from HuggingFace ...", dataset_config)
    ds = load_dataset("wikitext", dataset_config)

    if not tok_path.exists():
        logger.info("Training BPE tokenizer (vocab_size=%d) ...", vocab_size)
        tokenizer = BPETokenizer.train(_nonempty_texts(ds["train"]), vocab_size=vocab_size,
                                       path=tok_path)
        logger.info("Tokenizer vocab size: %d", tokenizer.vocab_size)
    else:
        tokenizer = BPETokenizer.load(tok_path)

    bos, eos = tokenizer.bos_id, tokenizer.eos_id
    for split in ("train", "validation"):
        texts = [t for t in _nonempty_texts(ds[split])]
        chunks = []
        batch_size = 5000
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            flat = []
            for enc in tokenizer.encode_batch(batch):
                flat.append(bos)
                flat.extend(enc)
                flat.append(eos)
            chunks.append(np.asarray(flat, dtype=np.int64))
        arr = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.int64)
        np.save(str(arrays[split]), arr)
        logger.info("%s split: %d tokens, array saved to %s", split, len(arr), arrays[split])

    return {s: np.load(str(arrays[s]), mmap_mode="r") for s in arrays}


def make_blocks(ids: np.ndarray, seq_len: int, copy: bool = True):
    """Create non-overlapping x/y blocks for next-token prediction."""
    # By default copy because a read-only mmap cannot back writable tensors.
    ids = np.array(ids, dtype=np.int64, copy=copy)
    n_blocks = (len(ids) - 1) // seq_len
    usable = n_blocks * seq_len
    x = ids[:usable].reshape(n_blocks, seq_len)
    y = ids[1:usable + 1].reshape(n_blocks, seq_len)
    return x, y


def get_dataloaders(cache_dir: str | Path = "data_cache/wikitext2",
                    seq_len: int = 256, vocab_size: int = 8192,
                    batch_size: int = 16, num_workers: int = 2,
                    tokenizer_path: Optional[str | Path] = None,
                    force_rebuild: bool = False,
                    dataset_config: str = "wikitext-2-raw-v1"):
    """Return (train_loader, val_loader, tokenizer)."""
    arrays = prepare_wikitext(seq_len=seq_len, vocab_size=vocab_size,
                              cache_dir=cache_dir, tokenizer_path=tokenizer_path,
                              force_rebuild=force_rebuild,
                              dataset_config=dataset_config)
    train_x, train_y = make_blocks(arrays["train"], seq_len)
    val_x, val_y = make_blocks(arrays["validation"], seq_len)

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0))
    tokenizer = BPETokenizer.load(Path(cache_dir) / "bpe_tokenizer.json")
    return train_loader, val_loader, tokenizer
