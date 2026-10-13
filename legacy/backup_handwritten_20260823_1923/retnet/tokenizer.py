"""Train/load a BPE tokenizer for the RetNet language model.

The BPE tokenizer is trained with the ``tokenizers`` library (data
pre-processing only).  The RetNet model itself does not use any ready-made
model component.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    # ------------------------------------------------------------------
    @staticmethod
    def train(texts: Iterable[str], vocab_size: int = 8192,
              min_frequency: int = 2, path: Optional[str | Path] = None) -> "BPETokenizer":
        """Train a byte-level BPE tokenizer from a stream of raw texts."""
        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train_from_iterator((t for t in texts if t and t.strip()), trainer)
        wrapper = BPETokenizer(tokenizer)
        if path is not None:
            wrapper.save(path)
        return wrapper

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def special_id(self, token: str) -> int:
        return self._tok.token_to_id(token)

    @property
    def bos_id(self) -> int:
        return self.special_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.special_id(EOS_TOKEN)

    @property
    def unk_id(self) -> int:
        return self.special_id(UNK_TOKEN)

    @property
    def pad_id(self) -> int:
        return self.special_id(PAD_TOKEN)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        return [enc.ids for enc in
                self._tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)
