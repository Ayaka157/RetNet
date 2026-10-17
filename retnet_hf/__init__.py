"""retnet_hf 包入口（**惰性导出**）。

## 为什么惰性

`retnet_hf/data.py` 只依赖 numpy 与 torch，不依赖 transformers。但 eager 的
`from ... import ...` 会让 `from retnet_hf.data import StagePackedDataset` 顺带把
整个模型模块拉进来 —— 本地环境 tokenizers 版本一旦与 transformers 不匹配
（实测 0.23.3 vs 要求 ≤0.23.0），**连纯数据层的单元测试都跑不起来**，
而这个问题在实例上并不存在。惰性导出把「用数据层」和「用模型层」解耦。

2026-10-17 加。
"""
from __future__ import annotations

import importlib

_LAZY = {
    "RetNetHFConfig": ("retnet_hf.modeling_retnet", "RetNetHFConfig"),
    "RetNetForCausalLM": ("retnet_hf.modeling_retnet", "RetNetForCausalLM"),
    "PackedChatDataset": ("retnet_hf.data", "PackedChatDataset"),
    "ChatDataCollator": ("retnet_hf.data", "ChatDataCollator"),
    "StagePackedDataset": ("retnet_hf.data", "StagePackedDataset"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
