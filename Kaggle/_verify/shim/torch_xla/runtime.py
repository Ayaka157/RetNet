"""Shim of ``torch_xla.runtime`` (single-replica CPU).

``world_size()`` / ``global_ordinal()`` can be faked with the env vars
``XLA_SHIM_WORLD_SIZE`` and ``XLA_SHIM_ORDINAL`` so that the notebook's
rank-dependent branches can be *executed* locally.  No collectives happen.
"""

import os

import torch


def world_size() -> int:
    return int(os.environ.get("XLA_SHIM_WORLD_SIZE", "1"))


def global_ordinal() -> int:
    return int(os.environ.get("XLA_SHIM_ORDINAL", "0"))


def local_ordinal() -> int:
    return global_ordinal()


def global_runtime_device_ordinal() -> int:
    return global_ordinal()


def local_device_ordinal() -> int:
    return 0


def addressable_device_count() -> int:
    return 1


def device_type() -> str:
    return "CPU"


def device_type_to_device(kind=None) -> torch.device:
    del kind
    return torch.device("cpu")


def set_device(dev) -> None:
    del dev


def is_initialized() -> bool:
    return True


def mark_step() -> None:
    return None


def get_memory_info(device=None):
    del device
    return {"kb_free": float("nan"), "kb_total": float("nan")}
