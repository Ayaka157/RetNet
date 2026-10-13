"""Shim of ``torch_xla.core.xla_model`` (single-replica CPU semantics).

``xm.all_reduce`` is an identity (optionally scaled).  That is only *correct*
for world_size == 1; when the harness fakes world_size > 1 the benchmark
numbers are meaningless -- the harness only uses that mode to execute
rank-dependent control flow.
"""

import torch

from .. import runtime

REDUCE_SUM = "sum"
REDUCE_MUL = "mul"
REDUCE_AND = "and"
REDUCE_OR = "or"
REDUCE_MAX = "max"
REDUCE_MIN = "min"

_ADDRESS = {"sum": "REDUCE_SUM"}


def is_master_ordinal(ordinal=None) -> bool:
    if ordinal is None:
        ordinal = runtime.global_ordinal()
    return int(ordinal) == 0


def get_ordinal() -> int:
    return runtime.global_ordinal()


def get_world_size() -> int:
    return runtime.world_size()


def xla_device(n=None, devkind=None) -> torch.device:
    del n, devkind
    return torch.device("cpu")


def xrt_world_size(defval=1) -> int:
    del defval
    return runtime.world_size()


def all_reduce(reduce_type, inputs, scale=1.0, groups=None, pin_layout=True,
               **kwargs):
    """Single-replica identity.  ``scale`` is applied like the real op does."""
    del groups, pin_layout, kwargs
    if reduce_type not in (REDUCE_SUM, REDUCE_MUL, REDUCE_AND, REDUCE_OR,
                           REDUCE_MAX, REDUCE_MIN):
        raise ValueError(f"unsupported reduce type {reduce_type!r}")
    if isinstance(inputs, (list, tuple)):
        return [i if scale == 1.0 else i * scale for i in inputs]
    return inputs if scale == 1.0 else inputs * scale


def mark_step() -> None:
    return None


def optimizer_step(optimizer, barrier=False, optimizer_args=None, groups=None,
                   pin_layout=True, **kwargs):
    del barrier, groups, pin_layout, kwargs
    if optimizer_args is None:
        optimizer_args = {}
    return optimizer.step(**optimizer_args)


def rendezvous(tag, payload=b"", groups=None) -> None:
    del tag, payload, groups
    return None


def save(data, file_or_path, master_only=True, global_master=False) -> None:
    if master_only and not is_master_ordinal(global_master):
        return None
    if isinstance(data, torch.Tensor):
        data = data.cpu()
    torch.save(data, file_or_path)
    return None


def optimization_barrier_(tensors):
    return tensors


def wait_device_ops(**kwargs) -> None:
    del kwargs


def set_rng_state(seed, device=None) -> None:
    del seed, device


def get_rng_state(device=None):
    del device
    return torch.get_rng_state()


def parallel_loader(*args, **kwargs):  # pragma: no cover - not used by cell 4
    raise NotImplementedError("not part of the T1 shim")
