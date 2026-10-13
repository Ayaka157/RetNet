"""Shim of ``torch_xla.utils.checkpoint.checkpoint``.

The real function (pytorch/xla ``torch_xla/utils/checkpoint.py``) wraps
``torch.utils.checkpoint.checkpoint`` with XLA-specific handling: it inserts
``xm.optimization_barrier_`` on the recompute inputs and (reentrant mode only)
re-seeds the XLA RNG for the recompute.

The shim keeps the *same call contract* so that cell 4's gradient-checkpointing
path is really executed, but it drops the XLA-specific RNG/barrier handling.
``XLA_SHIM_NO_CHECKPOINT=1`` makes it call the function directly (no
checkpointing at all) -- used only to isolate checkpoint-related behaviour.
"""

import os

import torch.utils.checkpoint as _torch_checkpoint


def checkpoint(function, *args, **kwargs):
    if os.environ.get("XLA_SHIM_NO_CHECKPOINT") == "1":
        return function(*args, **kwargs)
    if "use_reentrant" not in kwargs and "context_fn" not in kwargs:
        # torch >= 2.10 deprecated the old reentrant implementation; the
        # non-reentrant implementation is the one with matching semantics
        # (inputs kept by reference, recompute in backward).
        # XLA_SHIM_REENTRANT=1 selects the legacy reentrant implementation so
        # that the notebook's stated contract (torch_xla uses reentrant mode for
        # its RNG handling) can be exercised too.
        kwargs["use_reentrant"] = os.environ.get("XLA_SHIM_REENTRANT") == "1"
    return _torch_checkpoint.checkpoint(function, *args, **kwargs)
