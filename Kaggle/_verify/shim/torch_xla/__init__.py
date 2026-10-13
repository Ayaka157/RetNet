"""CPU-only stand-in for the third-party ``torch_xla`` package  (T1 harness).

WHY THIS EXISTS
---------------
cell 4 of ``Kaggle/202610011026.ipynb`` cannot even be *imported* on a machine
without ``torch_xla``::

    line 14:  import torch_xla
    line 21:  from torch_xla.utils.checkpoint import checkpoint as _xla_checkpoint

This package is injected by ``_verify/harness.py`` via ``sys.path`` so the
notebook's cell 4 can be executed **unmodified** on CPU.  It is NOT an XLA
implementation and it does NOT reproduce XLA semantics.

WHAT IT IS
----------
* single "device": ``torch.device("cpu")``
* single rank by default; ``XLA_SHIM_WORLD_SIZE`` / ``XLA_SHIM_ORDINAL`` env
  vars can fake a multi-rank world so that rank-dependent control flow can be
  executed (no real collectives are performed).
* ``torch_xla.device()`` -> ``torch.device('cpu')``
* ``xm.all_reduce`` is an identity (x1, or x``scale``) because world_size == 1
  is the only case with correct semantics; ``xm.optimizer_step`` is plain
  ``optimizer.step()``; ``xm.rendezvous`` / ``xm.mark_step`` are no-ops.

EVERY substitution is recorded in the run manifest written by
``harness.py`` (``_verify/logs/harness_injections.json``).
"""

import torch  # noqa: F401

from . import core, distributed, runtime, utils  # noqa: F401

__version__ = "0.0.0+cpu-shim"


def device(kind=None):
    """Mirror of ``torch_xla.device()`` (torch_xla >= 2.1).

    Real implementation returns an ``xla`` device; the shim returns CPU.
    """
    del kind
    return torch.device("cpu")


def device_of(tensor):
    return tensor.device
