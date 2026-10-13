"""Shim of ``torch_xla.distributed.xla_multiprocessing``.

Real ``spawn`` forks/creates ``nprocs`` TPU worker processes.  This shim has no
TPU to hand out: it calls the target once, in-process, with index 0.  It exists
so that ``import torch_xla.distributed.xla_multiprocessing`` (performed by
accelerate's TPU branch) does not explode when the shim package is on sys.path.

Note for D7: the real ``notebook_launcher`` calls
``xmp.spawn(launcher, args=args, start_method="fork")`` -- see the quoted source
in FINDINGS_V1.md.  Fork means every worker process *inherits* the notebook
kernel's module globals, which is why cell 4's top-level ``import torch_xla``
(line 14) makes ``torch_xla.device()`` resolvable inside ``train_worker``.
"""


def spawn(fn, args=(), nprocs=None, join=True, daemon=False, start_method=None):
    del nprocs, join, daemon, start_method
    return fn(0, *args)


def start_processes(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("not part of the T1 shim")
