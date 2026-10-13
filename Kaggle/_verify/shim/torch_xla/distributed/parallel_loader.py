"""Shim of ``torch_xla.distributed.parallel_loader.MpDeviceLoader``.

Real class prefetches H2D copies onto the XLA device and inserts
``mark_step()`` at batch boundaries.  The shim just moves tensors to the shim
device (CPU) so ``MpDeviceLoader`` behaves as a pass-through -- the shim
device *is* where the CPU DataLoader already produced the batch.
"""

import torch


class MpDeviceLoader:
    def __init__(self, loader, device, **kwargs):
        self._loader = loader
        self._device = device
        self._kwargs = kwargs

    def __iter__(self):
        for batch in self._loader:
            if isinstance(batch, dict):
                yield {
                    k: (v.to(self._device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
            else:
                yield batch

    def __len__(self):
        return len(self._loader)

    @property
    def loader(self):
        return self._loader


class ParallelLoader(MpDeviceLoader):
    pass
