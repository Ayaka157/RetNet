# Launch all TPU cores through Accelerate's Kaggle/XLA launcher.
# On TPU, notebook_launcher/xmp.spawn supplies the PJRT ordinal to its
# internal launcher, not to the user training function. Therefore train_worker()
# must take no positional index argument; it reads rank from torch_xla.runtime.
from accelerate import notebook_launcher

notebook_launcher(
    train_worker,
    args=(),
)
