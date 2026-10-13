## TPU training design

PyTorch/XLA 的多进程训练路径使用 `DistributedSampler` 做数据切分、`MpDeviceLoader` 做 host→TPU 输入流水，以及 `xm.optimizer_step()` 做 replica 间梯度聚合。

这里不再生成 `/kaggle/working/train_retnet_tpu.py`，也不再使用 `accelerate launch --num_processes=8`。Notebook 直接使用 `accelerate.notebook_launcher()` 启动 Kaggle TPU worker；它在 Kaggle TPU 环境下走 XLA multiprocessing。