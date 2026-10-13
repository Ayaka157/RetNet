# RetNet 272M · 8K · Kaggle TPU v5e-8

本 notebook 改为 **PyTorch/XLA 原生训练循环**，而不是 `Transformers Trainer + accelerate CLI` 混合模式。

核心约定：

- Kaggle TPU v5e-8 使用 PJRT，8 个 TPU core 做 data parallel。
- `torch_xla` 只在 `train_worker()` 内访问 TPU device。
- CPU `DataLoader` → `DistributedSampler` → `MpDeviceLoader`。
- optimizer 使用 `xm.optimizer_step()` 完成跨 core 梯度同步与 XLA step。
- 训练/验证 batch 固定，避免不必要的 XLA recompilation。
- checkpoint 仅由 rank 0 写入，其他 worker 用 rendezvous 同步。
- 保留原始 RetNet checkpoint 的参数命名；修复初始化函数错误清零 `GroupNorm.weight` / learned `gamma` 的问题。
- `chunkwise` 使用 `torch_xla.experimental.scan`，把原来的 Python chunk loop 表达为 XLA scan；训练后的 `state_dict` 与原模型完全同构。

**运行顺序：** 配置 → 安装依赖 → 定义模型与训练函数 → 启动 8-core TPU。

> 注意：不要在训练 worker 之外创建 `xm.xla_device()`、`Accelerator()` 或加载到 XLA 的模型。

**精度策略：** TPU 使用 `XLA_USE_BF16=1` + `torch.autocast("xla", dtype=torch.bfloat16)`；模型参数保持可加载的普通 PyTorch/HF `state_dict`，loss/reduction 明确保留 FP32。
