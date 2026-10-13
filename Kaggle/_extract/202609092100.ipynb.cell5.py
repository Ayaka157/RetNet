# Lightweight static/model sanity checks before launching TPU workers.
# No XLA device is touched here.
import inspect

assert CFG.seq_len == 8192
assert CFG.per_device_batch > 0
assert CFG.grad_accum > 0
assert CFG.max_steps > 0
assert CFG.model_mode in {"parallel", "chunkwise", "recurrent"}
assert CFG.chunk_size > 0 and CFG.seq_len % CFG.chunk_size == 0

config = RetNetHFConfig.from_pretrained(CFG.model_dir)
model_keys = set(RetNetForCausalLM(config).state_dict().keys())
print(f"[check] config d_model={config.d_model} layers={config.num_layers} "
      f"heads={config.num_heads} vocab={config.vocab_size}")
print(f"[check] checkpoint key count={len(model_keys)}")
print("[check] model definition and training config are structurally valid.")

