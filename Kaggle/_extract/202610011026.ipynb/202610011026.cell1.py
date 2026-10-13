# ============================================================
# Kaggle TPU v5e-8 setup and configuration
# ============================================================
import json
import os
from pathlib import Path

# Configure PJRT before importing torch_xla / XLA runtime.
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Kaggle images may expose stale legacy TPU environment variables.
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("CLOUD_TPU_TASK_ID", None)

# Keep optional JAX/TF stacks away from the TPU runtime. This notebook only
# uses PyTorch/XLA.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["WANDB_DISABLED"] = "true"

DATASET_DIR = "/kaggle/input/datasets/ayaka157/retnet-v11-8k-dataset/chat_pack_v11"
MODEL_DIR   = "/kaggle/input/datasets/ayaka157/retnet-272m-init-weights-2026-8-25/retnet_272m_rp"
OUTPUT_DIR  = "/kaggle/working/retnet_272m_v11_8k_tpu"

MODEL_MODE = "chunkwise"

SEQ_LEN          = 8192
PER_DEVICE_BATCH = 4
GRAD_ACCUM       = 1

MAX_STEPS        = 30_000
LR               = 1.1e-4
MIN_LR           = 1.1e-5
WARMUP           = 300
WEIGHT_DECAY     = 0.1
MAX_GRAD_NORM    = 1.0
CHUNK_SIZE       = 64

SAVE_STEPS       = 500
EVAL_STEPS       = 500
SAVE_LIMIT       = 3

def _must_dir(path, name):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")

def _must_file(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{name} file does not exist: {path}")

_must_dir(DATASET_DIR, "DATASET_DIR")
_must_dir(MODEL_DIR, "MODEL_DIR")

for filename in ("train.ids.npy", "train.wcode.npy",
                 "val.ids.npy", "val.wcode.npy"):
    _must_file(Path(DATASET_DIR) / filename, f"dataset/{filename}")

for filename in ("config.json", "model.safetensors"):
    _must_file(Path(MODEL_DIR) / filename, f"model/{filename}")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("[OK] Kaggle paths validated")
print(f"  dataset : {DATASET_DIR}")
print(f"  model   : {MODEL_DIR}")
print(f"  output  : {OUTPUT_DIR}")
print(f"  mode={MODEL_MODE} seq={SEQ_LEN} batch/device={PER_DEVICE_BATCH} "
      f"grad_accum={GRAD_ACCUM}")
