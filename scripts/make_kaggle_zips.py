"""Package the two Kaggle delivery zips.

  1. retnet_v11_8k_dataset.zip   -> chat_pack_v11/  (8K packed dataset)
  2. retnet_272m_init_weights.zip -> retnet_272m_rp/ (config + model.safetensors only)

Top-level folder is preserved so Kaggle extraction gives clean paths:
  /kaggle/input/<name>/chat_pack_v11/train.ids.npy
  /kaggle/input/<name>/retnet_272m_rp/model.safetensors
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def make_zip(out_name: str, src_dir: str, files: list[str]):
    out = ROOT / out_name
    src = ROOT / src_dir
    top = Path(src_dir).name  # "chat_pack_v11" / "retnet_272m_rp"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in files:
            p = src / rel
            assert p.is_file(), f"missing {p}"
            # archive name keeps ONLY the top-level folder: <top>/<rel>
            z.write(p, arcname=f"{top}/{rel}")
    print(f"[OK] {out.name}  ({out.stat().st_size / 1e9:.2f} GB)  <- {len(files)} files")


# --- dataset ---
make_zip(
    "retnet_v11_8k_dataset.zip",
    "data_cache/chat_pack_v11",
    ["train.ids.npy", "train.wcode.npy", "val.ids.npy", "val.wcode.npy",
     "bpe_tokenizer.json", "meta.json"],
)

# --- weights (essentials only, skip the 3 ~3.1GB checkpoint subdirs) ---
make_zip(
    "retnet_272m_init_weights.zip",
    "checkpoints/retnet_272m_rp",
    ["config.json", "model.safetensors"],
)

print("\n[DONE] both zips written to", ROOT)
