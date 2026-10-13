"""Shared helpers for the T1 repro scripts (all write inside _verify/)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

VERIFY = Path(__file__).resolve().parent
if str(VERIFY) not in sys.path:
    sys.path.insert(0, str(VERIFY))

import torch  # noqa: E402

import harness as H  # noqa: E402

V = H  # alias so the banner can read notebook/cell hashes
CPU = torch.device("cpu")


def banner(title: str) -> None:
    cmd = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    print("=" * 78)
    print(f"[{title}]")
    print(f"command : python {cmd}")
    print(f"cwd     : {os.getcwd()}")
    print(f"python  : {sys.version.split()[0]}  torch={torch.__version__}")
    print(f"notebook: {V.NOTEBOOK.name} sha256={V.NOTEBOOK_SHA256[:24]}")
    print(f"cell4   : {V.CELL4_REF} (lines={V.CELL4_LINES}) sha256={V.CELL4_SHA256[:24]}")
    print(f"          source read straight from the .ipynb JSON, executed UNMODIFIED")
    print("=" * 78)
    sys.stdout.flush()


def load_env(output_subdir: str = "tiny_out", **cfg_kw) -> dict:
    ds, md = H.setup_tiny_fixtures()
    out = H.SANDBOX / output_subdir
    env = H.exec_cell4(ds, md, out)
    H.tiny_cfg(env, ds, md, out, **cfg_kw)
    return env


def tiny_config(env: dict, **kw) -> object:
    base = {k: v for k, v in H.TINY.items() if k not in ("seq_len", "chunk_size")}
    base.update(kw)
    return env["RetNetHFConfig"](**base)


def build_msr(env: dict, num_heads: int = 1, d_model: int = 64,
              value_dim: int = 64, use_gla: bool = False):
    """A MultiScaleRetention whose key_dim matches the real 1024/16 = 64."""
    cfg = tiny_config(env, d_model=d_model, num_heads=num_heads,
                      value_dim=value_dim, num_layers=1, use_gla=use_gla)
    return env["MultiScaleRetention"](cfg)


def line_of(text: str) -> int:
    return text


def show(label: str, value) -> None:
    print(f"    {label:<52} {value}")


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 74 - len(title)))
    sys.stdout.flush()


def dump_json(name: str, obj) -> Path:
    H.LOGS.mkdir(parents=True, exist_ok=True)
    p = H.LOGS / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p
