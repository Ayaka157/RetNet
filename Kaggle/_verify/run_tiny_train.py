"""End-to-end CPU execution of the notebook's real train_worker().

RUN:
    python run_tiny_train.py fresh         # clean output dir, full loop + export
    python run_tiny_train.py resume        # 2nd run: hits D1 (_latest_checkpoint)
    python run_tiny_train.py resume-re     # resume with `re` injected (harness-only)

Everything happens on the shim device (cpu); the tiny fixtures are in
_verify/_sandbox.  `train_worker` is cell 4's own function, unmodified.
"""

from __future__ import annotations

import shutil
import sys
import time
import traceback

import torch

import vcommon as V

V.banner("E2E  train_worker() on CPU")

mode = sys.argv[1] if len(sys.argv) > 1 else "fresh"
ds, md = V.H.setup_tiny_fixtures()
out = V.H.SANDBOX / "e2e_out"

env = V.H.exec_cell4(ds, md, out)
if mode == "fresh":
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    V.H.tiny_cfg(env, ds, md, out, max_steps=6, save_steps=2, eval_steps=3,
                 per_device_batch=2)
elif mode == "resume":
    V.H.tiny_cfg(env, ds, md, out, max_steps=6, save_steps=2, eval_steps=3,
                 per_device_batch=2)
    # NOTE: no `re` injected on purpose -- this is the plain notebook resume.
else:
    V.H.tiny_cfg(env, ds, md, out, max_steps=10, save_steps=2, eval_steps=3,
                 per_device_batch=2)
    import re                                                  # harness-only
    env["re"] = re
    print("[harness] injected `re` into the cell-4 namespace to get past D1 "
          "(this is the 0909 fix, applied in memory only)")

sys.stdout.flush()
t0 = time.perf_counter()
status = "OK"
try:
    env["train_worker"]()
except BaseException as exc:  # noqa: BLE001
    status = type(exc).__name__
    print(f"\n!!! train_worker() FAILED: {type(exc).__name__}: {exc}")
    tb = traceback.format_exc().splitlines()
    for ln in tb[-12:]:
        print("   " + ln.strip())
print(f"\n[e2e] elapsed {time.perf_counter() - t0:.1f}s  status={status}")

V.section("output directory contents")
if out.exists():
    for p in sorted(out.iterdir()):
        n = len(list(p.iterdir())) if p.is_dir() else p.stat().st_size
        print(f"  {p.name:<24} {'dir with ' + str(n) + ' entries' if p.is_dir() else str(n) + ' bytes'}")

ckpts = sorted((p for p in out.glob("checkpoint-*") if p.is_dir()),
               key=lambda p: int(p.name.split("-")[-1]))
print(f"  checkpoints on disk: {[p.name for p in ckpts]}")
if ckpts:
    ck = ckpts[-1] / "training.pt"
    print(f"  loading {ck.name} with torch.load(map_location='cpu') ...")
    state = torch.load(ck, map_location="cpu", weights_only=False)
    print(f"    keys={sorted(state)}")
    print(f"    step={state['step']} best_eval_loss={state['best_eval_loss']}")
    print(f"    model tensors={len(state['model'])} "
          f"optimizer param_groups={len(state['optimizer']['param_groups'])} "
          f"optimizer state entries={len(state['optimizer']['state'])}")
    print(f"    scheduler keys={sorted(state['scheduler'])}")
    print(f"    config snapshot keys={len(state['config'])}")

V.section("final HF export (line 981 model.save_pretrained with tied weights)")
if (out / "model.safetensors").exists():
    from safetensors import safe_open
    with safe_open(out / "model.safetensors", framework="pt") as f:
        keys = list(f.keys())
        gamma_dtype = (f.get_slice("blocks.0.msr.gamma").get_dtype()
                       if "blocks.0.msr.gamma" in keys else "n/a")
    print(f"  model.safetensors has {len(keys)} tensors")
    print(f"  has lm_head.weight: {'lm_head.weight' in keys}   "
          f"has embedding.weight: {'embedding.weight' in keys}"
          f"   (tied weight deduplication)")
    print(f"  blocks.0.msr.gamma dtype in the export = {gamma_dtype}")
    reloaded = env["RetNetForCausalLM"].from_pretrained(out)
    print(f"  from_pretrained round trip: {len(reloaded.state_dict())} state_dict entries")
    print(f"  tied: lm_head.weight is embedding.weight -> "
          f"{reloaded.lm_head.weight is reloaded.embedding.weight}")
    print(f"  gamma dtype after reload = {reloaded.blocks[0].msr.gamma.dtype}, "
          f"#==1.0 = {int((reloaded.blocks[0].msr.gamma.float() == 1.0).sum())}/"
          f"{reloaded.blocks[0].msr.gamma.numel()}")
else:
    print("  no model.safetensors (run did not reach the final export)")

V.section("summary")
print(f"  mode={mode} status={status}")
if mode == "resume" and status == "NameError":
    print("  D1 PROVEN END-TO-END: the second train_worker() call dies in")
    print("  _latest_checkpoint because a checkpoint-* dir now exists.")
if mode == "resume-re" and status == "OK":
    print("  resume past D1 works: _load_checkpoint -> model/optimizer/scheduler")
    print("  all load (2-group optimizer matches), training continues from the")
    print("  saved step (D9 not reproducible on the notebook's own path).")
