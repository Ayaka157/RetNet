"""Forensics on the observed Kaggle crash (BrokenProcessPool at step 1).

This script does NOT claim a root cause.  It extracts what the saved artifacts
actually prove, labels each statement proven/reasoned, and lists the evidence
the next TPU run must capture.

RUN:  python repro_crash_forensics.py
"""

from __future__ import annotations

import re
from collections import Counter

import vcommon as V

V.banner("CRASH FORENSICS  what the saved Kaggle output proves")

EX = V.H.REPO / "Kaggle" / "_extract"
stream_f = EX / "0909_cell6_stream.txt"
dump_f = EX / "cell6_output_dump.txt"
stream = stream_f.read_text(encoding="utf-8", errors="replace")
dump = dump_f.read_text(encoding="utf-8", errors="replace")

cur = V.H.CELL4_SRC.splitlines()
old_f = EX / "202609092100.ipynb.cell4.py"
old = old_f.read_text(encoding="utf-8").splitlines() if old_f.exists() else []

V.section("proven fact 1: which notebook revision produced the crash log?")
def find_mark_step(lines):
    return [i + 1 for i, l in enumerate(lines) if l.strip() == "xm.mark_step()"]
print(f"  current 202610011026 cell4 xm.mark_step() lines : {find_mark_step(cur)}")
print(f"  older   202609092100 cell4 xm.mark_step() lines : {find_mark_step(old)}")
print(f"  the stream warns at /tmp/ipykernel_14/109047568.py:935 "
      f"-> matches the CURRENT notebook (935 is one of its mark_step lines)")
print("  line 934 of the current notebook is the `... / world_size` use.")

V.section("proven fact 2: how many worker processes reached the step-1 block?")
print(f"  occurrences of ':935' in the stream : {stream.count(':935')}")
print(f"  occurrences of '[XLA]' banner       : {stream.count('[XLA]')}")
print(f"  occurrences of '[train] step='      : {stream.count('[train] step=')}")
print(f"  occurrences of 'Traceback'          : {stream.count('Traceback')}")
print("  8 warnings at :935 == 8 worker processes each called line 935 once")
print("  (a single process calling it repeatedly is impossible: line 935 runs")
print("   once per step and step 1 prints; the '[train] step=1' line never")
print("   appeared, and the loop's other steps are 20/30000.)")

V.section("proven fact 3: the failure was native, not a Python exception")
hits = {
    "SIGTERM received by PID": [m for m in re.findall(r"SIGTERM received by PID (\d+) \(TID \d+\) on cpu \d+ from PID (\d+)", stream)],
    "FailureSignalHandler": [l for l in stream.splitlines() if "FailureSignalHandler" in l],
}
print(f"  SIGTERM senders -> `from PID` column: "
      f"{Counter(src for _, src in hits['SIGTERM received by PID'])}")
print(f"  SIGTERM targets : {sorted(pid for pid, _ in hits['SIGTERM received by PID'])}")
for ln in hits["FailureSignalHandler"]:
    print(f"  {ln}")
print("  reading: one worker was ALREADY inside absl's FailureSignalHandler")
print("  ('Signal 15 raised ... while already in FailureSignalHandler') when the")
print("  kernel (PID 14) SIGTERMed the rest -> a native fault / abort preceded the")
print("  teardown.  A Python exception in a child is re-raised in the parent by")
print("  ProcessPoolExecutor with a RemoteTraceback; the dump has only")
print("  BrokenProcessPool, no traceback -> the child did not raise in Python.")
print("  (proven: the log lines; reasoned: the causal reading above)")

V.section("proven fact 4: the crash is at the FIRST XLA materialization")
print("  The last normal output before the signals is the 8x mark_step warning at")
print("  line 935.  Everything before it (forward, backward, clip_grad_norm_,")
print("  xm.optimizer_step, the all_reduce at 932-934) is lazy: on XLA only")
print("  mark_step()/sync() materializes the graph.  So step 1 died while the")
print("  graph was being executed/compiled, not while Python was running cell 4's")
print("  code.  Also: the master's '[train] step=1' print sits AFTER line 935 and")
print("  never appeared -> the master did not get past the sync either.")

V.section("proven fact 5: D7/D8 are not the crash (see repro_d7/d8)")
print("  device=xla:0 in the banner was produced by torch_xla.device() at line 766")
print("  -> the kernel-global name resolved in the workers (fork inherits it).")
print("  8 processes reached line 935, three lines past the `/ world_size` use at")
print("  line 934 -> no NameError on any rank.")

V.section("what cell 4 does at that point (facts, for whoever debugs on TPU)")
for n in range(919, 936):
    print(f"  {n:4d}: {cur[n-1]}")
print("  quantities (real config): params 272M bf16 = 0.54 GB, grads 0.54 GB,")
print("  AdamW state 2 x 0.54 GB, plus activations of 20 blocks x 128 chunks at")
print("  seq 8192/batch 4 with block-level gradient checkpointing; the whole")
print("  step-1 graph (20 blocks x 128 unrolled chunks) is materialized in ONE")
print("  sync at line 935.  (arithmetic = reasoned only)")

V.section("torch_xla 2.9.0 source facts (fetched from pytorch/xla v2.9.0) that "
          "shape the diagnosis")
facts = [
    ("xm.optimizer_step(barrier=False) does NOT sync",
     "optimizer_step() calls reduce_gradients() + optimizer.step(), and only "
     "calls torch_xla.sync() when barrier=True.  cell 4 passes barrier=False, so "
     "the first materialization of the step-1 graph really is line 935."),
    ("xm.mark_step() is a thin deprecated alias",
     "@deprecated -> torch_xla.sync(wait, reset_scope).  This is why the stream "
     "shows the deprecation warning exactly at the call site."),
    ("xm.save() converts NESTED XLA tensors",
     "save() -> _maybe_convert_to_cpu(data) -> ToXlaTensorArena/xu.for_each_instance, "
     "which walks nested dict/list/tuple containers.  So _save_checkpoint's nested "
     "{model,optimizer,scheduler} dict is NOT a defect (checked, not assumed)."),
    ("xm.all_reduce(pin_layout=True) is documented as a compile-crash risk",
     "docstring: 'Layout pining can prevent potential data corruption ... but it "
     "might cause some xla compilation to fail. Unpin the layout when you see "
     "error message like \"HloModule has a mix of layout constrained\".'  cell 4 "
     "uses the default pin_layout=True at lines 750 and 932."),
    ("xm.get_memory_info() is the right OOM probe",
     "returns {'bytes_used', 'bytes_limit', 'peak_bytes_used'} for the device."),
]
for i, (what, why) in enumerate(facts, 1):
    print(f"  {i}. {what}\n     {why}")
print("  source: https://raw.githubusercontent.com/pytorch/xla/v2.9.0/torch_xla/core/xla_model.py "
      "(matches the version in the crash traceback paths)")

V.section("candidate mechanisms -- all REASONED ONLY, none provable locally")
cands = [
    ("libtpu/XLA resource exhaustion (HBM OOM) during the first sync",
     "sizes: 272M bf16 params + grads + 2 Adam states + checkpointed activations, "
     "single unrolled 128-chunk graph; libtpu aborts the process natively"),
    ("XLA compile-time failure of the unrolled chunkwise graph",
     "128 chunks x 20 layers with mixed bf16/fp32 islands from the fp32 gate "
     "restore; a compile abort also kills the process without a Python traceback"),
    ("a collective that never completes (all_reduce at 932 / optimizer_step) "
     "while another rank is stuck",
     "would hang, not abort; the teardown would look the same from the kernel"),
    ("layout-pinned all_reduce failing XLA compilation (pin_layout=True default)",
     "documented failure mode in the torch_xla 2.9 docstring; testable by passing "
     "pin_layout=False at lines 750 and 932"),
]
for i, (what, why) in enumerate(cands, 1):
    print(f"  {i}. {what}\n     why plausible: {why}")
print("  None of these can be settled with the captured artifact: the worker")
print("  stderr that contained the real error is not in the saved output.")

V.section("what the next TPU run MUST capture")
todo = [
    "child stderr redirected to a file (open('/kaggle/working/worker.log','w') in "
    "the worker) -- the kernel stream misses the fatal libtpu message",
    "PYTHONFAULTHANDLER=1, TORCH_SHOW_CPP_STACKTRACES=1, XLA_IR_DEBUG=1, "
    "XLA_HLO_DEBUG=1 exported before PJRT init",
    "wrap the train_worker body in try/except + traceback.print_exc() + "
    "sys.stderr.flush() so any Python-level failure is visible",
    "bisect the graph: xm.mark_step() right after model.to(device) and right "
    "after the first forward, then after backward, to find which sync dies",
    "shrink test: max_steps=1, per_device_batch=1, seq_len=1024/2048, "
    "grad checkpointing on/off -> size-dependent = OOM, size-independent = bug",
    "xr.get_memory_info(device) / xm.get_memory_info before and after the first "
    "sync to see HBM headroom",
    "try without XLA_USE_BF16=1 (cell 1 line 10) to rule the mixed bf16/fp32 "
    "islands in or out",
]
for t in todo:
    print("  - " + t)

V.section("verdict")
print("  UNDETERMINED.  What is established: the current notebook reached the")
print("  first XLA sync of step 1 on all 8 workers, then a native fault killed a")
print("  worker before any Python exception could be reported, and the launcher")
print("  tore the pool down (SIGTERM from the kernel PID 14).")
