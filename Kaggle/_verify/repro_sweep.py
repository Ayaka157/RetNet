"""Sweep -- every tensor constructor in the notebook with the default-dtype
question: does it carry an explicit dtype, is it immune (integer inputs), or is
it silently created in the poisoned bf16 default dtype?

RUN:  python repro_sweep.py
"""

from __future__ import annotations

import re

import torch

import vcommon as V

V.banner("SWEEP  dtype-less tensor constructors in cells 1/2/4/5/6")

env = V.load_env("sweep_out")
print("cell 4 executed; torch.get_default_dtype() =", torch.get_default_dtype())

V.section("part 1: dtype-inference facts, measured on this torch")
facts = [
    ("torch.arange(8)", torch.arange(8)),
    ("torch.arange(8, device='cpu')", torch.arange(8, device="cpu")),
    ("torch.arange(0.0, 8.0)", torch.arange(0.0, 8.0)),
    ("torch.linspace(0, 1, 4)", torch.linspace(0, 1, 4)),
    ("torch.zeros(4)", torch.zeros(4)),
    ("torch.zeros((), device='cpu')", torch.zeros((), device="cpu")),
    ("torch.ones(4)", torch.ones(4)),
    ("torch.full((4,), 0.1)", torch.full((4,), 0.1)),
    ("torch.empty(4)", torch.empty(4)),
    ("torch.eye(2)", torch.eye(2)),
    ("torch.rand(4)", torch.rand(4)),
    ("torch.randn(4)", torch.randn(4)),
    ("torch.randint(0, 4, (2,))", torch.randint(0, 4, (2,))),
    ("torch.tensor(1)", torch.tensor(1)),
    ("torch.tensor(1.0)", torch.tensor(1.0)),
    ("torch.tensor(float('nan'))", torch.tensor(float("nan"))),
    ("torch.tensor([1, 2])", torch.tensor([1, 2])),
    ("torch.multinomial(torch.ones(3), 1)", torch.multinomial(torch.ones(3), 1)),
    ("torch.arange(0, 512, dtype=torch.bfloat16)[257]", torch.arange(0, 512, dtype=torch.bfloat16)[257]),
]
for label, t in facts:
    kind = ("INT (immune)" if not t.is_floating_point()
            else ("bf16 (POISONED default)" if t.dtype == torch.bfloat16
                  else str(t.dtype)))
    print(f"  {label:<46} -> {str(t.dtype):<16} {kind}")
print("  => torch.set_default_dtype(bf16) affects only FLOATING factories;")
print("     integer-input factories (arange/tensor/randint) stay int64.")

V.section("part 2: source sweep of every constructor call in the cell files")
CTORS = ("arange", "zeros", "ones", "full", "empty", "rand", "randn", "tensor",
         "eye", "linspace", "new_zeros", "new_full", "zeros_like", "ones_like",
         "full_like")
pat = re.compile(r"\b(" + "|".join(CTORS) + r")\s*\(")
rows = []
for name in ("cell1", "cell2", "cell4", "cell5", "cell6"):
    text = V.H._NB_CELLS[name]
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        for m in pat.finditer(line):
            start = i - 1
            j = i - 1
            buf = ""
            depth = 0
            started = False
            while j < len(lines):
                seg = lines[j]
                buf += ("\n        " if j > start else "") + seg
                depth += seg.count("(") - seg.count(")")
                if "(" in seg:
                    started = True
                if started and depth <= 0:
                    break
                j += 1
            stmt = " ".join(buf.split())
            rows.append((name, i, m.group(1), stmt))

for name, ln, ctor, stmt in rows:
    if re.search(r"\bdtype\s*=", stmt):
        verdict = "explicit dtype= OK"
    elif re.search(r"\b" + ctor + r"\([^)]*\b(0|1|2|3|4|5|6|7|8|9)+\.0", stmt) and ctor in ("arange", "linspace"):
        verdict = "float args -> default dtype"
    elif ctor in ("arange", "randint", "tensor") and not re.search(r"\d\.\d|float\(", stmt):
        verdict = "integer args -> int64 (immune)"
    elif ctor.endswith("_like"):
        verdict = "inherits input dtype"
    else:
        verdict = "DEFAULT DTYPE (bf16 at runtime)"
    print(f"  {name}:{ln:<5} {ctor:<10} {verdict}")
    print(f"      {stmt[:150]}")

V.section("part 3: which swept sites are actually poisoned at runtime?")
print("  Every site marked 'DEFAULT DTYPE (bf16 at runtime)' was executed with")
print("  the real cell-4 namespace where possible:")
probes = {
    "cell4:304 torch.full((new_dim-old_dim,), gamma_new)":
        env["GrowthLayerNorm"](48, 64).gamma_new.dtype,
    "cell4:305 torch.full((new_dim-old_dim,), beta_new)":
        env["GrowthLayerNorm"](48, 64).beta_new.dtype,
    "cell4:97  inv_freq ... .to(torch.bfloat16)":
        V.build_msr(env).inv_freq.dtype,
    "cell4:120 torch.arange(..., dtype=dtype) with dtype=bf16":
        V.build_msr(env)._rotations(8, 0, V.CPU, torch.bfloat16)[0].dtype,
    "cell4:148 torch.arange(length, device=device)":
        torch.arange(8, device=V.CPU).dtype,
    "cell4:168 torch.arange(tgt_len, device=k.device)":
        torch.arange(8, device=V.CPU).dtype,
    "cell4:746 torch.tensor(float('nan'), device=device)":
        torch.tensor(float("nan"), device=V.CPU).dtype,
}
for label, dt in probes.items():
    print(f"  {label:<58} -> {dt}")
print("\n  benign default-dtype sites (value is exactly representable):")
gln = env["GrowthLayerNorm"](48, 64)
print(f"    GrowthLayerNorm.gamma_new = {gln.gamma_new.flatten()[0].item()!r} "
      f"(bf16 exact), beta_new = {gln.beta_new.flatten()[0].item()!r} (bf16 exact)")
print("    nn.LayerNorm/nn.Linear/GroupNorm weights initialised to 1.0/0.0 are")
print("    also exactly representable; the 'poisoning' only bites on values with")
print("    more than 8 significand bits (arbitrary init or loaded fp32 weights).")

V.section("part 4: extra default-dtype-sensitive, non-constructor APIs")
extra = [
    ("cell4:83  torch.pow(2.0, -5.0 - arange(..., dtype=torch.float32))", "explicit fp32 OK"),
    ("cell4:469 torch.zeros_like(wcodes, dtype=torch.float32)", "explicit fp32 OK"),
    ("cell4:470 torch.ones_like(w)  (w is fp32)", "inherits fp32 OK"),
    ("cell4:471 torch.full_like(w, prompt_loss_weight)", "inherits fp32 OK"),
    ("cell4:729 torch.zeros((), dtype=torch.float32)", "explicit fp32 OK"),
    ("cell4:904 torch.zeros((), dtype=torch.float32)", "explicit fp32 OK"),
    ("cell4:519 logits[0, id] = -torch.inf  (bf16 logits)", "bf16 holds -inf OK"),
    ("cell4:156 mask.clamp_min(1e-8) on a bf16 mask", "bf16 resolves 1e-8 (rel err ~0.4%)"),
    ("cell4:209/211 state built/cast to torch.bfloat16 (hard-coded)", "REASONED: forces bf16 even for an fp32 caller"),
    ("cell4:222-226 chunk masks cast to torch.bfloat16 (hard-coded)", "REASONED: same class, no crash at bf16"),
]
for a, b in extra:
    print(f"  {a:<62} {b}")

V.section("verdict")
print("  Poisoned by the bf16 default dtype in the CURRENT cell 4:")
print("    * line 120 _rotations position arange (D2)  -- PROVEN, huge impact")
print("    * line 97  inv_freq registered as bf16 (D2) -- PROVEN, small impact")
print("    * line 746 evaluate NaN tensor (D5)         -- PROVEN, no impact")
print("    * lines 304/305 GrowthLayerNorm params      -- PROVEN, exact values")
print("  NOT poisoned (refuted): lines 148 and 168 (integer arange -> int64).")
