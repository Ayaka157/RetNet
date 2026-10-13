"""D6 -- the fp32 decay-gate restore in train_worker (lines 83-97, 834-842).

Three separate facts, all measured here:

  (a) `isinstance(_gate, nn.Parameter)` is False on the `use_gla=False` branch,
      where gamma is a *persistent buffer* -> that branch is never restored and
      gamma stays bf16 after `model.to(dtype=bfloat16)`.
  (b) On the use_gla=True branch the restore only fixes the DTYPE.  The VALUES
      were already irreversibly rounded to bf16 by `model.to(dtype=bfloat16)`;
      `_gate.data.to(torch.float32)` upcasts the rounded values, it does not
      recover the fp32 master weights.
  (c) Magnitude depends on the gate VALUES:
      * with the code's own initialisation 1-2**(-5-i) (a run that does not
        resume from a trained checkpoint) 12/16 heads round to exactly 1.0;
      * the SHIPPED checkpoint `retnet_272m_init_weights` carries already
        trained gates (0.786 .. 0.9923), so 0/16 heads hit 1.0 there -- but the
        values are still quantized and 4 heads are merged into one value.

RUN:  python repro_d6.py
"""

from __future__ import annotations

import torch
from safetensors import safe_open

import vcommon as V

V.banner("D6  fp32 gamma restore: missing branch + irreversible bf16 value loss")

env = V.load_env("d6_out")
assert torch.get_default_dtype() == torch.bfloat16

src = V.H.CELL4_SRC.splitlines()
V.section("source under test (cell 4 = 202610011026.ipynb::cell4)")
for n in (83, 84, 90, 91, 92, 93, 94, 834, 835, 836, 837, 838, 839, 840, 841, 842):
    print(f"  {n:4d}: {src[n-1]}")

REAL = (V.H.REPO / "Kaggle" / "archived" / "retnet_272m_init_weights"
        / "retnet_272m_rp" / "model.safetensors")
CKPT_CFG = REAL.parent / "config.json"

# ---------------------------------------------------------------- part 1
V.section("part 1: gate initialisation 1 - 2**(-5-i) -- the 'fresh run' scenario")
g32 = 1.0 - torch.pow(2.0, -5.0 - torch.arange(16, dtype=torch.float32))
g_bf = g32.to(torch.bfloat16)
print("   bf16 has 8 significand bits: in [0.5,1) the spacing is 2**-8, so any")
print("   value within 2**-9 below 1.0 rounds to 1.0 (ties-to-even).")
print()
print("   i   gamma(fp32)             gamma(bf16)   ==1.0   gamma^64(fp32)        gamma_bf16^64")
collapsed = []
for i in range(16):
    v32, vbf = g32[i].item(), g_bf[i].item()
    hit = vbf == 1.0
    collapsed.append(i) if hit else None
    print(f"  {i:2d}   {v32!r:<22} {vbf!r:<12} {str(hit):<7} {v32 ** 64!r:<20} {vbf ** 64!r}")
print()
print(f"   heads collapsing to exactly 1.0 at init: {len(collapsed)}/16 -> {collapsed}")
print(f"   chunk_decay for those heads = gamma**64 = {g_bf[4] ** 64!r}"
      f"   (= no cross-chunk decay: state is never forgotten)")

# ---------------------------------------------------------------- part 2
V.section("part 2: the SHIPPED checkpoint's actual gate values")
with safe_open(REAL, framework="pt") as f:
    allg = torch.stack([f.get_tensor(k).float()
                        for k in sorted(f.keys()) if k.endswith("msr.gamma")])
blk0 = allg[0].clone()
print(f"   {tuple(allg.shape)} = 20 layers x 16 heads, dtype fp32; "
      f"config use_gla={V.json.loads(CKPT_CFG.read_text())['use_gla']}")
print(f"   blk0 values: {[round(v, 7) for v in blk0.tolist()]}")
print(f"   equal to the init formula? {bool(torch.equal(blk0, g32))}  "
      f"-> the checkpoint carries TRAINED gates (max value {blk0.max():.7f})")
ones = int((allg.to(torch.bfloat16) == 1.0).sum())
rel = ((allg.to(torch.bfloat16).float() - allg).abs() / allg)
print(f"   heads == exactly 1.0 after a bf16 cast: {ones}/{allg.numel()}"
      f"   -> the 12/16 collapse does NOT happen with these values")
print(f"   bf16 rounding error: max rel {rel.max().item():.3e}, "
      f"mean rel {rel.mean().item():.3e}, max abs {(allg.to(torch.bfloat16).float() - allg).abs().max().item():.3e}")
print(f"   distinct fp32 values in block 0: {len(set(blk0.tolist()))}/16, "
      f"distinct after bf16: {len(set(allg[0].to(torch.bfloat16).float().tolist()))}/16")
print(f"   block0 heads 12..15 fp32 : {[round(v, 7) for v in blk0[12:].tolist()]}")
print(f"   block0 heads 12..15 bf16 : {[v for v in allg[0].to(torch.bfloat16).float()[12:].tolist()]}")
print("   -> 4 distinct gates merge into ONE value (head-wise gate diversity lost)")
d64 = ((allg.to(torch.bfloat16).float() ** 64) - (allg ** 64)).abs() / (allg ** 64)
print(f"   relative error of chunk_decay = gamma**64 after the bf16 cast: "
      f"max {d64.max().item():.3e}")

# ---------------------------------------------------------------- part 3
V.section("part 3: the notebook's OWN code path (from_pretrained -> .to(bfloat16) "
          "-> lines 839-842)")
model_cls = env["RetNetForCausalLM"]


def run_notebook_path(gamma_values, tag):
    cfg = V.tiny_config(env, d_model=64, num_heads=16, value_dim=64, num_layers=2,
                        use_gla=True)
    torch.set_default_dtype(torch.float32)
    model = model_cls(cfg)
    with torch.no_grad():
        for blk in model.blocks:
            blk.msr.gamma.copy_(gamma_values)
    out_dir = V.H.SANDBOX / f"gamma_model_{tag}"
    model.save_pretrained(out_dir, safe_serialization=True)
    torch.set_default_dtype(torch.bfloat16)

    m = model_cls.from_pretrained(out_dir)
    print(f"\n  [{tag}] 1) after from_pretrained (default dtype={torch.get_default_dtype()})")
    print(f"        gamma dtype={m.blocks[0].msr.gamma.dtype}  "
          f"Parameter={isinstance(m.blocks[0].msr.gamma, torch.nn.Parameter)}  "
          f"#==1.0: {int((m.blocks[0].msr.gamma.float() == 1.0).sum())}/16")
    master = m.blocks[0].msr.gamma.detach().float().clone()
    m.to(device=V.CPU, dtype=torch.bfloat16)
    print(f"     2) after model.to(device=device, dtype=torch.bfloat16): "
          f"gamma dtype={m.blocks[0].msr.gamma.dtype}  "
          f"#==1.0: {int((m.blocks[0].msr.gamma.float() == 1.0).sum())}/16")
    for _block in m.blocks:                      # verbatim lines 839-842
        _gate = _block.msr.gamma
        if isinstance(_gate, torch.nn.Parameter):
            _gate.data = _gate.data.to(torch.float32)
    after = m.blocks[0].msr.gamma
    vals_equal_master = torch.equal(after.detach().float(), master)
    print(f"     3) after the notebook restore loop: gamma dtype={after.dtype}  "
          f"#==1.0: {int((after.float() == 1.0).sum())}/16")
    print(f"        values bitwise equal to the pre-.to() fp32 master? {vals_equal_master}")
    print(f"        values bitwise equal to the exact input gamma?       "
          f"{torch.equal(after.detach().float(), gamma_values)}")
    print(f"        max |gamma_after - gamma_input| = "
          f"{(after.detach().float() - gamma_values).abs().max().item():.3e}")
    return m, after.detach().float().clone()


m_formula, got_formula = run_notebook_path(g32, "init-formula gates")
print("        => restore fixes the DTYPE only; 12 heads are 1.0 forever.")
run_notebook_path(blk0, "shipped-checkpoint gates")

# ---------------------------------------------------------------- part 4
V.section("part 4: use_gla=False -> gamma is a persistent BUFFER, branch skipped")
cfg = V.tiny_config(env, d_model=64, num_heads=16, value_dim=64, num_layers=2,
                    use_gla=False)
torch.set_default_dtype(torch.float32)
m2 = model_cls(cfg)
torch.set_default_dtype(torch.bfloat16)
g = m2.blocks[0].msr.gamma
print(f"  gamma dtype={g.dtype}  isinstance(nn.Parameter)={isinstance(g, torch.nn.Parameter)}"
      f"  persistent in state_dict={'blocks.0.msr.gamma' in m2.state_dict()}")
m2.to(device=V.CPU, dtype=torch.bfloat16)
for _block in m2.blocks:
    _gate = _block.msr.gamma
    if isinstance(_gate, torch.nn.Parameter):
        _gate.data = _gate.data.to(torch.float32)
g2 = m2.blocks[0].msr.gamma
print(f"  after model.to(bfloat16) + the notebook restore loop:")
print(f"    gamma dtype={g2.dtype} (NOT restored)  #==1.0: {int((g2.float() == 1.0).sum())}/16")
print(f"    gamma is an optimizer parameter? "
      f"{any(n.endswith('msr.gamma') for n, _ in m2.named_parameters())}"
      f"   -> a use_gla=False gate can never be learned (it is a frozen bf16 buffer)")

# ---------------------------------------------------------------- part 5
V.section("part 5: numerical impact (chunkwise logits, seq_len=256)")
with torch.no_grad():
    ids = torch.randint(0, 256, (1, 256), dtype=torch.long)
    m_formula.eval()
    out_dead = m_formula(ids, mode="chunkwise", chunk_size=64).logits.float()
    for blk in m_formula.blocks:
        blk.msr.gamma.data = g32.clone()
    out_ok = m_formula(ids, mode="chunkwise", chunk_size=64).logits.float()
print(f"  ||logits(12 dead gates) - logits(exact fp32 gates)|| / ||exact|| = "
      f"{((out_dead - out_ok).norm() / out_ok.norm()).item():.4f}")
print(f"  max |logit difference| = {(out_dead - out_ok).abs().max().item():.6f}")
print("  (12 of 16 heads lose all decay -> they attend uniformly over the whole")
print("   window instead of forgetting; the remaining 4 heads cannot compensate)")

V.section("verdict")
print("  D6(a) use_gla=False buffer branch never restored            = PROVEN")
print("        (real hole, NOT exercised by this checkpoint: config use_gla=true)")
print("  D6(b) the restore cannot recover the fp32 VALUES            = PROVEN")
print("        (bf16 cast of an fp32 master weight is irreversible; dtype fp32")
print("         afterwards is cosmetic)")
print("  D6(c) magnitude:")
print(f"        * init-formula gates  -> {len(collapsed)}/16 heads pinned to 1.0")
print(f"        * SHIPPED checkpoint  -> {ones}/{allg.numel()} heads pinned to 1.0,")
print(f"          but quantized gates (max rel {rel.max().item():.2e}) and 4 heads merged")
print("        => the briefing's '12 of 16 heads collapse' is correct ONLY for the")
print("           init formula, not for the shipped (already trained) gates.")
