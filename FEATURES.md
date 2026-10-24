# FEATURES — what this project is and what it can do

This document describes the **capabilities and design of the codebase**, not its run
state. It contains no losses, no throughput figures, no training progress and no dates:
anything that changes when a job runs belongs in a log, not here. For "which artifact
is authoritative" see [`CANONICAL.md`](CANONICAL.md); for operations see
[`RUNBOOK.md`](RUNBOOK.md); for the survey that organised this repository see
[`archive/INDEX.md`](archive/INDEX.md).

---

## 1. What this is

A retentive-network language model written **from scratch**. No `torch.nn.Transformer`,
no prepackaged RetNet block, no copied implementation: the recurrence, the chunkwise
scan, the tokenizer wrapper, the packing format and the chat encoder are all hand-written
here. It follows [Retentive Network](https://arxiv.org/abs/2307.08621).

The only PyTorch primitives used are `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`,
`nn.GroupNorm` and `nn.Dropout`. That constraint is deliberate and is the reason the
three computation forms below could be made to agree rather than merely interoperate.

## 2. Model

| | |
| --- | --- |
| Parameters | 504,179,896 |
| Layers | 20 |
| `d_model` / heads | 1408 / 22 |
| Vocabulary | 20,000 (ByteLevel BPE) |
| Context | 2,048 |
| Chunk size | 128 |

A 272M configuration and a 143M configuration are also supported. **Width growth** is a
first-class feature, not a migration: a narrower checkpoint can be widened in place, so
model size is a training decision rather than a restart.

## 3. Architecture features

**Multi-scale retention with a learnable per-head decay gate.** Each head carries its own
learnable decay, so different heads can settle on different memory horizons. The gate is
kept in **fp32 even under bf16 autocast** — bf16 has seven mantissa bits near 1.0, which
collapses heads onto exactly 1.0 and erases the multi-scale structure the mechanism exists
to provide. This is not a tuning detail; the feature stops working without it.

**Three equivalent computations over one set of weights:**

| Form | Used for | Property |
| --- | --- | --- |
| Parallel | prefill | one matmul-shaped pass over the whole prompt |
| Chunkwise | training | closed-form state scan; no per-chunk Python loop |
| Recurrent | decoding | O(1) state per layer, independent of context length |

The parallel and chunkwise paths agree in value but not bit-for-bit at production scale;
the tests assert `1e-4` only at toy scale, and that limitation is stated rather than
hidden.

**Function-preserving width growth.** Widening a checkpoint initialises the new input
columns to zero and the new affine parameters so the wider model computes exactly what
the narrower one computed at step zero. Drift is measurable and measured.

**A known weakness of the current growth implementation.** Initialising the new gate to
zero also zeroes its own gradient path, so newly added channels can remain permanently
inert. The counterfactual initialisation has been **derived but never tested**; treat the
current growth feature as producing capacity that may not come alive.

## 4. Training

- **HF `Trainer` wrapper** (`retnet_hf/`) over the handwritten core, with safetensors
  export so checkpoints are loadable by name.
- **Config-driven runs**: one JSON per run under `tasks/`, launched by `run_task.py`,
  with preflight validation, a watchdog, automatic restart on crash, and resume.
- **Reply-only loss masking** via a per-token weight code — `0` padding, `1` assistant
  reply, `2` prompt and persona (down-weighted), `3` plain replay text. The model is
  trained to answer, not to model questions.
- **Sample-aligned packing**: every training row is one aligned sample beginning with
  BOS, ending with EOS, truncating from the head. Earlier stream-then-slice packing put
  most samples mid-conversation and is the specific bug this replaced.
- **One encoder for training and inference** (`retnet/chat.py`), so token boundaries
  cannot drift between the two.
- **8-bit AdamW is required, not optional**: fp32 optimizer state alone does not fit
  alongside the model at this size.
- **bf16 is the supported autocast mode.** fp16 is a documented dead route — it diverges
  to NaN gradients — and the numeric-hardening fork that explored it is archived.

## 5. Data

A tokenised chat corpus with a companion weight-code array, plus a pretraining token
stream, all in a fixed 20k vocabulary. `data_cache/index.json` is the machine-readable
inventory: it records, per dataset, its dtype and shape, token count, provenance, live
consumers, and its retention class (`CURRENT`, `BUILD_INPUT`, `DERIVABLE`, `SUPERSEDED`,
`ROOT`, `INCOMPATIBLE`, `NO_MASK_CORRUPT`).

The chat packs are a **lineage, not a collection**: each version is a re-packing of a
previous one, and the index records which pack each was built from so a stored derivative
can be regenerated rather than mistaken for a new corpus. Two English baselines in the
directory use 16k and 8k vocabularies and are therefore **incompatible** with the 20k
model; they are excluded from training by construction.

## 6. Serving

`serve/retnet_play_server.py` is a FastAPI app that streams tokens over SSE, with a single
worker thread draining a FIFO queue so one slow request cannot starve the others. Decoding
uses the recurrent path, so per-token cost does not grow with context. Prompt encoding is
shared with the interactive CLI (`play_v2.py`) rather than reimplemented, which is why the
server imports from it.

## 7. Evaluation

`bench/` is a frozen, self-contained benchmark: a declared item bank, E/M/H difficulty
tiers, a **44-predicate grading DSL**, and per-request RNG streams so an item scores the
same whether it runs alone or in a batch. A bit-exact decoder exists alongside a faster
one, and the two are tested for equivalence — batching is known to perturb sampling at the
1e-6 level, so reproducible scoring defaults to batch size 1.

Reports include paired significance testing and per-item discrimination diagnostics, so a
near-zero category can be distinguished from a benchmark that simply cannot resolve it.

## 8. Tests

| File | Asserts |
| --- | --- |
| `tests/test_model.py` | equivalence of the three computation forms, toy scale only |
| `tests/test_fast_chunkwise.py` | closed-form scan against the reference recurrence |
| `tests/test_checkers.py` | all 44 grading predicates |
| `tests/test_bench_equivalence.py` | fast decoder vs reference decoder, bit-for-bit |

The suite also carries regression tests written specifically to fail if a previously fixed
defect is reintroduced.

## 9. Repository map

```
retnet/          handwritten core: model, recurrence, tokenizer, chat encoder, packing, mmap dataset
retnet_hf/       HuggingFace wrapper: Trainer, safetensors export, resume handling
train.py         handwritten training entry point
run_task.py      config-driven launcher over tasks/*.json
play.py          interactive inference for legacy handwritten .pt checkpoints
play_v2.py       interactive inference for current safetensors checkpoints (also used by serve)
bench/           benchmark item bank, predicate DSL, decoders, runner, analysis
scripts/         data builds, width growth, evaluation, profiling, index generation
tasks/           one JSON per training run
tests/           equivalence, grading and regression tests
serve/           SSE chat server and launch scripts
docs/            current engineering notes
docs/archive/    superseded notes and a restored 2026 legacy set
archive/         relocated recovery trees, with a per-item undo script and a ledger
data_cache/      tokenised corpora (not in git) + index.json
checkpoints/     weights (not in git)
legacy/          superseded scripts and a snapshot of the first handwritten version
Kaggle/          TPU notebooks from the 8k-context attempt
```

## 10. Scope — what this project does not do

Stated as design scope, not as a judgement of quality:

- It does **not** ship a conversational product. It is a model, a trainer, an evaluation
  harness and a serving demo.
- It does **not** target factual recall or arithmetic at this scale; the architecture is
  present, the token budget and corpus composition are the binding constraints.
- Its **code corpus is empty**, so code generation is out of scope until data changes —
  this is a data-composition fact, not a tuning gap.
- The **growth mechanism** can add capacity that never activates (section 3).
- `play.py` and `play_v2.py` are **not interchangeable**: the former reads only legacy
  handwritten checkpoints, the latter only current safetensors ones.
