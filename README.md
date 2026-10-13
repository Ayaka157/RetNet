# RetNet, handwritten

A retentive network (RetNet) language model written from scratch — no `torch.nn.Transformer`,
no ready-made RetNet module — grown from 23M parameters to 504M and trained into a working
Chinese chat model on single consumer and workstation GPUs.

Paper this follows: [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621).

The current model is **RetNet-0.5B**: 504,179,896 parameters, 20 layers, `d_model` 1408, 22 heads,
vocab 20000, context 2048, chunk size 128. Validation loss went from 2.6358 to **1.9459** over three
training legs, 1,656,209,408 tokens in total.

## Where things are

| | |
| --- | --- |
| Weights (final) | [`Ayaka157/RetNet-0.5B`](https://huggingface.co/Ayaka157/RetNet-0.5B) — epoch 3, val loss 1.9459 |
| Weights (earlier legs) | [`RetNet-0.5B-ep1`](https://huggingface.co/Ayaka157/RetNet-0.5B-ep1) · [`RetNet-0.5B-ep2`](https://huggingface.co/Ayaka157/RetNet-0.5B-ep2) |
| Weights (first chat model) | [`Ayaka157/RetNet-100M`](https://huggingface.co/Ayaka157/RetNet-100M) — 100,370,064 parameters, 2026-08-20 |
| Technical report | [`docs/26_RetNet-0.5B_技术报告.md`](docs/26_RetNet-0.5B_技术报告.md) (Chinese) · [zh PDF](docs/latex/zh/RetNet-0.5B-技术报告.pdf) · [en PDF](docs/latex/en/RetNet-0.5B-Technical-Report.pdf) |
| Figures | [`figures/`](figures/README.md) — every training curve as vector PDF + PNG + raw CSV |
| Engineering log | [`docs/INDEX.md`](docs/INDEX.md) — 26 notes in Chinese, written while the work ran |
| Operations | [`RUNBOOK.md`](RUNBOOK.md) |
| Earlier Chinese README | [`README.zh-CN.md`](README.zh-CN.md), frozen at the 272M stage |

The model repos carry `model.safetensors`, `config.json` with an `auto_map`, the modelling code the
config points at, a `PreTrainedTokenizerFast`, and an English model card. Loading needs
`trust_remote_code=True`.

Weights, tokenized corpora and run logs are not in this repository. They total about 120 GB and
live on disk and on Hugging Face.

## Timeline

Dates are when the artifacts were written, read off the checkpoint files.

| Date | Stage |
| --- | --- |
| 2026-08-19 | 23M and 97M models; the first versions that trained without diverging |
| 2026-08-20 | 100M pretrain, then the first chat model |
| 2026-08-21 | 100M chat v3 — sample-aligned packing and reply-only loss masking |
| 2026-08-23 | 100M roleplay v3; function-preserving width growth to 143M |
| 2026-08-24 | 143M roleplay, then the first 272M |
| 2026-08-25 | 272M v10 |
| 2026-08-26 | Kaggle TPU runs at 8k context |
| 2026-10-09 – 10-11 | 272M v13/v14 reworked; 500M grown from 272M |
| 2026-10-12 | 0.5B epoch 1 on a rented RTX 3090 |
| 2026-10-13 | epochs 2 and 3 on an RTX PRO 6000, technical report, live serving |

Per-stage perplexities and losses for the earlier generations are in
[`README.zh-CN.md`](README.zh-CN.md) and [`docs/12_训练状态盘点.md`](docs/12_训练状态盘点.md).

## Architecture

Standard RetNet, with the recurrence written out by hand in `retnet/model.py`; `retnet_hf/` wraps
the same math for the HuggingFace `Trainer`. Only `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`,
`nn.GroupNorm` and `nn.Dropout` are used.

- **Multi-scale retention with a learnable per-head decay gate** (`use_gla`). The gate stays in
  fp32: bf16 has 7 mantissa bits near 1.0, which collapses 12 of the 16 heads onto exactly 1.0 and
  erases the multi-scale structure.
- **Three equivalent computations over one set of weights** — parallel for prefill, chunkwise for
  training, recurrent for token-by-token decoding. A closed-form chunkwise state scan replaces the
  per-chunk recurrence loop. The two prefill paths agree in value but not bit-for-bit: the largest
  absolute difference measured at 0.5B scale is 3.395. `tests/test_model.py` asserts `1e-4`, which
  holds only at toy scale.
- **Width growth through a function-preserving layer norm** (`GrowthLayerNorm`). Widening a
  checkpoint sets the new input columns to zero and `gamma_new = beta_new = 0`, so the wider model
  computes exactly what the narrower one did. Measured drift Δ7.25e-4.
- **O(1) decoding state** per layer, independent of context length.

One thing worth knowing about the shipped checkpoint: **78 of its 82 growth affine tensors are
still exactly zero**, and 67.08% of all weight elements are zero — the new channels never came
back to life. The mechanism is in the report §2.4: `gamma_new = 0` makes `∂new/∂x_new = 0`, which
cuts the gradient to the new inputs; those inputs are themselves zero, so `gamma_new` receives no
gradient either. Initializing `gamma_new = 1.0` would have been just as function-preserving at
step 0 and would have let the channels unlock. That counterfactual was not tested; the report
labels it a derivation.

## Evaluation

`bench_v3` — 1,020 items, 15 categories, 136 families, E/M/H difficulty tiers, a 44-predicate
grading DSL, and per-request RNG streams so one item scores the same whether it runs alone or in a
batch. One seed.

| Category | n | Score | Category | n | Score |
| --- | --- | --- | --- | --- | --- |
| Safety | 35 | 0.2857 | Cloze | 120 | 0.0250 |
| Multi-turn | 90 | 0.2667 | Bilingual | 45 | 0.0222 |
| Roleplay | 100 | 0.2000 | Instruction following | 80 | 0.0125 |
| Consistency | 70 | 0.1429 | Language | 90 | 0.0111 |
| Robustness | 50 | 0.1000 | Coding | 25 | 0.0000 |
| QA | 120 | 0.1000 | Math | 25 | 0.0000 |
| Structured | 60 | 0.0000 | Continuation | 60 | 0.0000 |
| Long context | 50 | 0.0000 | | | |

Weighted total **0.1037** (95% CI 0.032–0.130). Think-leakage 59.61%, truncation 72.94%,
degenerate repetition 0.00%.

Read that number with two caveats:

1. Items allow 16–160 new tokens; the median training reply is about 570 tokens. 72.94% of items
   hit the ceiling. Among the 276 items that were **not** truncated, the leak rate is **0.00** and
   the score is 0.15217 — the model closes its thinking block and stops when it has room. Much of
   the near-zero result is "ran out of room", not "produced garbage". Only one seed was run, so
   the per-category numbers carry no statistical power.
2. Training budget is 3.28496 tokens per parameter. Chinchilla's 20× would be 6.09× more.

Behaviourally the model holds a conversation. Factual recall, arithmetic, structured output and
code are not there.

## Training

Three legs, one epoch each over the same 552M-token corpus:

| Leg | GPU | Batch × accum | Steps | Wall clock | Tokens/s |
| --- | --- | --- | --- | --- | --- |
| ep1 | RTX 3090 24 GB | 4 × 1 | 67,390 | 11.316 h | 13,551 |
| ep2 | RTX PRO 6000 96 GB | 16 × 1 | 16,848 | 2.790 h | 54,970 |
| ep3 | RTX PRO 6000 96 GB | 16 × 1 | 16,848 | 2.685 h | 57,111 |

Each leg restarts a full cosine from 1.1e-4. That costs accuracy before it pays it back: ep2's
first evaluation sits 0.0764 above ep1's best, and ep3's first sits 0.1310 above ep2's best. Both
legs recover and finish below where they started.

AdamW 8-bit is required rather than optional — fp32 optimizer state alone is 7.7 GiB at this size.
Batch 16 peaks at 77–80 GiB, leaving little headroom on a 96 GB card.

```bash
python scripts/check_task_config.py tasks/v17_ep3_500m_pro6000.json
python run_task.py --config tasks/v17_ep3_500m_pro6000.json
```

The cloud procedure, including the pre-flight probe and the watchdog, is in
[`docs/24_AutoDL上机手册.md`](docs/24_AutoDL上机手册.md).

## Serving

`serve/retnet_play_server.py` is a FastAPI app that streams tokens over SSE. One worker thread
drains a FIFO queue, so a slow request cannot starve the others. Decoding uses the recurrent path;
an RTX 3070 serves 25–34 tok/s.

```bash
python serve/retnet_play_server.py --model_dir checkpoints/retnet_500m_v15b_ep3 --port 8077
```

## Failures that raised no error

The parts I would want to read first when picking this up:

- **`--resume` overwrote the command-line batch size.** The HuggingFace `Trainer` replaces
  `self._train_batch_size` with the value stored in `trainer_state.json`, so a run launched with
  `--batch_size 16` trained at 4. One warning line was the only trace. It self-locks, too: the new
  checkpoint records 4 and the next resume keeps 4. `retnet_hf/train.py` now calls
  `_sync_resume_state` before `train()` — and that fix itself sat unused for a day until a local
  test caught that it was defined but never called.
- **A missing `optimizer.pt` also skips the LR scheduler.** `trainer.py` gates scheduler loading on
  the optimizer file existing, so a resume without it re-warms the LR from step 0.
- **`train_steps_per_second` is `max_steps / runtime`.** `trainer.py:2819-2825` passes
  `num_steps=self.state.max_steps`, so on any run that stopped early or resumed mid-epoch the
  figure is wrong. One segment reported 10.304 steps/s while actually running at 1.630.
- **A benchmark prefill flag applied to more items than intended.** `--lc_prefill` is global, so a
  cleanup batch of 56 items — both entire categories — ran a different prefill path from the other
  964. Re-running them with the default moved Coding from 0.120 to 0.000 and degenerate repetition
  from 0.29% to 0.00%. The report records the correction.
- **A table taller than a page was clipped.** Appendix C's evidence index was one unbreakable
  `tabular` about 1.3 pages tall; LaTeX pushed it past the page edge and roughly half of it was
  invisible in the PDF. It is a `longtable` now, and `docs/latex/pdfbounds.py` checks every page
  for overflow on each build.

## Repository layout

```
retnet/          handwritten core: model, tokenizer, chat format, packing, mmap dataset
retnet_hf/       HuggingFace wrapper: Trainer, safetensors export, resume fixes
train.py         handwritten training entry point
run_task.py      config-driven launcher, reads tasks/*.json
play.py          interactive inference
bench/           benchmark v3: predicate DSL, frozen item bank, bit-exact decoder
scripts/         data builds, growth, evaluation, profiling
tasks/           one JSON per training run
tests/           equivalence and grading tests
serve/           SSE chat server and launch scripts
docs/            engineering log and technical reports (Chinese)
legacy/          superseded scripts and a snapshot of the first handwritten version
Kaggle/          TPU notebooks from the 8k-context attempt
```

## Tests

```bash
python tests/test_model.py              # three-form equivalence, toy scale only
python tests/test_fast_chunkwise.py     # closed-form scan vs reference recurrence
python tests/test_checkers.py           # all 44 grading predicates
python tests/test_bench_equivalence.py  # fastgen vs generate, bit-for-bit
```

## License

MIT — see [LICENSE](LICENSE).
