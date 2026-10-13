# Figures

Every figure here comes from a real artifact in this repository. Nothing is hand-drawn and
nothing is interpolated. Each one ships as a vector PDF (open it in Illustrator or Inkscape if you
want to edit) and as a PNG rendered at 4× (1720 px wide on the long edge). Each PDF carries a title
band naming the project and what the figure shows; `raw/` holds the untitled versions exactly as
the technical report uses them.

Regenerate with:

```bash
python scripts/export_figures.py
```

## Screenshots of the live UI

`shots/` holds captures of the actual chat front end (`serve/index.html`) talking to the epoch-3
model over the real SSE endpoint, taken with Chrome DevTools Protocol at 2× pixel density. They are
cropped to the application, not to the screen.

| File | Conversation |
| --- | --- |
| `shots/cv-roleplay-zhongli` | Persona: Zhongli (*Genshin Impact*). "钟离先生，给我讲讲璃月的故事吧。" — 114 tokens, closes on its own |
| `shots/cv-roleplay-luxun` | Persona: Lu Xun. "先生，青年应当如何读书？" — 63 tokens, closes on its own |

These were generated at temperature 0.6 with `repetition_penalty` 1.1 and `no_repeat_ngram_size` 4 —
the decoding settings shown in each screenshot's hyper-parameter line. Generation is sampled, so a
different run of the same prompt gives a different reply. Captures were discarded and regenerated
when the reply (a) left its `<think>` block unclosed — the UI flags that with a warning banner,
(b) came back shorter than 40 tokens, which reads as truncated, or (c) contradicted the prompt (one
Doraemon run answered "能让我考砸道具的" to a question about doing *well* on an exam).

The failure rate is not uniform. Prompts that ask the model to reason leak their `<think>` block
most of the time — a machine-learning question leaked in 6 runs out of 6, at both 0.5 and 0.6 — while
role-play prompts answered in character did not (6 out of 6 clean). That asymmetry is what the
benchmark reports too: role-play and multi-turn score highest, reasoning and coding sit at zero.

## The 0.5B model

| File | What it shows |
| --- | --- |
| `fig-0p5b-loss` | Validation and training loss across all three legs of RetNet-0.5B, plotted against cumulative training tokens. Thin lines are every logged step, thick lines are a 25-step mean, dots are evaluations. The two hot restarts are annotated with their cost. |
| `fig-0p5b-lr` | The learning-rate schedule: one full cosine per leg, 1.1e-4 down to 1.1e-5, 500 warmup steps, restarted at the start of each leg. |
| `fig-0p5b-step` | Composition of one 575 ms optimization step: 61.6% GEMM, 27.9% elementwise, 8.4% reduction, and the 22–24 ms of structural redundancy that is safe to remove. |
| `fig-bench-cats` | `bench_v3` scores by category for the epoch-3 checkpoint, 1,020 items. |

## Earlier stages

| File | What it shows |
| --- | --- |
| `fig-loss` | The 272M run: training loss over 10,000 steps against the step axis. |
| `fig-lr` | The 272M learning-rate schedule. |
| `fig-steps` | Per-step time across the optimization configurations, 2176 ms down to 662 ms, with throughput for each. |
| `fig-truncation` | Truncation and think-leakage rates before and after the 272M continuation. |
| `fig-gemm` | Invocation counts of each GEMM shape in the compiled graph. |
| `fig-kernels` | Kernel-level attribution of a training step: GEMM 66.4%, Triton elementwise 25.2%, Triton reduction 4.2%, elementwise 4.2%. |

## Raw data

`data/0p5b_train_loss.csv` — 5,053 rows, every logged training step of all three legs:

| Column | Meaning |
| --- | --- |
| `leg` | `ep1`, `ep2`, `ep3` |
| `gpu` | Which card the leg ran on |
| `step` | Optimization step within the leg |
| `tokens` | Cumulative tokens for that leg, `step × tokens_per_step` |
| `loss` | Training loss at that step |
| `learning_rate` | Learning rate at that step |
| `grad_norm` | Gradient norm at that step |

`data/0p5b_eval_loss.csv` — 29 rows, every validation evaluation:

| Column | Meaning |
| --- | --- |
| `leg`, `gpu`, `step`, `tokens` | As above |
| `eval_loss` | Validation loss |
| `eval_runtime_s` | Wall clock of that evaluation |

Both CSVs are read straight from the `trainer_state.json` of each leg:

```
checkpoints/retnet_500m_v15b/checkpoint-67390/trainer_state.json  (ep1)
checkpoints/retnet_500m_v15b_ep2/trainer_state.json               (ep2)
checkpoints/retnet_500m_v15b_ep3/trainer_state.json               (ep3, the released model)
```

The 272M figures read from `checkpoints/retnet_272m_v14/checkpoint-10000/trainer_state.json`.

Figures are labelled in English because they were drawn for the bilingual technical report. If you
need another language, the PDFs are vector and the CSVs are plain text.
