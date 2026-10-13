"""Train a from-scratch RetNet language model on Wikitext-2 (or cached data).

Example
-------
python train.py --d_model 256 --num_layers 4 --num_heads 4 \
    --vocab_size 4096 --seq_len 128 --batch_size 16 --max_steps 300 \
    --mode parallel --amp bf16
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from retnet.chat import encode_prompt
from retnet.data import get_dataloaders, make_blocks
from retnet.model import RetNetConfig, RetNetLM, strip_optimized_prefix
from retnet.tokenizer import BPETokenizer

logger = logging.getLogger("retnet.train")


def parse_args():
    p = argparse.ArgumentParser(description="Train a from-scratch RetNet language model")
    # Data
    p.add_argument("--cache_dir", type=str, default="data_cache/wikitext2")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                   choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--force_rebuild", action="store_true",
                   help="re-tokenize the dataset even if cache files exist")
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--prepared_data_dir", type=str, default=None,
                   help="use an already prepared cache (train.ids.npy / val.ids.npy / bpe_tokenizer.json)")
    p.add_argument("--prompt_loss_weight", type=float, default=0.1,
                   help="loss weight for user/prompt tokens in packed chat data "
                        "(1.0 = plain LM loss, 0.0 = pure response-only SFT)")
    p.add_argument("--shard_dirs", nargs="*", default=None,
                   help="pretraining mode: directories of flat int16/int64 token shards "
                        "(streamed via mmap, never loaded into RAM)")
    # Model
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--value_dim", type=int, default=None)
    p.add_argument("--ffn_dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--init_from", type=str, default=None,
                   help="warm-start hidden layers from a previous RetNet checkpoint")
    p.add_argument("--resume_from", type=str, default=None,
                   help="resume optimizer/step from a training checkpoint")
    # Optimization
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--lr_total_steps", type=int, default=None,
                   help="global step count used by the cosine LR schedule; defaults to --max_steps")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # Retention mode
    p.add_argument("--mode", type=str, default="parallel",
                   choices=["parallel", "chunkwise"])
    p.add_argument("--chunk_size", type=int, default=64)
    # Hardware / logging
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda, cpu, or auto")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    p.add_argument("--compile", action="store_true",
                   help="use torch.compile (on Windows falls back to aot_eager if Triton is absent)")
    p.add_argument("--compile_backend", default="auto",
                   choices=["auto", "inductor", "aot_eager", "eager"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--generate_prompt", type=str,
                   default="The history of the English language begins")
    p.add_argument("--chat_format", action="store_true",
                   help="format the sample prompt as 用户/助手 dialogue")
    return p.parse_args()


def setup_device(args):
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        args.device = "cpu"
    return torch.device(args.device)


def get_prepared_dataloaders(prepared_dir, seq_len, batch_size, num_workers, tokenizer_path=None):
    import warnings
    prepared_dir = Path(prepared_dir)
    tokenizer = BPETokenizer.load(Path(tokenizer_path) if tokenizer_path else prepared_dir / "bpe_tokenizer.json")
    train_x, train_y = make_blocks(np.load(str(prepared_dir / "train.ids.npy"), mmap_mode="r"), seq_len, copy=False)
    val_x, val_y = make_blocks(np.load(str(prepared_dir / "val.ids.npy"), mmap_mode="r"), seq_len, copy=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0))
    return train_loader, val_loader, tokenizer


def is_packed_dir(prepared_dir) -> bool:
    """Packed chat data stores 2-D rows plus a per-token loss-weight code."""
    prepared_dir = Path(prepared_dir)
    return (prepared_dir / "train.wcode.npy").exists()


def get_packed_dataloaders(prepared_dir, batch_size, num_workers, tokenizer_path=None):
    """Loader for sample-aligned packed rows built by scripts/build_sft_samples.py."""
    import warnings
    prepared_dir = Path(prepared_dir)
    tokenizer = BPETokenizer.load(Path(tokenizer_path) if tokenizer_path else prepared_dir / "bpe_tokenizer.json")

    def load(split):
        ids = np.load(str(prepared_dir / f"{split}.ids.npy"), mmap_mode="r")
        code = np.load(str(prepared_dir / f"{split}.wcode.npy"), mmap_mode="r")
        if ids.ndim != 2:
            raise SystemExit(f"{split}.ids.npy must be 2-D for packed data, got {ids.shape}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            return TensorDataset(torch.from_numpy(ids[:, :-1]),
                                 torch.from_numpy(ids[:, 1:]),
                                 torch.from_numpy(code[:, 1:]))

    train_ds, val_ds = load("train"), load("val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0))
    return train_loader, val_loader, tokenizer


def get_shard_dataloaders(shard_dirs, seq_len, batch_size, num_workers, tokenizer_path):
    """Pretraining loader: flat token shards streamed from disk via mmap."""
    from retnet.flatdata import ShardedTokenBlocks, find_shards
    train_paths = find_shards(shard_dirs, include_val=False)
    val_paths = find_shards(shard_dirs, include_val=True)
    if not tokenizer_path:
        raise SystemExit("--shard_dirs requires --tokenizer_path")
    tokenizer = BPETokenizer.load(tokenizer_path)
    train_ds = ShardedTokenBlocks(train_paths, seq_len)
    val_ds = ShardedTokenBlocks(val_paths or train_paths[-1:], seq_len)
    logger.info("shards: %d train files (%.2fB tokens), %d val files (%.1fM tokens)",
                len(train_paths), train_ds.total_tokens / 1e9,
                len(val_paths), val_ds.total_tokens / 1e6)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0))
    return train_loader, val_loader, tokenizer


def code_to_weight(code: torch.Tensor, prompt_weight: float) -> torch.Tensor:
    """Map loss-weight codes (0 pad / 1 reply / 2 prompt / 3 plain LM) to floats."""
    w = torch.zeros_like(code, dtype=torch.float32)
    w = torch.where(code == 1, torch.ones_like(w), w)
    w = torch.where(code == 2, torch.full_like(w, prompt_weight), w)
    w = torch.where(code == 3, torch.ones_like(w), w)
    return w


def cosine_lr(step, warmup, total, peak, minimum):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, max_batches, amp_dtype=None, prompt_weight=1.0):
    """Average loss. For packed chat data this is the loss on reply tokens."""
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if len(batch) == 3:
            x, y, code = batch
            w = code_to_weight(code.to(device, non_blocking=True), prompt_weight)
        else:
            x, y = batch
            w = None
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, y = x.long(), y.long()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x, mode="parallel")
            token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                         y.reshape(-1), reduction="none")
        token_loss = token_loss.float()
        if w is None:
            total_loss += token_loss.sum().item()
            total_weight += y.numel()
        else:
            wf = w.reshape(-1)
            total_loss += (token_loss * wf).sum().item()
            total_weight += wf.sum().item()
    model.train()
    return total_loss / max(1e-9, total_weight)


def generate_sample(model, tokenizer, prompt, device, chat_format=False, max_new_tokens=50):
    model.eval()
    if chat_format:
        ids = encode_prompt(tokenizer, [], prompt)
    else:
        ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.7,
                            top_k=40, top_p=0.9, repetition_penalty=1.1,
                            no_repeat_ngram_size=4, eos_id=tokenizer.eos_id)
    model.train()
    text = tokenizer.decode(output[0].tolist()[len(ids):], skip_special_tokens=True)
    if chat_format:
        text = text.split("用户：")[0].split("助手：")[0].strip()
        return f"用户：{prompt}\n助手：{text}"
    return text


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    device = setup_device(args)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    packed = bool(args.prepared_data_dir) and is_packed_dir(args.prepared_data_dir)
    if args.shard_dirs:
        logger.info("Loading SHARDED pretraining corpus from %s ...", args.shard_dirs)
        train_loader, val_loader, tokenizer = get_shard_dataloaders(
            args.shard_dirs, args.seq_len, args.batch_size, args.num_workers,
            args.tokenizer_path)
        packed = False
    elif packed:
        logger.info("Loading PACKED chat data from %s (response-weighted loss, "
                    "prompt_loss_weight=%.2f) ...", args.prepared_data_dir, args.prompt_loss_weight)
        train_loader, val_loader, tokenizer = get_packed_dataloaders(
            args.prepared_data_dir, args.batch_size, args.num_workers,
            tokenizer_path=args.tokenizer_path)
        row_seq_len = train_loader.dataset.tensors[0].shape[1]
        if row_seq_len != args.seq_len:
            logger.warning("packed rows have seq_len=%d, overriding --seq_len %d",
                           row_seq_len, args.seq_len)
            args.seq_len = row_seq_len
    elif args.prepared_data_dir:
        logger.info("Loading prepared data from %s ...", args.prepared_data_dir)
        train_loader, val_loader, tokenizer = get_prepared_dataloaders(
            args.prepared_data_dir, args.seq_len, args.batch_size, args.num_workers,
            tokenizer_path=args.tokenizer_path)
    else:
        logger.info("Preparing %s data (this downloads/tokenizes on first run) ...",
                    args.dataset_config)
        train_loader, val_loader, tokenizer = get_dataloaders(
            cache_dir=args.cache_dir, seq_len=args.seq_len, vocab_size=args.vocab_size,
            batch_size=args.batch_size, num_workers=args.num_workers,
            tokenizer_path=args.tokenizer_path, force_rebuild=args.force_rebuild,
            dataset_config=args.dataset_config)
    logger.info("Data ready in %.1fs. vocab_size=%d train_blocks=%d val_blocks=%d",
                time.time() - t0, tokenizer.vocab_size,
                len(train_loader.dataset), len(val_loader.dataset))

    cfg = RetNetConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        value_dim=args.value_dim,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
    model = RetNetLM(cfg).to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        old_state = strip_optimized_prefix(ckpt.get("model_state", ckpt))
        transferred = 0
        skipped = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                old_param = old_state.get(name)
                if old_param is not None and old_param.shape == param.shape:
                    param.copy_(old_param.to(param.device))
                    transferred += 1
                else:
                    skipped += 1
        logger.info("Warm start from %s: transferred=%d tensors skipped=%d (embeddings are random)",
                    args.init_from, transferred, skipped)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("RetNet model: %d layers, d_model=%d, heads=%d, params=%.2fM",
                args.num_layers, args.d_model, args.num_heads, n_params / 1e6)
    if args.compile:
        backend = args.compile_backend
        if backend == "auto":
            try:
                import triton  # noqa: F401
                backend = "inductor"
            except Exception:
                backend = "aot_eager"
        logger.info("Compiling model with torch.compile backend=%s ...", backend)
        model = torch.compile(model, backend=backend)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    start_step = 1
    best_val_loss = float("inf")
    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        resume_state = strip_optimized_prefix(resume["model_state"])
        (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(resume_state)
        optimizer.load_state_dict(resume["optimizer_state"])
        start_step = int(resume.get("step", 0)) + 1
        best_val_loss = float(resume.get("val_loss", best_val_loss))
        # Preserve the true global-best validation loss across train_loop segments.
        best_path = save_dir / "best_model.pt"
        if best_path.exists():
            try:
                prev_best = torch.load(best_path, map_location="cpu", weights_only=False)
                best_val_loss = min(best_val_loss, float(prev_best.get("val_loss", best_val_loss)))
            except Exception:
                pass
        logger.info("Resumed from step %d, best_val_loss=%.3f", start_step - 1, best_val_loss)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device.type == "cuda"))

    train_iter = iter(train_loader)
    total_tokens = 0
    t_start = time.time()

    for step in range(start_step, args.max_steps + 1):
        lr = cosine_lr(step - 1, args.warmup_steps,
                       args.lr_total_steps or args.max_steps,
                       args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if len(batch) == 3:
            x, y, code = batch
            w = code_to_weight(code.to(device, non_blocking=True), args.prompt_loss_weight)
        else:
            x, y = batch
            w = None
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, y = x.long(), y.long()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x, mode=args.mode, chunk_size=args.chunk_size)
            if w is None:
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            else:
                token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                             y.reshape(-1), reduction="none")
                wf = w.reshape(-1)
                loss = (token_loss * wf).sum() / wf.sum().clamp_min(1.0)
            loss = loss / args.grad_accum

        if args.amp == "fp16" and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step % args.grad_accum) == 0:
            if args.amp == "fp16" and device.type == "cuda":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_tokens += y.numel()

        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            step_loss = loss.item() * args.grad_accum
            logger.info("step %d/%d | loss %.3f | ppl %.2f | lr %.2e | %.0f tok/s",
                        step, args.max_steps, step_loss,
                        math.exp(step_loss) if step_loss < 20 else float("inf"),
                        lr, total_tokens / elapsed)

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = evaluate(model, val_loader, device, args.eval_batches, amp_dtype,
                                prompt_weight=args.prompt_loss_weight)
            logger.info("eval step %d | val_loss %.3f | val_ppl %.2f",
                        step, val_loss,
                        math.exp(val_loss) if val_loss < 20 else float("inf"))
            torch.save({
                "step": step,
                "model_state": strip_optimized_prefix(model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "args": vars(args),
                "val_loss": val_loss,
            }, save_dir / "last_model.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "step": step,
                    "model_state": strip_optimized_prefix(model.state_dict()),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "args": vars(args),
                    "val_loss": val_loss,
                }
                torch.save(ckpt, save_dir / "best_model.pt")
                torch.save({"step": step, "model_state": strip_optimized_prefix(model.state_dict()),
                            "config": cfg, "val_loss": val_loss},
                           save_dir / "model.pt")
                logger.info("Saved best checkpoint to %s", save_dir / "best_model.pt")
            sample = generate_sample(model, tokenizer, args.generate_prompt, device,
                                     chat_format=args.chat_format)
            logger.info("sample: %s", sample)

    torch.save({
        "step": args.max_steps,
        "model_state": strip_optimized_prefix(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
        "args": vars(args),
    }, save_dir / "last_model.pt")
    logger.info("Training finished. Final val ppl %.2f. Checkpoints in %s",
                math.exp(best_val_loss) if best_val_loss < 20 else float("inf"), save_dir)


if __name__ == "__main__":
    main()
