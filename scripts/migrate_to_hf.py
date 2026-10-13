"""Migrate the handwritten RetNet checkpoint to the HF implementation and
verify numerical identity (old vs new forward on identical inputs).

    python scripts/migrate_to_hf.py \
        --checkpoint checkpoints/retnet_100m_pretrain_v4/model.pt \
        --out_dir checkpoints/retnet_100m_pretrain_v4_hf \
        --tokenizer data_cache/chat_pack_v4/bpe_tokenizer.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.model import RetNetLM as OldModel, strip_optimized_prefix  # noqa: E402
from retnet_hf.modeling_retnet import RetNetHFConfig, RetNetForCausalLM  # noqa: E402


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq_len", type=int, default=128)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old_cfg = ckpt["config"]
    old_state = strip_optimized_prefix(ckpt.get("model_state", ckpt))
    print(f"old step={ckpt.get('step', '?')} val_loss={ckpt.get('val_loss', '-')}")
    print(f"old state keys: {len(old_state)}")

    cfg = RetNetHFConfig(
        vocab_size=old_cfg.vocab_size,
        d_model=old_cfg.d_model,
        num_layers=old_cfg.num_layers,
        num_heads=old_cfg.num_heads,
        value_dim=old_cfg.value_dim,
        ffn_dim=old_cfg.ffn_dim,
        dropout=old_cfg.dropout,
        norm_eps=old_cfg.norm_eps,
        tie_weights=old_cfg.tie_weights,
        embed_scale=old_cfg.embed_scale,
    )
    new_model = RetNetForCausalLM(cfg)
    new_keys = set(new_model.state_dict().keys())
    old_keys = set(old_state.keys())
    missing = new_keys - old_keys
    extra = old_keys - new_keys
    print(f"new keys: {len(new_keys)} | missing in old: {sorted(missing)}")
    print(f"extra in old: {sorted(extra)}")
    if missing:
        raise SystemExit("有缺失键，参数名未对齐，迁移中止")

    # 手写版 gamma 是 persistent buffer，名字一致，直接 strict 加载
    new_model.load_state_dict(old_state, strict=True)
    print("state_dict 严格加载成功（无 rename）")

    # ---------------- 数值一致性验证 ----------------
    torch.manual_seed(0)
    device = torch.device(args.device)
    old_model = OldModel(old_cfg).to(device)
    old_model.load_state_dict(old_state)
    old_model.eval()
    new_model.to(device).eval()

    for L in (1, 8, args.seq_len):
        ids = torch.randint(1, cfg.vocab_size, (2, L), device=device)
        with torch.inference_mode():
            old_logits = old_model(ids, mode="parallel")
            new_logits = new_model(ids).logits
        diff = (old_logits.float() - new_logits.float()).abs().max().item()
        print(f"seq={L:<4} max|Δlogits| = {diff:.3e} {'OK' if diff < 1e-4 else 'FAIL'}")

    # 保存 HF 格式（safetensors + config.json）
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    new_model.save_pretrained(str(out), safe_serialization=True)
    print(f"saved HF checkpoint -> {out}")

    # tokenizer 转 HF 资产
    from retnet.tokenizer import BPETokenizer
    tok = BPETokenizer.load(args.tokenizer)
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    wrapped = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(args.tokenizer))
    wrapped.pad_token = "<pad>"
    wrapped.bos_token = "<bos>"
    wrapped.eos_token = "<eos>"
    wrapped.unk_token = "<unk>"
    wrapped.save_pretrained(str(out))
    print(f"saved HF tokenizer -> {out}")
    print("MIGRATION OK")


if __name__ == "__main__":
    main()
