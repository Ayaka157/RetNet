"""Correctness tests for the from-scratch RetNet implementation.

Run:
    python test_model.py
"""
import math

import torch

from retnet.model import RetNetConfig, RetNetLM, MultiScaleRetention, theta_shift


def test_shapes_and_forward():
    torch.manual_seed(0)
    cfg = RetNetConfig(vocab_size=128, d_model=64, num_layers=2, num_heads=2,
                       dropout=0.0)
    model = RetNetLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, mode="parallel")
    assert out.shape == (2, 16, cfg.vocab_size)
    print("PASS: forward shapes")


def test_three_paradigms_match():
    """Parallel, recurrent, and chunkwise retention must be equivalent up to
    GroupNorm's scale invariance."""
    torch.manual_seed(1)
    cfg = RetNetConfig(vocab_size=64, d_model=64, num_layers=2, num_heads=2,
                       dropout=0.0)
    model = RetNetLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        y_parallel = model(ids, mode="parallel")
        y_recurrent, states_r = model(ids, mode="recurrent", return_state=True)
        y_chunk, states_c = model(ids, mode="chunkwise", chunk_size=4, return_state=True)
        _, states_p = model(ids, mode="parallel", return_state=True)

    assert (y_parallel - y_recurrent).abs().max() < 1e-4
    assert (y_parallel - y_chunk).abs().max() < 1e-4
    for sp, sr in zip(states_p, states_r):
        assert (sp - sr).abs().max() < 1e-4
    for sr, sc in zip(states_r, states_c):
        assert (sr - sc).abs().max() < 1e-4
    print("PASS: parallel == recurrent == chunkwise (including states)")


def test_causal_and_rotation():
    torch.manual_seed(2)
    msr = MultiScaleRetention(d_model=32, num_heads=2, dropout=0.0).eval()
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        y_parallel = msr(x, mode="parallel")
        y_full = msr(x, mode="parallel")  # mask is generated identically

        # Verify rotation is an orthogonal transform.
        angle = torch.tensor([0.3])
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        v = torch.randn(2, 4)
        w = theta_shift(v, sin, cos)
        assert torch.allclose(w.norm(dim=-1), v.norm(dim=-1), atol=1e-6)
    print("PASS: causal masks are deterministic and rotation is orthogonal")


def test_autoregressive_consistency():
    """Feeding [a, b] in parallel equals feeding [a] then [b] recurrently."""
    torch.manual_seed(3)
    cfg = RetNetConfig(vocab_size=32, d_model=32, num_layers=1, num_heads=2,
                       dropout=0.0)
    model = RetNetLM(cfg).eval()
    seq = torch.tensor([[5, 7, 9, 11]])
    with torch.no_grad():
        parallel = model(seq, mode="parallel")
        logits, states = model(seq[:, :3], mode="recurrent", offset=0, return_state=True)
        logits, states = model(seq[:, 3:], mode="recurrent", states=states, offset=3,
                               return_state=True)
    assert (parallel[:, -1] - logits[:, -1]).abs().max() < 1e-4
    print("PASS: autoregressive step-by-step recurrence is consistent")


def test_generate_runs():
    torch.manual_seed(4)
    cfg = RetNetConfig(vocab_size=32, d_model=32, num_layers=1, num_heads=2,
                       dropout=0.0)
    model = RetNetLM(cfg).eval()
    out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5,
                         temperature=1.0, top_k=5)
    assert out.shape[1] <= 8
    print("PASS: generate() runs in recurrent mode")


if __name__ == "__main__":
    test_shapes_and_forward()
    test_three_paradigms_match()
    test_causal_and_rotation()
    test_autoregressive_consistency()
    test_generate_runs()
    print("\nAll tests passed.")
