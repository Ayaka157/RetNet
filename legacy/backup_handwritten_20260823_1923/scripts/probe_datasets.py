"""Probe candidate HuggingFace datasets: reachable? what schema? how big a sample?

Streaming-only, so nothing large is downloaded.  Used to pick the data mix for
the role-play chat model without guessing repo names.

    python scripts/probe_datasets.py --out logs/dataset_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets"))
sys.path.insert(0, str(ROOT))

# (label, repo, config, split, category)
CANDIDATES = [
    # ---------- Chinese role-play ----------
    ("haruhi_54k", "silk-road/ChatHaruhi-54K-Role-Playing-Dialogue", None, None, "rp_zh"),
    ("haruhi_expand", "silk-road/ChatHaruhi-Expand-118K", None, None, "rp_zh"),
    ("haruhi_zero", "silk-road/Haruhi-Zero-RolePlaying-movie-PIPPA", None, None, "rp_zh"),
    ("rolebench_zh", "ZenMoore/RoleBench", None, None, "rp_zh"),
    ("roleplay_zh_gpt4", "shibing624/roleplay-zh-sharegpt-gpt4-data", None, None, "rp_zh"),
    ("coser", "Neph0s/CoSER", None, None, "rp_zh"),
    ("rolellm_zh", "ZhuangXialie/role_play_chinese", None, None, "rp_zh"),
    # ---------- English role-play ----------
    ("pippa", "PygmalionAI/PIPPA", None, None, "rp_en"),
    ("limarp_aug", "grimulkan/LimaRP-augmented", None, None, "rp_en"),
    ("limarp", "lemonilia/LimaRP", None, None, "rp_en"),
    ("claude_rp_30k", "Norquinal/claude_multiround_chat_30k", None, None, "rp_en"),
    ("bluemoon", "Squish42/bluemoon-fandom-1-1-rp-cleaned", None, None, "rp_en"),
    # ---------- real user chats (great for casual style) ----------
    ("lmsys_chat", "lmsys/lmsys-chat-1m", None, None, "real_chat"),
    ("wildchat", "allenai/WildChat-1M", None, None, "real_chat"),
    # ---------- Chinese novels / story text (role-play language feel) ----------
    ("webnovel_cn", "zxbsmk/webnovel_cn", None, None, "novel_zh"),
    ("chinese_novels", "wangrui6/Chinese-Novels", None, None, "novel_zh"),
    ("skypile", "Skywork/SkyPile-150B", None, None, "pretrain_zh"),
    ("wiki_zh", "wikimedia/wikipedia", "20231101.zh", None, "pretrain_zh"),
    # ---------- Chinese chat / instruct at scale ----------
    ("belle_3_5m", "BelleGroup/train_3.5M_CN", None, None, "chat_zh"),
    ("sharegpt_zh_90k", "shareAI/ShareGPT-Chinese-English-90k", None, None, "chat_zh"),
    ("smoltalk_zh", "opencsg/smoltalk-chinese", None, None, "chat_zh"),
    ("moss_003", "YeungNLP/moss-003-sft-data", None, None, "chat_zh"),
    ("infinity_instruct", "BAAI/Infinity-Instruct", "7M", None, "chat_zh"),
    # ---------- pretraining (English / edu / synthetic) ----------
    ("fineweb_edu_100bt", "HuggingFaceFW/fineweb-edu", "sample-100BT", None, "pretrain_en"),
    ("cosmopedia_v2", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", None, "pretrain_en"),
    ("fineweb_dedup", "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", None, "pretrain_en"),
    ("chinese_fineweb_v2", "opencsg/Chinese-FineWeb-Edu-V2", None, None, "pretrain_zh"),
    # ---------- bonus: code / math / reasoning ----------
    ("python_edu", "HuggingFaceTB/smollm-corpus", "python-edu", None, "code"),
    ("open_web_math", "open-web-math/open-web-math", None, None, "math"),
    ("magicoder", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, None, "code"),
    ("metamath", "meta-math/MetaMathQA", None, None, "math"),
    ("openhermes", "teknium/OpenHermes-2.5", None, None, "reason"),
]


def probe(repo, config, split):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split, streaming=True)
    if hasattr(ds, "keys"):
        first = next(iter(ds.keys()))
        ds = ds[first]
        split_used = first
    else:
        split_used = split or "train"
    it = iter(ds)
    ex = next(it)
    keys = {}
    for k, v in list(ex.items())[:12]:
        s = str(v)
        keys[k] = s[:120].replace("\n", " ")
    return {"ok": True, "split": split_used, "keys": list(ex.keys()), "sample": keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/dataset_probe.json")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    results = {}
    for label, repo, config, split, category in CANDIDATES:
        if args.only and label not in args.only:
            continue
        try:
            info = probe(repo, config, split)
            info.update(repo=repo, config=config, category=category)
            results[label] = info
            print(f"OK    {label:<22} {repo} keys={info['keys']}", flush=True)
        except Exception as exc:
            results[label] = {"ok": False, "repo": repo, "config": config,
                              "category": category,
                              "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(f"FAIL  {label:<22} {repo} -> {type(exc).__name__}: {str(exc)[:120]}",
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    ok = sum(1 for v in results.values() if v.get("ok"))
    print(f"\n{ok}/{len(results)} reachable -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
