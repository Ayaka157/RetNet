"""把导出的模型目录整理成可直接 `from_pretrained` 的 HF 仓库并上传。

历史教训（2026-10-13）：本地的导出目录里 `config.json` **没有 `auto_map`**，
直接上传后 `AutoModelForCausalLM.from_pretrained(repo)` 会报
"Transformers does not recognize this architecture"。所以上传前必须：
1. 把 `modeling_retnet.py` / `retention_fast.py` / `fused_ce.py` 一并放进仓库；
2. 在 config.json 里写 `auto_map` 与 `architectures`；
3. 附上 tokenizer 四件套。

用法：
    python scripts/upload_hf.py --model checkpoints/retnet_500m_v21 \\
        --repo Ayaka157/RetNet-0.5B-v21 --dry-run     # 先看要传什么
    python scripts/upload_hf.py --model checkpoints/retnet_500m_v21 \\
        --repo Ayaka157/RetNet-0.5B-v21               # 真传
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_FILES = ["modeling_retnet.py", "retention_fast.py", "fused_ce.py"]
TOK_FILES = ["tokenizer.json", "bpe_tokenizer.json", "special_tokens_map.json",
             "tokenizer_config.json"]

README_TMPL = """---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags: [retnet, retention, custom-architecture]
---

# {repo_short}

RetNet 0.5B（504,179,896 参数）第 {gen} 代权重，从零训练。

## 这一代和上一代的区别

{highlights}

## 用法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo}", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("{repo}", trust_remote_code=True)

ids = tok("人工智能的未来是", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.8)
print(tok.decode(out[0], skip_special_tokens=True))
```

`trust_remote_code=True` 是必需的：这是自定义架构，代码随仓库发布
（`modeling_retnet.py`），不在 transformers 里。

## 训练与评测

见项目主页的 `docs/`。本代的关键数字：

| 指标 | 值 |
|---|---|
| 域内 held-out loss（5 个分片） | {holdout} |
| 上一代（ep3）基线 | 3.6287 |
| 相对变化 | {delta} |
| 训练 token | {tokens} |
| 硬件 | 2× RTX PRO 6000（DDP） |
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="导出的模型目录")
    ap.add_argument("--repo", required=True, help="HF 仓库，如 Ayaka157/RetNet-0.5B-v21")
    ap.add_argument("--gen", default="v21", help="写进 README 的代号")
    ap.add_argument("--highlights", default="（待补）")
    ap.add_argument("--holdout", default="（待测）")
    ap.add_argument("--delta", default="（待算）")
    ap.add_argument("--tokens", default="（待算）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.model)
    if not src.is_absolute():
        src = ROOT / src
    if not (src / "model.safetensors").exists():
        raise SystemExit(f"{src} 里没有 model.safetensors —— 训练可能还没导出")

    work = Path(tempfile.mkdtemp(prefix="hf_upload_"))
    try:
        # 1) 导出目录里的一切
        for p in src.iterdir():
            if not p.is_file() or p.suffix not in (".json", ".safetensors", ".txt"):
                continue
            if p.name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
                continue          # 训练状态不该进模型仓库
            shutil.copy2(p, work / p.name)
        print(f"  [1] 从 {src} 复制 {len(list(work.iterdir()))} 个文件")

        # 2) 自定义架构代码（缺了它 trust_remote_code 会失败）
        for name in CODE_FILES:
            f = ROOT / "retnet_hf" / name
            if f.exists():
                shutil.copy2(f, work / name)
            else:
                print(f"  ⚠️ 缺少 {name}，仓库将无法 from_pretrained")
        print(f"  [2] 架构代码 {[n for n in CODE_FILES if (work / n).exists()]}")

        # 3) tokenizer：本地只有 bpe_tokenizer.json，用 PreTrainedTokenizerFast
        #    生成 tokenizer.json / tokenizer_config.json / special_tokens_map.json，
        #    这样 HF 用户能直接 AutoTokenizer.from_pretrained。
        tok_src = ROOT / "data_cache/chat_pack_v13/bpe_tokenizer.json"
        if tok_src.exists():
            from transformers import PreTrainedTokenizerFast
            tk = PreTrainedTokenizerFast(
                tokenizer_file=str(tok_src),
                pad_token="<pad>", bos_token="<bos>",
                eos_token="<eos>", unk_token="<unk>",
                model_max_length=2048,
            )
            tk.save_pretrained(str(work))
            shutil.copy2(tok_src, work / "bpe_tokenizer.json")
        else:
            print(f"  ⚠️ 找不到 {tok_src}")
        print(f"  [3] tokenizer: {[n for n in TOK_FILES if (work / n).exists()]}")

        # 4) config.json 补 auto_map
        cfg_path = work / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["auto_map"] = {
            "AutoConfig": "modeling_retnet.RetNetHFConfig",
            "AutoModelForCausalLM": "modeling_retnet.RetNetForCausalLM",
        }
        cfg["architectures"] = ["RetNetForCausalLM"]
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"  [4] auto_map 已写入（{cfg.get('model_type')}，"
              f"{cfg.get('num_hidden_layers')} 层）")

        # 5) README
        (work / "README.md").write_text(
            README_TMPL.format(repo=args.repo, repo_short=args.repo.split("/")[-1],
                               gen=args.gen, highlights=args.highlights,
                               holdout=args.holdout, delta=args.delta,
                               tokens=args.tokens),
            encoding="utf-8")
        print("  [5] README 已生成")

        total = sum(p.stat().st_size for p in work.iterdir() if p.is_file())
        print(f"  [6] 待传 {len(list(work.iterdir()))} 个文件，共 {total / 1e9:.2f} GB")
        for p in sorted(work.iterdir()):
            if p.is_file():
                print(f"        {p.name:<28} {p.stat().st_size / 1e6:>9.1f} MB")

        if args.dry_run:
            print("  [dry-run] 不真传。暂存目录:", work)
            return 0

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", exist_ok=True,
                        private=args.private)
        t0 = time.time()
        api.upload_folder(folder_path=str(work), repo_id=args.repo,
                          repo_type="model")
        print(f"  ✅ 上传完成，用时 {time.time() - t0:.0f}s")
        print(f"  https://huggingface.co/{args.repo}")
        return 0
    finally:
        if not args.dry_run:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
