"""05 - score every candidate tokenizer on the identical cached test sets.

Candidates:
  prod_v13        data_cache/chat_pack_v13/bpe_tokenizer.json   (in production)
  ctl_*           trained by 04_train_controls.py
  ref_qwen2.5     Qwen/Qwen2.5-0.5B tokenizer  (vocab 151,643)
  ref_gpt2        openai-community/gpt2        (vocab 50,257)
  ref_deepseek    deepseek-ai/DeepSeek-LLM-7B-Base (vocab 100,015)

Emits out/05_compare.md + .json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, TOK_PATH, fmt_table  # noqa: E402
from measure import build_testsets, score  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

TESTS = ("bench_all", "bench_zh", "bench_en", "bench_lc", "aga_r1", "aga_wiki", "aga_ruozhi")
CJKCLS = ("cjk", "latin", "digit", "space", "mixed", "other")


def discover() -> dict[str, Path]:
    cands = {"prod_v13": Path(TOK_PATH)}
    for p in sorted((OUT / "out" / "tok").glob("*.json")):
        if p.name == "train_log.json":
            continue
        cands[p.stem] = p
    refdir = OUT / "out" / "ref"
    for p in sorted(refdir.glob("*.json")):
        cands["ref_" + p.stem] = p
    return cands


def main() -> None:
    ts = build_testsets()
    cands = discover()
    print("candidates:", list(cands))
    out: dict[str, dict] = {}
    for name, path in cands.items():
        tok = Tokenizer.from_file(str(path))
        out[name] = {"path": str(path.relative_to(OUT.parent.parent)),
                     "vocab_size": tok.get_vocab_size(), "scores": {}}
        for t in TESTS:
            out[name]["scores"][t] = score(tok, ts[t])
        print(f"  {name}: vocab={tok.get_vocab_size()} "
              f"bench={out[name]['scores']['bench_all']['chars_per_token']:.3f}", flush=True)

    # --------------------------------------------------------------- report
    L = ["# 05 对照实验：同一批测试文本上的 chars/token\n",
         "所有候选都在 `out/testsets.json` 里字节相同的测试集上打分。"
         "对照 tokenizer 由 `04_train_controls.py` 用生产同款配置"
         "（BPE + ByteLevel(add_prefix_space=True)、initial_alphabet=ByteLevel.alphabet()、"
         "min_frequency=2、specials=<pad>/<bos>/<eos>/<unk>）训练。\n"]
    L.append("## 候选清单\n")
    L.append(fmt_table([[n, f"`{Path(v['path']).name}`", v["vocab_size"]]
                        for n, v in out.items()], ["候选", "文件", "vocab"]))
    L.append("")

    for t in TESTS:
        L.append(f"## 测试集 `{t}`\n")
        rows = []
        base = out["prod_v13"]["scores"][t]["tokens"]
        bp = out["prod_v13"]["scores"][t]
        for n, v in out.items():
            s = v["scores"][t]
            rows.append([n, v["vocab_size"], f"{s['chars']:,}", f"{s['tokens']:,}",
                         f"{s['chars_per_token']:.3f}", f"{s['bytes_per_token']:.3f}",
                         f"{base / s['tokens']:.3f}"])
        L.append(fmt_table(rows, ["候选", "vocab", "字符", "token", "chars/token",
                                  "bytes/token", "prod token 数 / 本候选 token 数"]))
        L.append("")
        L.append(f"> prod_v13: {bp['chars']:,} 字符 → {bp['tokens']:,} token，"
                 f"{bp['chars_per_token']:.3f} chars/token，{bp['bytes_per_token']:.3f} bytes/token\n")

    # class detail for the main candidates on bench_all + aga_r1
    for t in ("bench_all", "aga_r1"):
        L.append(f"## 类别明细 `{t}`\n")
        rows = []
        for n, v in out.items():
            s = v["scores"][t]
            for cls in CJKCLS:
                ct = s["class_tokens"].get(cls, 0)
                if not ct:
                    continue
                rows.append([n, cls, f"{ct:,}", f"{s['class_chars'].get(cls, 0):,}",
                             f"{s['class_chars'].get(cls, 0) / ct:.3f}",
                             f"{ct / s['tokens'] * 100:.2f}%"])
        L.append(fmt_table(rows, ["候选", "类别", "token", "字符", "chars/token", "占预算"]))
        L.append("")
        L.append(f"### 纯净 run 上的 chars/token `{t}`\n")
        rows = []
        for n, v in out.items():
            s = v["scores"][t]
            for cls in ("cjk", "latin", "digit", "code"):
                tk = s["run_tokens"].get(cls, 0)
                if not tk:
                    rows.append([n, cls, 0, "-", "-"])
                    continue
                rows.append([n, cls, f"{s['run_chars'][cls]:,}", f"{tk:,}",
                             f"{s['run_chars'][cls] / tk:.3f}",
                             f"{s['run_bytes'][cls] / tk:.3f}"])
        L.append(fmt_table(rows, ["候选", "类别", "run 字符", "run token",
                                  "chars/token", "bytes/token"]))
        L.append("")

    (OUT / "out" / "05_compare.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "out" / "05_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote out/05_compare.{md,json}")


if __name__ == "__main__":
    main()
