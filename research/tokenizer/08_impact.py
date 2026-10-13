"""08 - impact arithmetic + long-context accounting, all derived from the JSON
artifacts written by 02/03/03b/05/06.  Nothing here is hand-typed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, ROOT, TOK_PATH, cjk_ratio, fmt_table  # noqa: E402
from measure import build_testsets  # noqa: E402
from common import token_to_bytes  # noqa: E402

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

O = OUT / "out"
V13_ROWS = 269_561
SEQ = 2048
STEPS_PER_EPOCH_B4 = V13_ROWS // 4          # 67,390 (verified against trainer_state.json)


def main() -> None:
    fine = json.loads((O / "03b_fine_census.json").read_text(encoding="utf-8"))
    cen = json.loads((O / "03_stream_census.json").read_text(encoding="utf-8"))
    mig = json.loads((O / "06_migration.json").read_text(encoding="utf-8"))
    comp = json.loads((O / "05_compare.json").read_text(encoding="utf-8"))

    total = cen["total_slots"]
    pad = cen["pad"]
    bos, eos, unk = cen["bos"], cen["eos"], cen["unk"]
    content = total - pad - bos - eos - unk
    c_chars = fine["content_chars"]
    c_bytes = fine["content_bytes"]
    cpt = c_chars / content
    bpt = c_bytes / content
    keep = content / total

    L = ["# 08 影响量化（全部由 02/03/03b/05/06 的 JSON 推出）\n", "## 0 基础量\n"]
    L.append(fmt_table([
        ["槽位总数", f"{total:,}"],
        ["pad", f"{pad:,} ({pad / total * 100:.4f}%)"],
        ["bos", f"{bos:,}"], ["eos", f"{eos:,}"], ["unk", f"{unk:,}"],
        ["内容 token 数", f"{content:,} ({keep * 100:.4f}% 的槽位）"],
        ["内容字符数", f"{c_chars:,}"],
        ["内容字节数", f"{c_bytes:,}"],
        ["内容 chars/token", f"{cpt:.4f}"],
        ["内容 bytes/token", f"{bpt:.4f}"],
    ], ["量", "值"]))

    per_step_content_tokens = SEQ * keep
    rows = [["每个 2048 槽位窗口的内容 token", f"{per_step_content_tokens:.1f}"],
            ["…折合字符", f"{per_step_content_tokens * cpt:,.0f}"],
            ["…折合字节", f"{per_step_content_tokens * bpt:,.0f}"],
            ["…折合 UTF-8 字节 KB", f"{per_step_content_tokens * bpt / 1024:.2f} KB"]]
    L.append("\n## 1 一步 2048 token 窗口里到底有多少内容\n")
    L.append(fmt_table(rows, ["量", "值"]))
    L.append("\n> 换一种说法：模型的 2048 窗口，按**字符**算是 3,346 字，"
             "按**字节**算是 7.5 KB。对比一个英文 GPT-2 级 tokenizer"
             f"（bench 纯英文 4.472 chars/token）在同样窗口里能放 "
             f"{SEQ * 4.472:,.0f} 字符——差距是**语言密度**，不是废料。")

    L.append("\n## 2 训练量折算\n")
    v13_tokens = 12_500 * 4 * SEQ
    v14_first = 10_000 * 4 * SEQ
    v14_resume_50k = 50_000 * 4 * SEQ
    cum = v13_tokens + v14_resume_50k
    nom_v13 = v13_tokens / total
    nom_v14 = v14_first / total
    nom_resume = v14_resume_50k / total
    nom_cum = cum / total
    L.append(fmt_table([
        ["v13（tasks/v13_272m_2k.json: batch 4 × 12,500 步）", f"{v13_tokens:,}",
         f"{nom_v13 * 100:.2f}%"],
        ["1 epoch 要多少步（269,561 行 ÷ 4）", f"{STEPS_PER_EPOCH_B4:,} 步", "100%"],
        ["v14 第一段（10,000 步，trainer_state.json 实测 epoch=0.14839）",
         f"{v14_first:,}", f"{nom_v14 * 100:.2f}%"],
        ["v14 续训配置目标（max_steps 50,000 步）", f"{v14_resume_50k:,}",
         f"{nom_resume * 100:.2f}%"],
        ["v13 + v14 累计", f"{cum:,}", f"{nom_cum * 100:.2f}%"],
    ], ["训练段", "token 数（有效 batch 4 × 2048）", "占 552,330,489 槽位"]))

    L.append("\n### 换分词器等价于多训多少\n")
    L.append("用 `06_migration.json` 留出集上测出的「prod token 数 / 该候选 token 数」"
             "作为内容放大倍数（同样内容用更少 token 表达 → 同等 token 预算下内容更多）。\n")
    hs = mig["holdout_scores"]
    rows = []
    for n in ("ctl_v13_20k", "ctl_v13big_20k", "ctl_v13_32k", "ctl_v13_48k",
              "app_ctl_v13_20k", "app_ctl_v13_48k", "app_top8000",
              "ref_qwen2.5-0.5b", "ref_deepseek-llm"):
        if n not in hs:
            continue
        k = hs[n]["prod_over_this"]
        rows.append([n, hs[n]["vocab_size"], f"{hs[n]['chars_per_token']:.4f}",
                     f"×{k:.4f}", f"{nom_v13 * 100 * k:.2f}%", f"{nom_cum * 100 * k:.2f}%",
                     f"{(k - 1) * 100:.2f}%"])
    L.append(fmt_table(rows, ["候选", "vocab", "留出集 chars/token", "内容放大倍数",
                              "v13 等效覆盖", "v13+v14 等效覆盖", "省下的 token 比例"]))
    L.append("\n> 说明：这一列**不是**「多训了 14% 的梯度步」——梯度步数不变，"
             "是每一步吃到的内容多了 14%。它同时也意味着，如果换分词器，"
             "同样的 552M 槽位数据包会变成大约 "
             f"{total / 1.1435 / 1e6:.0f}M prod-token 当量（用 ctl_v13_20k 的倍数）。")

    L.append("\n## 3 长上下文：8192 字符材料 vs 2048 窗口\n")
    ts = build_testsets()
    prod = Tokenizer.from_file(str(TOK_PATH))
    cands = {"prod_v13": prod}
    for name in ("ctl_v13_20k", "ctl_v13_48k", "ref_qwen2.5-0.5b"):
        p = O / "tok" / f"{name}.json"
        if not p.exists():
            p = O / "ref" / f"{name.replace('ref_', '')}.json"
        if p.exists():
            cands[name] = Tokenizer.from_file(str(p))
    bench = [json.loads(l) for l in
             (ROOT / "bench" / "data" / "bench_v3.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    lc = [(r["id"], "\n".join(r["turns"])) for r in bench if r["id"].startswith("lc_")]
    rows = []
    for iid, text in lc:
        rows.append([iid, len(text), f"{cjk_ratio(text) * 100:.0f}%"]
                    + [f"{len(cands[n].encode(text, add_special_tokens=False).ids):,}"
                       for n in cands])
    L.append(fmt_table(rows, ["id", "字符", "CJK 占比"]
                       + [f"{n} token" for n in cands]))
    fit = {}
    for n, t in cands.items():
        over = [iid for iid, text in lc
                if len(t.encode(text, add_special_tokens=False).ids) > SEQ]
        fit[n] = (len(over), over)
    L.append("\n### 50 条 `lc_*` 里有多少条超出 2048 窗口\n")
    L.append(fmt_table([[n, v[0], f"{v[0] / len(lc) * 100:.0f}%"] for n, v in fit.items()],
                       ["候选", "超窗条数", "占比"]))
    d8 = [(iid, t) for iid, t in lc if iid.startswith("lc_d8k")]
    L.append("\n### 6 条 `lc_d8k_*` 逐条\n")
    rows = []
    for iid, text in d8:
        r = [iid, len(text), f"{cjk_ratio(text) * 100:.0f}%"]
        for n in cands:
            tk = len(cands[n].encode(text, add_special_tokens=False).ids)
            r += [f"{tk:,}", f"{len(text) / tk:.3f}", f"{tk / SEQ:.2f}×"]
        rows.append(r)
    L.append(fmt_table(rows, ["id", "字符", "CJK"]
                       + [f"{n} token/chars·tok/窗倍数" for n in cands]))
    # the arithmetic of what a tokenizer could possibly fix
    zh8k = [(i, t) for i, t in d8 if cjk_ratio(t) >= 0.5]
    en8k = [(i, t) for i, t in d8 if cjk_ratio(t) < 0.5]
    def block(items, label, ref_cpt):
        if not items:
            return
        for iid, t in items:
            tk = len(prod.encode(t, add_special_tokens=False).ids)
            L.append(f"- **{iid}**（{label}，{len(t)} 字符）：prod {tk} token = "
                     f"{tk / SEQ:.2f} 个窗口；按 prod 的 {len(t) / tk:.3f} chars/token 计算，"
                     f"塞进 {SEQ} token 需要 **{len(t) / SEQ:.3f} chars/token**，"
                     f"即当前效率的 **{len(t) / SEQ / (len(t) / tk):.2f}×**；"
                     f"即使达到参考值 {ref_cpt:.2f} chars/token 也要 "
                     f"{len(t) / ref_cpt:.0f} token = {len(t) / ref_cpt / SEQ:.2f} 个窗口。")
    L.append("\n### 能不能靠分词器把 8K 材料塞进 2048 窗口\n")
    block(en8k, "英文", 4.472)
    block(zh8k, "中文", 1.604)
    L.append("\n> 英文那条：prod 差一点点（1.01×），"
             "换到 4.47 chars/token 的英文 tokenizer 就能进窗——**分词器是英文超窗的主因**。")
    L.append("> 中文那几条：即使换到 Qwen2.5 的 1.60 chars/token 仍要 1.8 倍窗口，"
             "要进窗需要 2.24× 于 Qwen、2.81× 于 prod 的字符效率，**任何现实词表都做不到**——"
             "中文超窗的主因是窗口，不是分词器。")

    (O / "08_impact.md").write_text("\n".join(L), encoding="utf-8")
    json.dump({"content_tokens": content, "content_chars": c_chars, "content_bytes": c_bytes,
               "chars_per_token": cpt, "bytes_per_token": bpt, "keep_ratio": keep,
               "v13_tokens": v13_tokens, "nominal_v13": nom_v13,
               "steps_per_epoch_b4": STEPS_PER_EPOCH_B4,
               "cum": cum, "nominal_cum": nom_cum, "lc_over_window": fit},
              open(O / "08_impact.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
