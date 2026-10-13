"""Build the master dataset table (18 columns) from HF survey + local knowledge.

Sources of truth:
- data_cache/dataset_survey.json  (HF API metadata: description/license/storage/tags)
- local filesystem (already-used datasets: real token counts from meta.json,
  real local disk usage)
- scripts/download_*.py / docs/05 (usage stage, provenance)

Outputs:
- docs/07_数据集总表.csv  (UTF-8 BOM, Excel friendly)
- docs/07_数据集总表.md  (markdown mirror)

Column rules: 没有=“ / ”  不适用=“ - ”
Token counts that are estimates are prefixed with ≈ (HF storage bytes / 4).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

survey = json.loads((ROOT / "data_cache/dataset_survey.json").read_text(encoding="utf-8"))

# 人工归纳笔记（基于 HF 原始描述逐条阅读填写；缺失时回退规则推断）
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_notes import NOTES  # noqa: E402

# ---------------- local facts for datasets already used ----------------
LOCAL = {
    # repo -> (local_bytes, local_tokens, stage)
    "HuggingFaceFW/fineweb-edu": (0, 0, "预训练(旧 500M+新 505M token)"),
    "opencsg/chinese-fineweb-edu": (0, 0, "预训练 500M token"),
    "opencsg/Chinese-FineWeb-Edu-V2": (0, 0, "预训练 1,014M token"),
    "HuggingFaceTB/smollm-corpus": (0, 0, "预训练 300M+403M token"),
    "Skywork/SkyPile-150B": (0, 0, "预训练 504M token"),
    "wikimedia/wikipedia": (0, 0, "预训练 203M token"),
    "open-web-math/open-web-math": (0, 0, "预训练 152M token"),
    "wikitext": (0, 0, "预训练(旧)+97M 英文模型"),
    "hysi-lab/LCCC": (0, 0, "闲聊 30M token(经 chat_mix_v20k/chat_recovery_mix)"),
    "svjack/GLM-Open-Dialogue-Chinese-Dataset": (0, 0, "闲聊 10-15M token"),
    "silk-road/ChatHaruhi-54K-Role-Playing-Dialogue": (25, 11_648_294, "RP-SFT v4-v8"),
    "silk-road/ChatHaruhi-Expand-118K": (378, 43_114_014, "RP-SFT v4-v8(场景挖掘)"),
    "silk-road/Haruhi-Zero-RolePlaying-movie-PIPPA": (9, 302_065, "RP-SFT v4-v8"),
    "Neph0s/CoSER": (399, 5_624_322, "RP-SFT v4-v8(portraying 过滤)"),
    "shibing624/roleplay-zh-sharegpt-gpt4-data": (6, 4_011_014, "RP-SFT v4-v8"),
    "grimulkan/LimaRP-augmented": (18, 124_416, "RP-SFT v4-v8"),
    "Norquinal/claude_multiround_chat_30k": (163, 2_848_796, "指令 v4-v8"),
    "BelleGroup/multiturn_chat_0.8M": (57, 4_602_666, "指令 v4-v8"),
    "BelleGroup/train_3.5M_CN": (514, 5_753_498, "指令 v4-v8"),
    "YeungNLP/moss-003-sft-data": (1563, 5_447_168, "指令 v4-v8"),
    "shareAI/ShareGPT-Chinese-English-90k": (478, 6_044_443, "指令 v4-v8"),
    "opencsg/smoltalk-chinese": (58, 4_015_727, "指令 v4-v8"),
    "YeungNLP/firefly-train-1.1M": (823, 8_183_088, "指令 v4-v8"),
    "wangrui6/Zhihu-KOL": (291, 2_796_806, "指令 v4-v8"),
    "hfl/ruozhiba_gpt4": (3, 706_051, "指令 v4-v8"),
    "HuggingFaceH4/ultrachat_200k": (331, 1_887_025, "指令 v4-v8"),
    "shibing624/sharegpt_gpt4": (287, 3_257_791, "指令 v4-v8"),
    "HuggingFaceTB/smoltalk": (2, 299_629, "闲聊 v4-v8"),
    "cornell-movie-dialog/cornell_movie_dialog": (24, 9_896_128, "RP-SFT v7-v8(电影对白)"),
    "wikimedia/wikisource": (1200, 3_224_347, "RP-SFT v7-v8(古典对白)+叙事待预训练"),
    "zxbsmk/webnovel_cn": (291, 20_577_889, "小说语感 v4-v8"),
    "m-a-p/COIG-CQIA": (27, 3_608_041, "指令/闲聊 v4-v8"),
    # non-HF datasets (survey not applicable)
    "_gutenberg": (22, 1_708_880, "莎士比亚 RP v8 + 叙事待预训练"),
    "_chinese-poetry": (91, 30_418_286, "预训练 shard poetry_zh.npy"),
}

# ---------------- classification helpers (from description/repo name) ----------------
def classify(repo: str, desc: str, tags):
    """领域分类（* = 由名称/描述推断，非官方声明）"""
    t = (repo + " " + (desc or "")).lower()
    cats = []
    if any(k in t for k in ("roleplay", "role-play", "role play", "角色", "chatbot", "catgirl", "waifu", "moe", "dialogue", "conversation", "聊天", "女友", "猫娘", "恋爱", "haruhi", "baize")):
        cats.append("对话/RP")
    if any(k in t for k in ("novel", "fiction", "story", "小说", "轻小说", "网文", "文学", "poetry", "诗", "词")):
        cats.append("文学/小说")
    if any(k in t for k in ("reasoning", "cot", "chain-of-thought", "math", "数学", "o1", "r1")):
        cats.append("推理/数学")
    if any(k in t for k in ("code", "coding", "programming", "coder", "stackoverflow", "git", "代码", "编程")):
        cats.append("代码")
    if any(k in t for k in ("law", "legal", "法律", "gaokao", "高考", "exam", "考试", "chemistry", "物理", "化学", "science")):
        cats.append("知识/考试")
    if any(k in t for k in ("web", "crawl", "corpus", "pretrain", "预训练", "text", "cpt", "idiom")):
        cats.append("预训练语料")
    if any(k in t for k in ("medical", "drug", "医学", "药", "health")):
        cats.append("医疗")
    if any(k in t for k in ("qa", "question", "answer", "问答")):
        cats.append("问答")
    if any(k in t for k in ("comment", "meme", "bilibili", "social", "社区")):
        cats.append("社交/评论")
    if any(k in t for k in ("translation", "translate", "翻译", "glossary", "术语")):
        cats.append("翻译/术语")
    if any(k in t for k in ("travel", "旅游")):
        cats.append("旅游")
    if any(k in t for k in ("bench", "benchmark", "评测")):
        cats.append("评测基准")
    return (cats[0] if cats else "其他") + "*", ("/".join(cats) if cats else "其他") + "*"


def distill_model(repo: str, desc: str):
    t = (repo + " " + (desc or "")).lower()
    r = " / "
    if "r1" in t or "deepseek" in t:
        r = "DeepSeek-R1"
    elif "o1" in t:
        r = "OpenAI o1"
    elif "gemma" in t:
        r = "Gemma"
    elif "opus" in t:
        r = "Claude Opus"
    elif "gpt4" in t or "gpt-4" in t:
        r = "GPT-4"
    elif "gpt" in t:
        r = "GPT 系"
    return r if r == " / " else r + "*"


def mode(repo: str, desc: str):
    t = (repo + " " + (desc or "")).lower()
    r = " / "
    if any(k in t for k in ("distill", "蒸馏", "synthetic", "合成", "generated")):
        r = "蒸馏/合成"
    elif any(k in t for k in ("crawl", "爬", "web", "scrape", "清洗", "filter")):
        r = "爬取/清洗"
    elif any(k in t for k in ("collect", "整理", "curat", "translation", "翻译", "hand")):
        r = "整理/收集"
    return r if r == " / " else r + "*"


def rating(repo: str, cats: str, stage: str):
    """A=强烈推荐用于本项目 B=可选 C=低优先级 D=不适用/不建议（* = 主观预测）"""
    if stage != "未使用":
        return "A（已用）"
    t = repo.lower()
    r = "D"
    if any(k in t for k in ("sex", "criminal", "red-criminal")):
        r = "D"
    elif any(k in t for k in ("haruhi", "roleplay", "catgirl", "novel", "novelist", "dialogue",
                              "anime", "story", "lightnovel", "role-play", "baize", "waifu", "moe")):
        r = "A"
    elif any(k in t for k in ("cot", "reasoning", "o1", "r1", "math", "openo1", "distill")):
        r = "B"
    elif any(k in t for k in ("code", "coder", "stackoverflow", "gitee", "linux")):
        r = "B"
    elif any(k in t for k in ("law", "gaokao", "chemistry", "legal", "cmb", "medical", "drug")):
        r = "C"
    else:
        r = "C"
    return r + "*"


def fmt_tokens(repo: str, storage_bytes):
    """已用数据集=本地真实计数（无标记）；未用=HF 存储估算（≈ 前缀）。"""
    if repo in LOCAL and LOCAL[repo][1]:
        return f"{LOCAL[repo][1]:,}"
    if not storage_bytes:
        return " / "
    est = storage_bytes / 4
    if est >= 1e9:
        return f"≈{est / 1e9:.2f}B"
    if est >= 1e6:
        return f"≈{est / 1e6:.0f}M"
    return f"≈{est / 1e3:.0f}K"


def fmt_disk(repo: str, storage_bytes):
    """磁盘空间=预计占用。已用=本地实占；未用=HF 存储值（≈ 前缀，预计占用）。"""
    if repo in LOCAL and LOCAL[repo][0]:
        mb = LOCAL[repo][0]
        return f"{mb}MB（本地实占）" if mb < 1024 else f"{mb / 1024:.1f}GB（本地实占）"
    if repo.startswith("_"):
        return " / "
    if not storage_bytes:
        return " / "
    gb = storage_bytes / 1024 ** 3
    if gb >= 1:
        return f"≈{gb:.1f}GB（HF 存储）"
    mb = storage_bytes / 1024 ** 2
    if mb >= 1:
        return f"≈{mb:.0f}MB（HF 存储）"
    return f"≈{storage_bytes / 1024:.0f}KB（HF 存储）"


def mode_of(repo: str, desc: str):
    return mode(repo, desc)


def rating_of(repo: str, cats: str, stage: str):
    return rating(repo, cats, stage)


def short_desc(desc: str):
    if not desc:
        return " / "
    d = re.sub(r"\s+", " ", desc).strip()
    return d[:180]


# ---------------- column order ----------------
COLS = ["数据集名字", "语言", "所属领域", "细分领域", "内容简述", "蒸馏模型", "数据产生方式",
        "是否清洗", "子集", "超集", "数据来源", "关联研究", "数据量(k/M/B Tokens)", "磁盘空间(MB/GB)",
        "适用度评级", "是否已投入使用", "使用阶段/任务", "数据集发布平台与链接", "使用许可/开源协议"]

USED_STAGE = {k: v[2] for k, v in LOCAL.items() if not k.startswith("_")}


def row_for(repo: str):
    info = survey.get(repo) or {}
    storage = info.get("used_storage_bytes")
    lic = info.get("license")
    lic = lic if isinstance(lic, str) else (" / " if not lic else json.dumps(lic, ensure_ascii=False))
    n = NOTES.get(repo)
    stage = USED_STAGE.get(repo, "未使用")
    is_used = "是" if stage != "未使用" else "否"
    if n:
        lang = n["lang"]
        main_cat = n["domain"]
        cats = n["sub"]
        desc = n["desc"]
        distill = n["distill"]
        mode = n["mode"]
        cleaned = n["cleaned"]
        subset = n["subset"]
        superset = n["superset"]
        source = n["source"]
        related = n["related"]
        est_stage = n["stage"]
        rating = n["rating"] + ("（已用）" if is_used == "是" else "*")
    else:
        # 回退：规则推断（标注 *）
        desc = info.get("description") or ""
        main_cat, cats = classify(repo, desc, info.get("tags") or [])
        lang = "中文*" if ("zh" in repo.lower() or "chinese" in repo.lower()) else " / "
        distill = distill_model(repo, desc)
        mode = mode_of(repo, desc)
        cleaned = " / "
        subset = " / "
        superset = " / "
        source = "HuggingFace"
        related = " / "
        est_stage = "候选（未归纳）"
        rating = rating_of(repo, cats, stage)
    return [
        repo.split("/")[-1],
        lang,
        main_cat,
        cats,
        desc,
        distill,
        mode,
        cleaned,
        subset,
        superset,
        source,
        related,
        fmt_tokens(repo, storage),
        fmt_disk(repo, storage),
        rating,
        is_used,
        est_stage,
        f"https://huggingface.co/datasets/{repo}",
        lic,
    ]


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    # ordered: user's 90 new first, then used/probed
    user90 = [r for r in survey if r in open(
        ROOT / "Datasets_Findings_From_User.txt", encoding="utf-8").read()]
    rest = [r for r in survey if r not in user90]
    repos = user90 + rest
    rows = [row_for(r) for r in repos if not r.startswith("_")]

    # non-HF rows (hand-filled, truthful; 18+1 columns)
    rows += [
        ["Project Gutenberg(古登堡精选32本)", "英文", "文学/小说", "文学/戏剧",
         "莎士比亚全集、傲慢与偏见、罪与罚、白鲸等公共领域经典(戏剧对白+小说叙事)",
         " - ", "公开下载", "部分", " - ", " - ", "gutenberg.org/gutendex API", " - ",
         "1,708,880(已用部分)", "22MB（本地实占）", "A（已用）", "是", "莎士比亚 RP v8 + 叙事待预训练",
         "https://www.gutenberg.org", "Public Domain"],
        ["chinese-poetry", "中文", "文学/诗词", "诗词曲赋",
         "全唐诗5.6万+宋词2.5万+元曲1万+诗经/楚辞/论语(33.5万首作品)",
         " - ", "整理/开放数据", "是", " - ", " - ", "GitHub 开源仓库", " - ",
         "30,418,286(已编码)", "91MB（本地实占）", "A（已用）", "是", "预训练 shard poetry_zh.npy",
         "https://github.com/chinese-poetry/chinese-poetry", "MIT/CC(仓库声明)"],
    ]

    csv_path = ROOT / "docs/07_数据集总表_v2.csv"
    md_path = ROOT / "docs/07_数据集总表_v2.md"
    with open(str(csv_path) + ".tmp", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    os.replace(str(csv_path) + ".tmp", str(csv_path))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 07 数据集总表\n\n")
        f.write(f"> 共 {len(rows)} 条（HF 探测 123 个 + 非 HF 2 个）。\n")
        f.write("> **标记约定**：`≈` = 估算值（HF 存储字节÷4 换算的 token 量；磁盘空间为 HF 报告的"
                "存储占用=预计占用）；`*` = 推断值（领域/语言/蒸馏模型/产生方式/评级由名称与描述"
                "推断，未经人工核实）；无 `≈`/`*` 的数值 = 本地真实计数或 HF 官方字段。\n")
        f.write("> 生成于 2026-08-23，数据来源：HF API 实查 + 本地账目。\n\n")
        f.write("| " + " | ".join(COLS) + " |\n")
        f.write("|" + "---|" * len(COLS) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |\n")
    print(f"rows={len(rows)} -> {csv_path} , {md_path}")


if __name__ == "__main__":
    main()
