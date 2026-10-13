"""Batch-survey HuggingFace datasets via the Hub API (authenticated).

Pulls real metadata (splits sizes, license, description) for:
1. the 90 datasets the user found via the HF tag search (Text Generation+Chinese),
2. every HF dataset already used or probed in this project (from docs/05).

Writes:
- data_cache/dataset_survey.json   (raw metadata cache)
- docs/07_数据集总表.csv            (final wide table)

The access token is read from "HF Access Token.txt" and used only in-process.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, dataset_info

USER_FILE = ROOT / "HF Access Token.txt"
NEW_DATASETS = [  # from Datasets_Findings_From_User.txt (user's HF tag search)
    "Egor-3926/CoT-XLang", "kobe73er/shiji-translation-dataset",
    "ALmonster/MATH-Hard-Chinese", "Azure99/blossom-v6-sft-stage1",
    "CSJianYang/CodeArena", "llamafactory/OpenO1-SFT",
    "Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
    "Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT",
    "inclusionAI/Ling-Coder-SyntheticQA", "inclusionAI/Ling-Coder-SFT",
    "benchang1110/Chinese-DeepSeek-R1-Distill-data-110k-opencc",
    "FunnySaltyFish/Better-Ruozhiba", "LLaMAX/BenchMAX_Question_Answering",
    "LAMDA-NeSy/ChinaTravel", "Midsummra/bilibilicomment", "bh2821/LightNovel5000",
    "Mxode/Meow-Reasoning-100K", "allenlin316/twinkle-dialogue-gemma3-2025-08",
    "Mxode/Chinese-StackOverflow-QA-C_Language", "Mxode/I_Wonder_Why-Chinese",
    "Oedon42/webnovelbench", "wobure/zhuzi-baijia-books",
    "liushuaiqian/Chinese-High-School-Chemistry-Correction-Dataset",
    "WNJXYK/LawQA", "keke0130/project-sekai-qa-zh-tw",
    "manuelcaccone/actuarial-global-glossary-multilingual",
    "bdx33/ted-talks-in-chinese-zhongwen", "liumindmind/NekoQA-10K",
    "kxdw2580/catgirl-dataset", "Seikaijyu/Sex-novel-filtered",
    "GeCDU/novel-f3-v0.1", "YCWTG/MoeGirlPedia_zh_cleaned_latest",
    "Minami-su/Anime_novel_datasets", "silk-road/ChatHaruhi-Waifu",
    "Lifan-Z/Chinese-poetries-txt", "silk-road/Haruhi-Zero-RolePlaying-movie-PIPPA",
    "silk-road/Haruhi-Baize-Role-Playing-Conversation", "AiCloser/sharegpt_cot_dataset",
    "Orion-zhen/sys-novel-cleaned", "THUdyh/Oryx-SFT-Data", "fzkuji/CMB",
    "jtatman/CoT_reformatted", "m-a-p/COIG-CQIA", "LooksJuicy/Chinese-Roleplay-SingleTurn",
    "shibing624/roleplay-zh-sharegpt-gpt4-data",
    "lianghsun/chinese-english-technical-patent-glossary",
    "opencsg/chinese-cosmopedia", "GaryYang123/zh-meme-sft-8k",
    "RX5950XT/silicon-girlfriend-dataset", "taoyang168/HundredCV-Chat",
    "c0mpleX999/Anime_novel_datasets", "Morton-Li/ChineseWebText2.0-HighQuality",
    "EnlistedGhost/role-play-bench", "OpenSakura/OpenSakura-DS-260220-LN-ja-zh-PT-Adam",
    "AhYin/Cantonese-Prototype-Roleplay-Dataset", "Dxniz/Novelist-CoT",
    "HeAAAAA/story_generation_sft", "shantoislamdev/linux-commands",
    "nyuuzyou/gitee-code", "phoenixcph/AtmosphericQA-1k-Chinese-Distillation",
    "hehuanhao/phy", "iosscu/Refined-Anime-Text", "zjuqx/ZhiHua_QA",
    "y1yang0/scholar-novels-curated",
    "xzxnoccc/Chinese-QA-Agriculture_Forestry_Animal_Husbandry_Fishery",
    "YCWTG/Better-Ruozhiba-ChatML", "dibao-research/wanli-dibao-corpus",
    "dibao-research/ming-qing-wenji-corpus",
    "YCWTG/Opus-4.6-Reasoning-3000x-filtered-ChatML", "qundao/data-zh-poetry",
    "cyberlangke/Nana-catgirl-dataset-110k",
    "callofthenight1/gaokao-sft-chinese-balanced",
    "callofthenight1/gaokao-sft-chinese-strict-abcd",
    "callofthenight1/gaokao-sft-chinese-large", "gujilab/chinese-classical-corpus",
    "zzoceanpie/yukari-synthetic", "zzoceanpie/yukari-thbwiki",
    "twnlp/ChineseErrorCorrector4-4B", "6shmqshy7q-dot/drug-dev-sft-dataset",
    "Umranz/Red-Criminal-v1", "IAAR-Shanghai/LongEvoRoleBench",
    "BearNetworkChain/Machine-Checkable-Blockchain-Execution-Specification",
    "mikuhhn1239/novel-agent-sft-dataset", "bhxdianzhang/ParaPaper",
    "noah248/chinese-legal-sft", "asd567557275/chinese_novel",
    "liumindmind/NekoCOT-30K", "jiviteshjn/mc4-zh-idiom-cpt",
    "laubonghaudoi/shunpao",
]
USED_DATASETS = [  # already used or probed (from docs/05_数据源档案.md)
    "HuggingFaceFW/fineweb-edu", "opencsg/chinese-fineweb-edu",
    "opencsg/Chinese-FineWeb-Edu-V2", "HuggingFaceTB/smollm-corpus",
    "Skywork/SkyPile-150B", "wikimedia/wikipedia", "open-web-math/open-web-math",
    "wikitext", "hysi-lab/LCCC", "svjack/GLM-Open-Dialogue-Chinese-Dataset",
    "silk-road/ChatHaruhi-54K-Role-Playing-Dialogue",
    "silk-road/ChatHaruhi-Expand-118K", "Neph0s/CoSER",
    "BelleGroup/multiturn_chat_0.8M", "BelleGroup/train_3.5M_CN",
    "YeungNLP/moss-003-sft-data", "shareAI/ShareGPT-Chinese-English-90k",
    "opencsg/smoltalk-chinese", "YeungNLP/firefly-train-1.1M",
    "wangrui6/Zhihu-KOL", "hfl/ruozhiba_gpt4", "HuggingFaceH4/ultrachat_200k",
    "shibing624/sharegpt_gpt4", "HuggingFaceTB/smoltalk",
    "cornell-movie-dialog/cornell_movie_dialog", "wikimedia/wikisource",
    "zxbsmk/webnovel_cn", "PygmalionAI/PIPPA", "lemonilia/LimaRP",
    "ZhuangXialie/role_play_chinese", "Aratako/Moss-003-SFT-Chinese",
    "llamafactory/alpaca_zh_demo", "lmsys/lmsys-chat-1m", "allenai/WildChat-1M",
    "ZenMoore/RoleBench", "MohamedRashad/Cornell_Movie-Dialogs_Corpus",
    "spawn99/CornellMovieDialogCorpus", "bookcorpus/bookcorpus",
    "grimulkan/LimaRP-augmented", "Norquinal/claude_multiround_chat_30k",
]


def read_token() -> str:
    # token is the FIRST line; the file also contains a permissions note.
    return USER_FILE.read_text(encoding="utf-8").splitlines()[0].strip()


def fmt_bytes(n):
    if n is None:
        return " / "
    gb = n / 1024 ** 3
    if gb >= 1:
        return f"{gb:.2f}GB"
    mb = n / 1024 ** 2
    if mb >= 1:
        return f"{mb:.0f}MB"
    return f"{n / 1024:.0f}KB"


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    token = read_token()
    api = HfApi(token=token)
    cache_path = ROOT / "data_cache/dataset_survey.json"
    results = {}
    if cache_path.exists():
        results = json.loads(cache_path.read_text(encoding="utf-8"))

    repos = list(dict.fromkeys(NEW_DATASETS + USED_DATASETS))
    for i, repo in enumerate(repos):
        if repo in results and results[repo].get("ok"):
            continue
        t0 = time.time()
        try:
            info = api.dataset_info(repo)
            card = getattr(info, "card_data", None) or {}
            if card is not None and not isinstance(card, dict):
                card = {k: getattr(card, k) for k in
                        ("license", "language", "task_categories", "size_categories")
                        if hasattr(card, k)}
            results[repo] = {
                "ok": True,
                "description": (info.description or "")[:600],
                "license": card.get("license") if isinstance(card, dict) else None,
                "language": card.get("language") if isinstance(card, dict) else None,
                "task_categories": card.get("task_categories") if isinstance(card, dict) else None,
                "size_categories": card.get("size_categories") if isinstance(card, dict) else None,
                "tags": (info.tags or [])[:24],
                "used_storage_bytes": getattr(info, "usedStorage", None),
                "downloads": getattr(info, "downloads", None),
                "gated": getattr(info, "gated", None),
                "last_modified": str(getattr(info, "last_modified", ""))[:10],
                "sec": round(time.time() - t0, 1),
            }
            r = results[repo]
            print(f"OK   {repo:<52} storage={fmt_bytes(r['used_storage_bytes'])} "
                  f"dl={r['downloads']} license={r['license']}", flush=True)
        except Exception as exc:
            results[repo] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:150]}",
                             "sec": round(time.time() - t0, 1)}
            print(f"FAIL {repo:<52} {type(exc).__name__}", flush=True)
        if (i + 1) % 25 == 0:
            cache_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                  encoding="utf-8")

    cache_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    ok = sum(1 for v in results.values() if v.get("ok"))
    print(f"\nDONE: {ok}/{len(repos)} reachable -> {cache_path}", flush=True)


if __name__ == "__main__":
    main()
