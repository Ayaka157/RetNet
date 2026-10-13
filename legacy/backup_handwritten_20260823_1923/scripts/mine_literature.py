"""Mine literature corpora into RP training samples.

Inputs (downloaded by scripts/download_literature.py):
- data_cache/literature/wikisource_zh.jsonl  — Chinese classics (novels,
  operas, poems; also laws — filtered out by title)
- data_cache/literature/wikisource_en.jsonl  — English classics
- data_cache/literature/cornell_movie_dialog.jsonl — movie dialogue lines

Outputs:
- data_cache/literature/{zh,en}_plain.jsonl   — narrative prose (for the plain
  /pretrain channel: teaches description & narrative voice)
- data_cache/literature/{zh,en}_rp.jsonl      — mined dialogue (role-name +
  line pairs, converted to 用户/助手 with a 设定 persona)
- data_cache/literature/cornell_rp.jsonl      — movie dialogue as RP pairs

    python scripts/mine_literature.py --limit 2000 (smoke test)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.chat import pairs_from_roles  # noqa: E402
from retnet.packing import clean, cjk_ratio  # noqa: E402

ROLE_LINE = re.compile(r"^([^\s:：]{1,20})\s*[:：]\s*(.+)$")
# title filters: keep literary works, drop laws/treaties/notices
KEEP_ZH = re.compile(r"(小说|传|记|演义|梦|录|赋|词|诗|曲|剧|歌|行|谣|铭|颂|志|话本|评话|弹词|鼓词|宝卷|传奇|杂剧|南戏|戏文|话|疏|论|考|解|序|跋)", re.I)
DROP_ZH = re.compile(r"(法|公约|条约|條約|通知|憲法|宪法|条例|條例|规定|规定|协定|协议|宣言|公报|决议|决定|办法|细则|章程|规则|标准|规划|计划|方案|报告|批复|答复|意见|函|令|谕|诏|碑|志铭|碑銘|佛|經卷|经卷|經/|經)", re.I)


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def is_literary_zh(title: str) -> bool:
    t = title or ""
    if not t:
        return False
    if DROP_ZH.search(t):
        return False
    return bool(KEEP_ZH.search(t))


def split_speaker_lines(text: str) -> List[Tuple[str, str]]:
    """Extract (speaker, line) pairs from classical dialogue text.

    Handles 「xxx道：」「xxx曰：」「xxx说：」 style plus 【/（） stage hints.
    """
    out = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^[【\[]([^】\]]+)[】\]]", "", s)   # strip stage direction
        m = ROLE_LINE.match(s)
        if m and len(m.group(1)) <= 16:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


def mine_wikisource(path: Path, is_zh: bool, limit: int):
    plain_out = path.parent / (path.stem + "_plain.jsonl")
    rp_out = path.parent / (path.stem + "_rp.jsonl")
    n_plain = n_rp = 0
    with open(plain_out, "w", encoding="utf-8") as fp, \
         open(rp_out, "w", encoding="utf-8") as fr:
        for obj in read_jsonl(path):
            if limit and n_plain + n_rp >= limit:
                break
            title = obj.get("title", "")
            text = obj.get("text", "") or ""
            if is_zh and not is_literary_zh(title):
                continue
            if len(text) < 200:
                continue
            # 1) RP: dialogue-heavy sections
            entries = split_speaker_lines(text)
            if len(entries) >= 4:
                # group consecutive speaker lines into scenes
                scene: List[Tuple[str, str]] = []
                for sp, ln in entries:
                    if sp == "旁白":
                        continue
                    scene.append((sp, ln))
                    if len(scene) >= 2:
                        msgs = []
                        for s, l in scene:
                            msgs.append(("user" if len(msgs) % 2 == 0 else "assistant", l))
                        pairs = pairs_from_roles(msgs)
                        if pairs:
                            fr.write(json.dumps({
                                "title": title,
                                "pairs": pairs[:6],
                            }, ensure_ascii=False) + "\n")
                            n_rp += 1
                            if limit and n_rp >= limit:
                                break
                        scene = []
            if limit and n_rp >= limit:
                break
            # 2) plain narrative prose (strip dialogue lines, HTML, keep narration)
            plain_lines = [re.sub(r"<[^>]+>", "", l).strip() for l in text.split("\n")
                           if l.strip() and not ROLE_LINE.match(l.strip())]
            plain = "\n".join(l for l in plain_lines if l)
            if len(plain) > 300 and (not is_zh or cjk_ratio(plain) > 0.6):
                fp.write(json.dumps({"title": title, "text": plain[:4000]},
                                    ensure_ascii=False) + "\n")
                n_plain += 1
    print(f"{path.name}: plain={n_plain} rp={n_rp} -> {plain_out.name}, {rp_out.name}")


def mine_cornell(path: Path, limit: int):
    rp_out = path.parent / "cornell_rp.jsonl"
    n = 0
    scene: List[str] = []
    with open(rp_out, "w", encoding="utf-8") as f:
        for line in path.open("r", encoding="utf-8"):
            if limit and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "") if isinstance(obj, dict) else obj
            except Exception:
                continue
            text = text.strip()
            m = re.match(r"^([A-Z][A-Z0-9_\- ]{0,24})\s+(.+)$", text)
            if not m:
                continue
            speaker, text = m.group(1).strip(), m.group(2).strip()
            if len(text) < 2 or len(text) > 300:
                continue
            scene.append((speaker, text))
            if len(scene) >= 4:
                msgs = []
                for sp, tx in scene[:8]:
                    msgs.append(("user" if len(msgs) % 2 == 0 else "assistant", tx))
                pairs = pairs_from_roles(msgs)
                if pairs and len(pairs) >= 2:
                    f.write(json.dumps({
                        "source": "cornell_movie_dialog",
                        "pairs": pairs[:6],
                    }, ensure_ascii=False) + "\n")
                    n += 1
                scene = scene[-2:]
    print(f"cornell: rp={n} -> {rp_out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lit_dir", default="data_cache/literature")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    d = Path(args.lit_dir)
    if (d / "wikisource_zh.jsonl").exists():
        mine_wikisource(d / "wikisource_zh.jsonl", True, args.limit)
    if (d / "wikisource_en.jsonl").exists():
        mine_wikisource(d / "wikisource_en.jsonl", False, args.limit)
    if (d / "cornell_movie_dialog.jsonl").exists():
        mine_cornell(d / "cornell_movie_dialog.jsonl", args.limit)


if __name__ == "__main__":
    main()
