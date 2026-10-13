"""Mine Project Gutenberg texts into RP dialogue + plain narrative.

Shakespeare-style plays use the format:

    HAMLET.
    To be, or not to be...

(ALL-CAPS speaker name on its own line, then the speech line(s)).  That is
already user/assistant-shaped role-play dialogue.  Novels go to plain narrative
(for the pretrain channel: description & narration voice).

    python scripts/mine_gutenberg.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retnet.chat import pairs_from_roles  # noqa: E402

SPEAKER = re.compile(r"^[A-Z][A-Z0-9\'\- ]{1,28}\.$")
STAGE = re.compile(r"^[\(\[][^\)\]]*[\)\]]$|^[A-Z\s\.\,\']+\.$")


def clean_line(s: str) -> str:
    s = s.strip()
    if len(s) < 2:
        return ""
    # strip trailing speaker markers like "HAMLET." that appear mid-line
    s = re.sub(r"\s+[A-Z][A-Z0-9\-\' ]{1,24}\.$", "", s)
    return s


def mine_play(path: Path, max_pairs: int = 4000):
    """Extract (speaker, line) pairs from a Gutenberg play text."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    # cut Gutenberg boilerplate
    start = raw.find("*** START")
    end = raw.find("*** END")
    if start >= 0 and end > start:
        raw = raw[start:end]
    entries = []
    cur_speaker = None
    buf = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if SPEAKER.match(s):
            if cur_speaker and buf:
                entries.append((cur_speaker, " ".join(buf).strip()))
            cur_speaker = s[:-1].strip()
            buf = []
            continue
        if cur_speaker:
            t = clean_line(s)
            if t and not STAGE.match(s):
                buf.append(t)
    if cur_speaker and buf:
        entries.append((cur_speaker, " ".join(buf).strip()))
    return entries


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    gdir = ROOT / "data_cache/nonhf/gutenberg"
    rp_out = ROOT / "data_cache/literature/gutenberg_rp.jsonl"
    plain_out = ROOT / "data_cache/literature/gutenberg_plain.jsonl"
    n_rp = n_plain = 0
    with open(rp_out, "w", encoding="utf-8") as fr, \
         open(plain_out, "w", encoding="utf-8") as fp:
        for path in sorted(gdir.glob("*.txt")):
            title = path.stem.replace("gutenberg_", "").split("_", 1)[-1]
            entries = mine_play(path)
            # play-like (has speaker lines) -> RP
            if len(entries) >= 60:
                msgs = []
                scene = []
                for sp, tx in entries:
                    if len(tx) < 2 or len(tx) > 400:
                        continue
                    scene.append((sp, tx))
                    if len(scene) >= 4:
                        m = []
                        for s, t in scene:
                            m.append(("user" if len(m) % 2 == 0 else "assistant", t))
                        pairs = pairs_from_roles(m)
                        if pairs and len(pairs) >= 2:
                            fr.write(json.dumps({
                                "title": f"{title}（{sp}）",
                                "pairs": pairs[:6],
                            }, ensure_ascii=False) + "\n")
                            n_rp += 1
                        scene = scene[-2:]
                print(f"  {path.name}: {len(entries)} speaker lines -> RP", flush=True)
            # narrative (non-play) -> plain
            raw = path.read_text(encoding="utf-8", errors="replace")
            start = raw.find("*** START")
            end = raw.find("*** END")
            body = raw[start:end] if 0 <= start < end else raw
            prose = [l.strip() for l in body.splitlines()
                     if len(l.strip()) > 40 and not SPEAKER.match(l.strip())]
            text = "\n".join(prose)
            if len(text) > 5000:
                fp.write(json.dumps({"title": title, "text": text[:120_000]},
                                    ensure_ascii=False) + "\n")
                n_plain += 1
    print(f"gutenberg: rp={n_rp} plain={n_plain} -> {rp_out.name}, {plain_out.name}", flush=True)


if __name__ == "__main__":
    main()
