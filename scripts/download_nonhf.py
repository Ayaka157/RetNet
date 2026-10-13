"""Non-HF data acquisition: Project Gutenberg classics + chinese-poetry.

Gutenberg: gutendex.com API -> top N English books -> download plaintext from
gutenberg.org (public domain, legal to keep locally).
chinese-poetry: the well-known GitHub repo (全唐诗/宋词/论语/etc., CC-ish data).

    python scripts/download_nonhf.py --gutenberg 60 --poetry 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dsh-retnet-data/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_gutenberg(n_books: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # gutendex: most-downloaded English books
    data = json.loads(fetch("https://gutendex.com/books?sort=popular&languages=en"))
    books = data["results"][:n_books]
    got, skipped = 0, 0
    for b in books:
        gid = b["id"]
        title = (b.get("title") or f"book{gid}").replace("/", "_").replace("\\", "_")[:60]
        dst = out_dir / f"gutenberg_{gid}_{title}.txt"
        if dst.exists():
            got += 1
            continue
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        try:
            raw = fetch(url)
            if len(raw) < 10_000:
                skipped += 1
                continue
            dst.write_bytes(raw)
            got += 1
            if got % 10 == 0:
                print(f"  gutenberg: {got}/{n_books} books, "
                      f"{sum(f.stat().st_size for f in out_dir.glob('*.txt')) / 1048576:.0f} MB",
                      flush=True)
            time.sleep(0.3)  # be polite to gutenberg.org
        except Exception as exc:
            skipped += 1
            print(f"  fail {gid} {title}: {type(exc).__name__}", flush=True)
    total = sum(f.stat().st_size for f in out_dir.glob("*.txt"))
    print(f"gutenberg done: {got} books, {skipped} skipped, "
          f"{total / 1048576:.0f} MB -> {out_dir}", flush=True)


def download_poetry(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "chinese-poetry.zip"
    if not zip_path.exists():
        print("downloading chinese-poetry.zip ...", flush=True)
        raw = fetch(
            "https://github.com/chinese-poetry/chinese-poetry/archive/refs/heads/master.zip",
            timeout=300)
        zip_path.write_bytes(raw)
        print(f"  zip: {zip_path.stat().st_size / 1048576:.0f} MB", flush=True)
    extract_dir = out_dir / "chinese-poetry"
    if not extract_dir.exists():
        print("extracting ...", flush=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        # rename extracted dir if needed
        kids = [p for p in out_dir.iterdir() if p.is_dir() and p.name != "chinese-poetry"]
        if kids:
            (out_dir / "chinese-poetry").mkdir(exist_ok=True)
            # move json dirs into place
            for kid in kids:
                for sub in kid.iterdir():
                    dest = extract_dir / sub.name
                    if not dest.exists():
                        sub.rename(dest)
                kid.rmdir()
    n_json = sum(1 for _ in (extract_dir / "json").glob("*.json")) if (extract_dir / "json").exists() else 0
    print(f"chinese-poetry ready at {extract_dir} (poet json files: {n_json})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gutenberg", type=int, default=0,
                    help="number of most-popular English books to fetch (0=skip)")
    ap.add_argument("--poetry", type=int, default=0, help="1 = fetch chinese-poetry")
    ap.add_argument("--out_dir", default="data_cache/nonhf")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    out = Path(args.out_dir)
    if args.gutenberg:
        download_gutenberg(args.gutenberg, out / "gutenberg")
    if args.poetry:
        download_poetry(out / "poetry")
    print("NONHF DONE", flush=True)


if __name__ == "__main__":
    main()
