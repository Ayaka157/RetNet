"""Standalone verifier: check that this snapshot's documents match its code.

No network access, no file modification. Run:

    python verify.py

Checks:

  A. every code block in docs/01_function_reference.md is byte-identical to the
     same line range in code/<path>;
  B. docs/01 covers every function/method of the 7 training-path files (AST);
  C. every explanation in docs/01 is >=100 chars and contains no placeholder;
  D. every 00 id cited in docs/01 exists in docs/00;
  E. every code block in docs/00F1_fix_matrix.md is byte-identical to source;
  F. every 00F1 explanation is >=200 chars and every 00 id has an entry;
  G. status consistency: `fix_status.py`, the docs/01 table, the docs/00F1
     `**Status**` lines and the README table must be identical (the previous
     submission never checked the status table, so "all checks passed" could
     coexist with a table saying nine fixes were missing);
  H. literal contracts: greppable invariants (defaults, flag names, thresholds)
     must exist / not exist in the source;
  I. MANIFEST.sha256: every listed sha256 matches and every file in the tree is
     listed (except the manifest itself and caches).

Scope (a boundary, not a defect): A/B cover only the 7 files in FILES.
plan.py, control.py, provenance.py, defaults.py, scripts/ and tests/ are outside
the per-function reference; their docstrings and tests are the documentation.
Exit code 1 if anything fails.
"""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
DOC00 = ROOT / "docs" / "00_error_table.md"
DOC01 = ROOT / "docs" / "01_function_reference.md"
DOCF1 = ROOT / "docs" / "00F1_fix_matrix.md"
README = ROOT / "README.md"
MANIFEST = ROOT / "MANIFEST.sha256"
STATUS_PY = CODE / "scripts" / "fix_status.py"

sys.path.insert(0, str(CODE / "scripts"))

FILES = [
    "retnet_hf/train.py",
    "retnet_hf/modeling_retnet.py",
    "retnet_hf/retention_fast.py",
    "retnet_hf/data.py",
    "retnet_hf/fused_ce.py",
    "run_task.py",
    "scripts/grow_500m.py",
]

HEAD01 = re.compile(r"^### `(.+)` \(L(\d+)–L(\d+)\)$")
FILE01 = re.compile(r"^## `(.+)`$")
REF_F1 = re.compile(r"^`(.+?)` L(\d+)–L(\d+)$")
HEAD_F1 = re.compile(r"^## (E\d{2}|N\d{2}) ")
ROW01 = re.compile(r"^\| (E\d{2}|N\d{2}) \| ([^|]+?) \|")
STAT_F1 = re.compile(r"^\*\*Status\*\*: (.+)$")

#: Literal contracts: (file, fragment that must appear, why). A leading "!"
#: means the fragment must NOT appear.
CLAIMS = [
    ("retnet_hf/defaults.py", "DEFAULT_SAVE_TOTAL_LIMIT = 4", "E17 single default"),
    ("retnet_hf/train.py", "load_best_model_at_end=False", "E15 in-training selection off"),
    ("retnet_hf/train.py", '"--allow_dead_params"', "E08 explicit pruned-checkpoint exemption"),
    ("retnet_hf/train.py", '"--dead_grad_tol"', "E07 gradient-coverage threshold"),
    ("retnet_hf/train.py", "trainer.add_callback(GradCoverageCallback(",
     "E07 gradient coverage is actually wired"),
    ("retnet_hf/train.py", "!grad.abs().max()==0",
     "E07 the old comment's non-existent check is gone"),
    ("retnet_hf/modeling_retnet.py", "DEFAULT_CE_CHUNK_BYTES = 64 << 20", "E22 byte budget"),
    ("retnet_hf/modeling_retnet.py", "!chunk_tokens: int = 262_144",
     "E22 fixed token count removed"),
    ("retnet_hf/plan.py", "require_eos: bool = True", "E12 Stage field"),
    ("retnet_hf/plan.py", '"require_eos": s.require_eos', "P2-2 hot reload passes it through"),
    ("retnet_hf/data.py", "return bool(meta.get(\"sources\")) and bool(meta.get(\"text_fields\"))",
     "E13 legacy producer proof"),
    ("scripts/build_flat_corpus.py", '"rebuild_cmd"', "E13 producer writes rebuild command"),
    ("scripts/eval_holdout.py", '"--ctx-ablation"', "P1-6 context probe"),
    ("scripts/reinit_gamma.py", "model.safetensors.index.json", "P1-7 shard merge"),
    ("scripts/grow_500m.py", "FINAL_GATE = 0.0", "P1-8 zero side branch at the output"),
]


def parse_01(text: str):
    lines = text.splitlines()
    cur = None
    out = []
    i = 0
    while i < len(lines):
        m = FILE01.match(lines[i])
        if m:
            cur = m.group(1)
            i += 1
            continue
        m = HEAD01.match(lines[i])
        if not m:
            i += 1
            continue
        qual, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        j = i + 1
        while j < len(lines) and not lines[j].startswith("```python"):
            j += 1
        k = j + 1
        while k < len(lines) and lines[k] != "```":
            k += 1
        code = "\n".join(lines[j + 1:k])
        e = k + 1
        buf = []
        while e < len(lines) and not lines[e].startswith("### ") and lines[e] != "---":
            buf.append(lines[e])
            e += 1
        out.append((cur, qual, a, b, code, "\n".join(buf).strip()))
        i = e
    return out


def parse_f1(text: str):
    lines = text.splitlines()
    cur = None
    blocks = []
    texts = {}
    status = {}
    i = 0
    while i < len(lines):
        m = HEAD_F1.match(lines[i])
        if m:
            cur = m.group(1)
            texts[cur] = []
            i += 1
            continue
        if cur:
            ms = STAT_F1.match(lines[i])
            if ms:
                status[cur] = ms.group(1).strip()
                i += 1
                continue
            m = REF_F1.match(lines[i])
            if m:
                k = i + 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if k < len(lines) and lines[k].startswith("```python"):
                    j = k + 1
                    buf = []
                    while j < len(lines) and lines[j] != "```":
                        buf.append(lines[j])
                        j += 1
                    blocks.append((cur, m.group(1), int(m.group(2)), int(m.group(3)),
                                   "\n".join(buf)))
                    i = j + 1
                    continue
            texts[cur].append(lines[i])
        i += 1
    return blocks, {k: "\n".join(v).strip() for k, v in texts.items()}, status


def parse_table(text: str, pattern) -> dict:
    out = {}
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def collect_funcs(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []

    def rec(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(prefix + child.name)
                rec(child, prefix + child.name + ".")
            elif isinstance(child, ast.ClassDef):
                rec(child, prefix + child.name + ".")

    rec(tree, "")
    return out


def read_src(rel: str):
    p = CODE / rel
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").splitlines()


def check_manifest(fails: list[str]) -> None:
    if not MANIFEST.exists():
        fails.append("[I] MANIFEST.sha256 is missing")
        return
    listed = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            fails.append(f"[I] malformed manifest line: {line!r}")
            continue
        digest, rel = parts[0], parts[1].strip()
        listed[rel] = digest
    missing, bad = [], []
    for rel, want in listed.items():
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            bad.append(rel)
    if missing:
        fails.append(f"[I] manifest lists missing files: {missing[:5]}")
    if bad:
        fails.append(f"[I] sha256 mismatch: {bad[:5]}")
    skip = {"MANIFEST.sha256"}
    actual = set()
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel in skip or "__pycache__" in rel or ".pytest_cache" in rel:
            continue
        actual.add(rel)
    unlisted = sorted(actual - set(listed))
    if unlisted:
        fails.append(f"[I] unlisted files in the tree: {unlisted[:5]}")
    print(f"[I] MANIFEST {len(listed)} entries, {len(missing)} missing, "
          f"{len(bad)} hash mismatches, {len(unlisted)} unlisted; "
          f"{len(actual)} files in tree")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    fails = []
    doc00 = DOC00.read_text(encoding="utf-8")
    ids = set(re.findall(r"\b(E\d{2}|N\d{2})\b", doc00))

    # ---------------- 01 ----------------
    secs = parse_01(DOC01.read_text(encoding="utf-8"))
    print(f"[A] parsed {len(secs)} function sections in 01")
    for f, q, a, b, code, expl in secs:
        src = read_src(f)
        if src is None:
            fails.append(f"[A] code file missing: {f}")
            continue
        if not (1 <= a <= b <= len(src)):
            fails.append(f"[A] {f}::{q} line range out of bounds L{a}-L{b}")
            continue
        if code != "\n".join(src[a - 1:b]):
            fails.append(f"[A] code block differs from source: {f}::{q}")
        if len(expl) < 100:
            fails.append(f"[C] 01 explanation under 100 chars ({len(expl)}): {f}::{q}")
        for bad in ("same as above", "omitted", "placeholder"):
            if bad in expl:
                fails.append(f"[C] 01 contains placeholder {bad!r}: {f}::{q}")

    covered = {(f, q) for f, q, *_ in secs}
    total = 0
    for rel in FILES:
        p = CODE / rel
        if not p.exists():
            fails.append(f"[B] missing file: {rel}")
            continue
        for q in collect_funcs(p):
            total += 1
            if (rel, q) not in covered:
                fails.append(f"[B] 01 misses function: {rel}::{q}")
    print(f"[B] {total} functions in code, {len(covered)} documented")

    doc01 = DOC01.read_text(encoding="utf-8")
    refs = set()
    for m in re.finditer(r"\*\*00 links\*\*: (.+)", doc01):
        refs |= set(re.findall(r"E\d{2}", m.group(1)))
    dangling = sorted(refs - ids)
    if dangling:
        fails.append(f"[D] 01 cites ids not present in 00: {dangling}")
    print(f"[D] 00 has {len(ids)} ids, 01 cites {len(refs)}, {len(dangling)} dangling")

    # ---------------- 00F1 ----------------
    f1_blocks, f1_text, f1_status = parse_f1(DOCF1.read_text(encoding="utf-8"))
    print(f"[E] 00F1 parsed {len(f1_text)} entries, {len(f1_blocks)} code blocks")
    for eid, rel, a, b, code in f1_blocks:
        src = read_src(rel)
        if src is None:
            fails.append(f"[E] 00F1 references a missing file: {rel}")
            continue
        if not (1 <= a <= b <= len(src)) or code != "\n".join(src[a - 1:b]):
            fails.append(f"[E] 00F1 code block mismatch: {eid} {rel} L{a}-{b}")
    for eid, txt in f1_text.items():
        if len(txt) < 200:
            fails.append(f"[F] 00F1 {eid} explanation is only {len(txt)} chars")
        for bad in ("same as above", "omitted", "placeholder"):
            if bad in txt:
                fails.append(f"[F] 00F1 {eid} contains placeholder {bad!r}")
    missing = sorted(ids - set(f1_text))
    if missing:
        fails.append(f"[F] 00F1 missing entries: {missing}")
    print(f"[F] 00F1 covers {len(set(f1_text) & ids)}/{len(ids)} ids")

    # ---------------- G status consistency ----------------
    from fix_status import FIX_STATUS  # noqa: E402

    t01 = parse_table(DOC01.read_text(encoding="utf-8"), ROW01)
    readme_table = parse_table(README.read_text(encoding="utf-8"), ROW01)
    for name, table in (("docs/01_function_reference.md", t01),
                        ("README", readme_table),
                        ("docs/00F1_fix_matrix.md", f1_status)):
        if table != FIX_STATUS:
            diff = {k: (FIX_STATUS.get(k), table.get(k))
                    for k in sorted(set(FIX_STATUS) | set(table))
                    if FIX_STATUS.get(k) != table.get(k)}
            fails.append(f"[G] {name} status differs from fix_status.py: {diff}")
    print(f"[G] status tables: fix_status {len(FIX_STATUS)}, 01 {len(t01)}, "
          f"README {len(readme_table)}, 00F1 {len(f1_status)}")

    # ---------------- H literal contracts ----------------
    for rel, needle, why in CLAIMS:
        src = CODE / rel
        if not src.exists():
            fails.append(f"[H] missing file: {rel}")
            continue
        text = src.read_text(encoding="utf-8")
        if needle.startswith("!"):
            if needle[1:] in text:
                fails.append(f"[H] {rel} still contains a forbidden fragment "
                             f"({why}): {needle[1:]}")
        elif needle not in text:
            fails.append(f"[H] {rel} is missing fragment ({why}): {needle}")
    print(f"[H] literal contracts: {len(CLAIMS)}")

    check_manifest(fails)

    if fails:
        print(f"\n{len(fails)} checks failed")
        for x in fails[:40]:
            print("   ", x)
        return 1
    print("\nPASS: 01/00F1 code blocks byte-identical, every function covered, "
          "explanation lengths OK, no dangling ids, status table consistent in all "
          "four places, literal contracts hold, MANIFEST complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
