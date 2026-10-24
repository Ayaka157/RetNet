"""发布前的脱敏扫描：只报告「文件:行号 + 规则 + 掩码片段」，绝不打印明文密钥。

用法: python logs/_scrub_scan.py [根目录] [--all]
默认跳过 大目录/二进制/已排除项；--all 时连 docs/legacy 等也扫。
"""
import os
import re
import sys

ROOT = r'E:\Workspaces\RetNet'
SKIP_DIRS = {'checkpoints', 'data_cache', 'hf_cache', 'logs', '__pycache__',
             '.git', 'node_modules', '.venv', '.regress'}
SKIP_EXT = {'.pdf', '.dmp', '.npy', '.npz', '.xlsx', '.safetensors', '.bin',
            '.pt', '.pth', '.png', '.jpg', '.jpeg', '.gif', '.zip', '.tgz',
            '.ids', '.wcode', '.ico', '.woff', '.woff2', '.ttf'}

# 规则：(名称, 正则)
RULES = [
    ('HF token',        re.compile(r'hf_[A-Za-z0-9]{20,}')),
    ('GitHub token',    re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}')),
    ('OpenAI key',      re.compile(r'sk-[A-Za-z0-9_\-]{20,}')),
    ('AWS key id',      re.compile(r'AKIA[0-9A-Z]{16}')),
    ('Google key',      re.compile(r'AIza[0-9A-Za-z_\-]{30,}')),
    ('Slack token',     re.compile(r'xox[baprs]-[A-Za-z0-9\-]{10,}')),
    ('Private key',     re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----')),
    ('Bearer header',   re.compile(r'[Bb]earer\s+[A-Za-z0-9._\-]{20,}')),
    ('JWT',             re.compile(r'eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}')),
    ('assignment',      re.compile(r'(?i)\b(api[_-]?key|secret|passwd|password|access[_-]?token|auth[_-]?token|hf[_-]?token|tunnel[_-]?token)\b\s*[:=]\s*[\'"]?([^\s\'"]{12,})')),
    ('base64 blob',     re.compile(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{60,}={0,2}(?![A-Za-z0-9+/])')),
    # 本项目特有的基础设施标识
    ('tunnel id',       re.compile(r'a9c74e4f-f174-4382-b26e-b8d134136ef2')),
    ('tunnel name',     re.compile(r'Ayaka157-(Claude|LLM-Services)')),
    ('secrets path',    re.compile(r'C:\\+Tools\\+secrets[^\s\'"`)]*')),
    ('private ip',      re.compile(r'\b(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b')),
    ('ssh alias',       re.compile(r'\b(autodl6000|autodl3090|autodl|llmbox|gpu-box)\b')),
    ('home path',       re.compile(r'C:\\+Users\\+Ayaka157')),
    ('email',           re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')),
    ('domain',          re.compile(r'\b(retnet|bn-sync)\.ayaka157\.icu\b')),
    ('cloudflared exe', re.compile(r'C:\\+Tools\\+cloudflared\.exe')),
    ('wrangler path',   re.compile(r'\.wrangler[\\/]config')),
]


def mask(s):
    s = s.strip()
    if len(s) <= 8:
        return '*' * len(s)
    return s[:4] + '…' + ('*' * 6) + '…' + s[-2:]


def main():
    hits = {}
    scanned = 0
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith('.git')]
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXT:
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, ROOT)
            if 'Kaggle' in rel and 'checks' in rel:
                continue
            try:
                if os.path.getsize(p) > 8 * 1024 * 1024:
                    continue
                txt = open(p, encoding='utf-8', errors='replace').read()
            except Exception:
                continue
            scanned += 1
            for i, line in enumerate(txt.splitlines(), 1):
                for name, rx in RULES:
                    m = rx.search(line)
                    if m:
                        hits.setdefault(name, []).append((rel, i, mask(m.group(0))))

    print('扫描文件数:', scanned)
    print()
    total = 0
    for name, _ in RULES:
        v = hits.get(name)
        if not v:
            continue
        total += len(v)
        print('### %s  —— %d 处' % (name, len(v)))
        seen = set()
        for rel, ln, mk in v:
            key = (rel, ln)
            if key in seen:
                continue
            seen.add(key)
            print('   %s:%d   %s' % (rel, ln, mk))
            if len(seen) >= 25:
                print('   …（更多同类命中已省略）')
                break
        print()
    print('合计命中:', total)


if __name__ == '__main__':
    main()
