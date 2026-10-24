"""打印指定文件行的上下文，把疑似密钥的片段掩码后再输出。"""
import re
import sys

ROOT = r'E:\Workspaces\RetNet\\'
TARGETS = [
    (r'docs\latex\en\RetNet-0.5B-Technical-Report.tex', 397),
    (r'docs\latex\reports\0p5b\en_supplement.tex', 256),
    (r'serve\DEPLOY.md', 55),
    (r'serve\DEPLOY.md', 83),
    (r'serve\DEPLOY.md', 85),
    (r'serve\DEPLOY.md', 117),
    (r'Kaggle\202609092100.ipynb', 86),
    (r'docs\24_AutoDL上机手册.md', 298),
    (r'scripts\linux\setup_autodl_ssh.ps1', 41),
]

MASKERS = [
    re.compile(r'hf_[A-Za-z0-9]{12,}'),
    re.compile(r'gh[pousr]_[A-Za-z0-9]{12,}'),
    re.compile(r'sk-[A-Za-z0-9_\-]{12,}'),
    re.compile(r'(?i)(TUNNEL_TOKEN|CF_API_TOKEN|HF_TOKEN)\s*[:=]\s*\S+'),
    re.compile(r'[A-Za-z0-9+/]{50,}={0,2}'),
]


def mask(s):
    for rx in MASKERS:
        s = rx.sub(lambda m: m.group(0)[:4] + '<<MASKED>>', s)
    return s


for rel, ln in TARGETS:
    p = ROOT + rel
    try:
        lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
    except Exception as e:
        print('!! 读不到', rel, e)
        continue
    print('--- %s:%d ---' % (rel, ln))
    for i in range(max(0, ln - 2), min(len(lines), ln + 1)):
        print('   %5d| %s' % (i + 1, mask(lines[i])[:190]))
    print()
