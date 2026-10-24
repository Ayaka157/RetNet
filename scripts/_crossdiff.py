"""中英两本 PDF 的跨语言数值机械 diff（Lead 终验工具）。

用法：python logs/_crossdiff.py
输出：只在 zh 出现、只在 en 出现、以及两版都出现的数字 token。
"""
import re
import sys
import os
import fitz

ZH = r'E:\Workspaces\RetNet\docs\latex\zh\RetNet-0.5B-技术报告.pdf'
EN = r'E:\Workspaces\RetNet\docs\latex\en\RetNet-0.5B-Technical-Report.pdf'

# 关键数字清单（Lead 指定，来自 facts.md 与终稿）
KEY = [
    '504,179,896', '2,016,755,680', '1,656,209,408', '552,330,489', '552,060,928',
    '552,058,880', '552,075,264', '3.28496', '9.9782', '2.6358', '1.9459',
    '0.0764', '0.1310', '60,447.71', '13,551.5', '54,970', '57,111', '45.8',
    '16.79', '16.75', '19.47', '18.74', '54.96', '338,196,736', '67.08',
    '165,983,160', '78', '70.2', '51.5', '0.5430', '575', '61.6', '27.9', '8.4',
    '1,346', '25.5', '1,020', '0.1074', '55.0', '0.29', '0.1029', '0.0923',
    '0.0717', '204', '509', '307', '72.16', '0.1479', '0.2857', '0.2667',
    '0.2000', '0.1429', '0.1200',
]

# 已被推翻、两版都不该出现的值
STALE = ['3.28494', '82/82', '552,140,800', '1,656,340,480', '14,100', '0.0765',
         '0.1309', '60,542.7', '待补', 'TODO']

NUM = re.compile(r'\d[\d,]*\.?\d*')


def text_of(path, label):
    if not os.path.exists(path):
        print('[%s] 不存在: %s' % (label, path))
        return None
    d = fitz.open(path)
    t = ''.join(d[i].get_text() for i in range(d.page_count))
    print('[%s] %d 页, 可抽取 %d 字符' % (label, d.page_count, len(t)))
    return t


zh = text_of(ZH, 'zh')
en = text_of(EN, 'en')
if zh is None or en is None:
    sys.exit(1)

print()
print('=== 1. 关键数字清单：zh / en 双侧存在性 ===')
bad = []
for k in KEY:
    a = k in zh
    b = k in en
    if a and b:
        tag = 'both'
    elif a and not b:
        tag = 'ONLY-ZH  <-- 英文缺失'
        bad.append(k)
    elif b and not a:
        tag = 'ONLY-EN  <-- 中文缺失'
        bad.append(k)
    else:
        tag = 'NEITHER <-- 两版都缺'
        bad.append(k)
    if tag != 'both':
        print('  %-16s %s' % (k, tag))
print('  双侧都在: %d / %d' % (len(KEY) - len(bad), len(KEY)))

print()
print('=== 2. 被推翻的旧值残留 ===')
for k in STALE:
    a = zh.count(k)
    b = en.count(k)
    flag = 'OK' if (a == 0 and b == 0) else 'FOUND'
    print('  %-16s zh=%d en=%d  %s' % (k, a, b, flag))

print()
print('=== 3. 数字 token 集合差异（仅统计，不做数量化对齐）===')
nz = set(NUM.findall(zh))
ne = set(NUM.findall(en))
only_zh = nz - ne
only_en = ne - nz
print('  zh 数字 token 种类: %d, en: %d' % (len(nz), len(ne)))
print('  只在 zh: %d 种' % len(only_zh))
print('  只在 en: %d 种' % len(only_en))
big_zh = sorted([x for x in only_zh if len(x) >= 5], key=len, reverse=True)[:40]
big_en = sorted([x for x in only_en if len(x) >= 5], key=len, reverse=True)[:40]
print('  只在 zh 的长数字（>=5 字符）前 40:', big_zh)
print('  只在 en 的长数字（>=5 字符）前 40:', big_en)

print()
print('=== 4. 结论 ===')
print('  关键清单未通过项: %d' % len(bad))
if bad:
    print('  未通过:', bad)
