"""中英两本 PDF 的跨语言数值一致性核对（按数值归一化，不受显示精度影响）。

用法：python logs/_crosscheck2.py
"""
import re
import os
import fitz

ZH = r'E:\Workspaces\RetNet\docs\latex\zh\RetNet-0.5B-技术报告.pdf'
EN = r'E:\Workspaces\RetNet\docs\latex\en\RetNet-0.5B-Technical-Report.pdf'

# 终稿应同时出现在两版的数值（数值，不是字符串）
KEY = [
    504179896, 2016755680, 1656209408, 552330489, 552060928, 552058880, 552075264,
    3.28496, 9.9782, 2.6358, 1.9459, 0.0764, 0.1310, 60447.71, 13551.5,
    54970.2, 57111.0, 45.8, 16.79, 16.75, 19.47, 18.74, 54.96,
    338196736, 67.08, 165983160, 70.2, 51.5, 0.5430, 575, 61.6, 27.9, 8.4,
    1346, 25.5, 1020, 0.1037, 0.032, 0.130, 59.61, 72.94, 0.00,
    0.1029, 0.0904, 0.0651, 204, 509, 307, 0.15217, 102.396, 276,
    0.2857, 0.2667, 0.2000, 0.1429, 0.0000, 0.0125, 0.0222, 0.0250,
    0.0111, 0.1000, 136, 608, 1014, 3.395, 0.0375, 16.2,
]
# 已被推翻、两版都不该出现
STALE = ['3.28494', '82/82', '552,140,800', '1,656,340,480', '14,100', '0.0765',
         '60,542.7', '待补', 'TODO', '0.1074', '0.1479', '72.16']

def tokens(path):
    d = fitz.open(path)
    t = ''.join(d[i].get_text() for i in range(d.page_count))
    out = []
    for m in re.finditer(r'\d[\d,]*\.?\d*', t):
        s = m.group(0).replace(',', '')
        try:
            out.append(float(s))
        except ValueError:
            pass
    return d.page_count, t, out


zp, zt, zn = tokens(ZH)
ep, et, en = tokens(EN)
print('zh: %d 页 / %d 数字 token' % (zp, len(zn)))
print('en: %d 页 / %d 数字 token' % (ep, len(en)))
print()


def has(vals, v, tol=2e-3):
    for x in vals:
        if abs(x - v) <= tol * max(1.0, abs(v)):
            return True
    return False


print('=== 1. 关键数值双侧存在（按数值比对，容忍显示精度）===')
bad = []
for v in KEY:
    a = has(zn, v)
    b = has(en, v)
    if not (a and b):
        bad.append((v, a, b))
        print('  %-14s zh=%s en=%s' % (v, '有' if a else '缺', '有' if b else '缺'))
print('  通过 %d / %d' % (len(KEY) - len(bad), len(KEY)))

print()
print('=== 2. 废弃旧值残留 ===')
for k in STALE:
    a, b = zt.count(k), et.count(k)
    print('  %-14s zh=%d en=%d %s' % (k, a, b, 'OK' if a == 0 and b == 0 else 'FOUND'))

print()
print('=== 3. 结论 ===')
print('  关键数值未通过: %d' % len(bad))
if bad:
    print('  ', bad)

