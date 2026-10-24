import fitz, re, sys, os

zh = r'E:\Workspaces\RetNet\docs\latex\zh\RetNet-0.5B-技术报告.pdf'
en = r'E:\Workspaces\RetNet\docs\latex\en\RetNet-0.5B-Technical-Report.pdf'


def check(path, label):
    if not os.path.exists(path):
        print(label, '=> 不存在:', path)
        return None
    d = fitz.open(path)
    t = ''.join(d[i].get_text() for i in range(d.page_count))
    print('===', label, '===')
    print('  页数:', d.page_count, ' 可抽取字符:', len(t))
    print('  pdftitle:', d.metadata.get('title'))
    return t


t = check(zh, 'zh')
if t:
    good = ['0.1074', '1,020', '504,179,896', '1,656,209,408', '3.28496', '0.0764', '0.1310',
            '60,447.71', '67.08', '0.5430', '13,551.5', '57,111', '78']
    bad = ['待补', '3.28494', '82/82', '552,140,800', '1,656,340,480', '14,100', '0.0765', '0.1309']
    print('  --- 应存在 ---')
    for k in good:
        print('    %-16s %s  count=%d' % (k, 'OK ' if k in t else 'MISS', t.count(k)))
    print('  --- 应不存在 ---')
    for k in bad:
        print('    %-16s %s  count=%d' % (k, 'OK ' if k not in t else 'FOUND', t.count(k)))
    print('  --- 前 260 字 ---')
    print('   ', t[:260].replace('\n', ' | '))

print()
log = r'E:\Workspaces\RetNet\docs\latex\zh\RetNet-0.5B-技术报告.log'
c = open(log, encoding='utf-8', errors='replace').read().splitlines()
errs = [l for l in c if l.startswith('!')]
und = [l for l in c if 'undefined' in l]
ovf = [l for l in c if 'Overfull' in l]
print('=== zh .log ===')
print('  error(!) 行:', len(errs))
for l in errs[:5]:
    print('    ', l[:160])
print('  undefined 行:', len(und))
for l in und[:5]:
    print('    ', l[:160])
print('  Overfull 行:', len(ovf), ' 最大 pt:', max([float(re.search(r'\(([\d.]+)pt', l).group(1)) for l in ovf if re.search(r'\(([\d.]+)pt', l)] or [0]))
