"""检查两本 PDF 是否有页面内容超出下边界（对应 LaTeX 的 Overfull \\vbox）。"""
import fitz

for label, path in [
    ('zh', r'E:\Workspaces\RetNet\docs\latex\zh\RetNet-0.5B-技术报告.pdf'),
    ('en', r'E:\Workspaces\RetNet\docs\latex\en\RetNet-0.5B-Technical-Report.pdf'),
]:
    d = fitz.open(path)
    print('===', label, d.page_count, '页, 页面尺寸', d[0].rect)
    worst = []
    for i in range(d.page_count):
        p = d[i]
        H = p.rect.height
        W = p.rect.width
        maxy = 0.0
        minx = 1e9
        maxx = 0.0
        for b in p.get_text('blocks'):
            x0, y0, x1, y1 = b[:4]
            if y1 > maxy:
                maxy = y1
            minx = min(minx, x0)
            maxx = max(maxx, x1)
        # 页脚/页码通常在底部，用 > H-40 视为越界
        over = maxy - (H - 36)
        worst.append((over, i + 1, round(maxy, 1), H, round(maxx, 1), W))
    worst.sort(reverse=True)
    print('  内容底边最靠下的 5 页 (超出量pt, 页, 内容maxY, 页高, 内容maxX, 页宽):')
    for w in worst[:5]:
        print('   ', w)
