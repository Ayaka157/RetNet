"""把项目的训练曲线导出成可复用的素材：PNG + 原始 CSV + 说明。

数据源全部是真实产物，不重算、不插值：
  checkpoints/retnet_500m_v15b/checkpoint-67390/trainer_state.json   (ep1, 3090)
  checkpoints/retnet_500m_v15b_ep2/trainer_state.json               (ep2, PRO 6000)
  checkpoints/retnet_500m_v15b_ep3/trainer_state.json               (ep3, PRO 6000, 线上)
  checkpoints/retnet_272m_v14/checkpoint-10000/trainer_state.json   (上一代 272M，仅参考)
"""
import csv
import json
import os
import shutil

import fitz

ROOT = r'E:\Workspaces\RetNet'
OUT = os.path.join(ROOT, 'figures')
DATA = os.path.join(OUT, 'data')
RAWDIR = os.path.join(OUT, 'raw')

BAND = 30          # 标题带高度（pt）
TITLES = {
    'fig-0p5b-loss':  'RetNet-0.5B \u2014 validation and training loss across three training legs',
    'fig-0p5b-lr':    'RetNet-0.5B \u2014 learning-rate schedule, one full cosine per leg',
    'fig-0p5b-step':  'RetNet-0.5B \u2014 composition of one 575 ms optimization step',
    'fig-bench-cats': 'RetNet-0.5B \u2014 bench_v3 score by category (1,020 items)',
    'fig-loss':       'RetNet-272M \u2014 training loss over 10,000 optimization steps',
    'fig-lr':         'RetNet-272M \u2014 learning-rate schedule',
    'fig-steps':      'RetNet-272M \u2014 step time and throughput by configuration',
    'fig-truncation': 'RetNet-272M \u2014 truncation and think-leakage before and after continuation',
    'fig-gemm':       'RetNet-272M \u2014 GEMM shape invocation counts in the compiled graph',
    'fig-kernels':    'RetNet-272M \u2014 kernel-level attribution of one training step',
}

LEGS = [
    ('ep1', 'RTX 3090 24 GB', 8192, r'checkpoints\retnet_500m_v15b\checkpoint-67390\trainer_state.json'),
    ('ep2', 'RTX PRO 6000 96 GB', 32768, r'checkpoints\retnet_500m_v15b_ep2\trainer_state.json'),
    ('ep3', 'RTX PRO 6000 96 GB (online)', 32768, r'checkpoints\retnet_500m_v15b_ep3\trainer_state.json'),
]


def titled(src_pdf, dst_pdf, title):
    """在原图上方加一条标题带；内容仍为矢量（show_pdf_page 不栅格化）。"""
    src = fitz.open(src_pdf)
    w, h = src[0].rect.width, src[0].rect.height
    out = fitz.open()
    page = out.new_page(width=w, height=h + BAND)
    page.show_pdf_page(fitz.Rect(0, BAND, w, h + BAND), src, 0)
    page.draw_line(fitz.Point(0, BAND), fitz.Point(w, BAND),
                   color=(0.78, 0.78, 0.78), width=0.7)
    page.insert_text(fitz.Point(0, BAND - 10), title, fontsize=10.5,
                     fontname='helv', color=(0.12, 0.12, 0.12))
    out.save(dst_pdf)
    out.close()
    src.close()


def main():
    os.makedirs(DATA, exist_ok=True)

    # ---- 1) CSV：逐步 loss / lr ----
    rows_all = []
    evals_all = []
    for name, gpu, tok_per_step, rel in LEGS:
        p = os.path.join(ROOT, rel)
        st = json.load(open(p, encoding='utf-8'))
        for e in st['log_history']:
            if 'loss' in e:
                rows_all.append({
                    'leg': name, 'gpu': gpu, 'step': e['step'],
                    'tokens': e['step'] * tok_per_step,
                    'loss': ('%.6f' % e['loss']),
                    'learning_rate': ('%.10g' % e['learning_rate']) if 'learning_rate' in e else '',
                    'grad_norm': ('%.4f' % e['grad_norm']) if 'grad_norm' in e else '',
                })
        for e in st['log_history']:
            if 'eval_loss' in e:
                evals_all.append({
                    'leg': name, 'gpu': gpu, 'step': e['step'],
                    'tokens': e['step'] * tok_per_step,
                    'eval_loss': ('%.10f' % e['eval_loss']),
                    'eval_runtime_s': e.get('eval_runtime', ''),
                })

    f1 = os.path.join(DATA, '0p5b_train_loss.csv')
    with open(f1, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['leg', 'gpu', 'step', 'tokens', 'loss', 'learning_rate', 'grad_norm'])
        w.writeheader()
        w.writerows(rows_all)

    f2 = os.path.join(DATA, '0p5b_eval_loss.csv')
    with open(f2, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['leg', 'gpu', 'step', 'tokens', 'eval_loss', 'eval_runtime_s'])
        w.writeheader()
        w.writerows(evals_all)

    print('  CSV: %s  (%d 行)' % (os.path.basename(f1), len(rows_all)))
    print('  CSV: %s  (%d 行)' % (os.path.basename(f2), len(evals_all)))

    # ---- 2) 原始矢量图另存 raw/，带标题的写入 figures/ ----
    srcdir = os.path.join(ROOT, 'docs', 'latex', 'figures')
    os.makedirs(RAWDIR, exist_ok=True)
    for fn in sorted(os.listdir(srcdir)):
        if not fn.endswith('.pdf'):
            continue
        shutil.copy2(os.path.join(srcdir, fn), os.path.join(RAWDIR, fn))
        stem = os.path.splitext(fn)[0]
        titled(os.path.join(srcdir, fn), os.path.join(OUT, fn),
               TITLES.get(stem, stem))

    # ---- 3) 渲染 PNG（4x ≈ 288 dpi，适配幻灯片/文档）----
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith('.pdf'):
            continue
        d = fitz.open(os.path.join(OUT, fn))
        pix = d[0].get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
        png = os.path.join(OUT, fn.replace('.pdf', '.png'))
        pix.save(png)
        print('  PNG: %-24s %d x %d px' % (os.path.basename(png), pix.width, pix.height))

    print('  输出目录:', OUT)


if __name__ == '__main__':
    main()
