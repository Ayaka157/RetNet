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

LEGS = [
    ('ep1', 'RTX 3090 24 GB', 8192, r'checkpoints\retnet_500m_v15b\checkpoint-67390\trainer_state.json'),
    ('ep2', 'RTX PRO 6000 96 GB', 32768, r'checkpoints\retnet_500m_v15b_ep2\trainer_state.json'),
    ('ep3', 'RTX PRO 6000 96 GB (online)', 32768, r'checkpoints\retnet_500m_v15b_ep3\trainer_state.json'),
]


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

    # ---- 2) 矢量 PDF 原样复制 ----
    srcdir = os.path.join(ROOT, 'docs', 'latex', 'figures')
    for fn in sorted(os.listdir(srcdir)):
        if fn.endswith('.pdf'):
            shutil.copy2(os.path.join(srcdir, fn), os.path.join(OUT, fn))

    # ---- 3) 渲染 PNG（3x ≈ 216 dpi，适配幻灯片/文档）----
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
