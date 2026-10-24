"""精确拆分 CPU 推理的时间：加载 / 预填 / 逐 token 解码。跑两轮看稳态。"""
import os
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = r'E:\Workspaces\RetNet\logs\_hf_stage\RetNet-0.5B'
P = psutil.Process(os.getpid())
rss = lambda: P.memory_info().rss / 1024 / 1024

print('torch.get_num_threads() =', torch.get_num_threads())
print('是否启用了 MKL/oneDNN 相关环境:', {k: v for k, v in os.environ.items()
                                         if 'OMP' in k or 'MKL' in k or 'THREAD' in k})

t0 = time.time()
tok = AutoTokenizer.from_pretrained(D, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(D, trust_remote_code=True, dtype=torch.float32).eval()
print('加载 %.2f s, RSS %.0f MB' % (time.time() - t0, rss()))

ids = tok('<bos>用户：你好，介绍一下你自己。\n助手：')['input_ids']
t = torch.tensor([ids])

for rnd in (1, 2):
    # 预填（走 parallel，一次前向吃掉整个 prompt）
    with torch.inference_mode():
        a = time.time()
        out = m(t)
        prefill = time.time() - a
    # 解码 32 个 token
    with torch.inference_mode():
        a = time.time()
        g = m.generate(t, max_new_tokens=32, temperature=0.6)
        dec = time.time() - a
    n = len(g[0]) - len(ids)
    print('第 %d 轮: 预填 %.3f s | 32 token 解码 %.2f s | %.2f tok/s | 峰值 RSS %.0f MB'
          % (rnd, prefill, dec, n / dec if dec else 0, rss()))
print('TEXT:', repr(tok.decode(g[0].tolist()[len(ids):])[:120]))
