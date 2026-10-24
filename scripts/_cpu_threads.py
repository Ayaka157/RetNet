"""线程数对解码速度的影响 —— 决定「2 vCPU 的便宜机器」到底够不够用。"""
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = r'E:\Workspaces\RetNet\logs\_hf_stage\RetNet-0.5B'
tok = AutoTokenizer.from_pretrained(D, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(D, trust_remote_code=True, dtype=torch.float32).eval()
ids = tok('<bos>用户：你好，介绍一下你自己。\n助手：')['input_ids']
t = torch.tensor([ids])

print('%8s %10s %10s %10s' % ('线程', '预填 s', '解码 s', 'tok/s'))
with torch.inference_mode():
    for nt in (8, 4, 2, 1):
        torch.set_num_threads(nt)
        m(t)                                        # 预热
        a = time.time(); m(t); pre = time.time() - a
        a = time.time()
        g = m.generate(t, max_new_tokens=32, temperature=0.6)
        dec = time.time() - a
        n = len(g[0]) - len(ids)
        print('%8d %10.3f %10.2f %10.2f' % (nt, pre, dec, n / dec))
