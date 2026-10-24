"""量 CPU-only 推理的真实内存与速度 —— 这决定最便宜的托管方案要多大机器。

不写任何文件；只打印 RSS / 峰值 RSS / tok/s。
"""
import os
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = r'E:\Workspaces\RetNet\logs\_hf_stage\RetNet-0.5B'
proc = psutil.Process(os.getpid())


def rss_mb():
    return proc.memory_info().rss / 1024 / 1024


print('起始 RSS: %.0f MB' % rss_mb())
t0 = time.time()
tok = AutoTokenizer.from_pretrained(D, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(D, trust_remote_code=True, dtype=torch.float32)
m.eval()
load_s = time.time() - t0
print('加载完（fp32 / CPU）: %.1f s，RSS 现在 %.0f MB' % (load_s, rss_mb()))

ids = tok('<bos>用户：你好，介绍一下你自己。\n助手：')['input_ids']
with torch.inference_mode():
    out = m.generate(torch.tensor([ids]), max_new_tokens=32, temperature=0.6)
n = len(out[0]) - len(ids)
peak = rss_mb()
print('线程数 torch.get_num_threads() = %d' % torch.get_num_threads())
print('生成 %d token（含首 token 预填）' % n)
print('峰值 RSS: %.0f MB' % peak)
print('TEXT:', repr(tok.decode(out[0].tolist()[len(ids):])))
