import re
d = open(r"C:\Users\Ayaka\RetNet\logs\v11_bf16_4k.log", encoding="utf-8", errors="replace").read()
# 找所有步数与耗时组合（进度条行可能有 \r，用 span）
# 抓 [MM:SS<HHH:MM:SS] 全部
pat = re.compile(r"\[(\d+):(\d+)<(\d+):(\d+):(\d+),")
found = pat.findall(d)
print("total progress rows:", len(found))
# 找到步数最近的条目
for i, (m, s, hh, mm, ss) in enumerate(found):
    pass
# 直接看文件里 "2248/30000" 附近的上下文
idx = d.rfind("2248/30000")
print("context around 2248:", repr(d[idx-60:idx+40]))
# 是否有多个 it/s 阶段不同（如开始时加载数据慢）
first = d.find("0/30000")
print("first 20 lines:", repr(d[:400]))
