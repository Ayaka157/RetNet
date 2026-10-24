import json, io, sys
sys.stdout.reconfigure(encoding="utf-8")

def cats(path, model=None):
    j = json.load(open(path, encoding="utf-8"))
    ms = j["models"]
    k = model or list(ms)[0]
    return k, ms[k]["cats"], ms[k]["total"]

for path, model in [(r"logs\bench_v3\report.T0.7_s1.json", None),
                    (r"logs\bench_v3\report.c10k.json", None)]:
    k, c, t = cats(path, model)
    print("==", path, "model:", k, "total:", round(t,4))
    for name, v in c.items():
        print(f"   {name:14} n={v['n_items']:4} score={v['score']:.4f} leak={v.get('leak')}")
