"""在实例上下载子代理核实过的 HF 语料（加固版）。

上一版死因：连通性走 AutoDL 学术代理（/etc/network_turbo），代理不稳定时
huggingface_hub 的默认 10 秒读超时会反复 ReadTimeout，重试 5 次后进程静默死掉，
日志只停在 "Fetching 28 files: 7%"。

加固点：
1. 必须在 source /etc/network_turbo 之后运行（本脚本会检查代理变量）
2. 放宽 HF_HUB_DOWNLOAD_TIMEOUT / ETAG_TIMEOUT，减少无谓超时
3. 每个数据集单独重试（最多 6 轮，指数退避），单轮失败不清空已下的分片
4. 单文件级重试交给 snapshot_download 自己，max_workers 降到 2 减少并发超时
"""
import json
import os
import time
from pathlib import Path

# 宽松超时：代理稳的时候没用，抖的时候能救命
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

BASE = Path("/root/autodl-tmp/RetNet/data/hf")
REPOS = [
    ("inclusionAI/Ling-Coder-SFT", "ling_coder", None, "代码 SFT，6.26 GB / 2.15B token"),
    ("opencsg/Fineweb-Edu-Chinese-V3", "fineweb_edu_zh_v3", None, "中文教材/学科，0.284 GB"),
    ("TiWu-Lab/math-zh", "math_zh", "FIRST_N_PARQUET", "中文数学，全量 16.36 GB，只取前 4 片"),
    # skypile 的替代：SkyPile 是 Skywork 社区协议（非开源），换成 apache-2.0 的高质量中文网页。
    # 只取前 13 片 ≈ 1.7 GB ≈ 约 5 亿 token，规模与原 skypile 相当。
    ("Morton-Li/ChineseWebText2.0-HighQuality", "chinese_web_hq",
     "FIRST_N_PARQUET:13", "中文网页（替代 skypile），apache-2.0"),
]
N_PARQUET = 4
MAX_ROUNDS = 6


def size_of(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def have_proxy() -> bool:
    return bool(os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY"))


def main() -> int:
    print(f"[env] https_proxy={os.environ.get('https_proxy') or '(未设置)'}", flush=True)
    print(f"[env] HF_HUB_DOWNLOAD_TIMEOUT={os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}", flush=True)
    if not have_proxy():
        print("!! 没有代理变量。必须先 `source /etc/network_turbo` 再运行本脚本，"
              "否则 huggingface.co 直连会 DNS/连接超时。", flush=True)
        return 2

    api = HfApi()
    BASE.mkdir(parents=True, exist_ok=True)
    grand_total = 0
    failed = []
    for repo, local, patterns, note in REPOS:
        dst = BASE / local
        print(f"\n===== {repo} -> {dst} =====", flush=True)
        print(f"  说明: {note}", flush=True)
        allow = None
        if isinstance(patterns, str) and patterns.startswith("FIRST_N_PARQUET"):
            n_take = N_PARQUET
            if ":" in patterns:
                n_take = int(patterns.split(":", 1)[1])
            try:
                info = api.dataset_info(repo, files_metadata=True)
                parqs = sorted(
                    (s for s in (info.siblings or []) if s.rfilename.endswith(".parquet")),
                    key=lambda s: s.rfilename,
                )
                allow = [p.rfilename for p in parqs[:n_take]]
                print(f"  仓库共 {len(parqs)} 个 parquet，取前 {n_take} 个", flush=True)
                for p in parqs[:n_take]:
                    print(f"    {p.rfilename}  {(p.size or 0) / 1e9:.2f} GB", flush=True)
            except Exception as exc:                            # noqa: BLE001
                print(f"  ⚠️ 取文件清单失败（会退回全量下载）: {type(exc).__name__}: {exc}",
                      flush=True)

        ok = False
        for rnd in range(1, MAX_ROUNDS + 1):
            try:
                before = size_of(dst) if dst.exists() else 0
                print(f"  第 {rnd}/{MAX_ROUNDS} 轮开始（已有 {before / 1e9:.2f} GB）", flush=True)
                snapshot_download(
                    repo_id=repo, repo_type="dataset", local_dir=str(dst),
                    allow_patterns=allow, max_workers=2,
                )
                ok = True
                break
            except Exception as exc:                            # noqa: BLE001
                after = size_of(dst) if dst.exists() else 0
                print(f"  第 {rnd} 轮失败（本轮增加 {(after - before) / 1e9:.2f} GB）: "
                      f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
                wait = min(60, 10 * rnd)
                print(f"  {wait}s 后重试", flush=True)
                time.sleep(wait)
        if not ok:
            failed.append(repo)
            print(f"  ❌ {MAX_ROUNDS} 轮都没成功", flush=True)
            continue
        n = size_of(dst)
        grand_total += n
        df = os.statvfs("/root/autodl-tmp")
        print(f"  ✅ 落地 {n / 1e9:.2f} GB，数据盘剩余 "
              f"{df.f_bavail * df.f_frsize / 1e9:.1f} GB", flush=True)

    print(f"\n===== 合计 {grand_total / 1e9:.2f} GB =====", flush=True)
    if failed:
        print(f"!! 失败的数据集: {failed}", flush=True)
    (BASE / "manifest.json").write_text(
        json.dumps({"repos": [{"repo": r, "local": l, "note": n} for r, l, _, n in REPOS],
                    "total_bytes": grand_total, "failed": failed},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
