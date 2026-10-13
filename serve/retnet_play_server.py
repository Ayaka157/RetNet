"""RetNet 0.5B 的本地聊天服务：FastAPI + SSE 流式输出。

    # 启动（用 AI 环境，见 serve/DEPLOY.md）
    C:\\Tools\\Miniconda3\\envs\\AI\\python.exe serve\\retnet_play_server.py \\
        --model_dir checkpoints/retnet_500m_v15b_ep3 --port 8077 --dtype bf16 --root_path /play

访问：``http://127.0.0.1:8077/play/``（公网经 cloudflared 隧道映射到 ``your-demo-host.example/play/``）

## 架构：为什么是「单 worker + FIFO 任务队列」而不是信号量

第一版用 ``Semaphore(1)`` + 在 SSE 生成器里 ``acquire/release``。实测踩到：
**客户端一断开（刷新页面、关标签页），生成器被取消，``finally`` 里的 release 不一定执行
→ 锁永久泄漏**，此后所有请求都卡在队列里（``/api/health`` 会显示 ``busy:true, queue:1``
却没有任何人在生成）。这是「资源所有权放在会被取消的协程里」的经典错误。

现在改成：**一个后台 worker 线程**按 FIFO 消费任务，每个任务自带一个事件队列；
``acquire/release`` 的所有权在 worker 线程里（它一定会跑完，因为生成被
``max_new_tokens`` 硬上限约束），HTTP 侧只负责读事件。
附带好处：排队位置天然公平（``qsize``）、队列有上限（满了直接 503）、
客户端断开后可以把**还没开始**的任务标记取消，不白烧 GPU。

## 多用户与并发（按"流量不大"设计）

* 并发度 = worker 数（默认 1）。3070 上并发生成只会让所有人一起变慢，所以默认 1。
* 每个浏览器的多轮上下文由**前端自己回传**（服务端无会话状态），互不干扰。
* 每个任务自带 ``context`` / ``system`` / 采样参数，全部在服务端夹紧，不信任前端。
* 每 IP 最小间隔 + 全局排队上限 + 单次 token 上限，三道闸。
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from play_v2 import encode_prompt_ids  # noqa: E402  （与终端脚本共用同一套提示词编码）

ARGS = None
S = {
    "model": None,
    "tok": None,
    "served": 0,
    "rejected": 0,
    "cancelled": 0,
    "last_tps": 0.0,
    "started": time.time(),
    "running_since": 0.0,
    "worker_busy": False,
}
JOBS: "queue.Queue[Job]" = queue.Queue()
_last_hit: dict[str, float] = {}
_hit_lock = threading.Lock()

STOPS = ("\n用户：", "\n助手：", "\n用户:", "\n助手:")
MAX_NEW_HARD = 512
MAX_PROMPT_CHARS = 2000
MAX_SYSTEM_CHARS = 1200
MIN_INTERVAL_S = 1.0        # 默认值；实际用 --min_interval


def find_stop(text: str) -> int:
    best = -1
    for m in STOPS:
        i = text.find(m)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class Job:
    prompt: str
    system: str | None
    history: list
    gen: dict
    context: int
    q: "queue.Queue[tuple[str, object]]" = field(default_factory=queue.Queue)
    enqueued: float = field(default_factory=time.time)
    cancelled: bool = False


# --------------------------------------------------------------------------- 生成

def _run_job(job: Job) -> None:
    """在 worker 线程里跑一次生成，把事件推进 ``job.q``。

    事件：``delta``（新增可见文本）/ ``done`` / ``error``。结束一定发 ``__eof__``。
    """
    model, tok = S["model"], S["tok"]
    S["worker_busy"] = True
    S["running_since"] = time.time()
    try:
        ids_in = encode_prompt_ids(tok, job.history, job.prompt, system=job.system)
        budget = job.context - job.gen["max_new_tokens"]
        if budget < 64:
            budget = 64
        if len(ids_in) > budget:
            ids_in = ids_in[-budget:]

        produced: list[int] = []
        # holdback：停止标记（"\n用户：" 等）可能跨 token 才完整出现。若把可见文本立刻吐出去，
        # 后面才匹配到的标记就切不回来了。实测踩到过：模型写 "\n用户" 时先吐出"用户"，
        # 再收到"："才匹配上 "\n用户：" —— 那句"用户"已经送到前端了。
        # 所以尾巴先扣住 HOLDBACK 个字符不发，并在 done 事件里回传最终文本兜底。
        state = {"visible": "", "sent": "", "cut": None, "texts": []}
        HOLDBACK = 4

        def on_token(tid: int) -> bool:
            produced.append(tid)
            cur = tok.decode(list(produced), skip_special_tokens=False)
            state["texts"].append(cur)
            cut = find_stop(cur)
            visible = cur if cut == -1 else cur[:cut]
            # 字节级 BPE 会把一个汉字劈在相邻两个 token 上，前缀解码会得到半个字符
            # （U+FFFD）。先扣住不吐，等下一个 token 补齐。
            while cut == -1 and visible.endswith("\ufffd"):
                visible = visible[:-1]
            if cut != -1:
                safe = visible                      # 匹配到标记：可见文本已定型
            elif len(visible) > HOLDBACK:
                safe = visible[:-HOLDBACK]          # 尾巴留 4 字符再发
            else:
                safe = ""
            if len(safe) > len(state["sent"]):
                job.q.put(("delta", safe[len(state["sent"]):]))
                state["sent"] = safe
            state["visible"] = visible
            if cut != -1:
                state["cut"] = cut
                return True
            return False

        t0 = time.time()
        model.generate(
            torch.tensor([ids_in], device=ARGS.device, dtype=torch.long),
            max_new_tokens=job.gen["max_new_tokens"],
            temperature=job.gen["temperature"],
            top_k=job.gen["top_k"],
            top_p=job.gen["top_p"],
            repetition_penalty=job.gen["repetition_penalty"],
            no_repeat_ngram_size=job.gen["no_repeat_ngram_size"],
            penalize_prompt=False,
            eos_id=tok.eos_id,
            device=ARGS.device,
            on_token=on_token,
        )
        dt = max(time.time() - t0, 1e-6)
        S["last_tps"] = len(produced) / dt
        S["served"] += 1

        # 回传**回复的 token id**：多轮上下文必须回灌 id，不能拿文本重新 encode
        # （词表里 <、>、/ 是独立特殊 token，decode 会插空格 → encode(decode(ids)) != ids）。
        if state["cut"] is None:
            keep = list(produced)
        else:
            keep = []
            for i, s in enumerate(state["texts"]):
                if len(s) <= state["cut"]:
                    keep = produced[:i + 1]
                else:
                    break
        # 结束原因要分开记：命中回合边界 / 模型自己收尾（EOS）/ 撞到 max_new_tokens。
        # NOTE：EOS 不能只靠 `eos_id in produced` 判断 —— 实测那个 token 不会进 on_token 回调
        # （8 token 的"你好"回复被判成 max_tokens）。所以用"没命中回合标记、但**提前停了**"
        # 作为"模型自己收尾"的判据。
        if state["cut"] is not None:
            stop_reason = "turn"
        elif (tok.eos_id in produced) or (len(produced) < job.gen["max_new_tokens"]):
            stop_reason = "eos"
        else:
            stop_reason = "max_tokens"
        think_open = ("<think>" in state["visible"]) and ("</think>" not in state["visible"])
        job.q.put(("done", {
            "gen_tokens": len(produced),
            "seconds": round(dt, 2),
            "tok_per_s": round(len(produced) / dt, 1),
            "prompt_tokens": len(ids_in),
            "finished": state["cut"] is not None,
            "stop_reason": stop_reason,
            "think_open": think_open,
            "text": state["visible"],          # 最终干净文本（前端用它覆盖流式过程中已渲染的内容）
            "reply_ids": keep,
        }))
    except Exception as exc:                                    # noqa: BLE001
        job.q.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        S["worker_busy"] = False
        job.q.put(("__eof__", None))


def worker_loop(idx: int) -> None:
    while True:
        job = JOBS.get()
        try:
            if job.cancelled:
                S["cancelled"] += 1
            else:
                _run_job(job)
        finally:
            JOBS.task_done()


# --------------------------------------------------------------------------- HTTP

app = FastAPI(title="RetNet 0.5B play", docs_url=None, redoc_url=None)
# 路由都挂在**内层 app**（裸路径）上，由 main() 按 --root_path 挂载到外层。
# 为什么不在装饰器里直接用 --root_path：装饰器在 **import 时**执行，
# 而 argparse 要等 main() —— 第一版就是这么炸的：
#   AttributeError: 'NoneType' object has no attribute 'root_path'
api = FastAPI(docs_url=None, redoc_url=None)


@api.get("/api/health")
def health():
    return {
        "ok": True,
        "model": str(ARGS.model_dir),
        "device": ARGS.device,
        "dtype": ARGS.dtype,
        "params": ARGS._params,                             # noqa: SLF001
        "workers": ARGS.workers,
        "queue": JOBS.qsize(),
        "generating": S["worker_busy"],
        "generating_for_s": round(time.time() - S["running_since"], 1) if S["worker_busy"] else 0,
        "served": S["served"],
        "rejected": S["rejected"],
        "cancelled": S["cancelled"],
        "last_tok_per_s": round(S["last_tps"], 1),
        "uptime_s": round(time.time() - S["started"], 1),
        "context_max": ARGS.context,
    }


def _check_token(request: Request) -> None:
    if not ARGS.token:
        return
    got = request.query_params.get("t") or request.headers.get("x-auth-token") or ""
    if got != ARGS.token:
        S["rejected"] += 1
        raise PermissionError("口令不对")


def _parse_body(body: dict) -> tuple[Job | None, JSONResponse | None]:
    prompt = (body.get("prompt") or "").strip()[:MAX_PROMPT_CHARS]
    if not prompt:
        return None, JSONResponse({"error": "prompt 为空"}, status_code=400)

    history = []
    for h in (body.get("history") or [])[-8:]:
        ids = h.get("reply_ids") or []
        if isinstance(h.get("user"), str) and isinstance(ids, list):
            history.append((h["user"][:MAX_PROMPT_CHARS], [int(i) for i in ids][:4096]))

    ctx = int(_clamp(int(body.get("context") or ARGS.context), 256, ARGS.context))
    gen = {
        "max_new_tokens": int(_clamp(int(body.get("max_new_tokens") or 200), 16, MAX_NEW_HARD)),
        "temperature": float(_clamp(float(body.get("temperature") or 0.7), 0.05, 1.5)),
        "top_p": float(_clamp(float(body.get("top_p") or 0.9), 0.05, 1.0)),
        "top_k": int(_clamp(int(body.get("top_k") or 40), 1, 200)),
        "repetition_penalty": float(_clamp(float(body.get("repetition_penalty") or 1.1), 1.0, 1.5)),
        "no_repeat_ngram_size": int(_clamp(int(body.get("no_repeat_ngram_size") or 4), 0, 12)),
    }
    if gen["max_new_tokens"] > ctx - 64:
        gen["max_new_tokens"] = max(16, ctx - 64)
    system = (body.get("system") or "").strip()[:MAX_SYSTEM_CHARS] or None
    return Job(prompt=prompt, system=system, history=history, gen=gen, context=ctx), None


@api.post("/api/chat")
async def chat(request: Request):
    try:
        _check_token(request)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=401)

    # 坏请求体必须挡成 400，不能让它冒成 500 —— 公网端点会收到各种垃圾
    # （实测过一次：客户端 JSON 被 shell 拆碎 → JSONDecodeError → 500）
    try:
        body = await request.json()
    except Exception as exc:                                    # noqa: BLE001
        S["rejected"] += 1
        return JSONResponse({"error": f"请求体不是合法 JSON：{exc}"}, status_code=400)
    job, err = _parse_body(body)
    if err is not None:
        return err

    ip = request.client.host if request.client else "?"
    now = time.time()
    with _hit_lock:
        last = _last_hit.get(ip, 0.0)
        if now - last < ARGS.min_interval:
            S["rejected"] += 1
            return JSONResponse({"error": f"太频繁了，{ARGS.min_interval:.1f} 秒内只允许一次"},
                                status_code=429)
        _last_hit[ip] = now

    if JOBS.qsize() >= ARGS.max_waiting:
        S["rejected"] += 1
        return JSONResponse({"error": "排队的人太多了，稍后再试"}, status_code=503)
    # ⚠️ 位置必须在 put **之前**算，而且要算上"正在生成的那个"：
    # worker 取任务极快，put 之后 qsize 常常是 1，于是后到的人总以为自己排第一，
    # 排队提示永远不显示（实测踩到）。ahead = 队列里现有的 + 正在生成的那一个。
    ahead = JOBS.qsize() + (1 if S["worker_busy"] else 0)
    JOBS.put(job)
    position = ahead + 1          # 1 = 下一个就轮到你

    def event_stream():
        if position > 1:
            yield _sse({"type": "queue", "position": position,
                        "ahead": position - 1,
                        "message": f"前面还有 {position - 1} 位，排队中…"})
        try:
            while True:
                kind, payload = job.q.get()
                if kind == "__eof__":
                    break
                if kind == "delta":
                    yield _sse({"type": "delta", "text": payload})
                elif kind == "done":
                    yield _sse({"type": "done", **payload})
                elif kind == "error":
                    yield _sse({"type": "error", "message": payload})
        finally:
            # 客户端断开：任务已经跑完的无法回收，但**还没开始**的可以取消
            job.cancelled = True

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@api.get("/")
def index():
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 启动

def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser(description="RetNet 0.5B 聊天服务")
    ap.add_argument("--model_dir", default="checkpoints/retnet_500m_v15b_ep3")
    ap.add_argument("--tokenizer", default=str(ROOT / "data_cache/chat_pack_v13/bpe_tokenizer.json"))
    ap.add_argument("--host", default="127.0.0.1", help="默认只监听本机；公网走 cloudflared 隧道")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--root_path", default="/play", help="应用挂载路径，例如 /play")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--context", type=int, default=2048, help="最大上下文（模型窗口）")
    ap.add_argument("--workers", type=int, default=1, help="并发生成数；3070 上建议 1")
    ap.add_argument("--max_waiting", type=int, default=8, help="排队上限，超过直接 503")
    ap.add_argument("--min_interval", type=float, default=1.0,
                    help="同一 IP 两次请求的最小间隔秒数（防刷；压测时调小）")
    ap.add_argument("--token", default=None, help="设置后 /api/* 需要 ?t= 或 X-Auth-Token")
    ap.add_argument("--warmup", action="store_true", help="启动后跑一次 8 token 生成，报出 tok/s")
    ARGS = ap.parse_args()
    ARGS.root_path = ("/" + ARGS.root_path.strip("/")) if ARGS.root_path.strip("/") else ""

    # 挂载：--root_path /play ⇒ /play/api/chat、/play/ 都在内层 app 上；
    # 同时把根路径 302 到 /play/，否则朋友只输域名会看到 404。
    app.mount(ARGS.root_path or "/", api)
    if ARGS.root_path:
        app.add_api_route("/", lambda: RedirectResponse(url=ARGS.root_path + "/"), methods=["GET"])

    import os
    if ARGS.token is None:
        ARGS.token = os.environ.get("RETNET_PLAY_TOKEN") or None
    if ARGS.device == "auto":
        ARGS.device = "cuda" if torch.cuda.is_available() else "cpu"

    from retnet.tokenizer import BPETokenizer
    from retnet_hf.modeling_retnet import RetNetForCausalLM

    t0 = time.time()
    model = RetNetForCausalLM.from_pretrained(ARGS.model_dir)
    if ARGS.dtype == "bf16":
        model = model.to(torch.bfloat16)
    model = model.to(ARGS.device).eval()
    S["model"], S["tok"] = model, BPETokenizer.load(ARGS.tokenizer)
    ARGS._params = sum(p.numel() for p in model.parameters())   # noqa: SLF001

    for i in range(ARGS.workers):
        threading.Thread(target=worker_loop, args=(i,), daemon=True).start()

    print(f"[serve] 模型 {ARGS.model_dir}  {ARGS._params/1e6:.2f}M 参数"     # noqa: SLF001
          f"  {ARGS.device}/{ARGS.dtype}  加载 {time.time()-t0:.1f}s", flush=True)
    print(f"[serve] http://{ARGS.host}:{ARGS.port}{ARGS.root_path}/"
          f"  口令={'开' if ARGS.token else '关'}  worker={ARGS.workers}"
          f"  排队上限={ARGS.max_waiting}", flush=True)

    if ARGS.warmup:
        job = Job(prompt="你好", system=None, history=[],
                  gen={"max_new_tokens": 8, "temperature": 0.7, "top_p": 0.9, "top_k": 40,
                       "repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
                  context=ARGS.context)
        JOBS.put(job)
        while True:
            kind, payload = job.q.get()
            if kind == "__eof__":
                break
            if kind == "done":
                print(f"[serve] 预热完成 {payload['tok_per_s']} tok/s", flush=True)
            elif kind == "error":
                print(f"[serve] 预热失败 {payload}", flush=True)

    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
