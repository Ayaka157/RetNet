"""验证 play_v2 的 prompt 构造与训练数据逐 token 一致。

``train.ids.npy`` 是 **packed** 的：一行里塞了若干条互不相干的对话，每条以
``<bos>`` 开头，最后一条以 ``<eos>`` 收尾，剩下的用 ``<pad>`` 填满。
所以切分单位不是「行」，而是「行内的对话」。

单条对话的布局::

    <bos> [设定：…\n] 用户：U1\n助手： <think>R1</think>A1 [\n用户：U2\n助手： R2] <eos>

``train.wcode.npy`` 用 2 标 prompt、1 标助手回复、0 标 padding。按 wcode 的
**连续段**切分，prompt 段 ``p_j`` 的最后一位就是第 j 轮 prompt 的最后一个
token，于是待证命题是::

    encode_prompt_ids(tok, history[:j], U_j, system) == ids[dialogue_start : p_j.end + 1]

历史里的回复直接取 ``r_i`` 段的**原始 id**，不做文本往返 —— 这正是旧
``play.py`` 做不到的地方：``<`` ``>`` ``/`` 在词表里是独立特殊 token，
``decode()`` 会在它们两侧插空格，``encode(decode(ids)) != ids``。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from play_v2 import encode_prompt_ids  # noqa: E402
from retnet.chat import ASSISTANT, USER  # noqa: E402
from retnet.tokenizer import BPETokenizer  # noqa: E402

PACK = ROOT / "data_cache/chat_pack_v13"
TOK = None


def dialogues_of(ids_row, code_row) -> list[tuple[int, int]]:
    """按 ``<bos>`` 把一行切成若干条对话，返回 ``[(start, end_exclusive), ...]``。"""
    live = [i for i, c in enumerate(code_row) if c != 0]
    if not live:
        return []
    end = live[-1] + 1
    starts = [i for i in range(end)
              if ids_row[i] == TOK.bos_id and code_row[i] == 2]
    if not starts or starts[0] != 0:
        return []
    bounds = starts + [end]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def runs_in(code_row, s: int, e: int) -> list[tuple[int, int, int]]:
    """段内按 wcode 连续段切分，返回 ``[(wcode, start, end), ...]``。"""
    out, start = [], s
    for i in range(s + 1, e + 1):
        if i == e or code_row[i] != code_row[start]:
            out.append((int(code_row[start]), start, i - 1))
            start = i
    return out


def _head(ids_row, s: int, end: int) -> str:
    return TOK.decode(ids_row[s: end + 1].tolist(), skip_special_tokens=False)


def user_of(ids_row, s: int, end: int) -> str | None:
    """抽出某一轮的用户话；prompt 被截断（缺 ``助手：``）时返回 None。"""
    head = _head(ids_row, s, end)
    a = head.rfind(USER + "：")
    b = head.rfind(ASSISTANT + "：")
    if a == -1 or b < a:
        return None
    return head[a + len(USER + "："): b].strip("\n")


def system_of(ids_row, s: int, end: int) -> str | None:
    """取出角色设定块；「设定：」不在最前时返回 None。"""
    head = _head(ids_row, s, end)
    a = head.find("设定：")
    if a == -1:
        return None
    b = head.find(USER + "：")
    if b < a:
        return None
    return head[a + len("设定："): b].strip("\n ") or None


def verify(ids_row, code_row):
    """返回 ``(checks, skipped)``；checks 为 ``[(对话序号, 轮次, got, want), ...]``。

    被 packed 的 512 截断切掉尾巴的样本（prompt 段里没有 ``助手：``）无法构造出
    一轮完整问答，计入 skipped 而不算失败。
    """
    checks, skipped = [], 0
    for d, (s, e) in enumerate(dialogues_of(ids_row, code_row)):
        runs = runs_in(code_row, s, e)
        prompts = [(a, b) for c, a, b in runs if c == 2]
        replies = [(a, b) for c, a, b in runs if c == 1]
        if not prompts or not replies:
            continue
        system = system_of(ids_row, s, prompts[0][1])
        for j, (ps, pe) in enumerate(prompts):
            if j >= len(replies):
                break
            user = user_of(ids_row, s, pe)
            if user is None:
                skipped += 1
                continue
            history = []
            ok = True
            for i in range(j):
                prev = user_of(ids_row, s, prompts[i][1])
                if prev is None:
                    ok = False
                    break
                history.append((prev,
                                ids_row[replies[i][0]: replies[i][1] + 1].tolist()))
            if not ok:
                skipped += 1
                continue
            got = encode_prompt_ids(TOK, history, user, system=system)
            checks.append((d, j, got, ids_row[s: pe + 1].tolist()))
    return checks, skipped


def main():
    global TOK
    TOK = BPETokenizer.load(PACK / "bpe_tokenizer.json")
    ids = np.load(PACK / "train.ids.npy", mmap_mode="r")
    code = np.load(PACK / "train.wcode.npy", mmap_mode="r")
    print(f"pack {ids.shape} · tokenizer vocab={TOK.vocab_size}")

    n_rows = n_dlg = n_checks = 0
    by_turn: dict[int, int] = {}
    with_system = 0
    multi_dlg = 0
    bad = []

    # v13 包里绝大多数是单轮（一问一答一条 <bos>），多轮是少数派。均匀抽样会
    # 只捞到个位数的多轮样本，而多轮恰好是对话脚本最依赖的路径，所以这里用
    # 小步长扫，并且把「凑够 multi_target 条多轮」也当作停止条件。
    MULTI_TARGET, MAX_ROWS = 200, 2600
    n_skipped = 0
    for i in range(0, len(ids), 37):
        res, skipped = verify(ids[i], code[i])
        n_skipped += skipped
        dlg_count = len(dialogues_of(ids[i], code[i]))
        if not dlg_count:
            continue
        n_rows += 1
        n_dlg += dlg_count
        for d, j, got, want in res:
            n_checks += 1
            by_turn[j] = by_turn.get(j, 0) + 1
            if got != want:
                first = next((k for k in range(min(len(got), len(want)))
                              if got[k] != want[k]), min(len(got), len(want)))
                bad.append((i, d, j, len(got), len(want), first))
        # 每条对话的首轮 + 是否有第二轮
        for s, e in dialogues_of(ids[i], code[i]):
            runs = runs_in(code[i], s, e)
            p = [(a, b) for c, a, b in runs if c == 2]
            if not p:
                continue
            if system_of(ids[i], s, p[0][1]):
                with_system += 1
            if len(p) > 1:
                multi_dlg += 1
        if n_rows >= MAX_ROWS or (multi_dlg >= MULTI_TARGET and n_rows >= 600):
            break

    print(f"抽查 {n_rows} 行 / {n_dlg} 条对话 / {n_checks} 个 prompt 段："
          f"逐 token 一致 {n_checks - len(bad)}，不一致 {len(bad)}")
    print("  轮次分布：" + ", ".join(f"第{j + 1}轮×{c}" for j, c in sorted(by_turn.items())))
    print(f"  其中多轮对话 {multi_dlg} 条（{multi_dlg / max(n_dlg, 1):.1%}），"
          f"带「设定：」角色块 {with_system} 条")
    print(f"  跳过被 512 截断、凑不出一轮完整问答的 prompt 段 {n_skipped} 个")
    for row, d, j, lg, lw, k in bad[:10]:
        print(f"  ✗ row={row} 对话{d} 第{j + 1}轮 got={lg} want={lw} 首个分歧@{k}")
    print("PASS" if not bad else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
