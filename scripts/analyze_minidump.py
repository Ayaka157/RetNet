"""不用 WinDbg 解析 minidump：把出错指令地址映射到「哪个模块 + 偏移」。

动机
----
桌面在 2026-10-11 07:49 报 BugCheck **0x3b**（SYSTEM_SERVICE_EXCEPTION）：

    0x0000003b (0x00000000c0000005, 0xfffff8067b13a4a0, 0xffff8b80c4266350, 0x0)

对 0x3b，**参数 2 就是引发异常的那条指令的地址**（参数 1 是异常码 0xc0000005 访问违例，
参数 3 是 context record 地址）。所以只要能从 minidump 里读出**已加载模块表**
（MINIDUMP_MODULE_LIST），就能判断 `0xfffff8067b13a4a0` 落在哪个驱动里。

这条路完全不需要 WinDbg / dbgeng.dll / 符号服务器 —— 那些反而都不可用
（图吧工具箱里的 windbg.exe 是孤立的 642 KB exe，旁边没有 dbgeng.dll）。

    python scripts/analyze_minidump.py <dump> [--addr 0xfffff8067b13a4a0]

只读文件，不做任何写操作。
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# ---- MINIDUMP 常量 -------------------------------------------------------
MODULE_LIST_STREAM = 4
EXCEPTION_STREAM = 6
SYSTEM_INFO_STREAM = 7
MODULE_SIZEOF = 108          # MINIDUMP_MODULE 的字节数（x64）


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def _string(b, rva):
    """MINIDUMP_STRING：Length(u32, 字节数) + UTF-16LE 内容。"""
    n = _u32(b, rva)
    return b[rva + 4: rva + 4 + n].decode("utf-16-le", "replace")


def parse(path: Path):
    b = path.read_bytes()
    if b[:4] != b"MDMP":
        raise SystemExit(f"不是 minidump（前 4 字节 {b[:4]!r}，应为 b'MDMP'）")
    nstreams = _u32(b, 8)
    dir_rva = _u32(b, 12)
    streams = {}
    for i in range(nstreams):
        o = dir_rva + i * 12
        stype, size, rva = _u32(b, o), _u32(b, o + 4), _u32(b, o + 8)
        streams.setdefault(stype, []).append((size, rva))
    return b, streams


def modules(b, streams):
    out = []
    for _size, rva in streams.get(MODULE_LIST_STREAM, []):
        n = _u32(b, rva)
        for i in range(n):
            o = rva + 4 + i * MODULE_SIZEOF
            base = _u64(b, o)
            img_size = _u32(b, o + 8)
            name_rva = _u32(b, o + 20)
            try:
                name = _string(b, name_rva)
            except Exception:                                    # noqa: BLE001
                name = "?"
            out.append((base, img_size, name))
    return out


def exception(b, streams):
    for _size, rva in streams.get(EXCEPTION_STREAM, []):
        er = rva + 8                                        # 跳过 ThreadId+Alignment
        code = _u32(b, er)
        addr = _u64(b, er + 16)
        nparam = _u32(b, er + 24)
        params = [_u64(b, er + 32 + k * 8) for k in range(min(nparam, 15))]
        return code, addr, params
    return None


def sysinfo(b, streams):
    for _size, rva in streams.get(SYSTEM_INFO_STREAM, []):
        arch = struct.unpack_from("<H", b, rva)[0]
        return arch
    return None


ARCH = {0: "x86", 9: "x64", 5: "ARM", 12: "ARM64"}


def resolve(addr, mods):
    """带回绕的地址归属判定：取 base 最大且 addr 落在 [base, base+size) 的模块。"""
    hit = None
    for base, size, name in mods:
        if base <= addr < base + max(size, 0x1000):
            if hit is None or base > hit[0]:
                hit = (base, size, name)
    if hit is None:
        return None
    base, size, name = hit
    return name, addr - base


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--addr", default="0xfffff8067b13a4a0",
                    help="要定位的地址（默认：0x3b 的出错指令地址）")
    ap.add_argument("--top", type=int, default=12, help="按地址排序后显示的模块数")
    args = ap.parse_args()

    if not args.dump.exists():
        print(f"❌ 找不到 {args.dump}")
        return 2
    try:
        b, streams = parse(args.dump)
    except PermissionError:
        print(f"❌ 读取被拒绝：{args.dump}\n   需要管理员权限（或先把它复制到工作区）")
        return 3

    print(f"文件      : {args.dump}  ({args.dump.stat().st_size:,} 字节)")
    print(f"流        : " + ", ".join(
        f"{t}({len(v)})" for t, v in sorted(streams.items())))
    arch = sysinfo(b, streams)
    print(f"架构      : {ARCH.get(arch, arch)}")

    exc = exception(b, streams)
    if exc:
        code, addr, params = exc
        print(f"\n异常记录  : ExceptionCode=0x{code:08x}  ExceptionAddress=0x{addr:016x}")
        for k, p in enumerate(params):
            print(f"    Param{k + 1} = 0x{p:016x}")
        if code == 0xC0000005:
            print("    （0xc0000005 = ACCESS_VIOLATION，与 bugcheck 0x3b 的参数 1 一致）")

    mods = modules(b, streams)
    print(f"\n模块数    : {len(mods)}")
    if not mods:
        print("⚠️ 没有模块表 —— 可能 dump 不完整（volmgr 在 07:49:09 报过「创建转储期间出错」）")
        return 1

    target = int(args.addr, 16)
    print(f"\n把地址 0x{target:016x} 映射到模块：")
    r = resolve(target, mods)
    if r is None:
        print("    ⚠️ 没有命中的模块 —— dump 的模块表可能不含内核驱动，或地址属于未列出的映像")
    else:
        name, off = r
        print(f"    ✅ {name}  + 0x{off:x}")
        print(f"       （基址 0x{dict((m[2], m[0]) for m in mods)[name]:016x}）")

    print(f"\n已加载模块（按基址升序，前 {args.top} 个）：")
    for base, size, name in sorted(mods)[:args.top]:
        print(f"    0x{base:016x}  {size:>10,}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
