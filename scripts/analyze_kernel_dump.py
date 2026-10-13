"""解析 Windows 内核「小内存转储」(PAGEDU64 / TRIAGE_DUMP64)，把出错地址定位到驱动。

为什么不用 WinDbg：图吧工具箱那份 windbg.exe 是孤立的 642 KB exe（旁边没有
dbgeng.dll），新版商店 WinDbg 的 MCP 又要求先有一个活着的 WinDbg 会话。
而 0x3b 的 **参数 2 就是出错指令地址**，转储里又带着完整的**驱动表**
（TRIAGE_DUMP64.DriverListOffset + StringPoolOffset），所以自己解析就够了：
不需要调试器、不需要符号服务器、不需要联网。

    python scripts/analyze_kernel_dump.py <dump> [--addr 0x...]

只读文件。
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

TRIAGE_OFF = 0x2000                     # DUMP_HEADER64 之后

BUGCHECK = {
    0x3B: "SYSTEM_SERVICE_EXCEPTION",
    0x141: "VIDEO_ENGINE_TIMEOUT_DETECTED",
    0x142: "VIDEO_TDR_APPLICATION_BLOCKED",
    0x143: "VIDEO_DRIVER_INIT_FAILURE",
    0x1E: "KMODE_EXCEPTION_NOT_HANDLED",
    0x50: "PAGE_FAULT_IN_NONPAGED_AREA",
    0x7E: "SYSTEM_THREAD_EXCEPTION_NOT_HANDLED",
    0xA: "IRQL_NOT_LESS_OR_EQUAL",
    0x133: "DPC_WATCHDOG_VIOLATION",
    0x9F: "DRIVER_POWER_STATE_FAILURE",
}

EXC = {0xC0000005: "ACCESS_VIOLATION", 0xC0000096: "PRIVILEGED_INSTRUCTION",
       0xC000001D: "ILLEGAL_INSTRUCTION", 0x80000003: "BREAKPOINT"}


def parse_header(b: bytes) -> dict:
    mach, nproc = struct.unpack_from("<II", b, 0x30)
    bc = struct.unpack_from("<I", b, 0x38)[0]
    p = struct.unpack_from("<QQQQ", b, 0x40)
    return {
        "major": struct.unpack_from("<I", b, 8)[0],
        "minor": struct.unpack_from("<I", b, 12)[0],
        "ps_loaded_module_list": struct.unpack_from("<Q", b, 0x20)[0],
        "machine": mach, "nproc": nproc, "bugcheck": bc, "params": p,
    }


def parse_triage(b: bytes) -> dict:
    """TRIAGE_DUMP64 前几个字段。"""
    o = TRIAGE_OFF
    f = struct.unpack_from("<19I", b, o)          # 先取 19 个 u32
    keys = ["ServicePackBuild", "SizeOfDump", "ValidOffset", "ContextOffset",
            "ExceptionOffset", "MmOffset", "UnloadedDriversOffset", "PrcbOffset",
            "ProcessOffset", "ThreadOffset", "CallStackOffset", "SizeOfCallStack",
            "DriverListOffset", "DriverCount", "StringPoolOffset", "StringPoolSize",
            "BrokenDriverOffset", "TriageOptions", "reserved19"]
    return {k: v for k, v in zip(keys, f)}


def wide(b: bytes, off: int, maxlen: int = 260) -> str:
    end = off
    while end + 1 < len(b) and end - off < maxlen * 2 and b[end:end + 2] != b"\x00\x00":
        end += 2
    return b[off:end].decode("utf-16-le", "replace")


def drivers(b: bytes, tri: dict):
    """解析 triage 驱动表。

    字段布局**不是靠记忆猜的，是靠自洽性实测定的**：
    每个条目 144 字节（(StringPoolOffset-DriverListOffset)/DriverCount 恰好整除），
    偏移 0 = 文件名偏移（指向「u32 字符数 + UTF-16LE」的池内条目），
    偏移 0x38(字段7) = DllBase，偏移 0x48(字段9) = SizeOfImage。
    判据：用 (字段7, 字段9) 解释时 231 个模块**两两重叠 0 次**；
    换成 (字段3, 字段9) 则有 71 次重叠 —— 所以前者才是基址/大小。
    """
    dl, dc = tri["DriverListOffset"], tri["DriverCount"]
    if not (0 < dl < len(b)) or dc == 0 or dc > 5000:
        return []
    stride = (tri["StringPoolOffset"] - dl) // dc
    if stride <= 0:
        return []
    out = []
    for i in range(dc):
        o = dl + i * stride
        if o + stride > len(b):
            break
        name_off = struct.unpack_from("<Q", b, o)[0]
        base = struct.unpack_from("<Q", b, o + 0x38)[0]
        size = struct.unpack_from("<Q", b, o + 0x48)[0]
        try:
            n = struct.unpack_from("<I", b, name_off)[0]
            name = (b[name_off + 4: name_off + 4 + n * 2].decode("utf-16-le", "replace")
                    if 0 < n < 300 and name_off + 4 + n * 2 <= len(b) else "")
        except Exception:                                          # noqa: BLE001
            name = ""
        out.append((base, size, name))
    return out


def self_check(mods) -> tuple[int, int]:
    """正/负向对照：正确布局下模块应当两两不重叠。返回 (重叠数, 参与项数)。"""
    v = sorted([(x, y) for x, y, _ in mods if x and y])
    n = sum(1 for i in range(len(v) - 1) if v[i][0] + v[i][1] > v[i + 1][0])
    return n, len(v)


def resolve(addr: int, mods):
    hit = None
    for base, size, name in mods:
        if size and base <= addr < base + size:
            if hit is None or base > hit[0]:
                hit = (base, size, name)
    return None if hit is None else (hit[2], addr - hit[0], hit[0], hit[1])


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                          # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--addr", default=None, help="要定位的地址；默认用 bugcheck 参数 2")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    b = args.dump.read_bytes()
    if b[:4] != b"PAGE" or b[4:8] != b"DU64":
        print(f"不是 PAGEDU64 内核转储（前 8 字节 {b[:8]!r}）")
        if b[:4] == b"MDMP":
            print("这是用户态 minidump，请用 scripts/analyze_minidump.py")
        return 2

    h = parse_header(b)
    bc = h["bugcheck"]
    print(f"文件        : {args.dump}  ({len(b):,} 字节)")
    print(f"格式        : PAGEDU64 (内核转储)  版本 {h['major']}.{h['minor']}  "
          f"build {h['minor']}  {'AMD64' if h['machine'] == 0x8664 else hex(h['machine'])}")
    print(f"处理器数    : {h['nproc']}")
    print(f"BugCheck    : 0x{bc:08x}  {BUGCHECK.get(bc, '?')}")
    for i, v in enumerate(h["params"], 1):
        extra = ""
        if i == 1:
            extra = f"   ({EXC.get(v, '')})"
        print(f"  Param{i}    : 0x{v:016x}{extra}")
    p2 = h["params"][1]
    print(f"\n=> 出错指令地址 = Param2 = 0x{p2:016x}（0x3b 的定义：参数 2 是引发异常的指令地址）")

    tri = parse_triage(b)
    print(f"\nTRIAGE_DUMP64: SizeOfDump={tri['SizeOfDump']:,}（文件 {len(b):,}）  "
          f"DriverCount={tri['DriverCount']}  CallStackOffset={tri['CallStackOffset']:,}  "
          f"SizeOfCallStack={tri['SizeOfCallStack']:,}")
    print(f"              DriverListOffset={tri['DriverListOffset']:,}  "
          f"StringPoolOffset={tri['StringPoolOffset']:,}  StringPoolSize={tri['StringPoolSize']:,}")

    mods = drivers(b, tri)
    named = [m for m in mods if m[2]]
    print(f"\n驱动表      : 解析到 {len(mods)} 项，其中带名字 {len(named)} 项")
    ov, tot = self_check(mods)
    print(f"自洽性对照  : (DllBase=0x38, SizeOfImage=0x48) 下模块两两重叠 {ov}/{tot} —— "
          f"{'✅ 布局正确' if ov == 0 else '⚠️ 布局可能不对'}")
    if mods:
        stride = (tri["StringPoolOffset"] - tri["DriverListOffset"]) // max(tri["DriverCount"], 1)
        print(f"              结构体步长 = {stride} 字节"
              f"（{tri['DriverCount']} × {stride} = {tri['DriverCount'] * stride:,} = "
              f"StringPoolOffset − DriverListOffset，恰好整除）")

    target = int(args.addr, 16) if args.addr else p2
    r = resolve(target, mods)
    print(f"\n把 0x{target:016x} 映射到模块：")
    if r:
        name, off, base, size = r
        print(f"    ✅ {name}")
        print(f"       基址 0x{base:016x}  大小 0x{size:x}  偏移 +0x{off:x}")
    else:
        print("    ⚠️ 未命中任何驱动（驱动表可能解析不完整，见下方原始项）")

    # 命名驱动按基址排序，便于人工核对
    print(f"\n前 {args.top} 个驱动（按基址）：")
    for base, size, name in sorted(named)[:args.top]:
        print(f"    0x{base:016x}  {size:>10,}  {name}")
    # 专门看显示驱动
    print("\n与显示/GPU 相关的驱动：")
    for base, size, name in named:
        if any(k in name.lower() for k in ("nvlddmkm", "dxgkrnl", "dxgmms", "nvlddm")):
            inr = "  ← 出错地址落在这里" if resolve(target, [(base, size, name)]) else ""
            print(f"    0x{base:016x}  {size:>10,}  {name}{inr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
