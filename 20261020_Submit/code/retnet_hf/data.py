"""Packed chat dataset + collator for the HF Trainer (retnet_hf)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedChatDataset(Dataset):
    """Reads the packed int16 rows written by scripts/build_chat_mix_v4.py.

    Row layout (seq_len + 1 tokens): [x_0 ... x_{seq-1} x_seq]
    - input_ids = row[:-1], labels = row[1:]
    - wcodes    = per-target weight codes (0 pad / 1 reply / 2 prompt / 3 plain)
    """

    def __init__(self, ids_path, wcode_path, seq_len: int):
        self.ids = np.load(ids_path, mmap_mode="r")
        self.codes = np.load(wcode_path, mmap_mode="r")
        assert self.ids.ndim == 2 and self.ids.shape == self.codes.shape
        assert self.ids.shape[1] == seq_len + 1, \
            f"rows are {self.ids.shape[1]} wide, expected seq_len+1={seq_len + 1}"
        self.seq_len = seq_len

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        row = self.ids[idx]
        return {
            "input_ids": torch.from_numpy(row[:-1].astype(np.int64)),
            "labels": torch.from_numpy(row[1:].astype(np.int64)),
            "wcodes": torch.from_numpy(self.codes[idx][1:].astype(np.int64)),
        }


class ChatDataCollator:
    def __call__(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "wcodes": torch.stack([b["wcodes"] for b in batch]),
        }


# --------------------------------------------------------------- 多阶段数据集

class _PackedStage:
    """一段 packed 语料：二维 ``(rows, seq_len+1)`` 的 ids + wcode。"""

    def __init__(self, name, data_dir, seq_len, weight=None):
        self.name = name
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        ids = self.data_dir / "train.ids.npy"
        wcode = self.data_dir / "train.wcode.npy"
        if not ids.exists():
            raise FileNotFoundError(f"阶段「{name}」找不到 {ids}")
        self.ids = np.load(ids, mmap_mode="r")
        self.codes = None if weight is not None else np.load(wcode, mmap_mode="r")
        assert self.ids.ndim == 2 and self.ids.shape[1] == seq_len + 1, \
            f"阶段「{name}」行宽 {self.ids.shape[1]} != seq_len+1={seq_len + 1}"
        if self.codes is not None:
            assert self.ids.shape == self.codes.shape, f"阶段「{name}」ids/wcode 形状不一致"
        self.weight = weight

    def __len__(self):
        return self.ids.shape[0]

    def files(self) -> list[Path]:
        """本阶段占用的磁盘文件（供 GC 删除）。"""
        out = [self.data_dir / "train.ids.npy"]
        if self.codes is not None:
            out.append(self.data_dir / "train.wcode.npy")
        return [p for p in out if p.exists()]

    def row(self, idx):
        row = self.ids[idx]
        if self.codes is not None:
            wc = self.codes[idx][1:].astype(np.int64)
        else:
            wc = np.full(self.seq_len, int(self.weight), dtype=np.int64)
        return row[:-1], row[1:], wc


class _FlatStage:
    """一段 flat 语料：目录下若干一维 token 流（int16/int32），不打包、不复制。

    行数 = 每个分片 ``len // (seq_len+1)`` 之和（分片末尾余数丢弃，占比 <0.05%）。
    这样 ``data_cache/pretrain_v2`` 那种「已 tokenize 但没打包」的语料可以直接用，
    不必先花几小时生成一份 5 GB 的重复文件。
    """

    def __init__(self, name, data_dir, seq_len, weight=1, exclude=None):
        self.name = name
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.weight = int(weight)
        #: 要跳过的文件名前缀，例如 ["skypile_"] 表示不用该来源（license 原因等）
        self.exclude = tuple(str(x) for x in (exclude or ()))
        self.shards = []
        self._starts = []
        self._rows = 0
        shards, starts, total = self._scan()
        if total == 0:
            raise FileNotFoundError(
                f"阶段「{name}」在 {self.data_dir} 下没有可用的 *.npy（或文件都还没写完）")
        self.shards, self._starts, self._rows = shards, starts, total
        self._wc = np.full(seq_len, self.weight, dtype=np.int64)

    def _scan(self):
        """扫描目录，返回 (shards, starts, rows)。

        **跳过打不开的分片**：数据可能正在从外面传进来，文件被截断时
        ``np.load(mmap_mode=...)`` 会抛 ``ValueError: mmap length is greater than
        file size``。这里选择跳过而不是崩——下一轮 ``refresh()`` 会把它收进来。
        """
        width = self.seq_len + 1
        files = sorted(p for p in self.data_dir.glob("*.npy") if "val" not in p.name.lower())
        if self.exclude:
            files = [p for p in files if not p.name.startswith(self.exclude)]
        shards, starts, total = [], [], 0
        for p in files:
            try:
                arr = np.load(p, mmap_mode="r")
            except Exception:                                   # noqa: BLE001
                continue          # 正在写入 / 损坏，等下次 refresh
            if arr.ndim != 1:
                continue
            n = arr.shape[0] // width
            if n == 0:
                continue
            shards.append((p, arr, n))
            starts.append(total)
            total += n
        return shards, starts, total

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    def refresh(self) -> bool:
        """重新扫描目录；有新分片就接过，返回是否发生变化。"""
        shards, starts, total = self._scan()
        if len(shards) == len(self.shards) and total == self._rows:
            return False
        self.shards, self._starts, self._rows = shards, starts, total
        return True

    def files(self) -> list[Path]:
        """本阶段占用的磁盘文件（供 GC 删除）。"""
        return [p for p, _a, _n in self.shards]

    def close(self) -> None:
        """释放 mmap 句柄。

        ⚠️ 删除文件前必须调：Windows 上 mmap 会占用文件句柄，不先关就
        ``unlink`` 会报 WinError 32（Linux 上能删，但显式关闭更干净）。
        """
        for _p, arr, _n in self.shards:
            try:
                mm = getattr(arr, "_mmap", None)
                if mm is not None:
                    mm.close()
            except Exception:                                   # noqa: BLE001
                pass
        self.shards, self._starts, self._rows = [], [], 0

    def __len__(self):
        return self._rows

    def row(self, idx):
        width = self.seq_len + 1
        # 行号 -> 分片（二分查找）
        import bisect
        si = bisect.bisect_right(self._starts, idx) - 1
        local = idx - self._starts[si]
        _p, arr, _n = self.shards[si]
        off = local * width
        chunk = arr[off:off + width]
        return chunk[:-1], chunk[1:], self._wc


class StagePackedDataset(Dataset):
    """**可在训练中途切换语料阶段**的数据集。

    DataLoader 与优化器都不重建：``set_stage(i)`` 只改内部指针。
    阶段是惰性打开的 —— 某一段没轮到之前不会 mmap 它的文件。

    ``__getitem__`` 对行号取模，避免「切段后旧 sampler 的索引越界」：
    HF Trainer 的 RandomSampler 是按上一次的 ``__len__`` 采样的，切段那一刻
    手上可能还攥着旧范围的行号。
    """

    def __init__(self, stages: list[dict], seq_len: int):
        self._seq_len = int(seq_len)
        self._specs: list[dict] = []
        self._cache: dict[int, object] = {}
        self._active = 0
        self.update_stages(stages)

    # -- 阶段管理 ---------------------------------------------------
    def update_stages(self, stages: list[dict]) -> None:
        """替换阶段列表（计划热更新时调用）。已打开的阶段尽量复用。"""
        new_specs = []
        for s in stages:
            new_specs.append({
                "name": str(s["name"]),
                "data_dir": str(s["data_dir"]),
                "mode": str(s.get("mode", "auto")),
                "weight": s.get("weight"),
                "gc": bool(s.get("gc", False)),
                "min_shards": s.get("min_shards"),
                "exclude": list(s.get("exclude") or []),
            })
        # 按 (name, data_dir) 复用已打开的阶段，避免热更新时重复 mmap
        reuse = {}
        for i, sp in enumerate(self._specs):
            if i in self._cache:
                reuse[(sp["name"], sp["data_dir"])] = self._cache[i]
        self._specs = new_specs
        self._cache = {}
        for i, sp in enumerate(new_specs):
            key = (sp["name"], sp["data_dir"])
            if key in reuse:
                self._cache[i] = reuse[key]
        self._active = min(self._active, max(0, len(self._specs) - 1))

    def _stage(self, i: int):
        st = self._cache.get(i)
        if st is None:
            sp = self._specs[i]
            mode = sp["mode"]
            if mode == "auto":
                mode = "packed" if (Path(sp["data_dir"]) / "train.ids.npy").exists() else "flat"
            if mode == "packed":
                st = _PackedStage(sp["name"], sp["data_dir"], self._seq_len, sp.get("weight"))
            elif mode == "flat":
                st = _FlatStage(sp["name"], sp["data_dir"], self._seq_len,
                                exclude=sp.get("exclude"))
            else:
                raise ValueError(f"未知的 stage mode {mode!r}（阶段「{sp['name']}」）")
            self._cache[i] = st
            print(f"[stage] 打开阶段 {i}「{sp['name']}」mode={mode} rows={len(st)}",
                  flush=True)
        return st

    def set_stage(self, i: int) -> None:
        if not (0 <= i < len(self._specs)):
            raise IndexError(f"stage {i} 越界（共 {len(self._specs)} 段）")
        self._stage(i)                     # 提前打开，失败就在这里响亮报错
        self._active = int(i)

    # -- 就绪性 / 刷新 / 回收（训练中「边传边用」的基础）---------------
    def stage_ready(self, i: int) -> tuple[bool, str]:
        """该阶段现在能不能打开。不抛异常，供计划回调决定「等还是切」。

        这是「训练中上传下一轮数据」的关键：数据没传完时返回 (False, 原因)，
        回调继续用当前阶段，每 N 步重试一次，传完了自然就切过去。
        """
        if not (0 <= i < len(self._specs)):
            return False, f"stage {i} 越界"
        try:
            st = self._stage(i)
        except Exception as exc:                                # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        min_shards = self._specs[i].get("min_shards")
        if min_shards and hasattr(st, "shard_count") and st.shard_count < int(min_shards):
            return False, f"分片数 {st.shard_count} < 要求 {min_shards}"
        return True, f"rows={len(st)}"

    def refresh_stage(self, i: int) -> bool:
        """重新扫描该阶段目录，接过新到的分片。返回是否有变化。"""
        st = self._cache.get(i)
        if st is None:
            return False
        if not hasattr(st, "refresh"):
            return False
        if st.refresh():
            print(f"[stage] 阶段 {i}「{self._specs[i]['name']}」刷新："
                  f"{len(st)} 行 / {getattr(st, 'shard_count', '?')} 分片", flush=True)
            return True
        return False

    def gc_stage(self, i: int) -> int:
        """删除第 i 段的**数据文件**（已消费完才调）。返回释放字节数。

        安全约束（宁可少删也不能删错）：
        1. 只删本阶段自己登记的文件，不 rm -rf 整个目录；
        2. 该目录若被别的阶段引用，跳过；
        3. 该目录若是验证集目录，跳过。
        """
        if not (0 <= i < len(self._specs)):
            return 0
        sp = self._specs[i]
        d = Path(sp["data_dir"]).resolve()
        for j, other in enumerate(self._specs):
            if j != i and Path(other["data_dir"]).resolve() == d:
                print(f"[gc] 跳过阶段 {i}「{sp['name']}」：目录与阶段 {j} 相同", flush=True)
                return 0
        st = self._cache.get(i)
        if st is None or not hasattr(st, "files"):
            return 0
        # 2026-10-20（docs/00 E13）：删除前必须证明"可重建"。
        # 上一轮只查了"是否与别的阶段同目录""是否验证集"，然后就删了，
        # 结果 hf_pack_* 与 pretrain_v2 的分片全部不可复核。
        # 现在要求目录里有一份 meta.json，并且写明 source_ids 或 rebuild_cmd；
        # 缺一个就拒绝删除——宁可少删，不可删错。
        meta = d / "meta.json"
        if not meta.exists():
            print(f"[gc] 拒绝删除阶段 {i}「{sp['name']}」：缺少 {meta}，"
                  f"无法证明这份数据可重建", flush=True)
            return 0
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as exc:                                # noqa: BLE001
            print(f"[gc] 拒绝删除阶段 {i}「{sp['name']}」：{meta} 读不了（{exc}）",
                  flush=True)
            return 0
        if not (m.get("rebuild_cmd") or m.get("source_ids")):
            print(f"[gc] 拒绝删除阶段 {i}「{sp['name']}」：{meta} 里没有 "
                  f"rebuild_cmd / source_ids", flush=True)
            return 0
        targets = st.files()
        if hasattr(st, "close"):
            st.close()                 # 先松 mmap 句柄，否则 Windows 删不掉
        freed = 0
        for p in targets:
            try:
                sz = p.stat().st_size
                p.unlink()
                freed += sz
            except FileNotFoundError:
                pass
            except Exception as exc:                            # noqa: BLE001
                print(f"[gc] 删除 {p} 失败: {exc}", flush=True)
        self._cache.pop(i, None)
        print(f"[gc] 阶段 {i}「{sp['name']}」已回收 {freed / 1e9:.2f} GB 磁盘"
              f"（{d}）", flush=True)
        return freed

    @property
    def active_index(self) -> int:
        return self._active

    def files(self) -> list[Path]:
        """当前阶段占用的磁盘文件（诊断用；GC 走 gc_stage）。"""
        st = self._cache.get(self._active)
        return st.files() if st is not None and hasattr(st, "files") else []

    @property
    def active_name(self) -> str:
        return self._specs[self._active]["name"]

    # -- Dataset ----------------------------------------------------
    def __len__(self):
        return len(self._stage(self._active))

    def __getitem__(self, idx):
        st = self._stage(self._active)
        n = len(st)
        if n <= 0:
            raise IndexError(f"阶段「{st.name}」为空")
        inp, lab, wc = st.row(int(idx) % n)
        return {
            "input_ids": torch.from_numpy(inp.astype(np.int64)),
            "labels": torch.from_numpy(lab.astype(np.int64)),
            "wcodes": torch.from_numpy(wc.astype(np.int64)),
        }

