# legacy/ — 已被取代的旧脚本

这些脚本是早期阶段用过的，现在都有更好的替代品，保留只为查历史。

| 旧脚本 | 现在用什么 |
| --- | --- |
| `chat.py` | `play.py --mode chat`（支持角色扮演设定、top-p、n-gram 去重） |
| `generate.py` | `play.py --mode text` |
| `prepare_chat_corpus.py` | `scripts/build_chat_mix_v4.py`（对齐 + 掩码 + persona） |

**运行注意**：它们原本放在项目根目录，直接 `python legacy/chat.py` 会找不到
`retnet` 包。需要时从项目根目录这样跑：

```powershell
cd E:\Workspaces\RetNet
$env:PYTHONPATH = "E:\Workspaces\RetNet"
C:\Tools\Miniconda3\envs\AI\python.exe legacy\chat.py --prompt "你真好看"
```

`prepare_chat_corpus.py` 还会把 `HF_HOME` 指到脚本所在目录的 `hf_cache`，
如果真要重跑它，建议先把它挪回根目录。
