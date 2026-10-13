# RetNet 0.5B 分享部署（本地后端 + 前端 + Cloudflare 隧道）

**线上地址（已验证）：<https://your-demo-host.example/play/>**

* 本地服务：`http://127.0.0.1:8077/play/`（只监听 127.0.0.1）
* 公网入口：Cloudflare 隧道 `retnet-demo-tunnel` → Public Hostname `your-demo-host.example` → `http://127.0.0.1:8077`
* 模型：`checkpoints/retnet_500m_v15b_ep3`（504,179,896 参数，val loss 1.9459）

---

## 1. 一键启动 / 停止 / 看状态

```powershell
.\serve\start_play.ps1                  # 启动（后台常驻）：后端 + 连接器 + 看护进程，打印地址后退出
.\serve\start_play.ps1 -Attach          # 同上，但本脚本留在前台当"主人"：Ctrl+C / 关窗口 → 自动收尾
.\serve\start_play.ps1 -Status          # 现在谁在跑（读状态文件 + 实测端口/健康）
.\serve\start_play.ps1 -Stop            # 收尾：先打印汇总，再停后端/连接器/临时隧道/看护，删状态文件
.\serve\start_play.ps1 -Clean           # 强制清理残留（状态文件丢了/上次被硬杀时用；启动时也会自动跑）
.\serve\start_play.ps1 -Foreground      # 前台跑后端（调试，Ctrl+C 直接停；不启隧道）
.\serve\start_play.ps1 -QuickTunnel     # 额外起一条 trycloudflare 临时隧道
.\serve\start_play.ps1 -NoTunnel        # 只起后端
.\serve\start_play.ps1 -Token "口令"     # 开分享口令（分享链接为 …/play/?t=口令）
```

**意外终止怎么收尾**（都用真实场景测过，不是设计承诺）：

| 场景 | 谁收尾 | 实测证据 |
|---|---|---|
| `-Attach` 下 Ctrl+C | 脚本 `finally` + 看护兜底 | — |
| 直接关掉终端窗口 / 强杀启动器 | 看护进程发现"启动器 pid 没了" | 强杀 launcher → **4 秒内**后端+连接器停止、状态文件删除 |
| 后端自己崩了 | 看护把连接器一起停掉（避免"网页 502 但进程还挂着"） | 强杀后端 → 15 秒内连接器停止、状态文件删除 |
| 连接器自己崩了 | 同上，反向 | （同一条代码路径） |
| 机器关机 / 断电 / 任务管理器硬杀 | 进程随系统消失；**残留状态文件**由下次启动自动 `-Clean` 收掉 | `-Clean` 在无状态文件时也能靠端口 + cmdline 识别残留 |
| 用 `-Status` 排查 | 显示状态文件里每个 pid 的 alive/DEAD + 实测端口/连接器/健康 | — |

机制要点：

* **状态文件** `logs/play_state.json`：记录 backend / connector / quickTunnel / supervisor / launcher 的 pid，`-Status`/`-Stop`/`-Clean` 都靠它精确定位。
* **看护进程** `serve/_play_supervisor.ps1`：每 3 秒轮询，发现异常就收尾并删状态文件（日志 `logs/supervisor.out`）。
* **只停"认得出来是自己的"进程**：后端要求 pid 的 cmdline 含 `retnet_play_server.py`；连接器要求含 `--no-autoupdate`；临时隧道要求含 `--url http://127.0.0.1:8077`。
  **你那条 `cloudflared tunnel run --token eyJhIjoi…`（服务隧道 `other-service-tunnel`）不在任何一类里，永远不会被这个脚本碰。**


## 2. 手动等价命令（脚本出问题时用）

```powershell
$py = "C:\Tools\Miniconda3\envs\AI\python.exe"

# 后端
Start-Process $py -ArgumentList @("serve\retnet_play_server.py","--model_dir","checkpoints/retnet_500m_v15b_ep3",
  "--port","8077","--dtype","bf16","--root_path","/play") -WorkingDirectory "E:\Workspaces\RetNet" `
  -RedirectStandardOutput "E:\Workspaces\RetNet\logs\serve.out" -RedirectStandardError "E:\Workspaces\RetNet\logs\serve.err" -WindowStyle Hidden

# 连接器（token 走环境变量，不进命令行、不进日志）
$env:TUNNEL_TOKEN = (Get-Content "%TUNNEL_TOKEN_FILE%" -Raw).Trim()
Start-Process "cloudflared" -ArgumentList @("tunnel","--no-autoupdate","run") `
  -WorkingDirectory "E:\Workspaces\RetNet" -RedirectStandardOutput "E:\Workspaces\RetNet\logs\tunnel-llm.out" `
  -RedirectStandardError "E:\Workspaces\RetNet\logs\tunnel-llm.err" -WindowStyle Hidden

# 停止
Get-NetTCPConnection -LocalPort 8077 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" |
  Where-Object { $_.CommandLine -match '--no-autoupdate' -and $_.CommandLine -notmatch '--url' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

服务参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--root_path` | `/play` | 应用挂载路径；域名后面跟的就是它 |
| `--dtype` | `bf16` | 3070 上实测 **25–34 tok/s**；`fp32` 更稳但慢一点 |
| `--workers` | 1 | 并发生成数。3070 上**必须 1** |
| `--max_waiting` | 8 | 排队上限，满则 503 |
| `--min_interval` | 1.0 | 同 IP 最小间隔（防刷；压测可临时调 0.2） |
| `--token` | 无 | 开启口令；前端从 `?t=` 读取并记住 |

## 3. Cloudflare 接入（当前配置，已验证）

| 项 | 值 |
|---|---|
| 隧道名 | `retnet-demo-tunnel` |
| 隧道 ID | `TUNNEL_ID_REDACTED` |
| 类型 | 远程托管（token 模式，路由在 Cloudflare 后台） |
| token 文件 | `%TUNNEL_TOKEN_FILE%`（184 字符，单段 base64） |
| Public Hostname | `your-demo-host.example` → **HTTP** → `127.0.0.1:8077`（Path 留空） |
| 连接器 | 本机普通进程 `cloudflared tunnel --no-autoupdate run`（4 条边缘连接：mfm01×2 / hkg12 / hkg13） |
| DNS | Cloudflare 自动建的 CNAME，proxied → 172.67.160.56 / 104.21.90.193 |

**加路由的界面路径**：Networks → Tunnels → 选中隧道 → **Public Hostname** → Add a public hostname →
Subdomain `retnet` + Domain `<redacted-host>` + **Path 留空** + Service **HTTP** + `127.0.0.1:8077`。

> ⚠️ 两个容易填错的字段：**Path 必须留空**（填 `^/play` 会让 `https://your-demo-host.example/` 的 302 跳转进不来，直接 404）；
> **Service 必须选 http 而不是 https**（本机是明文 HTTP，选 https 会 502/522）。

> **旧隧道 `other-service-tunnel`（无路由）保持原样不动**。它和本项目无关；要清理单独说。

## 4. 端口 / 防火墙 / 端口映射

**走隧道：一个端口都不用开，也不要做端口映射。**

* cloudflared 是**出站**连接（QUIC/UDP 443、HTTPS 7844/TCP），不需要入站规则、不需要公网 IP、不需要 DDNS。
* 后端只监听 `127.0.0.1:8077`，**故意不监听 0.0.0.0** —— 局域网设备、端口扫描都碰不到它。
* 所以：不需要 `New-NetFirewallRule`，不需要路由器映射。

如果哪天真要端口直连（不推荐）：要同时满足 ① 改 `--host 0.0.0.0`；② 防火墙放行 TCP 8077；
③ 路由器映射 8077 → `192.168.1.1009:8077`；④ **本机得有真公网 IP**。
本机是 `192.168.1.1009`（NAT 后面），网卡里还有 Radmin / PrivadoVPN 多条隧道、出网 IP `60.246.60.119`——
多 NAT/多 VPN 环境下直连基本不可行，这就是推荐隧道的根本原因。

## 5. 安全与并发（当前实现）

* **单 worker FIFO 队列**：一次服务一个人；后到的人先收到 `{"type":"queue","position":N}`，前端显示"前面还有 N-1 位"。
* **三道闸**：每 IP 最小间隔、排队上限（满则 503）、单次 `max_new_tokens ≤ 512`，全部服务端夹紧。
* **多用户隔离**：服务端**无会话状态**，多轮上下文由前端回传**回复的 token id**（不是文本：
  词表里 `<`、`>`、`/` 是独立特殊 token，`encode(decode(ids)) != ids`）。
* **凭据纪律（重要）**：本项目**只读用户明确指定的凭据文件**（当前为 `%TUNNEL_TOKEN_FILE%`）。
  不读剪贴板、不读环境变量、不翻会话归档。token 只经环境变量传给 cloudflared，不进命令行、不进日志、不打印明文（只打掩码）。
* **想加登录门槛**：Cloudflare Zero Trust → Access → Applications，给 `your-demo-host.example` 加策略（邮箱/一次性 PIN）。
  免费额度对小圈子够用。不想开 Zero Trust 就用 `--token` 口令 + 分享链接 `…/play/?t=…`。

## 6. 已踩过的坑（都实测过）

| 现象 | 根因 | 处置 |
|---|---|---|
| 刷新页面后所有人都卡在队列里，health 显示 `busy:true, queue:1` | 第一版在 SSE 生成器里 `Semaphore.acquire/release`，客户端断开时 `finally` 不执行 → 锁泄漏 | 改成 worker 线程持有所有权；客户端断开只标记"未开始的任务"取消 |
| 后到的人永远显示"第 1 位" | 排队位置算在 `JOBS.put()` 之后，worker 秒取任务 | 位置在 put 之前算，且加上"正在生成的那个" |
| 请求返回 500 `JSONDecodeError` | 客户端 JSON 非法（测试脚本被 shell 拆碎过） | 服务端挡成 400 |
| `AttributeError: 'NoneType' object has no attribute 'root_path'` | 路由装饰器在 import 时执行，而 `--root_path` 要等 argparse | 路由挂内层 app，`main()` 里 `app.mount(root_path, api)` |
| 用 3 段 JWT 的正则解析隧道 token 失败 | **Cloudflare 隧道 token 是单段 base64**（`{"a":…,"t":…,"s":…}`），不是三段 JWT | 解析时按单段 base64 处理 |
| `Set-Content … "ey..."` 写进文件却没生效 | 占位符被当成字面量；且当时目录/环境不对 | 用记事本保存或 `Get-Clipboard \| Set-Content`；核对长度应为 ~184 字符 |

## 7. 验证记录（2026-10-13）

```
Resolve-DnsName your-demo-host.example  → 172.67.160.56 / 104.21.90.193（Cloudflare 代理）
GET  https://your-demo-host.example/play/api/health  → 200，JSON: model=retnet_500m_v15b_ep3, cuda/bf16, 504.2M
GET  https://your-demo-host.example/play/            → 200，17820 字节，<title>RetNet 0.5B · play</title>
GET  https://your-demo-host.example/                 → 307 → https://your-demo-host.example/play/
POST https://your-demo-host.example/play/api/chat    → 80 个 delta 事件，done: 80 tok / 3.13 s / 25.5 tok/s
```

## 8. 稳定性提醒

* 本机睡眠会掐断服务；当前电源设置已确认 **交流/直流均为 0（永不睡眠）** ✓。
* 现在服务与连接器都是**普通进程**，重启电脑后不会自启。要长期挂着可以注册计划任务（涉及改系统配置，未擅自做）。
* 模型只有 504M 参数、角色扮演向：**知识题会一本正经地编**，页面里已写明。
