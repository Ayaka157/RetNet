# 一键把 AutoDL 实例配成免密（之后 AI 助手和我都能直接用 `ssh gpu-host`，无需密码）。
#
#   pwsh -NoProfile -File scripts\linux\setup_autodl_ssh.ps1 `
#        -SshHost <主机名> -Port 12345
#
# 你只需要做两件事：
#   1) 把 AutoDL 实例页那行「ssh -p 12345 root@<gpu-host>」里的
#      HostName 和 Port 填到上面的参数里
#   2) 脚本运行到推公钥那一步时，**由你自己输入密码**（ssh 会提示）
#
# ⚠️ 密码不要贴给任何人/AI。本脚本全程只让它出现在 ssh 自己的密码提示里。
#
# 脚本做的事（都可重复执行，不会重复写配置）：
#   1. 检查/生成 ssh 密钥对
#   2. 往 ~/.ssh/config 写入（或复用）Host gpu-host 别名
#   3. 把公钥追加到实例的 ~/.ssh/authorized_keys（这一步会问你一次密码）
#   4. 用 BatchMode 验证免密**真的**生效（失败会明确报出来，不会假装成功）
#   5. 打印实例信息（GPU 型号/显存/核数/内存/数据盘）

param(
    [Parameter(Mandatory = $true)][string]$SshHost,
    [Parameter(Mandatory = $true)][int]$Port,
    [string]$User = 'root',
    [string]$Alias = 'gpu-host',
    # 下面两个只为**可测试性**存在：让本脚本能对一份配置副本做演练，
    # 从而在不动真实 ~/.ssh/config、也不连网的前提下验证覆盖逻辑。
    [string]$ConfigPath = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$sshDir = Join-Path $env:USERPROFILE '.ssh'
$cfgPath = if ($ConfigPath) { $ConfigPath } else { Join-Path $sshDir 'config' }
$keyPath = Join-Path $sshDir 'id_ed25519'
$pubPath = "$keyPath.pub"

New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
if ($ConfigPath) { New-Item -ItemType Directory -Force -Path (Split-Path $ConfigPath -Parent) | Out-Null }

# ---------- 0. 入参清洗：把 ssh 命令里常见的粘贴错误直接吃掉 ----------
# 实测教训：用户会把 `ssh -p 24121 root@connect.xxx` 里的 `root@` 一起粘进 HostName，
# 于是 ssh 去解析 "root@connect.xxx" 这个主机名，报 Could not resolve hostname。
# 与其指望人不犯错，不如在入口就把它规范化（并明确告诉用户改了什么）。
if ($SshHost -match '@') {
    $fixed = ($SshHost -split '@')[-1]
    Write-Host "!! -SshHost 里含 '@'（你大概是把 ssh 命令里的 root@ 一起粘进来了）" -ForegroundColor Yellow
    Write-Host "   已自动改成: $fixed"
    $SshHost = $fixed
}
$SshHost = $SshHost.Trim().TrimEnd('/')
if ($SshHost -match 'xxxx|\.\.\.|your|placeholder|<|>') {
    throw "-SshHost 看起来还是占位符：'$SshHost'。请填 AutoDL 实例页 SSH 登录列里真实的主机名。"
}
if ($Port -lt 1 -or $Port -gt 65535 -or "$Port" -match '^0+$') {
    throw "-Port 不合法：'$Port'。请填实例页显示的真实端口（通常 5 位数字）。"
}

# ---------- 1. 密钥 ----------
# ⚠️ 这里**故意不自动生成密钥**。原因是空的 -N 参数在 PowerShell 传给原生程序时
# 很容易被误写成 -N '""'，那会把口令设成两个字符的 `""`（本机实测确认过：
# 生成的密钥会卡在口令提示上）。宁可不做，也不要把你后续所有自动化都拖进坑里。
if (-not (Test-Path $pubPath)) {
    Write-Host "!! 没找到 $pubPath" -ForegroundColor Red
    Write-Host "   请先手动生成一个**不设口令**的密钥（一路回车即可）："
    Write-Host "     ssh-keygen -t ed25519 -f `"$keyPath`" -C dsh-gpu-host"
    Write-Host "   然后重新运行本脚本。"
    exit 1
}
$key = (Get-Content $pubPath -Raw).Trim()
if ($key -notmatch '^ssh-(ed25519|rsa)\s') {
    throw "公钥格式不对：$pubPath（内容是 '$($key.Substring(0, [Math]::Min(20, $key.Length)))...'）"
}
Write-Host ("公钥: {0}...{1}" -f $key.Substring(0, 24), $key.Substring($key.Length - 12))
# 顺带确认私钥**没有口令**（自动化里带口令的密钥会一直弹提示）
$probe = & ssh-keygen -y -f $keyPath -P '' 2>&1
if ($LASTEXITCODE -ne 0 -or $probe -notmatch '^ssh-') {
    Write-Host "!! 私钥 $keyPath 带口令 —— 自动化使用会卡在提示上。" -ForegroundColor Yellow
    Write-Host "   建议改用无口令密钥，或先 ssh-add 到 agent。继续执行可能失败。"
} else {
    Write-Host "私钥无口令 OK"
}

# ---------- 2. 别名：**总是校验并覆盖写入**（不做「存在就复用」）----------
# 原实现「已存在就复用」是个坏设计：它对段内的值毫无校验，
# 于是一个写错的 HostName 会被永久沿用，而且每次运行都打印「复用」假装正常。
$lines = if (Test-Path $cfgPath) { @(Get-Content $cfgPath) } else { @() }
$start = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "^\s*Host\s+$([regex]::Escape($Alias))\s*$") { $start = $i; break }
}
if ($start -ge 0) {
    # 找段尾（下一个 Host 行或文件末尾），并把它前面紧邻的注释行一起吃掉
    $end = $lines.Count
    for ($j = $start + 1; $j -lt $lines.Count; $j++) {
        if ($lines[$j] -match '^\s*Host\s') { $end = $j; break }
    }
    $cut = $start
    if ($cut -gt 0 -and $lines[$cut - 1] -match '^\s*#.*AutoDL.*') { $cut-- }
    $kept = @()
    if ($cut -gt 0) { $kept += $lines[0..($cut - 1)] }
    if ($end -lt $lines.Count) { $kept += $lines[$end..($lines.Count - 1)] }
    Write-Host "发现旧的 Host $Alias 段（$($end - $start) 行）—— 已移除并重建" -ForegroundColor Yellow
    $lines = $kept
}
$block = @(
    '',
    '# AutoDL 实例（由 scripts\linux\setup_autodl_ssh.ps1 写入/更新）',
    "Host $Alias",
    "    HostName $SshHost",
    "    Port $Port",
    "    User $User",
    '    StrictHostKeyChecking no',
    '    ServerAliveInterval 30'
)
$lines = @($lines) + $block
Set-Content -Path $cfgPath -Value $lines -Encoding utf8NoBOM
Write-Host "已写入 Host $Alias -> $User@${SshHost}:$Port"
Write-Host "（注意 HostName 只填主机名，不要带 root@ —— 本脚本现在会自动剥掉）"

# 让用户能眼见为实
Write-Host "  当前段内容:"
($lines | Select-Object -Last $block.Count) | ForEach-Object { Write-Host "    $_" }

# ---------- 3. 推公钥（**这里会提示你自己输入密码**）----------
if ($DryRun) {
    Write-Host ""
    Write-Host "-DryRun：跳过 ssh 推送/验证/取信息三步。配置文件已按上述内容写入 $cfgPath" -ForegroundColor Yellow
    exit 0
}
Write-Host ""
Write-Host "接下来 ssh 会提示输入密码 —— 请自己输入（本脚本不接触它）。" -ForegroundColor Yellow
$remote = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && " +
          "grep -qxF '$key' ~/.ssh/authorized_keys || echo '$key' >> ~/.ssh/authorized_keys; " +
          "chmod 600 ~/.ssh/authorized_keys && echo KEY_OK"
& ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no $Alias $remote
if ($LASTEXITCODE -ne 0) {
    Write-Host "!! 推公钥失败（退出码 $LASTEXITCODE）。检查 Host/Port 是否填对。" -ForegroundColor Red
    exit 1
}

# ---------- 4. 验证免密真的生效 ----------
Write-Host ""
Write-Host "用 BatchMode 验证免密（BatchMode 下不会弹密码提示，所以能真正证明密钥生效）："
& ssh -o BatchMode=yes -o ConnectTimeout=15 $Alias "echo PUBKEY_AUTH_OK"
if ($LASTEXITCODE -ne 0) {
    Write-Host "!! 免密未生效。常见原因：实例的 authorized_keys 被镜像重置 / 权限不对。" -ForegroundColor Red
    exit 1
}
Write-Host "✅ 免密已生效，之后可以直接：ssh $Alias" -ForegroundColor Green

# ---------- 5. 打印实例信息 ----------
Write-Host ""
Write-Host "=== 实例信息 ==="
& ssh $Alias "echo '--- GPU ---'; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; echo '--- CPU/内存 ---'; nproc; free -g | head -2; echo '--- 磁盘 ---'; df -h / /root/gpu-scratch 2>/dev/null | tail -3; echo '--- python/torch ---'; python3 -c 'import sys;print(\"python\",sys.version.split()[0])' 2>&1; python3 -c 'import torch;print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda,torch.cuda.is_available())' 2>&1"

Write-Host ""
Write-Host "把上面那段「实例信息」贴回来即可。" -ForegroundColor Cyan
