"""从用户点名的文件读取 HF token 并登录（编码健壮版）。

设计约束：
- token 明文只在本进程内存在，不打印、不落盘到其它文件、不进日志。
- 失败只打印异常**类型**，不打印正文。
- 只报告 token 的**形态**（长度、字符集是否符合 hf_ + 字母数字），不报告内容。
"""
import os
import re
import sys

TOKEN_FILE = r'E:\Workspaces\RetNet\HF Access Token.txt'
SHAPE = re.compile(r'hf_[A-Za-z0-9]{20,}')


def load_token():
    raw = open(TOKEN_FILE, 'rb').read()
    print('  文件字节数: %d' % len(raw))
    print('  前 4 字节: %s' % raw[:4].hex())
    for enc in ('utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'ascii', 'latin-1'):
        try:
            s = raw.decode(enc)
        except Exception:
            continue
        s = s.replace('\ufeff', '')
        m = SHAPE.search(s)
        if m:
            extra = len(s.strip()) - len(m.group(0))
            print('  ✅ 编码 %s：提取到合法 token（%d 字符），文件里另有 %d 个装饰字符'
                  % (enc, len(m.group(0)), extra))
            return m.group(0)
    for enc in ('utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be'):
        try:
            s = raw.decode(enc).replace('\ufeff', '')
        except Exception:
            continue
        bad = [c for c in s if not (c.isascii() and (c.isalnum() or c in '_-'))]
        print('  编码 %-10s 解出 %d 字符, 非 token 字符 %d 个, 含 hf_ : %s'
              % (enc, len(s), len(bad), 'hf_' in s))
    return None


def main():
    if not os.path.exists(TOKEN_FILE):
        print('  找不到 token 文件:', TOKEN_FILE)
        return 1

    tok = load_token()
    if not tok:
        print('  ❌ 该文件里没有找到形态合法的 HF token（不打印内容）')
        return 1

    import huggingface_hub as hh
    try:
        info = hh.whoami(token=tok)
    except Exception as e:
        print('  ❌ whoami 失败:', type(e).__name__)
        return 1

    print('  ✅ 账号:', info.get('name'), ' 类型:', info.get('type'))
    auth = (info.get('auth') or {}).get('accessToken') or {}
    perms = auth.get('permissions') or []
    print('  权限:')
    for p in perms:
        print('     -', p)

    try:
        hh.login(token=tok, add_to_git_credential=False)
        print('  ✅ 已登录')
    except Exception as e:
        print('  ❌ login 失败:', type(e).__name__)
        return 1

    del tok
    return 0


if __name__ == '__main__':
    sys.exit(main())
