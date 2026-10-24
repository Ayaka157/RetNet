"""把 logs/_hf_stage/ 下的三个模型目录上传到 Hugging Face。

前置：先完成 HF 登录（本脚本不接触明文 token）：
    C:\\Tools\\Miniconda3\\envs\\AI\\Scripts\\hf.exe auth login

用法：
    python logs/_hf_upload.py            # 上传全部
    python logs/_hf_upload.py RetNet-0.5B  # 只传一个
"""
import os
import sys

from huggingface_hub import HfApi, whoami

STAGE = r'E:\Workspaces\RetNet\logs\_hf_stage'
REPOS = ['RetNet-0.5B', 'RetNet-0.5B-ep2', 'RetNet-0.5B-ep1', 'RetNet-100M']


def main():
    me = whoami()
    user = me['name']
    print('HF 账号:', user, ' 类型:', me.get('type'))
    print()

    only = sys.argv[1:] or REPOS
    api = HfApi()
    for name in only:
        d = os.path.join(STAGE, name)
        if not os.path.isdir(d):
            print('  跳过（不存在）:', name)
            continue
        repo_id = '%s/%s' % (user, name)
        print('---', repo_id)
        api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True, private=False)
        api.upload_folder(
            repo_id=repo_id,
            repo_type='model',
            folder_path=d,
            commit_message='Upload %s' % name,
        )
        print('    ✅ https://huggingface.co/%s' % repo_id)


if __name__ == '__main__':
    main()
