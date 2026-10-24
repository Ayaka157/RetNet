"""为 Hugging Face 上传准备目录（不含任何凭据；产物放在 logs/ 下，不进 git）。

- config.json / model.safetensors / 建模代码 / 原生 HF tokenizer
- 给 config.json 加 auto_map，使 trust_remote_code=True 可直接加载
- tokenizer 用 PreTrainedTokenizerFast 三件套（AutoTokenizer 直接可用，无需自定义代码）
"""
import json
import os
import shutil

ROOT = r'E:\Workspaces\RetNet'
STAGE = os.path.join(ROOT, 'logs', '_hf_stage')

CODE = ['retnet_hf/modeling_retnet.py', 'retnet_hf/retention_fast.py', 'retnet_hf/fused_ce.py']
TOK_FILES = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
TOK_SRC = 'checkpoints/retnet_100m_chat_hf'   # 三个语料包的 bpe_tokenizer.json 逐字节相同

AUTO_MAP = {
    'AutoConfig': 'modeling_retnet.RetNetHFConfig',
    'AutoModelForCausalLM': 'modeling_retnet.RetNetForCausalLM',
}

# (产物名, 源权重目录, 是否 HF 格式)
TARGETS = [
    ('RetNet-0.5B',     'checkpoints/retnet_500m_v15b_ep3', True),
    ('RetNet-0.5B-ep2', 'checkpoints/retnet_500m_v15b_ep2', True),
    ('RetNet-0.5B-ep1', 'checkpoints/retnet_500m_v15b',     True),
    ('RetNet-100M',     'checkpoints/retnet_100m_chat_hf',  True),
]

GITATTR = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"


def main():
    for name, srcdir, _ in TARGETS:
        src = os.path.join(ROOT, srcdir)
        dst = os.path.join(STAGE, name)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)

        for f in ('config.json', 'model.safetensors'):
            p = os.path.join(src, f)
            if not os.path.exists(p):
                print('  !! 缺文件:', p)
                continue
            shutil.copy2(p, os.path.join(dst, f))

        cfgp = os.path.join(dst, 'config.json')
        if os.path.exists(cfgp):
            cfg = json.load(open(cfgp, encoding='utf-8'))
            cfg['auto_map'] = AUTO_MAP
            # 训练用的 checkpoint 里 mode=chunkwise；而 chunkwise 要求 seq_len 是
            # chunk_size 的整数倍，任何短提示的裸前向都会直接报错。推理安全的默认值
            # 是 parallel（generate() 本来就硬编码 parallel/recurrent，
            # train.py:230 也会显式覆盖这个字段，所以改它不影响训练）。
            cfg['mode'] = 'parallel'
            json.dump(cfg, open(cfgp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

        for c in CODE:
            shutil.copy2(os.path.join(ROOT, c), os.path.join(dst, os.path.basename(c)))
        for t in TOK_FILES:
            shutil.copy2(os.path.join(ROOT, TOK_SRC, t), os.path.join(dst, t))
        open(os.path.join(dst, '.gitattributes'), 'w').write(GITATTR)

        files = sorted(os.listdir(dst))
        sz = sum(os.path.getsize(os.path.join(dst, f)) for f in files)
        print('  %-16s %d 个文件 / %5.2f GB' % (name, len(files), sz / 1024**3))
        print('      %s' % ', '.join(files))


if __name__ == '__main__':
    main()
