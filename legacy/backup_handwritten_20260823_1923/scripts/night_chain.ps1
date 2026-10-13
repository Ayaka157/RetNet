# 夜间训练链：先继续预训练，再做角色扮演 SFT
# 两段都用完整 cosine 计划，段间自动衔接，无需人工干预。
$ErrorActionPreference = 'Continue'
$py = "C:\Tools\Miniconda3\envs\AI\python.exe"
Set-Location E:\Workspaces\RetNet

$DEADLINE = Get-Date "2026-08-21 07:35:00"

function Say($msg) { "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg }

# ---------------- Stage 1: 继续预训练（4.06B token 语料池，跑 70k 步 ≈ 363M token） ----------------
Say "STAGE 1 START: continued pretraining -> checkpoints/retnet_100m_pretrain_v3"
& $py scripts\train_loop.py `
    --total_steps 70000 `
    --segment_steps 24000 `
    --save_dir checkpoints/retnet_100m_pretrain_v3 `
    --log_dir logs/pretrain_v3 `
    -- `
    --shard_dirs data_cache/pretrain_v2 data_cache/pretrain_merged `
    --tokenizer_path data_cache/chat_pack_v4/bpe_tokenizer.json `
    --vocab_size 20000 `
    --d_model 768 --num_layers 12 --num_heads 12 `
    --seq_len 512 --batch_size 10 `
    --warmup_steps 1000 --lr 1.5e-4 --min_lr 1.5e-5 `
    --mode parallel --amp bf16 --device cuda --compile `
    --init_from checkpoints/retnet_100m_rp/model.pt `
    --eval_interval 4000 --eval_batches 40 `
    --generate_prompt "人工智能的发展"
Say "STAGE 1 DONE (exit=$LASTEXITCODE)"

# ---------------- Stage 2: 角色扮演 SFT（优先用 v5，没有就退回 v4） ----------------
if ((Get-Date) -ge $DEADLINE) { Say "past deadline, skipping stage 2"; exit 0 }

$data = if (Test-Path data_cache\chat_pack_v5\train.ids.npy) { "data_cache/chat_pack_v5" } else { "data_cache/chat_pack_v4" }
$tok  = "$data/bpe_tokenizer.json"
$init = if (Test-Path checkpoints\retnet_100m_pretrain_v3\model.pt) {
            "checkpoints/retnet_100m_pretrain_v3/model.pt"
        } else { "checkpoints/retnet_100m_rp/model.pt" }
Say "STAGE 2 START: role-play SFT on $data (init=$init)"

& $py scripts\train_loop.py `
    --total_steps 35000 `
    --segment_steps 35000 `
    --save_dir checkpoints/retnet_100m_rp_v2 `
    --log_dir logs/rp_v2 `
    -- `
    --prepared_data_dir $data `
    --tokenizer_path $tok `
    --vocab_size 20000 `
    --d_model 768 --num_layers 12 --num_heads 12 `
    --seq_len 512 --batch_size 10 `
    --warmup_steps 800 --lr 1.1e-4 --min_lr 1.1e-5 `
    --prompt_loss_weight 0.1 `
    --mode parallel --amp bf16 --device cuda --compile `
    --init_from $init `
    --eval_interval 2500 --eval_batches 40 `
    --chat_format --generate_prompt "你真好看"
Say "STAGE 2 DONE (exit=$LASTEXITCODE)"
Say "NIGHT CHAIN COMPLETE"
