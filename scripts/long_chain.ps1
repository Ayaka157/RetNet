# 长跑训练链：继续预训练(约26h) -> 角色扮演SFT(v6) -> 自动评测
# 全部段间自动衔接；每段崩溃可重试，随时可中断续训。
$ErrorActionPreference = 'Continue'
$py = "C:\Tools\Miniconda3\envs\AI\python.exe"
Set-Location E:\Workspaces\RetNet

function Say($msg) { "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg }

# ---------------- Stage 1: 继续预训练（池 4.06B token，跑 520k 步 ≈ 2.66B） ----------------
Say "STAGE 1 START: long continued pretraining -> checkpoints/retnet_100m_pretrain_v4"
& $py scripts\train_loop.py `
    --total_steps 520000 `
    --segment_steps 40000 `
    --save_dir checkpoints/retnet_100m_pretrain_v4 `
    --log_dir logs/pretrain_v4 `
    -- `
    --shard_dirs data_cache/pretrain_v2 data_cache/pretrain_merged `
    --tokenizer_path data_cache/chat_pack_v4/bpe_tokenizer.json `
    --vocab_size 20000 `
    --d_model 768 --num_layers 12 --num_heads 12 `
    --seq_len 512 --batch_size 10 `
    --warmup_steps 1500 --lr 1.5e-4 --min_lr 1.2e-5 `
    --mode parallel --amp bf16 --device cuda --compile `
    --init_from checkpoints/retnet_100m_pretrain_v3/model.pt `
    --eval_interval 5000 --eval_batches 40 `
    --generate_prompt "人工智能的发展"
Say "STAGE 1 DONE (exit=$LASTEXITCODE)"

# ---------------- Stage 2: 角色扮演 SFT（优先 v8，退回 v7/v6/v5） ----------------
# 手动停止开关：存在 logs/STOP_AFTER_STAGE1.flag 时跳过 Stage 2（改用成熟框架后由新脚本接管）
if (Test-Path logs\STOP_AFTER_STAGE1.flag) {
    Say "STOP_AFTER_STAGE1.flag 存在：跳过 Stage 2，等待成熟框架迁移"
    exit 0
}
$data = if (Test-Path data_cache\chat_pack_v8\train.ids.npy) { "data_cache/chat_pack_v8" } `
    elseif (Test-Path data_cache\chat_pack_v7\train.ids.npy) { "data_cache/chat_pack_v7" } `
    elseif (Test-Path data_cache\chat_pack_v6\train.ids.npy) { "data_cache/chat_pack_v6" } `
    else { "data_cache/chat_pack_v5" }
$tok  = "$data/bpe_tokenizer.json"
$init = if (Test-Path checkpoints\retnet_100m_pretrain_v4\model.pt) {
            "checkpoints/retnet_100m_pretrain_v4/model.pt"
        } else { "checkpoints/retnet_100m_rp_v2/model.pt" }
Say "STAGE 2 START: role-play SFT on $data (init=$init)"

& $py scripts\train_loop.py `
    --total_steps 45000 `
    --segment_steps 45000 `
    --save_dir checkpoints/retnet_100m_rp_v3 `
    --log_dir logs/rp_v3 `
    -- `
    --prepared_data_dir $data `
    --tokenizer_path $tok `
    --vocab_size 20000 `
    --d_model 768 --num_layers 12 --num_heads 12 `
    --seq_len 512 --batch_size 10 `
    --warmup_steps 1000 --lr 1.1e-4 --min_lr 1.1e-5 `
    --prompt_loss_weight 0.1 `
    --mode parallel --amp bf16 --device cuda --compile `
    --init_from $init `
    --eval_interval 2500 --eval_batches 40 `
    --chat_format --generate_prompt "你真好看"
Say "STAGE 2 DONE (exit=$LASTEXITCODE)"

# ---------------- Stage 3: 自动评测（报告在下次唤醒时直接引用） ----------------
if (Test-Path checkpoints\retnet_100m_rp_v3\model.pt) {
    Say "STAGE 3 START: eval battery on rp_v3"
    & $py scripts\eval_chat.py `
        --checkpoint checkpoints\retnet_100m_rp_v3\model.pt `
        --tokenizer data_cache\chat_pack_v6\bpe_tokenizer.json `
        --tag rp_v3_final
    Say "STAGE 3 DONE"
} else {
    Say "STAGE 3 SKIPPED: no rp_v3 checkpoint"
}
Say "LONG CHAIN COMPLETE"
