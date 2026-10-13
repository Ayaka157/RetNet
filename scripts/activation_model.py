"""逐张量枚举 chunkwise_batched 训路里被 autograd 保存的激活。

    python scripts/activation_model.py

目的是把「梯度检查点省多少显存」从线性外推换成结构计算，
并找出哪个中间量随 chunk_size 二次增长。
"""
MIB = 2 ** 20

# 各配置的每头维度。key_dim = d/heads = 64 恒定；value_head_dim = value_dim/heads
CONFIGS = [
    # name,     L,  d,  heads,  seq
    ("272M",    20, 1024, 16, 2048),
    ("probe_500m", 20, 1408, 22, 2048),
    ("probe_600m", 24, 1408, 22, 2048),
]


def per_layer_mib(L_layers, d, H, seq, C, dtype_bytes=2, mask_bytes=4, verbose=False):
    """一层的保存量（MiB）。dtype_bytes=2 -> bf16；inner_mask 是 fp32 会提升乘积。"""
    nc = seq // C
    kd = d // H
    vd = 2 * d // H          # value_dim = 2*d_model
    items = {}

    def add(name, nelem, b):
        items[name] = nelem * b / MIB

    # 线性层保存的输入
    add("norm1(x) [1,L,d]", seq * d, dtype_bytes)
    # q,k,g: [1,H,L,kd]；v: [1,H,L,vd]
    for nm in ("q", "k", "g"):
        add(f"{nm} [1,H,L,kd]", H * seq * kd, dtype_bytes)
    add("v [1,H,L,vd]", H * seq * vd, dtype_bytes)
    # (1) vr * value_decay -> [1,H,nc,C,vd]
    add("vr*value_decay", H * seq * vd, dtype_bytes)
    add("U [1,H,nc,kd,vd]", H * nc * kd * vd, dtype_bytes)
    add("T [H,nc+1,nc] fp32", H * (nc + 1) * nc, 4)
    add("S_all [1,H,nc+1,kd*vd]", H * (nc + 1) * kd * vd, dtype_bytes)
    # (3) inner_scores = matmul(qr,kr^T) * inner_mask  -> 与 fp32 mask 相乘被提升
    add("** inner_scores [1,H,nc,C,C] **", H * seq * C, mask_bytes)
    add("inner [1,H,nc,C,vd]", H * seq * vd, dtype_bytes)
    add("qr*cross_decay [1,H,nc,C,kd]", H * seq * kd, dtype_bytes)
    add("cross [1,H,nc,C,vd]", H * seq * vd, dtype_bytes)
    add("out [1,H,L,vd]", H * seq * vd, dtype_bytes)

    # ---- MLP 路径（上一版漏了这块，是脚本的错）----
    ffn = 2 * d
    add("y = x + msr_out [1,L,d]", seq * d, dtype_bytes)
    add("norm2(y) [1,L,d]", seq * d, dtype_bytes)
    add("ffn hidden [1,L,ffn]", seq * ffn, dtype_bytes)
    add("GELU(hidden) [1,L,ffn]", seq * ffn, dtype_bytes)
    add("ffn out [1,L,d]", seq * d, dtype_bytes)
    # ---- 其余线性层的激活（out_proj / group_norm 前后）----
    add("pre out_proj [1,L,vd]", seq * vd, dtype_bytes)
    add("group_norm 输出 [1,L,vd]", seq * vd, dtype_bytes)

    tot = sum(items.values())
    if verbose:
        for k, v in sorted(items.items(), key=lambda kv: -kv[1]):
            print(f"      {k:<34} {v:7.2f} MiB")
    return tot, items


def main():
    print("=== 单层被保存的激活（seq 2048, chunk 128, bf16，inner_mask fp32）===")
    tot0, _ = per_layer_mib(20, 1024, 16, 2048, 128, verbose=True)
    print(f"      {'合计':<34} {tot0:7.2f} MiB")
    print(f"      × 20 层 = {tot0*20:7.1f} MiB = {tot0*20/1024:.2f} GiB")

    print("\n=== inner_scores 随 chunk_size 的二次增长（单层）===")
    print(f"  {'C':>6}{'nc':>5}{'inner_scores':>15}{'单层合计':>12}{'×20层':>10}")
    for C in (64, 128, 256, 512, 1024):
        t, it = per_layer_mib(20, 1024, 16, 2048, C)
        print(f"  {C:>6}{2048//C:>5}{it['** inner_scores [1,H,nc,C,C] **']:>12.2f} MiB"
              f"{t:>9.2f} MiB{t*20/1024:>8.2f} GiB")

    print("\n=== 各配置的保存量（chunk 128）===")
    print(f"  {'配置':<12}{'L':>4}{'d':>6}{'H':>4}{'单层':>10}{'全部层':>12}{'层输入':>10}")
    for name, L, d, H, seq in CONFIGS:
        t, it = per_layer_mib(L, d, H, seq, 128)
        stored_in = L * seq * d * 2 / MIB
        print(f"  {name:<12}{L:>4}{d:>6}{H:>4}{t:>7.2f} MiB{t*L/1024:>9.2f} GiB"
              f"{stored_in:>7.0f} MiB")

    print("\n=== 与实测对照（关键：这个缺口没有解释）===")
    print("  [实测] 272M 关检查点的非状态内存 = 4,342 MiB")
    print(f"  [推导] 本模型算出的「全部层被保存的激活」 = {tot0*20:.0f} MiB")
    print(f"  缺口 = 4342 - {tot0*20:.0f} = {4342-tot0*20:.0f} MiB，**未解释**")
    print()
    print("  ⚠️ 这条缺口决定梯度检查点到底省多少：")
    print("     若缺口是「被 autograd 保存但本模型没数到的张量」-> 检查点能省掉它；")
    print("     若缺口是「分配器碎片 / Triton·cuBLAS workspace / 8-bit Adam 反量化副本」")
    print("       -> 检查点省不掉它，那么开检查点后的非状态内存会远高于 262~467 MiB，")
    print("          0.5B/0.6B 的余量（2,783 / 1,875 MiB）可能**不够**。")
    print("     实验 E1 就是为这件事设计的。")


if __name__ == "__main__":
    main()
