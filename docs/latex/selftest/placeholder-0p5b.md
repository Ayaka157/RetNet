# RetNet-0.5B：流水线自测占位稿

> 日期：2026-10-13 ｜ 对象：`docs/latex/selftest/placeholder-0p5b.md`
> 说明：**这不是真实报告**，只用来验证两条链路——「中文单语 md → 中文版 PDF」与
> 「英文补充稿 → 英文版 PDF」——在真实报告（`docs/26_RetNet-0.5B_技术报告.md`）交付前可构建。
> 全部数字取自 `docs/_report_0p5b/facts.md` 已核实的 `trainer_state.json`。

---

## 摘要

本占位稿由 `mktex.py` 的 `selftest` 配置读入：中文版直接取本文件正文，英文版全文来自
`selftest/en/_supplement.tex`。这一节同时验证「摘要由 Markdown 提供、排在目录之前、标题进目录但不编号」
的处理方式；若把 `## 摘要` 写成 `#` 之外的其它层级，构建器会打印一条 front-matter 警告。

## 1. 引言

这一段验证中文正文的行内渲染：行内代码 `RETNET_FAST_CHUNKWISE=1`、加粗 **重点结论**、
链接 [README](README.md)、以及行内数学 $d_{\mathrm{model}} = 1408$。
换行与长代码 `--resume checkpoints/retnet_500m_v15b/checkpoint-67390` 应当正常断行。

### 1.1 列表与引用

- 第一项：验证无序列表；
- 第二项：验证嵌套列表：

  1. 嵌套有序项一；
  2. 嵌套有序项二。

> 这一段是引用块，应当渲染成以 Note. 起头的缩进段落，而不是灰底框。

## 2. 架构

### 2.1 模型配置表

| 项 | 值 | 出处 |
| --- | --- | --- |
| 参数量 | 504,179,896 | `checkpoints/retnet_500m_v15b_ep3/model.safetensors` |
| d_model / layers / heads | 1408 / 20 / 22 | `config.json` |
| 存储精度 | `float32`（训练 bf16） | `config.json` |

### 2.2 代码块

```python
def retention(x, gamma):
    """自测用代码块：验证 listings 样式与中文注释。"""
    return x * gamma
```

### 2.3 热重启代价

| 衔接 | 上一段最优 | 下一段首次评测 | 差 |
| --- | --- | --- | --- |
| ep1 → ep2 | 2.1035 @65000 | 2.1800 @2106 | +0.0764 |
| ep2 → ep3 | 1.9865 @16848 | 2.1174 @2106 | +0.1310 |

三段训练共 1,656,340,480 token，token/参数 = 3.285。

## 3. 参考文献

1. Retentive Network: A Successor to Transformer for Large Language Models. arXiv:2307.08621.
2. Gated Linear Attention Transformers with Hardware-Efficient Training. arXiv:2312.06635.

## 附录 A：复现命令

```powershell
python makefigs.py
python mktex.py --report selftest
```
