"""项目相关论文的统一编号表。

编号来自 `Papers/` 目录的文件前缀（01–17），与项目里引用的顺序一致。
生成器与手写文档共用这一份，避免两处编号漂移。
"""
from __future__ import annotations

REFS: list[tuple[str, str, str, str]] = [
    # (编号, 标题, 本地文件, 链接)
    ("01", "Retentive Network: A Successor to Transformer for Large Language Models",
     "Papers/01_Retentive-Network_2307.08621.pdf", "https://arxiv.org/abs/2307.08621"),
    ("02", "A Length-Extrapolatable Transformer",
     "Papers/02_Length-Extrapolatable-Transformer_2212.10554.pdf", "https://arxiv.org/abs/2212.10554"),
    ("03", "Gated Linear Attention Transformers with Hardware-Efficient Training",
     "Papers/03_Gated-Linear-Attention_2312.06635.pdf", "https://arxiv.org/abs/2312.06635"),
    ("04", "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
     "Papers/04_Transformers-are-RNNs_2006.16236.pdf", "https://arxiv.org/abs/2006.16236"),
    ("05", "Group Normalization",
     "Papers/05_Group-Normalization_1803.08494.pdf", "https://arxiv.org/abs/1803.08494"),
    ("06", "GLU Variants Improve Transformer",
     "Papers/06_GLU-Variants_2002.05202.pdf", "https://arxiv.org/abs/2002.05202"),
    ("07", "Net2Net: Accelerating Learning via Knowledge Transfer",
     "Papers/07_Net2Net_1511.05641.pdf", "https://arxiv.org/abs/1511.05641"),
    ("08", "Network Morphism",
     "Papers/08_Network-Morphism_1603.01670.pdf", "https://arxiv.org/abs/1603.01670"),
    ("09", "bert2BERT: Towards Reusable Pretrained Language Models",
     "Papers/09_bert2BERT_2110.07143.pdf", "https://arxiv.org/abs/2110.07143"),
    ("10", "Efficient Training of BERT by Progressively Stacking",
     "Papers/10_StackBERT_PMLR-v97-gong19a.pdf", "https://proceedings.mlr.press/v97/gong19a.html"),
    ("11", "Progressively Stacking 2.0: A Multi-stage Layerwise Training Method for BERT Training Speedup",
     "Papers/11_Progressively-Stacking-2.0_2011.13635.pdf", "https://arxiv.org/abs/2011.13635"),
    ("12", "Learning to Grow Pretrained Models for Efficient Transformer Training",
     "Papers/12_LiGO_2303.00980.pdf", "https://arxiv.org/abs/2303.00980"),
    ("13", "ReZero is All You Need: Fast Convergence at Large Depth",
     "Papers/13_ReZero_2003.04887.pdf", "https://arxiv.org/abs/2003.04887"),
    ("14", "Fixup Initialization: Residual Learning Without Normalization",
     "Papers/14_Fixup_1901.09321.pdf", "https://arxiv.org/abs/1901.09321"),
    ("15", "Training Compute-Optimal Large Language Models",
     "Papers/15_Chinchilla_2203.15556.pdf", "https://arxiv.org/abs/2203.15556"),
    ("16", "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale",
     "Papers/16_FineWeb_2406.17557.pdf", "https://arxiv.org/abs/2406.17557"),
    ("17", "Skywork: A More Open Bilingual Foundation Model",
     "Papers/17_Skywork_2310.19341.pdf", "https://arxiv.org/abs/2310.19341"),
]


def cite(*ids: str) -> str:
    """行内引用：cite("01","03") -> "（参 [01][03]）"。"""
    return "（参 " + "".join(f"[{i}]" for i in ids) + "）"


def refs_md(title: str = "参考文献") -> str:
    lines = [f"## {title}", "",
             "> 编号与 `Papers/` 目录的文件前缀一一对应；本文件只引用实际用到的论文。", "",
             "| 编号 | 原文标题 | 本地文件 | 链接 |", "|---|---|---|---|"]
    for i, t, local, url in REFS:
        lines.append(f"| [{i}] | {t} | `{local}` | <{url}> |")
    lines.append("")
    return "\n".join(lines)
