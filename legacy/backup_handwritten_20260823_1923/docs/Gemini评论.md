# **模型训练不足，建议加训**

[https://gemini.google.com/app/09cf32f8f340368f](https://gemini.google.com/app/09cf32f8f340368f)

*User prompt: At 8000 ppl102. 10k likely \~95. At 6000 ppl117. At 5000 ppl130. Slower. At 10k maybe 80\. At 9000 ppl98. Final 10000 \~95. Chat quality maybe okay but could benefit from more training. Should we extend? 10000 steps only sees 41M tokens of 58M; one epoch not complete. Training another 5000 steps could push ppl to \~75 and improve chat. Time \~7 min per 1k? Wait throughput 18.4k, step=4096 tokens \=\>4.5 steps/sec, 1000 steps=222s \~3.7min. 5000 steps \~18.5min. Could do. User wants better chat; worth extending to 15000 or 20000\. 100M的模型这样是不是有点拉跨*

Response: **不算模型拉跨，主要是“喂饭”严重不足（训练极度不充分）。**

PPL 停在 \~95（对应 Cross Entropy Loss *L*\=ln(95)≈4.55）并不是 100M 参数模型的性能瓶颈，而是因为目前训练量连模型的“起跑线”都还没达到。

**核心瓶颈分析**

> * **数据与参数比例严重失衡** 根据大模型 Scaling Law（如 Chinchilla 原理），100M（1亿）参数的模型通常需要 **20亿 (2B) tokens** 左右才能充分收敛（Tokens : Params 最佳比例约为 20:1）。目前 10k steps 仅消耗了 **41M tokens**（比例仅 0.4:1），连 58M 的总数据集都没跑完 1 个 Epoch。  
> * **PPL 95 属于正常起步水平** Loss 在 4.5 左右意味着模型刚建立起基础的词频分布和简单语法规则，尚未进入复杂的语义表达学习阶段。在这个 Token 吞吐量下，任何 100M 架构都会在此区间。

**建议与后续操作**

> * **立即加训至 15,000 – 30,000 steps**  
  * **成本极低**：按 18.4k throughput 计算，再跑 5000 steps 仅需不到 19 分钟。  
  * **收益显著**：从 5000 步到 9000 步的 PPL 下降轨迹（130 → 117 → 102 → 98）表明模型处于**快速学习期**，尚未进入平台期。跑 2\~3 个 Epoch（120M tokens 左右）可以将 PPL 压至更低，显著改善文本生成的连贯性。  
> * **密切关注过拟合（Overfitting）**  
  * 58M 数据对 100M 模型来说极小。跑多 Epoch 时需监控 Validation Loss，若 Val Loss 开始上升而 Train Loss 继续下降，即到达过拟合拐点。  
> * **补充数据集（若要提升 Chat 质量）**  
  * 58M tokens（仅约几百 MB 文本）容量太小，模型难以掌握丰富的知识与对话模式。若想获得实用的 Chat 能力，建议至少扩充至 **500M \~ 1B tokens** 的通用与对话混合语料。

当前表现完全是“训练量太少”导致的正常现象，建议直接延长训练到 15k-20k steps。

---

*User prompt: 有没有推荐的数据集，中英文的*

Response: 针对 100M 级别的模型，建议优先选用**数据质量极高、经过严苛过滤或提炼**的数据集（小模型对噪音极其敏感，高质量语料能显著加速收敛）。

**预训练数据集（Pre-training）**

| 语种 | 数据集名称 | 适用特点与推荐理由 | Hugging Face ID / 来源 |
| :---- | :---- | :---- | :---- |
| **英文** | **FineWeb-Edu** | 当前公认质量最高的高清网页语料之一，Edu 版本经打分筛选，教育/学术与逻辑性极强 | HuggingFaceFW/fineweb-edu |
| **英文** | **SmolLM-Corpus** | Hugging Face 专门为小模型（100M\~1B）定制的预训练数据集，混合了优质合成数据 | HuggingFaceTB/smollm-corpus |
| **英文** | **SlimPajama-627B** | 经典 RedPajama 的深度去重精简版，覆盖 Wikipedia、图书、代码等通用知识 | cerebras/SlimPajama-600B |
| **中文** | **Chinese-FineWeb-Edu** | FineWeb-Edu 评分机制移植到中文语料的成果，过滤掉了大量垃圾文本 | opencsg/chinese-fineweb-edu |
| **中文** | **SkyPile-150B** | 昆仑万维开源的高质量中文预训练数据集，以高质网页文本为主 | Skywork/SkyPile-150B |
| **中文** | **Wan卷 (WanJuan-1.0)** | 上海人工智能实验室开源，经过严格清洗和敏感过滤的高质量中文语料 | OpenDataLab/WanJuan1.0 |

**指令微调与 Chat 数据集（SFT / Instruction）**

| 语种 | 数据集名称 | 适用特点与推荐理由 | Hugging Face ID / 来源 |
| :---- | :---- | :---- | :---- |
| **中文** | **COIG-CQIA** | 严格贴近真实中文用户交互（知乎、问答社区等），文本极其自然，适合增强 Chat 感觉 | m-a-p/COIG-CQIA |
| **中文** | **SmolTalk (Chinese)** | 针对小模型精炼的中文 SFT 数据集，包含常用对话与问答 | opencsg/smoltalk-chinese |
| **中文** | **BELLE / Firefly** | 经典的中文多任务指令集，覆盖总结、翻译、角色扮演等多种任务 | BelleGroup/train\_0.5M\_CN |
| **英文** | **UltraChat / UltraFeedback** | 高质量多轮对话数据集，语义丰富、逻辑清晰 | Stingning/ultrachat |
| **英文** | **SmolTalk** | SmolLM 官方使用的精炼 SFT 数据集，剔除了冗余和低质量样本 | HuggingFaceTB/smoltalk |

**实操混合建议**

> 1. **预训练扩展**：按 **5:5 或 6:4** 比例混合 FineWeb-Edu 与 Chinese-FineWeb-Edu（或 SkyPile），截取约 **1B\~2B tokens** 喂给模型，跑 10\~20 万步，PPL 会直接降到 20\~30 级别。  
> 2. **Chat 对齐**：100M 参数不宜喂太多复杂的长指令。预训练完成后，抽取 **2万 \~ 5万条** 优质的 COIG-CQIA 和 SmolTalk 进行 2\~3 个 Epoch 的微调即可获得很不错的对话连贯度。

---

*User prompt: 能再加一些对话相关的中、英文数据集吗？最好是方便下载的（如直接在HF上）*

Response: **中文对话数据集（可以直接通过 datasets.load\_dataset 加载）**

| 数据集名称 | Hugging Face ID | 类型与样本量 | 特点与推荐理由 |
| :---- | :---- | :---- | :---- |
| **COIG-CQIA** | m-a-p/COIG-CQIA | 单/多轮 QA (\~160k) | 贴近中国互联网用户真实提问风格（知乎、贴吧等），回答极具人味 |
| **ShareGPT 4 (中文)** | shibing624/sharegpt\_gpt4 | 多轮对话 (\~18k) | 真实用户与 GPT-4 对话沉淀，多轮逻辑连贯，适合训练流畅的语气 |
| **BELLE 多轮对话** | BelleGroup/multiturn\_chat\_0.8M | 多轮对话 (\~80万) | 包含大量的日常问答、角色扮演与角色对话，覆盖面极广 |
| **弱智吧精选 QA** | hfl/ruozhiba\_gpt4 | 单轮 QA (\~2.5k) | 包含逻辑陷阱、谐音梗与双关语，**专门训练小模型的“脑筋急转弯”与抗误导能力** |
| **Fineweb-Edu-CN V2.2 (SFT)** | opencsg/Fineweb-Edu-Chinese-V2.2 | 单轮精炼 QA (1.43M) | 基于高质语料萃取的问答对，事实密度极高，能让小模型回答更有干货 |

**英文对话数据集**

| 数据集名称 | Hugging Face ID | 类型与样本量 | 特点与推荐理由 |
| :---- | :---- | :---- | :---- |
| **UltraChat 200k** | HuggingFaceH4/ultrachat\_200k | 多轮对话 (200k) | 经过严格过滤的优秀英文多轮 Chat 数据，语气自然且指令类型丰富 |
| **OpenAssistant (OASST1)** | OpenAssistant/oasst1 | 多轮树状对话 (\~161k) | 真实人类标注的 Assistant 问答，覆盖多种对话风格与安全边界 |
| **SmolTalk (Everyday)** | HuggingFaceTB/smoltalk | 多轮与单轮混合 | **专为 100M\~1B 小模型设计**，包含打招呼、日常闲聊与实用工具调用的精炼对话 |
| **OpenHermes 2.5** | teknium/OpenHermes-2.5 | 指令与对话混合 (\~1M) | 综合性极强的提纯数据集，包含大量解释性对话与推理过程 |

**100M 小模型微调的“对齐配方”建议**

对于 100M 参数量的模型，**数据不在多，而在精**。过多复杂的长文回复容易导致模型产生严重幻觉或断句异常。

> * **数据量配置**：总计提取 **3万 \~ 8万条** 即可（例如 3 万条 COIG-CQIA \+ 2 万条 UltraChat \+ 5000 条 ShareGPT \+ 2000 条弱智吧）。  
> * **格式过滤**：筛选回答长度在 **128 \~ 512 tokens** 之间的对话，截断或丢弃太长（\>1024 tokens）的深度推导回答。  
> * **快捷加载示例**：  
>   `from datasets import load_dataset`

>   `# 加载 COIG-CQIA (中文)`  
>   `coig = load_dataset("m-a-p/COIG-CQIA", split="train")`

>   `# 加载 UltraChat (英文)`  
>   `ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")`

**Sources**

> 1. [opencsg/Fineweb-Edu-Chinese-V2.2 · Datasets at Hugging Face](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.2)

---

*User prompt: 只有8G的显存能整吗？*

Response: **完全可以，甚至可以说是“绰绰有余”。**

对于 100M（1 亿）参数量的模型，8G 显存不仅能做指令微调（SFT），即便做全参数预训练（Pre-training）也毫无压力。

**8G 显存运行 100M 模型的显存开销拆解**

| 占用项 | FP16 / BF16 混合精度 | 备注 |
| :---- | :---- | :---- |
| **模型权重 (Weights)** | \~0.2 GB (200MB) | 108×2 Bytes |
| **梯度 (Gradients)** | \~0.2 GB (200MB) | 与权重相当 |
| **优化器状态 (AdamW)** | \~0.8 GB (800MB) | 标准 32-bit AdamW 状态 |
| **激活值与上下文 (Context)** | \~1.0 – 2.0 GB | Batch Size \= 8/16, Context \= 1024 |
| **框架与系统预留** | \~1.0 GB | PyTorch 开销 |
| **实际总显存峰值** | **仅约 2.5 GB – 4.0 GB** | 仅占 8GB 显存的一半左右 |

**8GB 显存实操推荐配置**

在使用 Hugging Face SFTTrainer 或 Trainer 时，建议采用以下参数组合，以兼顾训练速度与稳定度：

> * **混合精度**：开启 bf16=True（NVIDIA RTX 30/40 系列卡推荐）或 fp16=True（RTX 20 系列卡）。  
> * **批次大小 (Batch Size)**：  
  * 单卡设置：per\_device\_train\_batch\_size \= 16  
  * 梯度累积：gradient\_accumulation\_steps \= 2 或 4  
  * 相当于等效总 Batch Size 为 32 \~ 64，非常利于收敛。  
> * **序列长度 (Max Length)**：设置为 1024 或 2048。  
> * **加速插件 (可选)**：如果安装了 flash-attn，开启后显存和速度还会再优化 20%\~30%。

8G 显存唯一的瓶颈是**无法支撑过大的 Batch Size 或极长文本（如 8k/16k context）**，但在 1024\~2048 上下文的常规对话训练中，完全可以放开手脚去跑。

---

*User prompt: 但是还要算上数据集加载的显存啊 如果都算上的话，整个预训练，最大能跑到多少参数量？*

Response: 数据集加载**主要占用的是电脑的主内存 (System RAM)**，而不是显卡上的显存 (VRAM)。

通过 DataLoader 加载数据时，数据是在 CPU 和内存中读取、洗牌、分批（Batching）的，显存里**只会存放当前正在计算的那一小批（Mini-batch）数据**。这一批文本（例如 16 个样本 × 1024 Token）在显存里只占几兆（MB），对 8GB 显存几乎可以忽略不计。

**8G 显存全参数预训练的“上限”计算**

在排除数据集的影响后，决定显存上限的只有**模型参数**、**梯度**、**优化器状态（Optimizer States）** 以及 **前向传播激活值（Activations）**。

以常见的 **FP16 / BF16 混合精度 \+ Standard AdamW** 为例，训练每个参数平均需要 **16 字节 (Bytes)** 的静态显存开销：

> * 主权重 (FP32) \+ 显存模型 (FP16) \+ 梯度 (FP16) \+ AdamW 动量与方差 (FP32) \= 4 \+ 2 \+ 2 \+ 8 \= **16 Bytes/参数**

扣除系统与 CUDA Context 预留的 \~1.5GB 显存，不同优化策略下的预训练上限如下：

| 训练配置 / 优化策略 | 单参数静态显存 | 8G 显存预训练上限参数 | 算力与体验评估 |
| :---- | :---- | :---- | :---- |
| **原生 AdamW (BF16/FP16)** | \~16 Bytes | **\~300M \- 350M** | **最顺畅**：显存留白充足，可以使用较大 Batch Size 跑满 GPU 算力。 |
| **8-bit AdamW \+ 梯度检查点** | \~10 Bytes | **\~500M \- 600M** | **性价比最高**：利用 bitsandbytes 将优化器压到 8-bit，可轻松跑 0.5B 模型。 |
| **ZeRO-Offload (优化器卸载至 CPU)** | \~4 Bytes (GPU) | **\~1.0B \- 1.5B** | **极限吃满**：将 AdamW 状态全部塞进系统内存，但 PCIe 传输会成为速度瓶颈。 |

> * 家用 8G 显卡（如 RTX 3060/4060）跑全参数预训练的最佳甜蜜点（Sweet Spot）是 **300M \~ 500M 参数**（例如 Qwen2.5-0.5B、SmolLM-360M 这种架构）。  
> * 如果只是**微调 (SFT)**，因为不需要维护大量的预训练激活值，开启 LoRA 甚至可以微调 **7B \~ 8B** 模型。

---

*User prompt: 我只有一张3070 8GB，跑刚才的那个100M的模型预训练，你估计要跑多久？*

Response: 用一张 RTX 3070 8GB 预训练 100M 模型**非常快**。按大模型充分收敛标准的 **2B (20亿) Tokens** 计算，全过程仅需 **12 \~ 30 小时**（约 0.5 到 1.2 天）。

**RTX 3070 预训练 100M 模型耗时预估**

| 训练目标 (Tokens) | 训练阶段与定位 | 默认速度 (\~18.4k tokens/s) | 深度优化后 (\~45k tokens/s) |
| :---- | :---- | :---- | :---- |
| **200M (2亿)** | 当前 58M 数据跑 3\~4 个 Epoch | **\~3 小时** | **\~1.2 小时** |
| **2B (20亿)** | **Chinchilla 最佳收敛点**（建议目标） | **\~30 小时 (1.2天)** | **\~12 小时** |
| **5B (50亿)** | 充分“过训练”（语法与知识极其扎实） | **\~75 小时 (3.1天)** | **\~31 小时 (1.3天)** |

**3070 跑小模型提速关键点**

100M 参数模型的计算量极小（每 Token 仅需 0.6 GFLOPs），算力往往不会卡在 GPU 显存或计算核心上，而是卡在 **CPU 调度与 GPU 内核启动开销（Kernel Launch Overhead）**。

> * **开启 torch.compile(model)**：小模型提升最明显的武器，通过图编译将大量小算子融合，吞吐量通常能直接提升 30%\~50%。  
> * **原生 BF16 \+ FlashAttention-2**：RTX 3070 (Ampere 架构) 原生支持 BF16，搭配 PyTorch 的 attn\_implementation="sdpa"，显存带宽占用大幅降低。  
> * **拉满 DataLoader 读取**：由于 GPU 算得太快，CPU 喂数据容易跟不上。设置 num\_workers=4 或 8 并开启 pin\_memory=True，避免 GPU 挂起等待数据。

直接准备 2B 左右的混合文本（如 FineWeb-Edu \+ Chinese-FineWeb-Edu），挂在后台跑一天即可完成整个 100M 模型的完整预训练。

---

