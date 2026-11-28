# ILVAD 复现与改进实验 · 交接文档

> 面向接手者（人或模型）。本文档自包含：读完即可知道**做过什么、结论是什么、代码在哪、怎么复跑**。
> 最后更新：2026-11-28

---

## 0. 一页速览

**课题**：复现并改进 ILVAD（arXiv:2605.20965）——一种 training-free 的推理期注意力干预方法，用于缓解多模态大模型的物体幻觉。

**做了什么**：在 RTX 3090 上用**未改动的官方代码**复现了 ILVAD，然后在 POPE / CHAIR / LLaVA-Wild 三个基准上做了 6 组对照实验，包括 4 个自研改进方向的完整消融。

**最重要的 5 个结论**：

1. **方法有效但增益小**：POPE +2.20（1000 题，p=0.0013）、+2.80（500 题，p=0.007）。
2. **α 是一道悬崖**：2.5 无效、5 最优、7.5 崩盘（−15）、10 更惨（−30.8）。论文未讨论。
3. **增益机制是"压假阳性"**：precision +5.4、recall 不动。空白图实验证明它**确实在看图**（图像贡献 36.4 点）。
4. **四个改进方向里只有 D 有效**：只在决策点干预 → CHAIR 复读率 47.2%→19.2%，代价是 CHAIR_S 16.34→19.87。
5. **ILVAD 的主要副作用（复读/早停）来自"每步都干预"，不是"注意力干预"本身。**

**没做的事**：ASCD（AAAI 26）的对照实验；POPE 上的 D 结果（代码卡死）；MMHal-Bench；多 seed。

---

## 1. 环境与代码位置

### 服务器
```
主机：222.29.51.157（用户 liruixiang，SSH 免密已配置）
GPU：10 × RTX 3090 24GB（共享机，用前先 nvidia-smi 找空闲卡）
驱动 525.105.17 → 只能用 cu118 的 torch
```

### 目录（**重要：不要动官方代码**）
```
/home/liruixiang/ilvad/                     # 代码
  ├── official/                             # 官方仓库，一行未改（MD5 可校验）
  │   ├── main.py  ilvad.py  utils.py  load_data.py
  │   ├── eval/                             # chair.py / eval.py / cocoeval.py / llava_wild.py
  │   ├── data/
  │   │   ├── coco/val2014/                 # 667 张图（500 CHAIR + 167 POPE）
  │   │   └── coco/annotations/             # instances/captions_val2014.json
  │   ├── out_*/                            # 所有实验输出（见 §2）
  │   ├── main_ab.py                        # 自研扩展：A/B 预算
  │   └── main_abcd.py                      # 自研扩展：A/B/C/D + 空白图
  ├── run_server.py                         # 我自己的 notebook 实现的服务器版（另一条线）
  └── *.sh                                  # 各次实验的启动脚本

/data/liruixiang/ilvad_repro/               # 重资产（/home 只剩 50GB，不要往那写大文件）
  ├── envs/ilvad/                           # conda 环境（python 3.10 + torch 2.1.2+cu118 + transformers 4.45）
  ├── models/llava-1.5-7b-hf/               # 14 GB fp16
  ├── data/pope/*.parquet                   # POPE 三个 split
  ├── hf/                                   # HF 缓存
  └── logs/                                 # 所有运行日志
```

### 环境要点
```bash
PY=/data/liruixiang/ilvad_repro/envs/ilvad/bin/python
export HF_HOME=/data/liruixiang/ilvad_repro/hf
export TRANSFORMERS_OFFLINE=1
export DISABLE_TORCH_COMPILE=1     # 必须！否则动态 patch 会触发逐样本重编译，慢 15-40×
export CUDA_VISIBLE_DEVICES=<空闲卡>
```

**踩过的坑**：
- `numpy 2.x` 与 torch 2.1.2 不兼容 → 必须 `<2`
- hf-mirror 的 API 有 429 限流 → 数据一律本地化，不要用 `load_dataset(streaming=True)`
- 官方 `utils.py` 自带 `DISABLE_TORCH_COMPILE` 开关（不是我们加的）

---

## 2. 数据与评测协议

### 数据
| 数据 | 位置 | 规模 |
|---|---|---|
| POPE random/popular/adversarial | `data/pope/*.parquet` + 导出成 `data/coco/val2014/*.jpg` | 各 3000 题 |
| POPE 子集（本轮用的） | `data/pope/coco/coco_pope_random{40,500,1000}.json` | 前 N 题 |
| CHAIR | COCO val2014 500 张（`random.seed(0); random.sample`） | 500 图 |
| LLaVA-Bench-in-the-Wild | `data/llava-bench-in-the-wild/` | 60 题 / 24 图 |

### 评测协议
- **POPE**：官方规则 = 取回答第一句 → 按空格切词 → 出现 `No/no/not/Not` 判否，否则判是。输出含 `gt` 字段，直接算 acc/P/R。
- **CHAIR**：**官方评测器跑不起来**（缺 nltk punkt，GitHub 被墙、镜像也拉不到），因此用**自实现近似**：
  - 用 `eval/synonyms.txt`（80 个 COCO 类别 → 同义词表）
  - 用 WordNet 词形还原（`~/nltk_data/corpora` 里有 wordnet）
  - GT 用 `instances_val2014.json` 的类别集合
  - `CHAIR_S = 100 × Σ|幻觉物体| / Σ|提及物体|`，`CHAIR_I = 100 × Σ|幻觉物体| / 图数`
  - ⚠️ **绝对值会比官方略高**（缺 PTB 分词），**只能比相对变化**
- **LLaVA-Wild**：官方用 GPT-4o 打分，我们没有 API key → 只报**长度 / 复读率 / 截断率**

---

## 3. 方法清单（含公式与开关）

约定：`a_ij` = 第 i 个 query 对第 j 个视觉 token 的注意力；`Ŝ_j ∈ [0,1]` = 证据图；`α` = 强度。

### M0 · greedy（基线）
不干预。`out_greedy/`

### M1 · ILVAD（官方原版）
**式**：`a_ij ← a_ij · exp(α·Ŝ_j)`，行归一化。**只改"正在预测下一个 token"的那一行**（官方 `attn_weights[0,:,-1]`）。
**图**：前 T=10 步的注意力 → 层内 top-50% 头平均 → `Ā > τ·mean` 二值化 → 跨层正向差分计数 → min-max 到 [0,1]。
**开关**：`--method ilvad --alpha 5 --tau 5 --T 10 --layer_to_enhance 0..30`
**⚠️ 复现警告**：我最初把自己的实现写成"改所有注意力行"，结果符号反转（+2.8 → −3.2）。**必须只改一行。**

### M2 · 空白图基线（诊断用）
`--blank_image 1`：把 `pixel_values` 置零，测纯语言先验。
**结果**：POPE acc 50.00%，yes-rate 0.0% → 模型对所有问题都答 "No"，图像贡献 36.4 点。

### M3 · A：预算化干预
**动机**：α 是自由参数且尺度依赖，换成"注意力偏移预算 ε"。
**做法**：在 α 网格上解 `KL(q(α)‖p) ≤ ε` 的最大 α，用它替换固定 α。
**开关**：`--budget_mode A --eps 1.0`
**结果**：ε=1.0 → 选出 α 均值 4.23 → +1.40（p=0.12，不显著）；ε=2.5 → **每题都撞上限 12** → −32.40。
**结论**：**A 不成立**，它只是把 α 换了名字，悬崖依旧。

### M4 · B：支撑集下界
**动机**：坍缩的本质是有效支撑集太小。
**做法**：约束倾斜后每层 `perplexity = exp(H) ≥ ρ × 基线 perplexity`，取满足约束的最大 α，并与官方 α 取 min（作为上限）。
**开关**：`--budget_mode B --kratio 0.5`
**结果**：α 均值 3.24（上限锁在 5，不再出现 12）→ +1.80（p=0.022），**break 只有 2 条**（α=5 是 5 条）。
**结论**：**B 有稳健性收益但准确率不如 α=5**。基线 perplexity 实测均值 37.8（19.2—143.1）——所以 `kmin` 必须用相对比例，绝对设 60 会让 α 恒为 0。

### M5 · C：Query 条件化（null-prompt 减法）
**动机**：null-prompt 下 top-10 token 重叠 84—92%，说明大部分被看的 token 与问题无关（sink）。
**做法**：`Ŝ_final = relu(Ŝ(q, image) − λ·Ŝ(q_null, image))`，再走原流程。
**开关**：`--lam 1.0 --null_prompt ""`
**结果**：+0.80（p=0.34），比 α=5 低 2 点，yes-rate 反弹到 51.2%。
**结论**：**C 反而更差**。空问题下模型仍在看主要物体，相减把真证据也削掉了。

### M6 · D：决策点门控（**唯一有效的改进**）
**动机**：幻觉产生于"输出物体名词"那一刻，全程干预会让偏置在 KV 缓存里累积。
**做法**：用 `LogitsProcessor` 每步检查刚生成的词，若是 `a/an/the/no/any/some/:/−/is/are` 或为空则**打开**干预，否则把 `enhancer.ilvad.layers_to_modify` 置空**关闭**。
**开关**：`--gate 1`
**结果**（CHAIR）：复读 47.2%→**19.2%**，词数 68.9→**74.1**，CHAIR_S 16.34→19.87。
**结论**：**用 3/4 的幻觉抑制换回接近正常的文本**。长度匹配后的真增益：D −5.42 vs α=5 −7.30。
**⚠️ 关键坑**：`ilvad.py` 的状态挂在 `EnhanceEvidenceAttention` **实例**上，改模块属性完全无效（现象：门控计数器正常但 500 条输出逐位相同）。必须改 `enhancer.ilvad.layers_to_modify`。

### M7 · 同学的四条线（未在本轮复跑，结论来自其报告复核）
| 变体 | 做法 | 结论 |
|---|---|---|
| Idea 1 Prefill | 用 prompt 尾部 K=6 个 token 建图，省一遍 prefill | 速度真、CHAIR 提升主要是长度红利 |
| Idea 2 Value/Sink | `A×‖V‖` 加权 + null-prompt 去 sink + top-p | 机制成立，最终收益不稳定 |
| Idea 3 ECD | 空图对比解码 + APC | CHAIR 最好但 POPE 掉 4.2（yes-rate 冲到 71%） |
| v3 | 单遍 + 在线 handoff + 低熵门控更新 | 长度未控，增益存疑 |

### M8 · ASCD（**仅文献，未跑**）
AAAI 2026，[arXiv:2506.14766](https://arxiv.org/abs/2506.14766)，代码 <https://github.com/BroJunn/ASCD>。
**做法**：(i) 正转向——自动挖掘稳定的 text-centric head 并提升它们；(ii) 负转向——压制"关键视觉 token"；两者 logits 做**对比解码**。
**为什么值得做**：它是**双侧有界 + 对比解码**，从设计上避开了我们实测的两个病灶（无界倾斜、输出塌缩）。
**注意**：需要先做 head profiling（模型特定），不是纯零样本。

---

## 4. 结果总表

### POPE（random 前 500 题，同一批）

| 配置 | acc | yes-rate | precision | recall | 平均词数 | 净变化 | fix/break | p |
|---|---|---|---|---|---|---|---|---|
| **greedy** | 86.40% | 50.8% | 0.858 | 0.872 | 18.5 | — | — | — |
| greedy 空白图 | 50.00% | 0.0% | — | 0.000 | 19.3 | −36.40 | — | <0.001 |
| ILVAD α=2.5 | 86.80% | 51.6% | 0.857 | 0.884 | 17.7 | +0.40 | 3/1 | 0.625 |
| **ILVAD α=5** | **89.20%** | 47.6% | **0.912** | 0.868 | **5.5** | **+2.80** | 19/5 | **0.007** |
| ILVAD α=7.5 | 71.40% | 75.8% | 0.641 | 0.972 | 1.9 | −15.00 | 33/108 | <0.001 |
| ILVAD α=10 | 55.60% | 92.4% | 0.530 | 0.980 | 5.2 | −30.80 | 33/187 | <0.001 |
| A ε=1.0 | 87.80% | 49.8% | 0.880 | 0.876 | 12.2 | +1.40 | 11/4 | 0.118 |
| A ε=2.5 | 54.00% | 95.2% | 0.521 | 0.992 | 4.7 | −32.40 | 34/196 | <0.001 |
| B ρ=0.5 | 88.20% | 49.8% | 0.884 | 0.880 | 13.1 | +1.80 | 11/2 | 0.022 |
| C λ=1.0 | 87.20% | 51.2% | 0.863 | 0.884 | 10.1 | +0.80 | 7/3 | 0.344 |
| D 门控 | 89.20%* | 47.6% | 0.912 | 0.868 | 5.5 | +2.80* | 19/5 | 0.007* |

\* D 在 POPE 上的这行数字来自**门控未生效的旧版**（逐位等于 α=5），**不可采信**，见 §5。

### CHAIR（seed 0，500 图）

| 配置 | 平均词数 | 复读率 | 截断率 | CHAIR_S ↓ | CHAIR_I ↓ |
|---|---|---|---|---|---|
| greedy | 90.8 | 12.8% | 0.0% | 29.77 | 91.80 |
| ILVAD α=5 | 68.9 | 47.2% | 5.0% | 16.34 | 31.20 |
| **D 门控（修正后）** | **74.1** | **19.2%** | 6.0% | 19.87 | 43.40 |

**长度匹配后的真增益**：`greedy@68词 = 23.64`（α=5 −7.30）；`greedy@74词 = 25.29`（D −5.42）。
→ **ILVAD 的 CHAIR 增益里约一半是长度红利。**

### LLaVA-Wild（60 题）

| 配置 | 平均词数 | 最长 | 复读率 | 截断率 |
|---|---|---|---|---|
| greedy | 89.5 | 307 | 21.7% | 1.7% |
| ILVAD α=5 | 100.6 | 509 | **43.2%** | **31.8%** |

---

## 5. 已知问题（接手必读）

1. **POPE + 修正后门控会卡死**：加载模型后无任何输出，150s/240s/3天都试过。**同一份代码、同一组参数在 CHAIR 上正常**。原因未定位。
   - 已排除：数据集缺失、图片损坏（`--gate 0` 同数据正常）
   - 待查方向：`LogitsProcessor` 在 `max_new_tokens=32` 的短序列下与 `output_attentions=True` 的首轮 pass 的交互
   - **影响**：D 在 POPE 上是否保持 +2.8 未知
2. **CHAIR 指标是自实现近似**（无 PTB 分词），绝对值偏高，只能比相对变化
3. **D 的触发词是启发式**，未用词性标注；实际干预比例 66%（偏多）
4. **单 seed**：CHAIR 只跑了 seed 0，POPE 只跑了 random 的前 500/1000 题
5. **α 悬崖**：任何换模型/换精度/换层区间的复现都要重新扫 α，**不要直接用 5**
6. **`/home` 只剩约 50 GB**，大文件一律放 `/data/liruixiang/ilvad_repro/`

---

## 6. 建议的下一步（按性价比）

1. **修 POPE 的 D 卡死** → 补上 D 在 POPE 上的数字（决定 D 是"稳赚"还是"权衡"）
2. **跑 ASCD**（代码开源，半天）→ 唯一能回答"谁更强"的实验
   - 建议同时把**负转向单独拆出来**（只压制关键视觉 token），那是与我们的 C 最直接可比的部件
3. **补多 seed**（至少 3 个）→ 现在所有结论都是单 seed
4. **换成官方 CHAIR 评测器**（需解决 nltk punkt；可从内网镜像或手动放置 punkt.zip 到 `~/nltk_data/tokenizers`）
5. **把 D 的门控换成词性标注**（nltk `pos_tag` 或轻量规则）→ 看是否能进一步降复读

---

## 7. 文件索引

| 路径 | 内容 |
|---|---|
| `official/main.py` 等 | 官方代码，**未改动**（可 MD5 校验） |
| `official/main_ab.py` | A/B 实验（预算化 + 支撑集下界），`--budget_mode {none,A,B,AB}` |
| `official/main_abcd.py` | 全部扩展：A/B + C(`--lam`) + D(`--gate`) + 空白图(`--blank_image`) |
| `official/out_*/` | 所有输出 JSON（每行 `{question_id, text, gt, question}`） |
| `logs/*.log` | 运行日志（含 `chosen alpha` / `gate on/off` 等诊断） |
| `output/colab/docs/RESULTS_SUMMARY.md` | 精简结果表（GitHub 上） |
| `output/colab/docs/IMPROVEMENT_IDEAS.md` | 十条改进方向与本轮实测的对应 |
| `output/colab/docs/AB_CD_RESULTS.md` | A/B/C/D 的单独记录 |
| `output/colab/ILVAD_repro.ipynb` | 我自己的独立实现（Colab 版，可作交叉验证） |
| `output/pdf/ASCD_*.pdf` | ASCD 原文 |

---

## 8. 一句话交接

> 这个项目已经证明了 **ILVAD 有效但极脆**，并且找到了它副作用的具体来源（每步干预导致 KV 缓存累积污染）。
> 下一步最有价值的不是再调参，而是**在同等条件下与 ASCD 做一次公平对比**，并补上多 seed。
