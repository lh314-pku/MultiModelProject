# MultiModelProject

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lh314-pku/MultiModelProject/blob/main/ILVAD_repro.ipynb)

LVLM 幻觉缓解方法 **ILVAD**（*Finding the Correct Visual Evidence Without Forgetting*，ICML 2026，arXiv:2605.20965）的复现工程，外加一个拓展：**把「证据显著图」从注意力里蒸成轻量探针**（下称方向 B）。

仓库里同时放了配套的三份中文分析报告（`reports/`），从方法脉络讲到复现预算，再到下一步能改哪里。

## 仓库内容

| 路径 | 内容 |
|---|---|
| `ILVAD_repro.ipynb` | 主 notebook：加载 → 取注意力 → 式(4)—(10) → 评测 → 探针训练 |
| `tests/test_math.py` | CPU 单测，直接解析 notebook 里的函数，不需要 GPU / 模型 |
| `reports/ILVAD_技术路线与前置知识.pdf` | 16 页：方法怎么一步步推出来，读懂它需要哪些前置知识 |
| `reports/ILVAD_改进方向与技术分析.pdf` | 9 页：三个候选改进方向、相关论文、数学分析、Colab 可行性 |
| `reports/ILVAD_论文分析与复现硬件评估.pdf` | 结果可信度、代码核验、显存与耗时预算 |
| `requirements.txt` | 依赖（`transformers==4.45.0`，与论文官方仓库一致） |

> 中文全译稿未放入仓库：它是论文的衍生作品，公开再分发有版权风险。需要的话请放在本地或私有仓库。

## 一句话总结方法

幻觉往往来自「注意力没落在正确答案对应的图像区域」。ILVAD 用**相邻层之间由不显著变为显著的次数**来定位证据区域，
再把注意力按指数倾斜（等价于给视觉 token 加 logit 偏置）压回这些区域 —— 训练-free、即插即用。

```
式(4)  Ā[l,j] = 1/(T·|H_v|) · Σ_h Σ_{i≤T} A[l,h,i,j]
式(5)  Ã[l,j] = 1[ Ā[l,j] > τ · mean_j Ā[l,:] ]
式(6)  S[j]   = Σ_l max( Ã[l+1,j] − Ã[l,j], 0 ),   Ŝ = S / max(S)
式(8)  Â[i,j] = A[i,j] · exp(α·Ŝ[j]),      j ∈ X_v
式(10) Â[i,j] = A[i,j] · (ŵ[i] + β),       j ∈ X_t
```

## 这个 notebook 做了什么

```
Stage 0  用 eager attention 读出「前 T 个生成 token」对各视觉 token 的注意力
Stage 1  式(4)—(6)：按视觉敏感头平均 → 层内阈值二值化 → 跨层正向差分 → 归一化成 Ŝ
Stage 2  式(7)—(10)：视觉侧 A·exp(αŜ)，文本侧 A·(ŵ+β)，逐行归一化后写回
         （补丁式替换 attention forward，附带「关掉干预应与原模型一致」的等价性自检）
评测     yes/no 冒烟测试、AOPC（证据图是否 faithful）
拓展 B   用 (第 k 层隐状态, Ŝ) 训练两层 MLP，推理时不再物化注意力
扩展     τ 扫描 / POPE 真图对照 / 探针跨图验证（第 10 节）
```

## 状态

| 部分 | 状态 |
|---|---|
| 核心数学（式 4—6、重加权算子、因果掩码） | ✅ 单测通过（`tests/test_math.py`，对照手算样例） |
| fp16 / fp32 混合精度路径 | ✅ 单测通过（行和为 1，可参与 matmul） |
| 端到端流程（加载 7B、取注意力、生成） | ✅ 2026-11-23 在 T4 上跑通（见下方运行记录） |
| 探针（方向 B） | ✅ 管线跑通，但同场景数据下的精度不可信（见下文） |
| 论文表格复现 | ❌ 未做（需要 CHAIR 全量，免费 Colab 跑不完） |

> 本仓库是**机制复现**，不是论文数字复现。4bit 量化后的注意力与论文 fp16 不可直接比，只能比较 baseline 与 ILVAD 的差值。

## 首次运行记录（Colab T4 16 GB，2026-11-23）

环境：`Tesla T4`、`torch 2.14.1+cu130`、`transformers 4.45.0`、LLaVA-1.5-7B 4bit（nf4）。

| 检查项 | 结果 | 解读 |
|---|---|---|
| 视觉段 | `[5, 581)`，576 个 token | 与 `image_seq_length` 一致；起点取决于 chat template |
| 生成窗口 | 10 步，注意力形状 `(32, 32, 576)` | 32 层 × 32 头，符合 LLaVA-1.5 |
| 等价性自检 | `max |Δlogit| = 7.8e-3` → PASS | 4bit 下重算注意力带来的数值噪声，不是逻辑错误 |
| 证据图稀疏度 | 66/576 非零，`S ∈ [0, 7]` | `τ=5` 在这张图上合理，没有全空 |
| **AOPC** | **ILVAD 0.161 vs 随机 0.011（14.7×）** | 按图遮 token 掉的 margin 远多于随机，说明地图真的落在证据上 |
| mask 自检 | margin 3.555 → 0.047 | 遮掉全部视觉 token 后 margin 崩塌，确认 mask 通路有效 |
| 探针 val_MAE | 0.021 | ⚠️ **不可信**：训练/验证集都是同一张合成图的微扰，模型只是背下了这张图 |

最后一行是这次运行里唯一需要特别提醒的地方：**0.021 的 MAE 是过拟合，不是泛化**。要真正验证方向 B，训练集必须换成几百张不同的图（COCO 或 POPE 的图像），并且验证集不能与训练集同源。

AOPC 的 14.7× 也只来自**一个样本**，只能说明管线通了、地图方向对，不能当结论。

### 扩展实验（第 10 节，同一批环境）

| 实验 | 结果 | 解读 |
|---|---|---|
| τ 扫描（合成图） | τ=2 → −2.6×；τ=3 → 4.0×；**τ=5 → 14.7×**；τ=8 → 9.9× | 单峰：太小则图太脏（171/576 非零，遮掉反而更自信），太大则丢证据（41/576）。**论文默认 τ=5 得到支持**，两次运行逐位一致 |
| **α 扫描（POPE random n=40）** | baseline **85%** / α=2 **88%** / α=3 **88%** / α=5 **80%** | **α=5 是负收益**：四次翻转全是 yes→no；α=2—3 净 +1。α 与 yes-rate 呈倒 U（40% → 52.5% → 40%）。默认强度对这个 4bit 模型过猛 |
| 探针跨图（24 张 POPE 图） | MAE 0.0379 / **全零基线 0.0381** / top-30 IoU **0.280** / 相关 **0.601** | MAE 几乎等于「什么都不预测」，但 IoU 是随机（≈0.05）的 5.6 倍、相关 0.60：**探针抓到了大致位置，抓不到精确的 30 个 token**，还不足以替代教师 |

三条结论一次比一次具体：τ 选对了、α 选错了、探针方向对但精度不够。α 的结果说明**问题在干预强度而不是证据图**。

三个结论都只建立在 1 张合成图 + 20 条 POPE 上，**方向可用、数值不可引用**。

## 快速开始

### Colab

1. 用 Colab 打开本仓库的 `ILVAD_repro.ipynb`；
2. 运行时 → 更改运行时类型 → **T4 GPU**；
3. 从上往下跑。第 0 节的 `pip install` 若提示需要重启运行时，重启后从第 1 节继续。

### 本地

```bash
pip install -r requirements.txt
jupyter lab ILVAD_repro.ipynb
python tests/test_math.py        # 不需要 GPU
```

需要一张至少 8 GB 显存的 NVIDIA 卡（4bit 7B）；纯 CPU 不可行。

## 关键配置（notebook 第 0 节的 `Cfg`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `load_4bit` | True | nf4 量化，7B 压到约 4.5 GB |
| `window_T` | 10 | 式(4) 的窗口，取前 T 个生成 token |
| `tau` | 5.0 | 式(5) 的阈值倍数（图全空就降到 2—3） |
| `alpha` | 5.0 | 式(8) 的视觉增强强度 |
| `beta` | 1.0 | 式(10) 的文本保底系数 |
| `rho` | 0.5 | 保留的注意力头比例 |
| `probe_layer` | 16 | 方向 B 取第几层隐状态 |

## 怎么判断「跑对了」

1. **视觉段定位**：`[V0, V1)` 的长度应等于 576（LLaVA-1.5-336、patch 14）。
2. **等价性自检**：`max |Δlogit|` 应在 1e-3 以下（通常 ~1e-5）。FAIL 时后面所有干预段会自动跳过。
3. **证据图**：合成图上的 Ŝ 应该集中在红球，而不是均匀撒在草地或拍子上。
4. **AOPC**：ILVAD 的 AOPC 应显著高于随机图（随机基线的期望是 `(k/|V|)·margin`）。

## 已知限制

- **慢**：为可读性关掉了 KV cache，生成长文本时是 O(n²)。要快请自行加 cache。
- **4bit 漂移**：只比差值，不比绝对值。
- **合成图**：第 2 节自带一张可控合成图（真值已知），用于验证管线；接 POPE / COCO 需要自己补数据适配（见第 7 节末尾的代码注释）。
- **只支持 Llama 系 attention**：Qwen2-VL 等 GQA + 不同 RoPE 接口的模型要另写补丁。
- **未做 CHAIR 全量**：需要几千次长文本生成，超出免费额度。

## 下一步（按性价比）

- [ ] 扫 `tau` / `window_T` / `alpha`，看证据图与 AOPC 的敏感度（论文表 5、表 8 的微型版）
- [ ] 加「随机图 / sink 图」对照，确认收益来自证据而不是整体加大视觉权重
- [ ] 探针改成端到端（冻结 LVLM，只用下游 margin 反传）
- [ ] 换 Qwen2-VL-2B 验证跨架构可迁移性

## 引用

```bibtex
@inproceedings{xie2026ilvad,
  title     = {Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy},
  author    = {Xie, Yutong and Hua, Zhenglin and Wang, Ran and Ng, Wing W. Y. and Wang, Xizhao and Jia, Yuheng},
  booktitle = {ICML},
  year      = {2026}
}
```

## License

代码部分 MIT；论文、图表与报告文字版权归原作者。
