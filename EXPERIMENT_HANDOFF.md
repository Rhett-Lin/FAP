# 实验交接说明：FAP + Robust TAC

本文档用于在另一台服务器上继续当前实验。新服务器仍然需要遵守 `/work1/zixuan/AGENTS.md` 中的服务器使用限制：不要在 `/home/zixuan` 下写项目、数据、环境、缓存或输出；所有实验相关内容放在 `/work1/zixuan` 下。

## 一句话目标

我们想在 FAP 仓库中实现一个独立的 **Robust TAC** 实验管线：把 TAC 的外部文本语义引导接入到 CLIP 聚类中，并加入 label-free 对抗一致性训练，比较 Static、VPrompt、TPrompt 以及 FAP-style adversarial-aware weighting 在 clean/adversarial clustering 上的表现。

## 研究目的

原始 FAP 关注 few-shot adversarial prompt learning；TAC 关注无类别名先验的 external textual guidance 图像聚类。我们当前工作的目标是把二者的想法结合起来，但不直接改原始 FAP 主训练入口。

具体想验证：

1. 在无监督聚类场景中，TAC 的文本 counterpart 和 noun anchor 是否能帮助 CLIP 图像聚类。
2. 在对抗扰动下，clean image assignment 和 adversarial image assignment 是否可以通过 clean-adversarial consistency 保持稳定。
3. 视觉 prompt（VPrompt）或文本 prompt（TPrompt）是否能在冻结 CLIP backbone 的情况下提升鲁棒聚类。
4. FAP-style weighting，即用 clean/adv 特征相似度加权 external-guided KL，是否能改善 adversarial robustness。
5. 在 CIFAR-10 和 Caltech101 上，比较不同组合：
   - CA only
   - CA + EG text assignment
   - CA + EG anchor KL
   - CA + EG + FAP weight
   - TPrompt symmetric EG

## 重要原则

Robust TAC 代码是隔离实验模块，不应破坏原始 FAP 复现路径。

已明确保持不修改：

- `train.py`
- `trainers/fap.py`
- `attack/pgd.py`
- `datasets/caltech101.py`
- FAP 原始 trainer/loss/attack/evaluation 主流程

新增和改动主要集中在：

```bash
/work1/zixuan/projects/FAP/experiments/robust_tac
```

## 当前工作区

主路径：

```bash
WORK_ROOT=/work1/zixuan
FAP_DIR=/work1/zixuan/projects/FAP
TAC_DIR=/work1/zixuan/projects/2024-ICML-TAC
DASSL_DIR=/work1/zixuan/projects/Dassl.pytorch
FAP_ENV=/work1/zixuan/envs/conda_envs/fap
DATA_ROOT=/work1/zixuan/data/fap
OUTPUT_ROOT=/work1/zixuan/projects/FAP/outputs/robust_tac
```

服务器限制文件：

```bash
/work1/zixuan/AGENTS.md
/work1/zixuan/projects/FAP/AGENTS.md
```

## 我们已经做的主要改动

### 1. 新增 Robust TAC 实验模块

目录：

```bash
experiments/robust_tac
```

核心文件：

```bash
experiments/robust_tac/train_robust_tac.py
experiments/robust_tac/generate_caltech101_guidance.py
experiments/robust_tac/data.py
experiments/robust_tac/guidance.py
experiments/robust_tac/losses.py
experiments/robust_tac/models.py
experiments/robust_tac/attacks.py
experiments/robust_tac/metrics.py
experiments/robust_tac/text_prompt.py
experiments/robust_tac/utils.py
```

这个模块实现了一个最小可控的 Robust TAC pipeline：

- 使用 frozen CLIP ViT-B/32 作为图像编码器。
- 支持 `static`、`vprompt`、`tprompt` 三种模式。
- 支持 label-free PGD attack。
- 训练时不使用 label；label 只用于最后计算 ACC/NMI/ARI。
- 输出每个实验的 `summary.csv`。

### 2. 支持 Caltech101 和 CIFAR-10 数据

文件：

```bash
experiments/robust_tac/data.py
```

已支持：

- `Caltech101`
- `CIFAR10` / `CIFAR-10`

CIFAR-10 使用 `torchvision.datasets.CIFAR10`，数据根目录是：

```bash
/work1/zixuan/data/fap
```

实际 CIFAR-10 文件夹：

```bash
/work1/zixuan/data/fap/cifar-10-batches-py
```

### 3. 构造 TAC text guidance

文件：

```bash
experiments/robust_tac/generate_caltech101_guidance.py
```

虽然文件名仍然叫 `generate_caltech101_guidance.py`，但现在参数里支持：

```bash
--dataset Caltech101
--dataset CIFAR10
```

这个脚本会：

1. 用 frozen CLIP 提取 train/test image embeddings。
2. 从 TAC 项目的 WordNet noun embedding 或 noun list 中得到候选 noun embeddings。
3. 对图像 embedding 做 semantic center clustering。
4. 为每个 semantic center 选 top nouns。
5. 构造 `noun_anchor_bank`。
6. 为每个样本检索 `text_counterpart`。
7. 保存为 `.npz`。

当前已有 guidance cache：

```bash
experiments/robust_tac/guidance_cache/caltech101_tac_guidance_full.npz
experiments/robust_tac/guidance_cache/caltech101_tac_guidance_smoke.npz
experiments/robust_tac/guidance_cache/cifar10_tac_guidance_full.npz
```

CIFAR-10 guidance 文件大小约 215MB。

`cifar10_tac_guidance_full.npz` 的内容：

```text
train_text_counterpart: (50000, 512), float32
test_text_counterpart:  (10000, 512), float32
noun_anchor_bank:       (50, 512), float32
train_image_embedding:  (50000, 512), float32
test_image_embedding:   (10000, 512), float32
selected_noun_indices:  (50,), int64
selected_nouns:         (50,), str
train_paths:            (50000,), str
test_paths:             (10000,), str
dataset:                scalar
num_clusters:           scalar int64
semantic_centers:       scalar int64
tau_retrieval:          scalar float32
guidance_type:          scalar
noun_source:            scalar
```

### 4. Guidance 加载逻辑

文件：

```bash
experiments/robust_tac/guidance.py
```

支持两种 guidance：

- `placeholder`：直接使用 clean image features 作为占位 counterpart，只用于 plumbing/smoke test。
- `tac_text`：加载 `.npz` 中的 `train_text_counterpart`、`test_text_counterpart`、`noun_anchor_bank`。

当使用：

```bash
--guidance-type tac_text
--guidance-path experiments/robust_tac/guidance_cache/cifar10_tac_guidance_full.npz
```

训练脚本会从 guidance npz 中读取文本 counterpart 和 noun anchors。

### 5. 训练入口和损失

文件：

```bash
experiments/robust_tac/train_robust_tac.py
experiments/robust_tac/losses.py
```

训练目标包含：

- TAC clean loss：让 image assignment 和 text assignment 对齐，同时保持 assignment balance。
- CA loss：clean assignment 与 adversarial assignment 的 KL consistency。
- EG loss：
  - `text_assignment`：让 adversarial image assignment 对齐 text assignment。
  - `anchor_kl`：让 clean/adv image feature 到 noun anchors 的关系保持一致。
- FAP-style weighted EG：
  - `mean((cosine_similarity(v_clean, v_adv) + 1.0) * per_sample_eg_kl)`

关键参数：

```bash
--lambda-ca
--lambda-eg
--lambda-fap
--eg-type none|text_assignment|anchor_kl
--use-fap-weight
--tau-relation
--eg-symmetric
--eta-sym
```

### 6. 模型模式

文件：

```bash
experiments/robust_tac/models.py
experiments/robust_tac/text_prompt.py
```

支持模式：

- `static`：冻结 CLIP，只训练 image/text cluster heads。
- `vprompt`：冻结 CLIP backbone，只训练 shallow visual prompt tokens 和 cluster heads；当前支持 `--prompt-depth 1`。
- `tprompt`：冻结 CLIP image/text backbone，只训练 learnable text context tokens 和 cluster heads。

TPrompt 使用 guidance npz 中的 `selected_nouns` 构造 prompt，例如：

```text
X X {noun}.
```

### 7. 对抗攻击

文件：

```bash
experiments/robust_tac/attacks.py
```

实现 label-free PGD：

- 在 raw pixel space 中攻击。
- 目标是最大化 `KL(stopgrad(p_clean) || p_adv)`。
- 不使用 ground-truth label 作为 attack target。

常用设置：

```bash
--eps 1/255
--eval-eps 1/255
--step-size 1/255
--train-steps 2
--eval-steps 100
```

### 8. 评估指标

文件：

```bash
experiments/robust_tac/metrics.py
```

输出 clean 和 adversarial clustering 指标：

- `clean_acc`
- `clean_nmi`
- `clean_ari`
- `adv_acc`
- `adv_nmi`
- `adv_ari`
- `cfr`
- `clean_adv_kl_eval`
- `eg_kl_eval`
- `mean_cos_factor_eval`

每个 run 会写：

```bash
outputs/robust_tac/<run_name>/summary.csv
```

## 已写好的运行脚本

### CIFAR-10 final10 对比

```bash
experiments/robust_tac/run_cifar10_final10.sh
```

默认 GPU：

```bash
GPU_ID=9
```

包含 7 个 CIFAR-10 10-epoch run：

- Static + CA + EG
- VPrompt + CA + EG
- TPrompt + CA + EG
- TPrompt + CA + EG symmetric 0.05
- TPrompt + CA + EG symmetric 0.1
- TPrompt + CA + EG + FAP weight
- TPrompt + CA + EG + FAP weight + symmetric 0.05

汇总输出：

```bash
outputs/robust_tac/cifar10_final10_summary.csv
```

### CIFAR-10 FAP-only 补跑

```bash
experiments/robust_tac/run_cifar10_final10_fap_only.sh
```

只跑 FAP weight 相关的两个 TPrompt 实验，适合前面部分实验已经完成时补跑。

### Caltech101 convergence/weight control

```bash
experiments/robust_tac/run_convergence_weight.sh
experiments/robust_tac/summarize_convergence_weight.py
```

包含：

- 5/10/20 epoch convergence 对比。
- 不同 `lambda_eg` 的权重扫描。
- FAP weight control。

汇总输出：

```bash
outputs/robust_tac/convergence_weight_summary.csv
```

## 新服务器继续实验的最小准备

在新服务器上先确保这些内容存在：

```bash
/work1/zixuan/AGENTS.md
/work1/zixuan/projects/FAP
/work1/zixuan/projects/2024-ICML-TAC
/work1/zixuan/envs/miniconda3
/work1/zixuan/envs/conda_envs/fap
/work1/zixuan/data/fap
/work1/zixuan/cache
```

推荐复制这些文件/目录：

```bash
rsync -avh OLD_SERVER:/work1/zixuan/AGENTS.md /work1/zixuan/AGENTS.md
rsync -avh OLD_SERVER:/work1/zixuan/EXPERIMENT_HANDOFF.md /work1/zixuan/EXPERIMENT_HANDOFF.md
rsync -avh OLD_SERVER:/work1/zixuan/projects/FAP/ /work1/zixuan/projects/FAP/
rsync -avh OLD_SERVER:/work1/zixuan/projects/2024-ICML-TAC/ /work1/zixuan/projects/2024-ICML-TAC/
rsync -avh OLD_SERVER:/work1/zixuan/data/fap/ /work1/zixuan/data/fap/
```

不建议直接复制整个 conda env。新服务器优先重新安装 Miniconda 和 `/work1/zixuan/envs/conda_envs/fap` 环境。

可以复制 guidance cache，避免重新生成：

```bash
rsync -avh OLD_SERVER:/work1/zixuan/projects/FAP/experiments/robust_tac/guidance_cache/ /work1/zixuan/projects/FAP/experiments/robust_tac/guidance_cache/
```

也可以复制已有输出用于汇总：

```bash
rsync -avh OLD_SERVER:/work1/zixuan/projects/FAP/outputs/robust_tac/ /work1/zixuan/projects/FAP/outputs/robust_tac/
```

## 新服务器环境验证

进入 FAP：

```bash
cd /work1/zixuan/projects/FAP
```

加载环境：

```bash
source /work1/zixuan/envs/miniconda3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache

conda activate /work1/zixuan/envs/conda_envs/fap
```

检查：

```bash
pwd
git status
which python
which pip
python --version
pip --version
nvidia-smi
```

验证 imports：

```bash
python - << 'PY'
import torch
import torchvision
import dassl
import clip
import numpy
import scipy
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("imports passed")
PY
```

## 如果需要重新生成 CIFAR-10 guidance

先检查 GPU 空闲情况：

```bash
nvidia-smi
```

然后运行：

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> /work1/zixuan/envs/conda_envs/fap/bin/python experiments/robust_tac/generate_caltech101_guidance.py \
  --dataset CIFAR10 \
  --root /work1/zixuan/data/fap \
  --tac-root /work1/zixuan/projects/2024-ICML-TAC \
  --output experiments/robust_tac/guidance_cache/cifar10_tac_guidance_full.npz \
  --semantic-centers auto_num_clusters \
  --top-nouns-per-center 5 \
  --tau-retrieval 0.005 \
  --batch-size 128
```

验证：

```bash
/work1/zixuan/envs/conda_envs/fap/bin/python - << 'PY'
import numpy as np
p = "experiments/robust_tac/guidance_cache/cifar10_tac_guidance_full.npz"
d = np.load(p, allow_pickle=False)
print(d.files)
for k in d.files:
    print(k, d[k].shape, d[k].dtype)
PY
```

## 继续跑 CIFAR-10 final10

先检查 GPU：

```bash
nvidia-smi
```

运行完整对比：

```bash
cd /work1/zixuan/projects/FAP
GPU_ID=<GPU_ID> bash experiments/robust_tac/run_cifar10_final10.sh
```

如果只补跑 FAP weight 两个实验：

```bash
cd /work1/zixuan/projects/FAP
GPU_ID=<GPU_ID> bash experiments/robust_tac/run_cifar10_final10_fap_only.sh
```

查看结果：

```bash
cat outputs/robust_tac/cifar10_final10_summary.csv
```

## 继续跑 Caltech101 convergence/weight

```bash
cd /work1/zixuan/projects/FAP
GPU_ID=<GPU_ID> bash experiments/robust_tac/run_convergence_weight.sh all
```

只跑 convergence：

```bash
GPU_ID=<GPU_ID> bash experiments/robust_tac/run_convergence_weight.sh convergence
```

只跑 weight scan：

```bash
GPU_ID=<GPU_ID> bash experiments/robust_tac/run_convergence_weight.sh weight
```

查看汇总：

```bash
cat outputs/robust_tac/convergence_weight_summary.csv
```

