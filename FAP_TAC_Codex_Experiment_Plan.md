# Codex 实验任务书：验证 FAP-style Adversarial Prompt / Adaptive Guidance 是否优于普通 Clean-Adv Consistency

> 目标读者：Codex / 编程助手  
> 基础仓库：`https://github.com/Rhett-Lin/FAP`  
> 参考仓库：`https://github.com/XLearning-SCU/2024-ICML-TAC`  
> 当前研究阶段：只验证一个命题——**FAP-style adversarial prompt / adaptive guidance 是否比普通 clean-adversarial consistency 更有价值**。  
> 已经默认成立的前提：  
> 1. 无监督聚类在攻击下确实不稳定；  
> 2. 外部引导比纯内部聚类信号更稳定。  

---

## 0. 总原则

请严格遵守以下原则：

1. **不要重写整个 FAP 或 TAC 框架**。本阶段只做最小可行实验。
2. **以 FAP 仓库为主代码库**，因为我们需要复用 FAP 的 CLIP、prompt learner、PGD attack、训练配置风格。
3. **TAC 仓库只作为参考和少量模块来源**，尤其参考：
   - `models.py` 中的 `ClusterHead`
   - `loss_utils.py` 中的 `DistillLoss`, `consistency_loss`, `entropy`
   - `train_head.py` 中 TAC 的训练流程
   - `filter_nouns.py`, `retrieve_text.py` 中外部文本引导的构造逻辑
4. **训练损失中不能使用真实标签**。真实标签只允许用于最后的 ACC / NMI / ARI 评价。
5. 第一阶段只跑 Caltech101数据集。
6. CLIP 主干默认冻结；可训练参数只能是：
   - cluster heads；
   - optional visual prompts；
   - optional prompt projection；
   - optional small parameter-matched adapter baseline。
7. 需要保存每个实验的：
   - config；
   - stdout log；
   - clean / adv metrics；
   - loss curve；
   - checkpoint；
   - final `.csv` 汇总表。

---

## 1. 实验核心问题

我们要验证的问题不是“是否可以让 TAC 更鲁棒”，而是：

> 在已经有普通 clean-adv consistency 的情况下，FAP-style prompt / adaptive guidance 是否还能提供额外且不可替代的鲁棒性收益？

因此，最重要的对比是：

```text
Static-TAC + CA
vs.
Prompt-TAC + CA
vs.
Prompt-TAC + CA + EG
vs.
Prompt-TAC + CA + EG + FAP-style weighting
```

其中：

- `CA` = clean-adversarial cluster assignment consistency
- `EG` = external-guided relational consistency
- `FAP-style weighting` = 使用 clean/adv visual feature cosine factor 调制鲁棒正则，借鉴 FAP 的 adversarial-aware objective

---

## 2. 建议新增文件结构

请在 FAP 仓库中新建以下目录和文件：

```text
experiments/
  robust_tac/
    README.md
    train_robust_tac.py
    eval_robust_tac.py
    models.py
    losses.py
    attacks.py
    data.py
    metrics.py
    guidance.py
    prompt_encoders.py
    utils.py
    configs/
      cifar10_static_tac.yaml
      cifar10_ca.yaml
      cifar10_vprompt_ca.yaml
      cifar10_vprompt_ca_eg.yaml
      cifar10_vprompt_ca_eg_fapweight.yaml
    scripts/
      run_cifar10_minimal.sh
      run_cifar10_ablation.sh
```

如果 FAP 仓库已有类似结构，可以复用已有风格，但请保证实验代码和原始 FAP 分类训练代码解耦，避免破坏原始功能。

---

## 3. 需要先审查的现有代码

请先运行并记录：

```bash
pwd
find . -maxdepth 3 -type f | sort | sed 's#^\./##' | head -200
```

重点阅读：

```text
train.py
trainers/fap.py
attack/pgd.py
clip/
configs/
datasets/
```

同时把 TAC 仓库放到同级目录或临时目录，例如：

```text
../2024-ICML-TAC/
```

重点参考：

```text
../2024-ICML-TAC/models.py
../2024-ICML-TAC/loss_utils.py
../2024-ICML-TAC/train_head.py
../2024-ICML-TAC/filter_nouns.py
../2024-ICML-TAC/retrieve_text.py
```

---

## 4. 数据与样本索引要求

第一阶段使用 CIFAR-10。

请优先实现一个**独立 torchvision CIFAR-10 dataloader**，不要强依赖 Dassl 数据集，因为我们需要样本 index 与 external guidance 对齐。

每个 batch 必须返回：

```python
{
    "image": image_tensor,      # raw image tensor in [0, 1], before CLIP preprocessing
    "label": label_tensor,      # only for evaluation, not for training loss
    "index": index_tensor       # used to fetch text counterpart / guidance
}
```

要求：

1. 训练集和测试集顺序固定。
2. 每个 sample 有稳定的 `index`。
3. `index` 用于取出：
   - `text_counterpart[index]`
   - nearest neighbor index
   - cached clean features, if needed
4. 所有训练 loss 禁止使用 `label`。

---

## 5. 外部引导数据的两种实现路径

### 路径 A：优先使用 TAC 缓存特征

如果已经能从 TAC 仓库生成以下文件：

```text
data/CIFAR-10_image_embedding_train.npy
data/CIFAR-10_image_embedding_test.npy
data/CIFAR-10_filtered_nouns_embedding.npy
data/CIFAR-10_retrieved_nouns_embedding.npy
data/CIFAR-10_labels_test.txt
```

则直接拷贝或软链接到：

```text
experiments/robust_tac/data/
```

并在代码中加载：

```python
image_guidance_clean = np.load("...CIFAR-10_image_embedding_train.npy")
noun_anchor_bank = np.load("...CIFAR-10_filtered_nouns_embedding.npy")
text_counterpart = np.load("...CIFAR-10_retrieved_nouns_embedding.npy")
```

### 路径 B：如果没有 TAC 缓存，则先实现 feature-level 简化版

先用 CLIP zero-shot text encoder 或 FAP 的 CLIP text encoder构造一个小型 noun anchor bank。

最低要求：

```text
anchor bank size: 200-1000
embedding dim: 512
normalize: yes
```

第一阶段并不要求完全复现 TAC 的 WordNet noun filtering。只要能构造稳定 external guidance，就可以先验证 prompt 是否比 CA 更有价值。

---

## 6. 模型组件

### 6.1 ClusterHead

参考 TAC 的 `ClusterHead`：

```python
class ClusterHead(nn.Module):
    def __init__(self, in_dim=512, num_clusters=10):
        ...
    def forward(self, text, image):
        logit_text = self.cluster_head_text(text)
        logit_image = self.cluster_head_image(image)
        return logit_text, logit_image
```

注意：

- 输出是 soft cluster assignment；
- `num_clusters=10` for CIFAR-10；
- image head 和 text head 结构保持与 TAC 一致；
- 不要在第一阶段引入更大的 head。

---

### 6.2 Image Encoder Variants

需要实现至少三种 image encoder 模式：

#### Mode 1: `static`

```text
Frozen CLIP image encoder.
No prompt.
Only cluster heads are trained.
```

#### Mode 2: `vprompt`

```text
Frozen CLIP backbone.
Train visual prompts from FAP.
Train cluster heads.
```

优先复用 FAP `trainers/fap.py` 中的 prompt learner 逻辑。

如果直接复用困难，可以实现一个 thin wrapper：

```python
class PromptedImageEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        ...
    def forward(self, images):
        # return normalized image features, shape [B, 512]
```

要求：

- 对于 `static` 和 `vprompt`，输出都必须是 normalized 512-d CLIP image feature；
- visual prompt 的参数量需要打印；
- CLIP 主干参数必须冻结。

#### Mode 3: `param_matched_adapter`（建议但非第一优先级）

为了回应“prompt 只是多了参数”的质疑，加入一个参数量接近 visual prompt 的 image-feature adapter baseline：

```python
z = z + adapter(z)
z = normalize(z)
```

该 baseline 不使用 prompt，不使用 text adaptation，只匹配参数量。

---

## 7. Loss 设计

设：

```python
v_c = image encoder(clean image)
v_a = image encoder(adversarial image)
t_i = text_counterpart[index]
T = noun_anchor_bank
p_c = image_cluster_head(v_c)
p_a = image_cluster_head(v_a)
q_i = text_cluster_head(t_i)
```

所有 feature 均需 L2 normalize。

---

### 7.1 TAC clean loss

参考 TAC：

```python
loss_distill = DistillLoss(p_c, q_neighbor) + DistillLoss(q_i, p_neighbor)
loss_consist = consistency_loss(q_i, p_c)
loss_entropy = entropy(q_i) + entropy(p_c)
loss_tac_clean = loss_distill + loss_consist - alpha_balance * loss_entropy
```

第一阶段如 neighbor 逻辑实现困难，可以先用简化版：

```python
loss_tac_clean = consistency_loss(q_i, p_c) - alpha_balance * (entropy(q_i) + entropy(p_c))
```

但最终至少需要支持 TAC-style neighbor distillation。

---

### 7.2 Clean-Adv Assignment Consistency: CA

这是最关键 baseline：

```python
loss_ca = KL(stopgrad(p_c) || p_a)
```

实现建议：

```python
def kl_consistency(p_teacher, p_student):
    return F.kl_div(
        torch.log(p_student.clamp_min(1e-8)),
        p_teacher.detach().clamp_min(1e-8),
        reduction="batchmean"
    )
```

---

### 7.3 External-Guided Consistency: EG

有两个版本，先实现 Version B，简单稳定。

#### Version A: noun-anchor relation consistency

```python
r_c = softmax((v_c @ T.T) / tau_relation)
r_a = softmax((v_a @ T.T) / tau_relation)
loss_eg = KL(stopgrad(r_c) || r_a)
```

#### Version B: text-to-adversarial assignment consistency

```python
loss_t2a = KL(stopgrad(q_i) || p_a)
```

建议第一阶段两个都实现，用参数控制：

```bash
--eg-type none
--eg-type anchor_kl
--eg-type text_assignment
```

---

### 7.4 FAP-style adversarial-aware weighting

借鉴 FAP 的思想：不强行让 clean/adversarial visual feature 完全一样，而是在保持外部关系一致时显式考虑 clean/adv visual feature 差异。

```python
cos_factor = cosine_similarity(v_c, v_a) + 1.0
loss_fap_weight = mean(cos_factor * per_sample_KL(stopgrad(r_c), r_a))
```

如果使用 text assignment 版本，则：

```python
loss_fap_weight = mean(cos_factor * per_sample_KL(stopgrad(q_i), p_a))
```

提供参数：

```bash
--use-fap-weight
```

---

### 7.5 总损失

```python
loss = (
    loss_tac_clean
    + beta_adv_tac * loss_tac_adv
    + lambda_ca * loss_ca
    + lambda_eg * loss_eg
    + lambda_fap * loss_fap_weight
)
```

第一阶段建议：

```yaml
alpha_balance: 5.0
beta_adv_tac: 0.0       # 先不启用 adv TAC distillation
lambda_ca: 1.0
lambda_eg: 0.5
lambda_fap: 0.5
tau_relation: 0.01
```

---

## 8. 对抗样本生成

不要使用真实标签攻击。

需要实现 clustering attack：

```python
delta = argmax_delta KL(stopgrad(p_c) || p_a)
```

如果启用 external-guided attack：

```python
delta = argmax_delta [
    KL(stopgrad(p_c) || p_a)
    + attack_lambda_eg * KL(stopgrad(r_c) || r_a)
]
```

PGD 参数：

```yaml
train_eps: 1/255
train_steps: 2
train_alpha: 1/255

eval_eps_list: [1/255, 2/255, 4/255]
eval_steps: 20
eval_alpha: eps / 4
```

实现要求：

1. delta 在 raw pixel space 中更新；
2. 每步 clamp 到 `[0, 1]`；
3. 每步调用 CLIP preprocessing；
4. 训练攻击默认 PGD-2；
5. 评估攻击默认 PGD-20；
6. 支持 `--eval-steps 100`，但第一阶段不必默认使用。

---

## 9. 实验组设计

### E0: Static TAC clean baseline

```bash
python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode static \
  --loss tac_clean \
  --epochs 20 \
  --output outputs/robust_tac/cifar10_static
```

目的：确保基础聚类训练正常。

---

### E1: Plain clean-adv consistency baseline

```bash
python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode static \
  --loss tac_ca \
  --lambda-ca 1.0 \
  --epochs 20 \
  --output outputs/robust_tac/cifar10_static_ca
```

这是最重要 baseline。

---

### E2: Visual prompt + CA

```bash
python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode vprompt \
  --loss tac_ca \
  --lambda-ca 1.0 \
  --prompt-depth 1 \
  --n-ctx 2 \
  --epochs 20 \
  --output outputs/robust_tac/cifar10_vprompt_ca
```

目的：验证 prompt 本身是否比普通 CA 有增益。

---

### E3: Visual prompt + CA + external-guided consistency

```bash
python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode vprompt \
  --loss tac_ca_eg \
  --lambda-ca 1.0 \
  --lambda-eg 0.5 \
  --eg-type anchor_kl \
  --prompt-depth 1 \
  --n-ctx 2 \
  --epochs 20 \
  --output outputs/robust_tac/cifar10_vprompt_ca_eg
```

目的：验证 adaptive prompt 与 external-guided relation 是否互补。

---

### E4: Visual prompt + CA + EG + FAP-style weighting

```bash
python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode vprompt \
  --loss tac_ca_eg_fapweight \
  --lambda-ca 1.0 \
  --lambda-eg 0.5 \
  --lambda-fap 0.5 \
  --eg-type anchor_kl \
  --use-fap-weight \
  --prompt-depth 1 \
  --n-ctx 2 \
  --epochs 20 \
  --output outputs/robust_tac/cifar10_vprompt_ca_eg_fapweight
```

目的：验证 FAP-style adversarial-aware weighting 是否带来额外收益。

---

### E5: Parameter-matched adapter baseline

```bash
python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode param_matched_adapter \
  --loss tac_ca_eg_fapweight \
  --lambda-ca 1.0 \
  --lambda-eg 0.5 \
  --lambda-fap 0.5 \
  --epochs 20 \
  --output outputs/robust_tac/cifar10_adapter_ca_eg_fapweight
```

目的：排除“只是多了参数”的解释。

---

## 10. 评价指标

每个实验必须输出：

```text
Clean ACC
Clean NMI
Clean ARI

Adv ACC @ eps=1/255
Adv NMI @ eps=1/255
Adv ARI @ eps=1/255

Adv ACC @ eps=2/255
Adv NMI @ eps=2/255
Adv ARI @ eps=2/255

Adv ACC @ eps=4/255
Adv NMI @ eps=4/255
Adv ARI @ eps=4/255

Delta ARI = Clean ARI - Adv ARI
Cluster Flip Rate, CFR
External Relation KL
Noun Top-k Overlap, optional
```

CFR 定义：

```python
CFR = mean(argmax(p_c) != argmax(p_a))
```

结果保存为：

```text
outputs/robust_tac/summary.csv
```

建议列：

```text
dataset,seed,mode,loss,eg_type,use_fap_weight,
clean_acc,clean_nmi,clean_ari,
adv_acc_eps1,adv_nmi_eps1,adv_ari_eps1,cfr_eps1,egkl_eps1,
adv_acc_eps2,adv_nmi_eps2,adv_ari_eps2,cfr_eps2,egkl_eps2,
adv_acc_eps4,adv_nmi_eps4,adv_ari_eps4,cfr_eps4,egkl_eps4,
trainable_params,total_params
```

---

## 11. 判断标准

### 继续做完整方法的标准

如果满足以下条件，说明命题值得继续：

```text
1. E2 > E1：vprompt + CA 在 adv NMI/ARI 或 CFR 上明显优于 static + CA；
2. E3 > E2：加入 external-guided consistency 后进一步提升；
3. E4 > E3：FAP-style weighting 进一步降低 CFR 或提升 adv ARI；
4. clean ARI 下降不超过 2-3 个点；
5. E4 优于 E5，说明不是单纯参数量造成。
```

最低可接受提升：

```text
adv ARI / adv NMI: 至少提升 2 个点以上
或
CFR: 相对下降 10% 以上
```

### 不建议继续 prompt 路线的标准

如果出现以下结果，应暂停 prompt 路线：

```text
E2 ≈ E1
E3 ≈ E1
E4 ≈ E1
```

说明 prompt / adaptive guidance 没有比普通 clean-adv consistency 提供额外价值。

如果：

```text
E5 ≥ E4
```

说明提升可能只是额外参数，不是 prompt 机制。

如果：

```text
clean ARI 明显下降，但 adv ARI 提升很小
```

说明方法只是牺牲 clean clustering 换取弱鲁棒性，不值得继续。

---

## 12. 需要输出的诊断图

请至少保存以下图：

```text
figures/cifar10_adv_ari_vs_eps.png
figures/cifar10_cfr_vs_eps.png
figures/cifar10_clean_adv_tradeoff.png
figures/cifar10_external_relation_kl_vs_eps.png
figures/cifar10_assignment_transition_matrix.png
```

其中：

1. `adv_ari_vs_eps`：不同方法随攻击强度变化；
2. `cfr_vs_eps`：cluster flip rate；
3. `clean_adv_tradeoff`：x-axis clean ARI, y-axis adv ARI；
4. `external_relation_kl_vs_eps`：外部关系漂移程度；
5. `assignment_transition_matrix`：clean cluster 到 adv cluster 的转移矩阵。

---

## 13. 代码质量要求

1. 所有新增函数要有 docstring。
2. 所有 loss 要单独写在 `losses.py`，不要塞进训练循环。
3. 所有 attack 要单独写在 `attacks.py`。
4. 所有 metrics 要单独写在 `metrics.py`。
5. 所有随机种子固定：
   ```python
   torch.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   ```
6. 每次运行都打印：
   ```text
   trainable params
   total params
   dataset size
   batch size
   eps
   PGD steps
   loss weights
   ```
7. 默认支持 CPU fallback，但主要面向 CUDA。
8. 任何新增依赖需要写入 `experiments/robust_tac/README.md`。

---

## 14. 快速 smoke test

请先实现一个极小测试，确认代码能跑：

```bash
python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode static \
  --loss tac_ca \
  --epochs 1 \
  --batch-size 64 \
  --max-train-samples 512 \
  --max-test-samples 512 \
  --eval-steps 2 \
  --output outputs/robust_tac/smoke_static_ca
```

然后：

```bash
python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode vprompt \
  --loss tac_ca_eg_fapweight \
  --epochs 1 \
  --batch-size 64 \
  --max-train-samples 512 \
  --max-test-samples 512 \
  --eval-steps 2 \
  --prompt-depth 1 \
  --n-ctx 2 \
  --output outputs/robust_tac/smoke_vprompt
```

smoke test 通过标准：

```text
1. 无 shape mismatch；
2. loss 非 NaN；
3. 反向传播成功；
4. 输出 clean/adv metrics；
5. 生成 summary.csv。
```

---

## 15. 第一阶段最终交付物

请完成后提供：

```text
1. 新增/修改文件列表
2. 关键实现说明
3. smoke test 命令与输出
4. CIFAR-10 ablation 表格
5. 5 张诊断图
6. 是否支持下一步扩展到 ImageNet-10 / DTD 的判断
```

---

## 16. 不要做的事情

第一阶段不要做：

```text
1. 不要跑 ImageNet-1K；
2. 不要训练 CLIP backbone；
3. 不要使用真实类别名作为训练监督；
4. 不要把 FAP 的分类 CE loss 直接用于聚类；
5. 不要只报告 ACC；
6. 不要只比较 TAC 和 full model；
7. 不要跳过 Static-TAC+CA baseline；
8. 不要跳过 parameter-matched baseline；
9. 不要为了追求结果而调很多超参数。
```

---

## 17. 推荐实现顺序

请按这个顺序实现：

```text
Step 1: 新建 experiments/robust_tac 目录和基础脚本
Step 2: 实现 CIFAR-10 indexed dataloader
Step 3: 实现 static CLIP feature encoder
Step 4: 实现 ClusterHead 和 TAC-style losses
Step 5: 实现 label-free PGD clustering attack
Step 6: 跑通 Static-TAC+CA smoke test
Step 7: 实现 visual prompt image encoder
Step 8: 跑通 VPrompt-TAC+CA smoke test
Step 9: 实现 EG loss
Step 10: 实现 FAP-style weighting
Step 11: 实现 metrics 和 summary.csv
Step 12: 实现 parameter-matched adapter baseline
Step 13: 跑 CIFAR-10 ablation
Step 14: 生成诊断图
```

---

## 18. 备注：关于 unsupervised 设置

虽然 CIFAR-10 有标签，但训练中不能用标签。标签只用于：

```text
ACC / NMI / ARI evaluation
```

训练中允许使用：

```text
image
sample index
external text counterpart
noun anchor bank
clean/adv model predictions
nearest neighbors computed from features
```

训练中禁止使用：

```text
ground-truth label
class name prompt
supervised cross entropy
few-shot labeled subset
```

---

## 19. 最终目标表格模板

请最终生成如下表格：

| Method | Clean ACC | Clean NMI | Clean ARI | Adv ARI eps1 | Adv ARI eps2 | Adv ARI eps4 | CFR eps2 ↓ | EG-KL eps2 ↓ | Trainable Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static-TAC | | | | | | | | | |
| Static-TAC + CA | | | | | | | | | |
| VPrompt-TAC + CA | | | | | | | | | |
| VPrompt-TAC + CA + EG | | | | | | | | | |
| VPrompt-TAC + CA + EG + FAP-weight | | | | | | | | | |
| ParamMatched Adapter + CA + EG + FAP-weight | | | | | | | | | |

最重要的结论来自这三个差值：

```text
Gain_prompt = VPrompt-TAC+CA - Static-TAC+CA
Gain_EG = VPrompt-TAC+CA+EG - VPrompt-TAC+CA
Gain_FAP_weight = VPrompt-TAC+CA+EG+FAP-weight - VPrompt-TAC+CA+EG
```

如果这三个 gain 都明显为正，继续做完整方法。
如果只有 `Gain_EG` 为正，论文主线应改成 external-guided consistency，而不是 FAP prompt。
如果三者都不明显，暂停该方向。
