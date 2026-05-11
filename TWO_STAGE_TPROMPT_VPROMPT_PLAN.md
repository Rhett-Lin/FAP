# Two-Stage TPrompt-to-VPrompt Robust Clustering Experiment

## 0. Goal

We want to test a two-stage prompt strategy for unsupervised adversarial robust clustering.

Motivation:

- Previous CIFAR10 results show:
  - `VPrompt+CA+EG` gives the best adversarial robustness.
  - `TPrompt` improves clean semantic alignment but does not improve adversarial robustness enough.
  - Simultaneous `DualPrompt` improves clean performance but weakens VPrompt robustness.

Hypothesis:

```text
Stage 1: TPrompt learns a better external semantic teacher.
Stage 2: Freeze this teacher, then train VPrompt for robust visual representations.
```

This is different from simultaneous DualPrompt.

---

## 1. Paths

Repository:

```text
/work1/zixuan/projects/FAP
```

Python:

```text
/work1/zixuan/envs/conda_envs/fap/bin/python
```

Experiment directory:

```text
experiments/robust_tac/
```

TAC repo:

```text
/work1/zixuan/projects/2024-ICML-TAC
```

GPU:

```text
CUDA_VISIBLE_DEVICES=9
```

Before every run, check:

```bash
nvidia-smi
```

If GPU 9 is busy, stop and report. Do not occupy other users' GPU processes.

---

## 2. Strict Restrictions

1. Do not modify original FAP main files:
   - `train.py`
   - `trainers/fap.py`
   - `attack/pgd.py`
   - `datasets/*`

2. Do not modify TAC repo.

3. Only modify files under:

```text
experiments/robust_tac/
```

4. Do not install new packages.

5. Training loss must not use ground-truth labels.

6. Labels can only be used for evaluation metrics:
   - ACC
   - NMI
   - ARI

7. CLIP image/text backbone must remain frozen.

8. Existing modes must not be broken:
   - `static`
   - `vprompt`
   - `tprompt`
   - `dualprompt`

---

## 3. Current Best Baselines on CIFAR10

Use these as reference:

```text
VPrompt+CA+EG:
clean_acc = 0.9062
adv_acc   = 0.7161
adv_ari   = 0.4929
cfr       = 0.2768

Best TPrompt clean:
clean_acc = 0.9230
clean_ari = 0.8400
```

The two-stage method should be judged by:

1. Whether clean accuracy improves over VPrompt.
2. Whether adversarial accuracy / ARI stays close to or exceeds VPrompt.
3. Whether CFR stays close to VPrompt.

Success target:

```text
clean_acc > 0.9062
adv_acc   >= 0.7161 or only slightly lower
adv_ari   >= 0.4929 or only slightly lower
cfr        <= 0.30
```

---

## 4. Conceptual Design

### Stage 1: Learn a TPrompt semantic teacher

Use text-side prompt only.

Image branch:

```text
v_clean = frozen_CLIP_image_encoder(x_clean)
p_clean = image_head(v_clean)
```

Text branch:

```text
selected nouns
-> learnable text prompt P_t
-> CLIP text encoder
-> prompted noun anchors T(P_t)
-> soft retrieval using v_clean
-> text counterpart
-> text_head
-> q_teacher
```

Stage 1 should learn:

```text
text prompt P_t
image cluster head
text cluster head
```

Frozen:

```text
CLIP image encoder
CLIP text encoder
visual prompt not used
```

Recommended Stage 1 loss:

```text
L_stage1 = L_TAC
```

where:

```text
L_TAC = consistency(q_teacher, p_clean) - 5 * balance_entropy
```

Do not use strong adversarial loss in the first version.

Optional weak-adversarial teacher version:

```text
L_stage1 = L_TAC + 0.2 * L_CA + 0.2 * L_EG
```

But implement clean-only first.

---

## 5. Stage 1 Output

After Stage 1, save:

```text
teacher_q_train: [N_train, K]
teacher_q_test:  [N_test, K]

teacher_text_counterpart_train: [N_train, 512]
teacher_text_counterpart_test:  [N_test, 512]

prompted_noun_anchor_bank: [M, 512]

text_prompt_state
text_head_state
image_head_state
metadata
```

Recommended files:

```text
outputs/robust_tac/cifar10_stage1_tprompt_clean_teacher/teacher.npz
outputs/robust_tac/cifar10_stage1_tprompt_clean_teacher/checkpoint.pt
```

Important:

- `teacher_q_train` and `teacher_q_test` must align with dataset stable indices.
- During Stage 2, use `teacher_q[index]` directly.
- `teacher_q` must be treated as fixed and stop-gradient.

---

## 6. Stage 2: Frozen Teacher + VPrompt Robust Adaptation

Stage 2 should train visual prompt using the frozen teacher from Stage 1.

Image branch:

```text
v_clean = CLIP_image_encoder(x_clean; P_v)
v_adv   = CLIP_image_encoder(x_adv; P_v)

p_clean = image_head(v_clean)
p_adv   = image_head(v_adv)
```

Teacher:

```text
q_teacher = fixed_teacher_q[index]
```

Trainable:

```text
visual prompt P_v
image cluster head
```

Recommended:

- Initialize image head from Stage 1 checkpoint.
- Do not train text prompt.
- Do not train text head.
- Do not recompute dynamic text counterpart in Stage 2 for the main version.

Frozen:

```text
CLIP image encoder backbone
CLIP text encoder backbone
text prompt
text head
q_teacher
```

Stage 2 loss:

```text
L_stage2 =
    L_TAC_fixed
    + lambda_ca * L_CA
    + lambda_eg * L_EG_fixed
```

where:

```text
L_TAC_fixed = consistency(q_teacher, p_clean) - 5 * entropy(mean(p_clean))

L_CA = KL(stopgrad(p_clean) || p_adv)

L_EG_fixed = KL(stopgrad(q_teacher) || p_adv)
```

Use:

```text
lambda_ca = 1.0
lambda_eg = 0.5
```

---

## 7. Required Code Changes

Read these files first:

```text
experiments/robust_tac/train_robust_tac.py
experiments/robust_tac/models.py
experiments/robust_tac/text_prompt.py
experiments/robust_tac/guidance.py
experiments/robust_tac/losses.py
experiments/robust_tac/README.md
```

Implement new functionality with minimal changes.

### New arguments

Add:

```text
--save-teacher-path
--teacher-path
--teacher-type
--load-stage1-heads
--stage1-checkpoint
```

Recommended values:

```text
--teacher-type none
--teacher-type fixed_q
```

### Stage 1 behavior

If `--save-teacher-path` is given:

1. Train normally with `mode=tprompt`.
2. After training, run full train/test forward pass.
3. Save fixed teacher outputs:
   - `teacher_q_train`
   - `teacher_q_test`
   - optional text counterparts
   - optional prompted noun anchors
4. Save checkpoint:
   - text prompt
   - text head
   - image head
   - args / metadata

### Stage 2 behavior

If:

```text
--teacher-type fixed_q
--teacher-path <path>
```

then:

1. Load `teacher_q_train` and `teacher_q_test`.
2. During training:
   - use `teacher_q_train[index]`
3. During evaluation:
   - use `teacher_q_test[index]`
4. Do not compute dynamic text counterpart.
5. Do not update teacher.
6. Use fixed teacher for:
   - `L_TAC_fixed`
   - `L_EG_fixed`

If `--stage1-checkpoint` and `--load-stage1-heads` are given:

1. Load image head from Stage 1 checkpoint.
2. Optionally load text head only if needed.
3. Do not load text prompt into trainable Stage 2 unless explicitly required.

---

## 8. Important Implementation Details

### Stable indices

Teacher loading must respect the dataset index returned by dataloader.

Batch should provide:

```python
batch["index"]
```

Then:

```python
q_teacher = teacher_q[index]
```

Check shape:

```text
q_teacher: [B, K]
p_clean:   [B, K]
p_adv:     [B, K]
```

If index is out of range, raise error.

---

### Teacher normalization

`teacher_q` should be valid probability distribution:

```text
q_teacher >= 0
q_teacher.sum(dim=1) ~= 1
```

If needed, apply softmax before saving or after loading.

Avoid double softmax if already saved as probability.

---

### Loss logging

Summary must include:

```text
method
stage
mode
teacher_type
clean_acc
clean_nmi
clean_ari
adv_acc
adv_nmi
adv_ari
cfr
clean_adv_kl_eval
eg_kl_eval
mean_cos_factor_eval
lambda_ca
lambda_eg
teacher_path
stage1_checkpoint
visual_prompt_trainable_params
text_prompt_trainable_params
cluster_head_trainable_params
total_trainable_params
```

---

## 9. Experiments to Run

### Experiment A: Stage 1 clean TPrompt teacher

Run 10 epochs.

```bash
CUDA_VISIBLE_DEVICES=9 /work1/zixuan/envs/conda_envs/fap/bin/python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode tprompt \
  --epochs 10 \
  --batch-size 32 \
  --train-steps 1 \
  --eval-steps 20 \
  --lambda-ca 0.0 \
  --lambda-eg 0.0 \
  --eg-type text_assignment \
  --n-ctx 2 \
  --num-clusters auto \
  --guidance-type tac_text \
  --save-teacher-path outputs/robust_tac/cifar10_stage1_tprompt_clean_teacher/teacher.npz \
  --output outputs/robust_tac/cifar10_stage1_tprompt_clean_teacher
```

If CIFAR10 requires `--root` or `--guidance-path`, use the same values as previous successful CIFAR10 runs. Do not guess; inspect README or existing command history in outputs.

Expected output:

```text
outputs/robust_tac/cifar10_stage1_tprompt_clean_teacher/teacher.npz
outputs/robust_tac/cifar10_stage1_tprompt_clean_teacher/checkpoint.pt
outputs/robust_tac/cifar10_stage1_tprompt_clean_teacher/summary.csv
```

---

### Experiment B: Stage 2 VPrompt with fixed clean teacher

Run 10 epochs.

```bash
CUDA_VISIBLE_DEVICES=9 /work1/zixuan/envs/conda_envs/fap/bin/python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode vprompt \
  --epochs 10 \
  --batch-size 32 \
  --train-steps 1 \
  --eval-steps 20 \
  --lambda-ca 1.0 \
  --lambda-eg 0.5 \
  --eg-type text_assignment \
  --prompt-depth 1 \
  --n-ctx 2 \
  --num-clusters auto \
  --guidance-type tac_text \
  --teacher-type fixed_q \
  --teacher-path outputs/robust_tac/cifar10_stage1_tprompt_clean_teacher/teacher.npz \
  --stage1-checkpoint outputs/robust_tac/cifar10_stage1_tprompt_clean_teacher/checkpoint.pt \
  --load-stage1-heads \
  --output outputs/robust_tac/cifar10_stage2_vprompt_fixed_tprompt_teacher
```

---

## 10. Optional Ablation Experiments

Only run after Experiments A and B complete.

### Experiment C: Stage 1 weak adversarial teacher

```bash
CUDA_VISIBLE_DEVICES=9 /work1/zixuan/envs/conda_envs/fap/bin/python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode tprompt \
  --epochs 10 \
  --batch-size 32 \
  --train-steps 1 \
  --eval-steps 20 \
  --lambda-ca 0.2 \
  --lambda-eg 0.2 \
  --eg-type text_assignment \
  --n-ctx 2 \
  --num-clusters auto \
  --guidance-type tac_text \
  --save-teacher-path outputs/robust_tac/cifar10_stage1_tprompt_weakadv_teacher/teacher.npz \
  --output outputs/robust_tac/cifar10_stage1_tprompt_weakadv_teacher
```

### Experiment D: Stage 2 VPrompt with weak-adversarial fixed teacher

```bash
CUDA_VISIBLE_DEVICES=9 /work1/zixuan/envs/conda_envs/fap/bin/python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --mode vprompt \
  --epochs 10 \
  --batch-size 32 \
  --train-steps 1 \
  --eval-steps 20 \
  --lambda-ca 1.0 \
  --lambda-eg 0.5 \
  --eg-type text_assignment \
  --prompt-depth 1 \
  --n-ctx 2 \
  --num-clusters auto \
  --guidance-type tac_text \
  --teacher-type fixed_q \
  --teacher-path outputs/robust_tac/cifar10_stage1_tprompt_weakadv_teacher/teacher.npz \
  --stage1-checkpoint outputs/robust_tac/cifar10_stage1_tprompt_weakadv_teacher/checkpoint.pt \
  --load-stage1-heads \
  --output outputs/robust_tac/cifar10_stage2_vprompt_fixed_tprompt_weakadv_teacher
```

---

## 11. Summary File

After finishing, generate:

```text
outputs/robust_tac/cifar10_two_stage_prompt_summary.csv
```

Include at least:

```text
Baseline VPrompt+CA+EG
Stage1 TPrompt clean teacher
Stage2 VPrompt fixed clean teacher
Optional Stage1 weakadv teacher
Optional Stage2 VPrompt fixed weakadv teacher
```

Fields:

```text
method
stage
mode
teacher_type
clean_acc
clean_nmi
clean_ari
adv_acc
adv_nmi
adv_ari
cfr
clean_adv_kl_eval
eg_kl_eval
mean_cos_factor_eval
teacher_path
stage1_checkpoint
visual_prompt_trainable_params
text_prompt_trainable_params
total_trainable_params
```

Use known VPrompt baseline if existing summary is not found:

```text
VPrompt+CA+EG:
clean_acc = 0.9062
adv_acc   = 0.7161
adv_ari   = 0.4929
cfr       = 0.2768
```

---

## 12. Reporting

Report only:

1. Modified files.
2. Whether Stage 1 teacher was saved successfully.
3. Teacher shapes:
   - `teacher_q_train`
   - `teacher_q_test`
4. Whether Stage 2 loaded fixed teacher correctly.
5. Whether Stage 2 loaded Stage 1 image head.
6. Whether there were NaNs or shape mismatches.
7. Final summary table.
8. Whether two-stage improves clean_acc over VPrompt.
9. Whether two-stage matches or exceeds VPrompt adv_acc / adv_ari.
10. Whether two-stage keeps CFR close to VPrompt.
11. Recommendation:
    - continue two-stage
    - tune weak-adversarial teacher
    - abandon two-stage and keep VPrompt as main method

---

## 13. Decision Rule

If Stage 2 achieves:

```text
clean_acc > 0.9062
adv_acc >= 0.7000
adv_ari >= 0.4700
cfr <= 0.3200
```

then the two-stage design is promising.

If Stage 2 has:

```text
clean_acc improves
but adv_acc / adv_ari drop strongly
or cfr rises clearly
```

then the teacher learned by TPrompt still harms robustness, and VPrompt+fixed original TAC guidance should remain the main method.

If Stage 2 outperforms VPrompt in both clean and adversarial metrics, then use two-stage as the main method.