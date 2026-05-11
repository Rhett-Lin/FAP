# Robust TAC Smoke Test

This directory contains minimal Static-TAC / VPrompt-TAC + clean-adversarial
consistency smoke tests for unsupervised robust clustering on Caltech101.

The code is intentionally isolated from the original FAP training path. It does
not modify `train.py`, `trainers/fap.py`, `attack/pgd.py`, or
`datasets/caltech101.py`.

## Scope

- Dataset: Caltech101 with the same ignored categories as FAP
  (`BACKGROUND_Google`, `Faces_easy`).
- Backbone: frozen CLIP ViT-B/32.
- Static trainable modules: image and text cluster heads only.
- VPrompt trainable modules: shallow visual prompt tokens and cluster heads only.
- DualPrompt trainable modules: shallow visual prompt tokens, learnable text
  context tokens, and cluster heads only.
- Training labels: not used.
- Evaluation labels: used only for ACC, NMI, ARI.
- Attack: label-free PGD in raw pixel space, maximizing
  `KL(stopgrad(p_clean) || p_adv)`.

## Guidance

The code reserves the TAC external guidance interface through:

- `text_counterpart[index]`
- `noun_anchor_bank`
- `guidance_type`
- `guidance_path`

If a Caltech101 TAC guidance cache is available, it can be loaded read-only via
`--guidance-path` or from `--tac-root/data`. If no cache is found, the smoke test
uses stop-gradient clean image features as placeholder guidance. This placeholder
is only for plumbing and smoke testing; it is not the final TAC external
guidance method.

## Example

```bash
/work1/zixuan/envs/conda_envs/fap/bin/python experiments/robust_tac/train_robust_tac.py \
  --dataset Caltech101 \
  --root /work1/zixuan/data \
  --mode static \
  --epochs 1 \
  --batch-size 32 \
  --max-train-samples 512 \
  --max-test-samples 512 \
  --train-steps 1 \
  --eval-steps 2 \
  --lambda-ca 1.0 \
  --num-clusters auto \
  --tac-root /work1/zixuan/projects/2024-ICML-TAC \
  --output outputs/robust_tac/smoke_caltech101_static_ca
```

VPrompt smoke test:

```bash
CUDA_VISIBLE_DEVICES=9 /work1/zixuan/envs/conda_envs/fap/bin/python experiments/robust_tac/train_robust_tac.py \
  --dataset Caltech101 \
  --root /work1/zixuan/data/fap \
  --mode vprompt \
  --epochs 1 \
  --batch-size 32 \
  --max-train-samples 512 \
  --max-test-samples 512 \
  --train-steps 1 \
  --eval-steps 2 \
  --lambda-ca 1.0 \
  --num-clusters auto \
  --prompt-depth 1 \
  --n-ctx 2 \
  --tac-root /work1/zixuan/projects/2024-ICML-TAC \
  --output outputs/robust_tac/smoke_caltech101_vprompt_ca
```

The current VPrompt implementation uses the FAP repository's modified CLIP
`VisionTransformer` VPT path with `prompt-depth=1`. It does not use the Dassl
trainer and does not modify CLIP source files.

## TAC Text Guidance

Generate a TAC-style Caltech101 text guidance cache from the TAC WordNet noun
list without using Caltech101 labels or class names:

```bash
CUDA_VISIBLE_DEVICES=9 /work1/zixuan/envs/conda_envs/fap/bin/python experiments/robust_tac/generate_caltech101_guidance.py \
  --root /work1/zixuan/data/fap \
  --tac-root /work1/zixuan/projects/2024-ICML-TAC \
  --output experiments/robust_tac/guidance_cache/caltech101_tac_guidance_full.npz \
  --semantic-centers auto_num_clusters \
  --top-nouns-per-center 5 \
  --tau-retrieval 0.005 \
  --batch-size 128
```

Then pass `--guidance-type tac_text --guidance-path <cache.npz>` to
`train_robust_tac.py`. The placeholder guidance mode remains available for
small plumbing tests only.

## TPrompt

`--mode tprompt` keeps the CLIP image and text backbones frozen and trains only
learnable text context tokens plus the cluster heads. It loads `selected_nouns`
from the TAC guidance npz, encodes prompts of the form `X X {noun}.`, builds a
prompted noun anchor bank each batch, and retrieves text counterparts from clean
image features via softmax retrieval. Add `--eg-symmetric --eta-sym 0.1` to let
the text prompt receive the reverse KL signal from the adversarial branch.

## DualPrompt

`--mode dualprompt` combines the VPrompt image path and TPrompt text path. Clean
and adversarial image features are encoded with the trainable visual prompt. The
text counterpart is dynamically retrieved from prompted noun anchors using the
prompted clean image feature:

```text
softmax(v_clean @ T(P_t).T / tau_retrieval) @ T(P_t)
```

Use it with TAC text guidance:

```bash
CUDA_VISIBLE_DEVICES=9 /work1/zixuan/envs/conda_envs/fap/bin/python experiments/robust_tac/train_robust_tac.py \
  --dataset CIFAR10 \
  --root /work1/zixuan/data/fap \
  --mode dualprompt \
  --epochs 1 \
  --batch-size 32 \
  --max-train-samples 512 \
  --max-test-samples 512 \
  --train-steps 1 \
  --eval-steps 2 \
  --lambda-ca 1.0 \
  --lambda-eg 0.5 \
  --eg-type text_assignment \
  --prompt-depth 1 \
  --n-ctx 2 \
  --num-clusters auto \
  --guidance-type tac_text \
  --guidance-path experiments/robust_tac/guidance_cache/cifar10_tac_guidance_full.npz \
  --output outputs/robust_tac/smoke_dualprompt_ca_eg
```

## EG And FAP Weight

`train_robust_tac.py` supports two external-guided losses:

- `--eg-type text_assignment`: aligns adversarial image assignments with
  `text_head(text_counterpart[index])`.
- `--eg-type anchor_kl`: aligns clean and adversarial relations to the
  `noun_anchor_bank` stored in the guidance npz.

Use `--lambda-eg` to weight EG. Add `--use-fap-weight --lambda-fap <value>` to
enable the FAP-style adversarial-aware term:

```text
mean((cosine_similarity(v_clean, v_adv) + 1.0) * per_sample_eg_kl)
```

These losses use no labels during training. Labels remain evaluation-only for
ACC/NMI/ARI.
