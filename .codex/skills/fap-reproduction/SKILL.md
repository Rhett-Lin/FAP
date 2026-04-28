---
name: fap-reproduction
description: Use this skill when reproducing, debugging, configuring, or extending experiments for the FAP repository on the shared RIKEN server.
---

# FAP Reproduction Skill

## Purpose

This skill guides reliable reproduction of **FAP: Few-Shot Adversarial Prompt Learning on Vision-Language Models** on a shared RIKEN server.

The goal is to reproduce the original repository first, not to redesign the method.

Use this skill for:

- Miniconda-based environment setup
- dependency checks
- server-specific path configuration
- few-shot reproduction
- debugging reproduction errors
- preparing scripts for controlled experiments
- organizing logs and outputs
- summarizing reproduction status

Do not use this skill to optimize, refactor, redesign, or extend the FAP method unless explicitly requested.

## Fixed Server Paths

Always use the following paths:

```bash
PROJECT_DIR=/work1/zixuan/projects/FAP
MINICONDA_DIR=/work1/zixuan/envs/miniconda3
CONDA_ENV_DIR=/work1/zixuan/envs/conda_envs/fap
DATA_DIR=/work1/zixuan/data/fap
OUTPUT_DIR=/work1/zixuan/outputs/FAP
CACHE_DIR=/work1/zixuan/cache
CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
PIP_CACHE_DIR=/work1/zixuan/cache/pip
TORCH_HOME=/work1/zixuan/cache/torch
XDG_CACHE_HOME=/work1/zixuan/cache
```

Never create projects, datasets, outputs, logs, checkpoints, virtual environments, conda environments, or caches under:

```bash
/home/zixuan/
```

## Safety and Server Constraints

Always follow these rules:

- Do not create projects under `/home/zixuan/`.
- Do not store datasets, checkpoints, logs, outputs, caches, or experiment results under `/home/zixuan/`.
- Use `/work1/zixuan/projects/FAP` as the repository path.
- Use `/work1/zixuan/envs/miniconda3` as the Miniconda installation path.
- Use `/work1/zixuan/envs/conda_envs/fap` as the FAP conda environment path.
- Use `/work1/zixuan/data/fap` as the dataset root.
- Use `/work1/zixuan/outputs/FAP` as the output root.
- Use `/work1/zixuan/cache` for cache files.
- Miniconda is allowed only because explicit permission has been obtained.
- Do not install Miniconda under `/home/zixuan/`.
- Do not create conda environments under `/home/zixuan/`.
- Do not use full Anaconda distribution unless explicitly permitted.
- Do not install Python packages globally.
- Do not install project packages into the conda base environment.
- Do not use system Python or system pip for this project.
- Do not create external tunnels, port forwarding, P2P, remote mapping, or unauthorized network processes.
- Do not store credentials, passwords, private keys, tokens, or private account information in files.
- Do not commit datasets, checkpoints, logs, generated results, caches, virtual environments, conda environments, or Miniconda installers.

## Directory Initialization

If required directories do not exist, create them with:

```bash
mkdir -p /work1/zixuan/envs
mkdir -p /work1/zixuan/envs/tools
mkdir -p /work1/zixuan/envs/conda_envs
mkdir -p /work1/zixuan/projects
mkdir -p /work1/zixuan/data/fap
mkdir -p /work1/zixuan/outputs/FAP
mkdir -p /work1/zixuan/cache/conda_pkgs
mkdir -p /work1/zixuan/cache/pip
mkdir -p /work1/zixuan/cache/torch
mkdir -p /work1/zixuan/cache/clip
```

Do not create equivalent directories under `/home/zixuan/`.

## Miniconda Installation Workflow

Check whether Miniconda already exists:

```bash
test -x /work1/zixuan/envs/miniconda3/bin/conda && echo "miniconda exists" || echo "miniconda missing"
```

If Miniconda is missing, install it under `/work1/zixuan/envs/miniconda3`.

Download installer:

```bash
mkdir -p /work1/zixuan/envs/tools
cd /work1/zixuan/envs/tools

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
```

If `wget` is unavailable, use:

```bash
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /work1/zixuan/envs/tools/miniconda.sh
```

Install Miniconda:

```bash
bash /work1/zixuan/envs/tools/miniconda.sh -b -p /work1/zixuan/envs/miniconda3
```

Do not install Miniconda under `/home/zixuan/`.

Do not run:

```bash
conda init
```

Instead, activate conda manually with:

```bash
source /work1/zixuan/envs/miniconda3/etc/profile.d/conda.sh
```

## Conda Environment Setup Workflow

Before creating or using the FAP environment, set server-safe cache paths:

```bash
export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache
```

Check whether the FAP conda environment exists:

```bash
test -x /work1/zixuan/envs/conda_envs/fap/bin/python && echo "fap conda env exists" || echo "fap conda env missing"
```

If the FAP conda environment is missing, create it with Python 3.8:

```bash
source /work1/zixuan/envs/miniconda3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache

conda create -y -p /work1/zixuan/envs/conda_envs/fap python=3.8 pip
```

Activate the environment:

```bash
source /work1/zixuan/envs/miniconda3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache

conda activate /work1/zixuan/envs/conda_envs/fap
```

Then verify:

```bash
which python
which pip
python --version
pip --version
conda info --envs
```

Expected:

```bash
/work1/zixuan/envs/conda_envs/fap/bin/python
/work1/zixuan/envs/conda_envs/fap/bin/pip
```

If `which python` or `which pip` points outside `/work1/zixuan/envs/conda_envs/fap`, stop and fix the environment before continuing.

## Cache Configuration

Always use server-safe cache paths:

```bash
export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache
```

If the cache directories do not exist, create them:

```bash
mkdir -p /work1/zixuan/cache/conda_pkgs
mkdir -p /work1/zixuan/cache/pip
mkdir -p /work1/zixuan/cache/torch
mkdir -p /work1/zixuan/cache/clip
```

Do not allow package caches, model weights, CLIP downloads, Torch caches, or conda package caches to be written under `/home/zixuan/`.

After activation, check:

```bash
echo $CONDA_PKGS_DIRS
echo $PIP_CACHE_DIR
echo $TORCH_HOME
echo $XDG_CACHE_HOME
```

Expected:

```bash
/work1/zixuan/cache/conda_pkgs
/work1/zixuan/cache/pip
/work1/zixuan/cache/torch
/work1/zixuan/cache
```

## Lightweight Environment Verification

Before running training, run only lightweight checks:

```bash
cd /work1/zixuan/projects/FAP

source /work1/zixuan/envs/miniconda3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache

conda activate /work1/zixuan/envs/conda_envs/fap

pwd
git status
which python
which pip
python --version
pip --version
conda info --envs
echo $CONDA_PKGS_DIRS
echo $PIP_CACHE_DIR
echo $TORCH_HOME
echo $XDG_CACHE_HOME
nvidia-smi
python train.py --help
```

Then run import verification:

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
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("imports passed")
PY
```

Do not run training until these checks pass.

## Dependency Installation Policy

Install packages only inside the activated FAP conda environment.

Before installing packages, always run:

```bash
which python
which pip
python --version
pip --version
```

Expected:

```bash
/work1/zixuan/envs/conda_envs/fap/bin/python
/work1/zixuan/envs/conda_envs/fap/bin/pip
```

Use:

```bash
python -m pip install ...
```

or:

```bash
pip install ...
```

only after confirming that `which pip` points to:

```bash
/work1/zixuan/envs/conda_envs/fap/bin/pip
```

Do not install packages globally.

Do not install project packages into the conda base environment.

Do not use system Python or system pip.

Do not blindly install the original `requirements.txt` if it contains invalid, duplicated, or conflicting entries.

If dependency versions conflict, report the conflict and propose the smallest compatible fix.

## PyTorch Installation Guidance

First inspect CUDA and Python:

```bash
python --version
nvidia-smi
nvcc --version
```

Candidate installation command:

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

After installation, verify:

```bash
python - << 'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
PY
```

If this fails, do not randomly upgrade PyTorch. First report:

- Python version
- CUDA information from `nvidia-smi`
- the exact pip error
- the exact import error, if any

Then propose a minimal compatible alternative.

## Dassl Installation Workflow

Dassl should be installed from a local clone under `/work1/zixuan/projects`.

If the directory does not exist:

```bash
cd /work1/zixuan/projects
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
```

Install inside the activated FAP conda environment:

```bash
cd /work1/zixuan/projects/Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
```

Verify:

```bash
python - << 'PY'
import dassl
print("Dassl imported successfully.")
PY
```

## FAP Dependency Workflow

Inside the FAP repository:

```bash
cd /work1/zixuan/projects/FAP
```

If the original `requirements.txt` causes conflicts, create and use a cleaned dependency file such as:

```bash
cat > requirements_clean.txt << 'EOF'
autoattack==0.1
einops==0.8.0
ftfy==6.3.0
numpy==1.24.3
Pillow==8.3.2
regex==2023.10.3
scipy==1.10.1
setuptools==59.5.0
tqdm==4.66.1
yacs==0.1.8
EOF
```

Install:

```bash
pip install -r requirements_clean.txt
```

Install CLIP:

```bash
pip install git+https://github.com/openai/CLIP.git
```

Then run the full import verification command from the lightweight verification section.

## Reproduction Workflow

Follow this sequence:

1. Inspect repository structure.
2. Identify training entry point.
3. Identify trainer implementation.
4. Identify config files.
5. Identify scripts under `scripts/Adv/fap/`.
6. Verify repository path.
7. Verify Miniconda installation.
8. Verify FAP conda environment.
9. Verify cache paths.
10. Verify dependencies.
11. Verify dataset paths.
12. Create or verify server-specific few-shot script.
13. Run minimal command.
14. Debug errors with minimal changes.
15. Summarize exact reproduction state.

Do not skip lightweight verification.

Do not start with base-to-new or cross-dataset experiments.

## Repository Structure Hints

Expected key files and directories:

```bash
train.py
trainers/fap.py
configs/
configs/datasets/
configs/trainers/FAP/
datasets/
attack/pgd.py
clip/
zsrobust/
scripts/Adv/fap/
```

The main entry point is:

```bash
train.py
```

The FAP trainer is:

```bash
trainers/fap.py
```

The first relevant script is:

```bash
scripts/Adv/fap/few_shot.sh
```

The server-specific copy should be:

```bash
scripts/Adv/fap/few_shot_zixuan.sh
```

## Script Modification Rules

If script paths are wrong:

1. Copy the original script.
2. Rename it with `_zixuan`.
3. Modify only server-specific paths.
4. Preserve the original experiment logic.
5. Do not modify the original script unless explicitly requested.

For few-shot reproduction:

```bash
cd /work1/zixuan/projects/FAP
cp scripts/Adv/fap/few_shot.sh scripts/Adv/fap/few_shot_zixuan.sh
```

Expected paths inside `few_shot_zixuan.sh`:

```bash
DATA=/work1/zixuan/data/fap
OUTPUT_DIR=/work1/zixuan/outputs/FAP
```

Do not modify:

- trainer logic
- model logic
- prompt learner logic
- loss functions
- PGD attack logic
- evaluation logic
- config semantics

unless explicitly requested.

## Dataset Verification

The dataset root must be:

```bash
/work1/zixuan/data/fap
```

For the first Caltech101 reproduction, check:

```bash
test -d /work1/zixuan/data/fap/caltech-101 && echo "caltech-101 directory exists" || echo "caltech-101 directory missing"
test -d /work1/zixuan/data/fap/caltech-101/101_ObjectCategories && echo "images exist" || echo "images missing"
test -f /work1/zixuan/data/fap/caltech-101/split_zhou_Caltech101.json && echo "split file exists" || echo "split file missing"
```

Expected first dataset paths:

```bash
/work1/zixuan/data/fap/caltech-101/101_ObjectCategories
/work1/zixuan/data/fap/caltech-101/split_zhou_Caltech101.json
```

If data is missing, report the missing path exactly.

Do not download datasets with P2P tools.

Do not place datasets under `/home/zixuan/`.

Do not commit datasets to Git.

## First Target Experiment

The first experiment should be:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/few_shot_zixuan.sh caltech101 16 0
```

Before running it:

```bash
cd /work1/zixuan/projects/FAP

source /work1/zixuan/envs/miniconda3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache

conda activate /work1/zixuan/envs/conda_envs/fap

nvidia-smi
```

Also verify:

```bash
test -f scripts/Adv/fap/few_shot_zixuan.sh && echo "few_shot_zixuan.sh exists" || echo "few_shot_zixuan.sh missing"
test -d /work1/zixuan/data/fap && echo "dataset root exists" || echo "dataset root missing"
test -d /work1/zixuan/outputs/FAP && echo "output root exists" || echo "output root missing"
```

Do not run other datasets before this succeeds.

Do not run base-to-new or cross-dataset experiments before this succeeds.

## GPU and Runtime Rules

Before launching any training:

```bash
nvidia-smi
```

Use `CUDA_VISIBLE_DEVICES` to select a GPU.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/few_shot_zixuan.sh caltech101 16 0
```

Use `tmux` for long-running experiments:

```bash
tmux new -s fap
cd /work1/zixuan/projects/FAP

source /work1/zixuan/envs/miniconda3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache

conda activate /work1/zixuan/envs/conda_envs/fap

CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/few_shot_zixuan.sh caltech101 16 0
```

Do not launch long-running jobs unless explicitly requested.

Do not launch multiple GPU jobs without checking GPU availability.

## Debugging Rules

When an error occurs:

1. Read the traceback.
2. Identify the smallest failing component.
3. Do not refactor unrelated code.
4. Prefer path, environment, and dependency fixes before code logic changes.
5. If code changes are needed, explain why they are necessary for reproduction.
6. Re-run the smallest command to verify the fix.
7. Report the key error message and the exact fix.

Do not hide failed commands.

Do not claim success unless the relevant verification command passed.

## Common Failure Categories

Check these first when debugging:

- wrong conda environment
- base conda environment used accidentally
- system Python used accidentally
- system pip used accidentally
- missing Dassl installation
- missing CLIP installation
- wrong dataset root
- wrong output root
- missing Caltech101 split file
- CUDA not available
- PyTorch / CUDA incompatibility
- invalid dependency version
- original script still using `/data` or `/output_dir`
- cache written to `/home/zixuan/`

## Git Rules

Before modifying files:

```bash
git status
```

Prefer working on:

```bash
reproduce/fap-server-setup
```

Do not commit:

- datasets
- checkpoints
- logs
- output directories
- caches
- virtual environments
- conda environments
- Miniconda installation files
- credentials
- server account information

After meaningful changes, summarize:

- files changed
- why they changed
- whether the changes were tested

## Output Report

At the end of each task, provide:

1. Current reproduction status.
2. What was inspected.
3. Files changed.
4. Why each change was necessary.
5. Commands executed.
6. Which commands succeeded.
7. Which commands failed.
8. Key error messages, if any.
9. Whether the repository is ready for the next step.
10. Next recommended action.

If a command fails, include the key error message and propose the smallest fix.

If no files were changed, explicitly state that no files were changed.