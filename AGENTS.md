# AGENTS.md

## Project Role

This repository is used to reproduce and manage experiments for **FAP: Few-Shot Adversarial Prompt Learning on Vision-Language Models**.

The primary goal is **reliable reproduction**. Do not optimize, refactor, redesign, or extend the method unless explicitly requested.

This repository follows a Dassl/CoOp-style experiment pipeline. The main entry point is `train.py`, and the FAP trainer is implemented in `trainers/fap.py`.

## Repository Information

- User fork: `https://github.com/Rhett-Lin/FAP`
- Original upstream: `https://github.com/lionel-w2/FAP`
- Local project path: `/work1/zixuan/projects/FAP`

Do not store server credentials, passwords, private keys, tokens, or personal account information in this repository.

## Server Rules

This repository is used on a shared RIKEN server.

Strict rules:

- Do not create projects under `/home/zixuan/`.
- Do not store datasets, checkpoints, logs, outputs, caches, or experiment results under `/home/zixuan/`.
- All project files must stay under `/work1/zixuan/projects/FAP`.
- All datasets must stay under `/work1/zixuan/data/fap`.
- All experiment outputs must stay under `/work1/zixuan/outputs/FAP`.
- All Python environments must stay under `/work1/zixuan/envs`.
- Miniconda is allowed only because explicit permission has been obtained.
- Install Miniconda only under `/work1/zixuan/envs/miniconda3`.
- Do not install Miniconda, conda environments, conda packages, pip packages, datasets, checkpoints, or caches under `/home/zixuan/`.
- Do not use full Anaconda distribution unless explicitly permitted.
- Do not install Python packages globally.
- Do not use system-wide pip.
- Do not create external tunneling, port forwarding, P2P, remote mapping, or unauthorized network processes.
- Do not write scripts that expose account information.
- Do not commit datasets, checkpoints, logs, caches, virtual environments, or generated experiment results.

## Required Local Paths

Use the following fixed paths:

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

Before running setup, installation, or experiments, make sure the required directories exist:

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

## Miniconda Policy

Miniconda may be used only under the following constraints:

- Miniconda must be installed to `/work1/zixuan/envs/miniconda3`.
- The FAP environment must be created at `/work1/zixuan/envs/conda_envs/fap`.
- Conda package cache must be stored under `/work1/zixuan/cache/conda_pkgs`.
- Pip cache must be stored under `/work1/zixuan/cache/pip`.
- Torch and CLIP caches must be stored under `/work1/zixuan/cache`.
- Do not run `conda init` unless explicitly requested.
- Do not modify `/home/zixuan/.bashrc` automatically.
- Do not activate or use the base environment for project work.
- Do not install packages into the base environment except when strictly required for conda itself.
- Do not use system Python or system pip for this repository.
- Do not use sudo or apt for environment setup.

## Miniconda Installation

If Miniconda is not installed, install it under `/work1/zixuan/envs/miniconda3`.

Recommended commands:

```bash
mkdir -p /work1/zixuan/envs/tools
cd /work1/zixuan/envs/tools

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

bash /work1/zixuan/envs/tools/miniconda.sh -b -p /work1/zixuan/envs/miniconda3
```

If `wget` is unavailable, use:

```bash
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /work1/zixuan/envs/tools/miniconda.sh

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

## Conda Environment Creation

Before creating or using the FAP environment, set safe cache paths:

```bash
export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache
```

Create the FAP environment with Python 3.8:

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

After activation, verify:

```bash
which python
which pip
python --version
pip --version
conda info --envs
```

Expected paths:

```bash
/work1/zixuan/envs/conda_envs/fap/bin/python
/work1/zixuan/envs/conda_envs/fap/bin/pip
```

If `which python` or `which pip` points outside `/work1/zixuan/envs/conda_envs/fap`, stop and fix the environment before continuing.

## Cache Paths

Always use server-safe cache paths:

```bash
export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache
```

If these variables are missing, set them before installing packages, importing CLIP, downloading model weights, or running training.

After activation, verify:

```bash
echo $CONDA_PKGS_DIRS
echo $PIP_CACHE_DIR
echo $TORCH_HOME
echo $XDG_CACHE_HOME
```

Expected values:

```bash
/work1/zixuan/cache/conda_pkgs
/work1/zixuan/cache/pip
/work1/zixuan/cache/torch
/work1/zixuan/cache
```

## Dependency Policy

Install packages only inside the activated FAP conda environment.

Before installing packages, always check:

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

only after confirming that pip belongs to the FAP environment.

Do not use system-wide pip.

Do not install packages globally.

Do not install packages into the conda base environment.

Do not blindly install the original `requirements.txt` if it contains invalid, duplicated, or conflicting entries.

If dependency conflicts occur, explain the conflict and propose the smallest compatible fix.

Prefer minimal dependency changes that preserve the original FAP reproduction setting.

## Suggested Dependency Setup

First inspect GPU and CUDA information:

```bash
nvidia-smi
nvcc --version
```

Candidate PyTorch installation command:

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

If this fails, do not randomly upgrade PyTorch. First inspect:

```bash
python --version
nvidia-smi
pip --version
```

Then propose a minimal compatible alternative.

Dassl should be installed from a local clone under `/work1/zixuan/projects`:

```bash
cd /work1/zixuan/projects
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git

cd /work1/zixuan/projects/Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
```

FAP dependencies should be installed inside the FAP repository.

If needed, create a cleaned dependency file such as `requirements_clean.txt` rather than blindly installing the original `requirements.txt`.

Example cleaned dependency file:

```bash
cd /work1/zixuan/projects/FAP

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

Install FAP dependencies:

```bash
pip install -r requirements_clean.txt
pip install git+https://github.com/openai/CLIP.git
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
echo $CONDA_PKGS_DIRS
echo $PIP_CACHE_DIR
echo $TORCH_HOME
echo $XDG_CACHE_HOME
nvidia-smi
python train.py --help
```

Then verify imports:

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

## Dataset Rules

The dataset root must be:

```bash
/work1/zixuan/data/fap
```

For the first reproduction run, the expected Caltech101 dataset structure is:

```bash
/work1/zixuan/data/fap/caltech-101/101_ObjectCategories
/work1/zixuan/data/fap/caltech-101/split_zhou_Caltech101.json
```

Do not place datasets under `/home/zixuan/`.

Do not commit datasets to Git.

Do not use P2P download tools.

If data is missing, report exactly which path is missing.

## Output Rules

The output root must be:

```bash
/work1/zixuan/outputs/FAP
```

Do not write outputs to:

```bash
/output_dir
/home/zixuan
```

Do not overwrite existing results unless explicitly requested.

Keep logs and checkpoints organized by dataset, shots, seed, and trainer.

## Reproduction Priority

Follow this order:

1. Verify repository location.
2. Verify Miniconda installation.
3. Verify FAP conda environment.
4. Verify cache paths.
5. Verify dependencies.
6. Verify `python train.py --help`.
7. Verify dataset path.
8. Create or verify server-specific few-shot script.
9. Run the smallest few-shot experiment.
10. Debug errors with minimal changes.
11. Extend to more seeds only after the first run succeeds.
12. Extend to more shots only after seed runs succeed.
13. Extend to more datasets only after the minimal dataset succeeds.
14. Consider base-to-new or cross-dataset experiments only after few-shot reproduction works.

## Minimal Target Experiment

The first target experiment is:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/few_shot_zixuan.sh caltech101 16 0
```

Do not launch large-scale experiments before this minimal run succeeds.

Do not run base-to-new or cross-dataset experiments before the minimal few-shot experiment succeeds.

## Script Modification Rules

If script paths are wrong, do not modify the original script directly.

Instead:

1. Copy the original script.
2. Rename it with `_zixuan`.
3. Modify only server-specific paths.
4. Preserve the original experiment logic.

For few-shot reproduction:

```bash
cp scripts/Adv/fap/few_shot.sh scripts/Adv/fap/few_shot_zixuan.sh
```

Expected path settings inside `few_shot_zixuan.sh`:

```bash
DATA=/work1/zixuan/data/fap
OUTPUT_DIR=/work1/zixuan/outputs/FAP
```

Do not change trainer logic, model logic, loss functions, adversarial attack logic, or evaluation logic unless explicitly requested.

## Coding Rules

- Prefer minimal changes.
- Do not change the core FAP method unless explicitly requested.
- Do not change model logic, loss functions, adversarial training logic, or evaluation logic unless necessary for reproduction.
- Preserve original scripts as references.
- Make path changes explicit and easy to review.
- Avoid hard-coded credentials or private information.
- Do not silently overwrite previous results.
- If code must be changed, explain why the change is necessary for reproduction.

## Experiment Rules

Before launching any experiment:

```bash
nvidia-smi
```

Use `CUDA_VISIBLE_DEVICES` to select a GPU.

Use `tmux` for long-running experiments.

Do not run multiple large jobs without checking GPU usage.

Do not launch long-running jobs unless explicitly requested.

For long experiments, use a command pattern like:

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

## Git Rules

Work on a reproduction branch, not directly on `main`, unless explicitly requested.

Recommended branch:

```bash
reproduce/fap-server-setup
```

Before modifying files:

```bash
git status
```

After meaningful changes, summarize the changed files.

Do not commit:

- datasets
- checkpoints
- logs
- outputs
- caches
- virtual environments
- private credentials
- server account information
- Miniconda installation files
- conda environments

Keep `.gitignore` updated for outputs, datasets, caches, and local environments.

## Reporting Rules

At the end of each task, report:

1. What was inspected.
2. What files were changed.
3. Why each change was necessary.
4. What commands were run.
5. Whether each command succeeded or failed.
6. Key error messages if any command failed.
7. Whether the repository is ready for the next reproduction step.
8. The next recommended action.

If a command fails, include the key error message and propose the smallest fix.

Do not hide failed commands.

Do not claim success unless the relevant verification command passed.