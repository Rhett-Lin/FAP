# AGENTS.md

## Project Role

This repository is used to reproduce and manage experiments for **FAP: Few-Shot Adversarial Prompt Learning on Vision-Language Models**.

The primary goal is **reliable reproduction**. Do not optimize, refactor, redesign, or extend the method unless explicitly requested.

The repository is maintained as the user's fork:

- User fork: `https://github.com/Rhett-Lin/FAP`
- Original upstream: `https://github.com/lionel-w2/FAP`

This project is based on the Dassl framework and follows the CoOp / CoCoOp / MaPLe-style experiment pipeline.

---

## Server Rules

This repository is used on a shared RIKEN server. All actions must follow the server usage rules.

Strict rules:

- Do not create projects under `/home/zixuan/`.
- Do not store datasets under `/home/zixuan/`.
- Do not store checkpoints under `/home/zixuan/`.
- Do not store logs under `/home/zixuan/`.
- Do not store experiment outputs under `/home/zixuan/`.
- The project path must be:

```bash
/work1/zixuan/projects/FAP
````

* The Python virtual environment path must be:

```bash
/work1/zixuan/envs/fap
```

* The dataset root must be:

```bash
/work1/zixuan/data/fap
```

* The output root must be:

```bash
/work1/zixuan/outputs/FAP
```

* Do not use Anaconda.
* Do not use conda.
* Use Python `venv` and `pip` only.
* Do not install packages globally.
* Do not modify system-level Python packages.
* Do not create external tunneling processes.
* Do not create port forwarding processes.
* Do not create P2P transfer processes.
* Do not create unauthorized remote mapping or external network exposure processes.
* Do not store server credentials, passwords, private keys, tokens, or personal account information in this repository.
* Do not write scripts that expose account information.
* Do not commit datasets, checkpoints, logs, generated outputs, or private configuration files.

---

## Directory Layout

Expected server-side directory layout:

```bash
/work1/zixuan/
├── envs/
│   └── fap/
├── projects/
│   └── FAP/
├── data/
│   └── fap/
└── outputs/
    └── FAP/
```

The repository should be located at:

```bash
/work1/zixuan/projects/FAP
```

The Python environment should be activated by:

```bash
source /work1/zixuan/envs/fap/bin/activate
```

Before running any command, verify the working directory and Python environment:

```bash
pwd
which python
which pip
python --version
```

---

## Reproduction Priority

Follow this order strictly:

1. Inspect the repository structure.
2. Verify the Python environment.
3. Verify lightweight imports.
4. Verify that `python train.py --help` works.
5. Prepare or validate the dataset path.
6. Create server-specific run scripts if necessary.
7. Run the smallest few-shot experiment first.
8. Debug errors with minimal changes.
9. Only after the smallest experiment works, extend to more seeds.
10. Only after seed reproduction works, extend to more shots.
11. Only after few-shot reproduction works, consider base-to-new experiments.
12. Only after base-to-new is understood, consider cross-dataset experiments.

Do not launch large-scale experiments before the minimal few-shot run succeeds.

---

## Minimal Target Experiment

The first target experiment should be:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/few_shot_zixuan.sh caltech101 16 0
```

This corresponds to:

* Dataset: `caltech101`
* Shots: `16`
* Seed: `0`

Do not run base-to-new or cross-dataset experiments before this minimal experiment succeeds.

---

## Coding Rules

General coding rules:

* Prefer minimal changes.
* Do not change the core FAP method unless explicitly requested.
* Do not change model architecture unless explicitly requested.
* Do not change loss functions unless explicitly requested.
* Do not change adversarial training logic unless explicitly requested.
* Do not change evaluation logic unless explicitly requested.
* Do not refactor unrelated code.
* Do not rename public functions or classes unless necessary.
* Do not silently remove existing functionality.
* Do not silently overwrite previous results.
* Do not hard-code credentials or private information.
* Do not add unnecessary dependencies.
* Preserve original experiment logic.

Treat the following files as sensitive:

* trainer implementations
* model definitions
* loss functions
* adversarial training logic
* dataset loaders
* evaluation code
* config files defining the experimental protocol

Only modify these files if the error cannot be solved by environment, dependency, path, or script-level fixes.

When path changes are needed:

* Do not directly modify the original script unless explicitly requested.
* Copy the original script and create a user-specific version.
* For example:

```bash
cp scripts/Adv/fap/few_shot.sh scripts/Adv/fap/few_shot_zixuan.sh
```

Expected path variables in server-specific scripts:

```bash
DATA=/work1/zixuan/data/fap
OUTPUT_DIR=/work1/zixuan/outputs/FAP
```

Server-specific scripts are for local reproduction only. They should not change the experimental protocol.

Keep original scripts as references.

---

## Environment Rules

Use only the virtual environment located at:

```bash
/work1/zixuan/envs/fap
```

Activate it with:

```bash
source /work1/zixuan/envs/fap/bin/activate
```

Before installing packages, always check:

```bash
which python
which pip
python --version
```

Do not use:

```bash
conda
anaconda
```

Do not install Python packages globally.

If dependency versions conflict:

1. Report the conflict.
2. Identify the exact failing package.
3. Propose the smallest compatible change.
4. Do not upgrade major dependencies unless required.
5. Avoid changing PyTorch version unless necessary.
6. Prefer compatibility with the original repository.

---

## Lightweight Verification Commands

Use these commands before running training:

```bash
pwd
git status
which python
which pip
python --version
nvidia-smi
```

Verify PyTorch:

```bash
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
PY
```

Verify key imports:

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
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("imports passed")
PY
```

Verify the training entry:

```bash
python train.py --help
```

These checks are lightweight and should be performed before launching long experiments.

---

## Experiment Rules

Before launching any training job:

1. Check GPU usage:

```bash
nvidia-smi
```

2. Select GPU explicitly by prefixing the training command:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/few_shot_zixuan.sh caltech101 16 0
```

Alternatively:

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/Adv/fap/few_shot_zixuan.sh caltech101 16 0
```

3. Confirm the dataset path exists:

```bash
ls /work1/zixuan/data/fap
```

4. Confirm the output path exists:

```bash
mkdir -p /work1/zixuan/outputs/FAP
```

5. Use `tmux` for long-running experiments.

Recommended tmux usage:

```bash
tmux new -s fap
```

Detach from tmux:

```text
Ctrl + B
D
```

Reattach:

```bash
tmux attach -t fap
```

List sessions:

```bash
tmux ls
```

Experiment outputs must be saved under:

```bash
/work1/zixuan/outputs/FAP
```

Keep logs and checkpoints organized by:

* trainer
* dataset
* shots
* seed
* configuration
* timestamp if necessary

Do not overwrite existing results unless explicitly requested.

---

## Dataset Rules

The dataset root must be:

```bash
/work1/zixuan/data/fap
```

Do not place datasets under:

```bash
/home/zixuan
```

For the first target dataset, the expected Caltech101 structure is usually:

```bash
/work1/zixuan/data/fap/caltech-101/
├── 101_ObjectCategories/
└── split_zhou_Caltech101.json
```

The script argument is `caltech101`, while the actual dataset folder may be `caltech-101`.

Before running the minimal experiment, check:

```bash
ls /work1/zixuan/data/fap/caltech-101
ls /work1/zixuan/data/fap/caltech-101/101_ObjectCategories
ls /work1/zixuan/data/fap/caltech-101/split_zhou_Caltech101.json
```

If the dataset is missing:

* Do not guess paths.
* Report the missing path.
* Explain what files or folders are expected.
* Do not download large datasets without explicit user approval.
* If a dataset is missing, report the expected dataset name, folder structure, and source instruction, but do not start downloading it automatically.
* Do not use P2P tools.
* Use only normal and allowed download methods.

---

## Script Rules

Original scripts should be preserved.

For server-specific experiments, create copies such as:

```bash
scripts/Adv/fap/few_shot_zixuan.sh
```

Do not modify:

```bash
scripts/Adv/fap/few_shot.sh
```

unless explicitly requested.

For `few_shot_zixuan.sh`, expected path settings are:

```bash
DATA=/work1/zixuan/data/fap
OUTPUT_DIR=/work1/zixuan/outputs/FAP
```

After creating or modifying a script, show the diff:

```bash
git diff scripts/Adv/fap/few_shot_zixuan.sh
```

---

## Debugging Rules

When an error occurs:

1. Read the full traceback.

2. Identify the smallest failing component.

3. Distinguish between:

   * environment error
   * dependency error
   * path error
   * dataset error
   * config error
   * code logic error

4. Prefer dependency and path fixes before code logic changes.

5. Do not refactor unrelated code.

6. Do not change model logic just to bypass an error.

7. If a code change is necessary, explain why.

8. Re-run the smallest relevant command to verify the fix.

When reporting an error, include:

* the command that failed
* the key traceback line
* the suspected cause
* the minimal proposed fix
* whether the fix was tested

---

## Git Rules

Work on a reproduction branch, not directly on `main`.

Recommended branch:

```bash
reproduce/fap-server-setup
```

Before making changes:

```bash
git status
```

After making changes:

```bash
git diff
```

Do not commit:

* datasets
* checkpoints
* logs
* generated outputs
* virtual environments
* credentials
* private keys
* tokens
* temporary files

Recommended commit style:

```bash
git add <changed-files>
git commit -m "Add server-specific FAP reproduction setup"
```

Before pushing:

```bash
git status
git log --oneline -5
```

---

## Reproduction Log Rules

Maintain reproduction notes in:

```bash
REPRODUCTION.md
```

When a meaningful step is completed, update the reproduction log with:

* date
* command
* environment
* result
* error if any
* next step

Do not exaggerate results.

Do not report an experiment as successfully reproduced unless the command actually finished and produced the expected output.

---

## Dependency Rules

Do not blindly install the original `requirements.txt` if it contains conflicting, duplicate, or invalid entries.

Before installing, inspect dependency files.

If creating a cleaned dependency file, name it clearly, for example:

```bash
requirements_clean.txt
```

Do not remove the original `requirements.txt`.

If a dependency issue occurs, first report:

* Python version
* PyTorch version
* CUDA availability
* failing package
* exact error message

Do not upgrade to the latest version of every package unless necessary.

---

## Codex Behavior Rules

When working as an agent:

* First inspect, then plan, then modify.
* Do not modify files during the initial inspection step.
* Do not run long training jobs without explicit instruction.
* Do not run multiple experiments at once.
* Do not create files outside the allowed project, environment, data, or output directories.
* Do not create files under `/home/zixuan/`.
* Do not store credentials.
* Do not use external tunneling or P2P tools.
* Do not make large code changes unless requested.
* Always summarize what was changed and why.
* Always state whether verification was completed or not.

---

## First Inspection Task

For the first Codex run, do not modify files.

The first task should inspect:

1. Repository structure.
2. Main training entry point.
3. Trainer implementation related to FAP.
4. Config files used by the few-shot script.
5. Scripts under `scripts/Adv/fap/`.
6. Expected dataset root.
7. Expected output root.
8. Path assumptions that must be changed for this server.
9. Minimal environment verification commands.
10. Minimal first reproduction command.

The first Codex run should only produce a reproduction plan.

---

## Expected First Reproduction Command

After environment, dataset, and script paths are verified, run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/few_shot_zixuan.sh caltech101 16 0
```

Before running, confirm:

```bash
pwd
git status
which python
which pip
python --version
nvidia-smi
ls /work1/zixuan/data/fap
ls /work1/zixuan/outputs/FAP
```

---

## Final Report Format

At the end of each task, report:

### Summary

Briefly state what was done.

### Files Inspected

List the important files inspected.

### Files Changed

List changed files. If no files were changed, say so.

### Commands Run

List commands that were executed.

### Results

State whether each command succeeded or failed.

### Key Error

If there was an error, include the key error message.

### Next Step

Suggest the next minimal step.

---

## Important Reminder

The goal is to reproduce the original FAP results in a controlled and traceable way.

Do not prioritize speed over correctness.

Do not make unnecessary changes.

Do not violate server usage rules.

Always keep the repository clean, reproducible, and easy to review.

