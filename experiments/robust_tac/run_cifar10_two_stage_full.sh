#!/usr/bin/env bash
set -euo pipefail

cd /work1/zixuan/projects/FAP

GPU_ID="${GPU_ID:-9}"
PY="/work1/zixuan/envs/conda_envs/fap/bin/python"
DATA_ROOT="/work1/zixuan/data"
TAC_ROOT="/work1/zixuan/projects/2024-ICML-TAC"
GUIDANCE="experiments/robust_tac/guidance_cache/cifar10_tac_guidance_full.npz"
OUTPUT_ROOT="/work1/zixuan/outputs/FAP/robust_tac"

STAGE1_OUT="${OUTPUT_ROOT}/cifar10_stage1_tprompt_clean_teacher"
STAGE2_OUT="${OUTPUT_ROOT}/cifar10_stage2_vprompt_fixed_tprompt_teacher"
LOG_DIR="${OUTPUT_ROOT}/cifar10_two_stage_logs"
SUMMARY="${OUTPUT_ROOT}/cifar10_two_stage_prompt_summary.csv"

TEACHER_PATH="${STAGE1_OUT}/teacher.npz"
STAGE1_CKPT="${STAGE1_OUT}/checkpoint.pt"

mkdir -p "${LOG_DIR}"

export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache

check_env() {
  echo "===== environment check ====="
  pwd
  "${PY}" --version
  echo "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
  echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
  echo "TORCH_HOME=${TORCH_HOME}"
  echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
  test -f "${GUIDANCE}"
  test -d "${DATA_ROOT}/cifar-10-batches-py"
}

check_gpu_free() {
  local log="$1"
  echo "===== $(date '+%F %T') GPU check before run =====" | tee -a "${log}"
  nvidia-smi | tee -a "${log}"
  local bus_id
  bus_id="$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F, -v id="${GPU_ID}" '$1+0==id {gsub(/^ /, "", $2); print $2}')"
  if [[ -z "${bus_id}" ]]; then
    echo "GPU ${GPU_ID} not found." | tee -a "${log}"
    exit 1
  fi
  local busy
  busy="$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits | grep "${bus_id}" || true)"
  if [[ -n "${busy}" ]]; then
    echo "GPU ${GPU_ID} is busy; stop without launching experiments." | tee -a "${log}"
    echo "${busy}" | tee -a "${log}"
    exit 1
  fi
}

run_stage1() {
  local log="${LOG_DIR}/cifar10_stage1_tprompt_clean_teacher.log"
  check_gpu_free "${log}"
  echo "===== $(date '+%F %T') start Stage 1: TPrompt clean teacher =====" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" experiments/robust_tac/train_robust_tac.py \
    --dataset CIFAR10 \
    --root "${DATA_ROOT}" \
    --mode tprompt \
    --epochs 10 \
    --batch-size 32 \
    --num-workers 2 \
    --train-steps 1 \
    --eval-steps 20 \
    --lambda-ca 0.0 \
    --lambda-eg 0.0 \
    --eg-type text_assignment \
    --n-ctx 2 \
    --num-clusters auto \
    --guidance-type tac_text \
    --guidance-path "${GUIDANCE}" \
    --tac-root "${TAC_ROOT}" \
    --save-teacher-path "${TEACHER_PATH}" \
    --output "${STAGE1_OUT}" 2>&1 | tee -a "${log}"
  echo "===== $(date '+%F %T') done Stage 1 =====" | tee -a "${log}"
}

run_stage2() {
  local log="${LOG_DIR}/cifar10_stage2_vprompt_fixed_tprompt_teacher.log"
  test -f "${TEACHER_PATH}"
  test -f "${STAGE1_CKPT}"
  check_gpu_free "${log}"
  echo "===== $(date '+%F %T') start Stage 2: VPrompt fixed TPrompt teacher =====" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" experiments/robust_tac/train_robust_tac.py \
    --dataset CIFAR10 \
    --root "${DATA_ROOT}" \
    --mode vprompt \
    --epochs 10 \
    --batch-size 32 \
    --num-workers 2 \
    --train-steps 1 \
    --eval-steps 20 \
    --lambda-ca 1.0 \
    --lambda-eg 0.5 \
    --eg-type text_assignment \
    --prompt-depth 1 \
    --n-ctx 2 \
    --num-clusters auto \
    --guidance-type tac_text \
    --guidance-path "${GUIDANCE}" \
    --tac-root "${TAC_ROOT}" \
    --teacher-type fixed_q \
    --teacher-path "${TEACHER_PATH}" \
    --stage1-checkpoint "${STAGE1_CKPT}" \
    --load-stage1-heads \
    --output "${STAGE2_OUT}" 2>&1 | tee -a "${log}"
  echo "===== $(date '+%F %T') done Stage 2 =====" | tee -a "${log}"
}

write_final_summary() {
  "${PY}" - <<'PY'
import csv
import os

output_root = "/work1/zixuan/outputs/FAP/robust_tac"
summary_path = os.path.join(output_root, "cifar10_two_stage_prompt_summary.csv")
runs = [
    {
        "method": "VPrompt+CA+EG baseline from plan",
        "stage": "baseline",
        "mode": "vprompt",
        "teacher_type": "none",
        "clean_acc": "0.9062",
        "clean_nmi": "",
        "clean_ari": "",
        "adv_acc": "0.7161",
        "adv_nmi": "",
        "adv_ari": "0.4929",
        "cfr": "0.2768",
        "clean_adv_kl_eval": "",
        "eg_kl_eval": "",
        "mean_cos_factor_eval": "",
        "teacher_path": "",
        "stage1_checkpoint": "",
        "visual_prompt_trainable_params": "",
        "text_prompt_trainable_params": "",
        "total_trainable_params": "",
    },
    {
        "method": "Stage1 TPrompt clean teacher",
        "path": os.path.join(output_root, "cifar10_stage1_tprompt_clean_teacher", "summary.csv"),
    },
    {
        "method": "Stage2 VPrompt fixed clean teacher",
        "path": os.path.join(output_root, "cifar10_stage2_vprompt_fixed_tprompt_teacher", "summary.csv"),
    },
]
fields = [
    "method", "stage", "mode", "teacher_type",
    "clean_acc", "clean_nmi", "clean_ari",
    "adv_acc", "adv_nmi", "adv_ari", "cfr",
    "clean_adv_kl_eval", "eg_kl_eval", "mean_cos_factor_eval",
    "teacher_path", "stage1_checkpoint",
    "visual_prompt_trainable_params", "text_prompt_trainable_params",
    "total_trainable_params",
]

rows = []
for run in runs:
    if "path" not in run:
        rows.append({field: run.get(field, "") for field in fields})
        continue
    with open(run["path"], newline="") as f:
        row = next(csv.DictReader(f))
    row["method"] = run["method"]
    rows.append({field: row.get(field, "") for field in fields})

os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"summary_csv: {summary_path}")
PY
}

check_teacher() {
  "${PY}" - <<'PY'
import numpy as np

path = "/work1/zixuan/outputs/FAP/robust_tac/cifar10_stage1_tprompt_clean_teacher/teacher.npz"
data = np.load(path, allow_pickle=False)
for key in [
    "teacher_q_train",
    "teacher_q_test",
    "teacher_text_counterpart_train",
    "teacher_text_counterpart_test",
    "prompted_noun_anchor_bank",
]:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, finite={np.isfinite(arr).all()}")
print(
    "teacher_q_train row_sum:",
    float(data["teacher_q_train"].sum(axis=1).min()),
    float(data["teacher_q_train"].sum(axis=1).max()),
)
print(
    "teacher_q_test row_sum:",
    float(data["teacher_q_test"].sum(axis=1).min()),
    float(data["teacher_q_test"].sum(axis=1).max()),
)
PY
}

main() {
  check_env
  run_stage1
  check_teacher
  run_stage2
  write_final_summary
  echo "===== completed CIFAR10 two-stage full experiment ====="
  echo "Stage 1 output: ${STAGE1_OUT}"
  echo "Stage 2 output: ${STAGE2_OUT}"
  echo "Final summary: ${SUMMARY}"
}

main "$@"
