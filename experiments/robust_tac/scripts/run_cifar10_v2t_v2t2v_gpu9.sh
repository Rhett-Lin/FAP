#!/usr/bin/env bash
set -euo pipefail

cd /work1/zixuan/projects/FAP

GPU_ID="${GPU_ID:-9}"
PY="/work1/zixuan/envs/conda_envs/fap/bin/python"
DATA_ROOT="/work1/zixuan/data/fap"
TAC_ROOT="/work1/zixuan/projects/2024-ICML-TAC"
GUIDANCE_PATH="experiments/robust_tac/guidance_cache/cifar10_tac_guidance_full.npz"
LOG_DIR="outputs/robust_tac/logs"

STAGE0_OUT="outputs/robust_tac/cifar10_stage0_vprompt_ca_eg"
STAGE1_OUT="outputs/robust_tac/cifar10_v2t_tprompt_teacher"
STAGE2_OUT="outputs/robust_tac/cifar10_v2t2v_fixed_teacher"
STAGE2_ABLATION_OUT="outputs/robust_tac/cifar10_v2t2v_fixed_teacher_no_stage1_heads"

mkdir -p "${LOG_DIR}"

export CONDA_PKGS_DIRS=/work1/zixuan/cache/conda_pkgs
export PIP_CACHE_DIR=/work1/zixuan/cache/pip
export TORCH_HOME=/work1/zixuan/cache/torch
export XDG_CACHE_HOME=/work1/zixuan/cache

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

run_stage0() {
  local log="${LOG_DIR}/cifar10_stage0_vprompt_ca_eg.log"
  check_gpu_free "${log}"
  echo "===== $(date '+%F %T') start Stage0 VPrompt+CA+EG =====" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" experiments/robust_tac/train_robust_tac.py \
    --dataset CIFAR10 \
    --root "${DATA_ROOT}" \
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
    --guidance-path "${GUIDANCE_PATH}" \
    --tac-root "${TAC_ROOT}" \
    --output "${STAGE0_OUT}" 2>&1 | tee -a "${log}"
  echo "===== $(date '+%F %T') done Stage0 =====" | tee -a "${log}"
}

run_stage1_v2t() {
  local log="${LOG_DIR}/cifar10_v2t_tprompt_teacher.log"
  test -f "${STAGE0_OUT}/checkpoint.pt"
  check_gpu_free "${log}"
  echo "===== $(date '+%F %T') start Stage1 V2T TPrompt teacher =====" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" experiments/robust_tac/train_robust_tac.py \
    --dataset CIFAR10 \
    --root "${DATA_ROOT}" \
    --mode tprompt \
    --epochs 10 \
    --batch-size 32 \
    --train-steps 1 \
    --eval-steps 20 \
    --lambda-ca 0.0 \
    --lambda-eg 0.0 \
    --eg-type text_assignment \
    --prompt-depth 1 \
    --n-ctx 2 \
    --num-clusters auto \
    --guidance-type tac_text \
    --guidance-path "${GUIDANCE_PATH}" \
    --tac-root "${TAC_ROOT}" \
    --frozen-visual-checkpoint "${STAGE0_OUT}/checkpoint.pt" \
    --freeze-loaded-visual-prompt \
    --save-teacher-path "${STAGE1_OUT}/teacher.npz" \
    --output "${STAGE1_OUT}" 2>&1 | tee -a "${log}"
  echo "===== $(date '+%F %T') done Stage1 =====" | tee -a "${log}"
}

run_stage2_v2t2v() {
  local log="${LOG_DIR}/cifar10_v2t2v_fixed_teacher.log"
  test -f "${STAGE0_OUT}/checkpoint.pt"
  test -f "${STAGE1_OUT}/teacher.npz"
  test -f "${STAGE1_OUT}/checkpoint.pt"
  check_gpu_free "${log}"
  echo "===== $(date '+%F %T') start Stage2 V2T2V fixed teacher =====" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" experiments/robust_tac/train_robust_tac.py \
    --dataset CIFAR10 \
    --root "${DATA_ROOT}" \
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
    --guidance-path "${GUIDANCE_PATH}" \
    --tac-root "${TAC_ROOT}" \
    --teacher-type fixed_q \
    --teacher-path "${STAGE1_OUT}/teacher.npz" \
    --stage0-checkpoint "${STAGE0_OUT}/checkpoint.pt" \
    --stage1-checkpoint "${STAGE1_OUT}/checkpoint.pt" \
    --load-stage1-heads \
    --output "${STAGE2_OUT}" 2>&1 | tee -a "${log}"
  echo "===== $(date '+%F %T') done Stage2 =====" | tee -a "${log}"
}

run_stage2_v2t2v_no_stage1_heads() {
  local log="${LOG_DIR}/cifar10_v2t2v_fixed_teacher_no_stage1_heads.log"
  test -f "${STAGE0_OUT}/checkpoint.pt"
  test -f "${STAGE1_OUT}/teacher.npz"
  check_gpu_free "${log}"
  echo "===== $(date '+%F %T') start Stage2 ablation no Stage1 heads =====" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" experiments/robust_tac/train_robust_tac.py \
    --dataset CIFAR10 \
    --root "${DATA_ROOT}" \
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
    --guidance-path "${GUIDANCE_PATH}" \
    --tac-root "${TAC_ROOT}" \
    --teacher-type fixed_q \
    --teacher-path "${STAGE1_OUT}/teacher.npz" \
    --stage0-checkpoint "${STAGE0_OUT}/checkpoint.pt" \
    --output "${STAGE2_ABLATION_OUT}" 2>&1 | tee -a "${log}"
  echo "===== $(date '+%F %T') done Stage2 ablation =====" | tee -a "${log}"
}

main() {
  echo "===== CIFAR10 V2T/V2T2V sequential prompt experiment ====="
  echo "GPU_ID=${GPU_ID}"
  echo "GUIDANCE_PATH=${GUIDANCE_PATH}"
  echo "DATA_ROOT=${DATA_ROOT}"
  test -f "${GUIDANCE_PATH}"
  run_stage0
  run_stage1_v2t
  run_stage2_v2t2v
  run_stage2_v2t2v_no_stage1_heads
  echo "===== completed all configured stages ====="
}

main "$@"
