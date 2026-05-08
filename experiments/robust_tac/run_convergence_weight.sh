#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-9}"
PYTHON_BIN="${PYTHON_BIN:-/work1/zixuan/envs/conda_envs/fap/bin/python}"
ROOT="${ROOT:-/work1/zixuan/data/fap}"
TAC_ROOT="${TAC_ROOT:-/work1/zixuan/projects/2024-ICML-TAC}"
GUIDANCE_PATH="${GUIDANCE_PATH:-experiments/robust_tac/guidance_cache/caltech101_tac_guidance_full.npz}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

check_gpu_free() {
  local line mem util attempt
  for attempt in 1 2; do
    line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' -v gpu="${GPU_ID}" '$1 == gpu {print $2 " " $3}')"
    if [[ -z "${line}" ]]; then
      echo "GPU ${GPU_ID} not found" >&2
      exit 1
    fi
    read -r mem util <<< "${line}"
    echo "GPU ${GPU_ID} check attempt ${attempt}: memory.used=${mem}MiB utilization=${util}%"
    if (( mem <= 100 && util == 0 )); then
      return 0
    fi
    if (( mem <= 100 && util != 0 && attempt == 1 )); then
      sleep 10
      continue
    fi
    echo "GPU ${GPU_ID} is not free; stopping before launching the next experiment." >&2
    exit 2
  done
}

run_exp() {
  local name="$1"
  shift
  local args=("$@")
  local output_dir=""
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--output" && $((i + 1)) -lt ${#args[@]} ]]; then
      output_dir="${args[$((i + 1))]}"
      break
    fi
  done
  if [[ "${SKIP_EXISTING}" == "1" && -n "${output_dir}" && -s "${output_dir}/summary.csv" ]]; then
    echo "===== SKIP ${name}: ${output_dir}/summary.csv exists ====="
    return 0
  fi
  check_gpu_free
  echo "===== START ${name} ====="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" experiments/robust_tac/train_robust_tac.py \
    --dataset Caltech101 \
    --root "${ROOT}" \
    --mode vprompt \
    --batch-size 32 \
    --train-steps 2 \
    --eval-steps 100 \
    --eps 1/255 \
    --eval-eps 1/255 \
    --step-size 1/255 \
    --num-clusters auto \
    --prompt-depth 1 \
    --n-ctx 2 \
    --guidance-type tac_text \
    --guidance-path "${GUIDANCE_PATH}" \
    --tac-root "${TAC_ROOT}" \
    "${args[@]}"
  echo "===== DONE ${name} ====="
}

run_convergence() {
  for epochs in 5 10 20; do
    run_exp "converge_ep${epochs}_ca" \
      --epochs "${epochs}" \
      --lambda-ca 1.0 \
      --lambda-eg 0.0 \
      --lambda-fap 0.0 \
      --eg-type none \
      --output "outputs/robust_tac/converge_ep${epochs}_ca"

    run_exp "converge_ep${epochs}_eg_text" \
      --epochs "${epochs}" \
      --lambda-ca 1.0 \
      --lambda-eg 0.5 \
      --lambda-fap 0.0 \
      --eg-type text_assignment \
      --output "outputs/robust_tac/converge_ep${epochs}_eg_text"

    run_exp "converge_ep${epochs}_eg_text_fapweight" \
      --epochs "${epochs}" \
      --lambda-ca 1.0 \
      --lambda-eg 0.5 \
      --lambda-fap 0.5 \
      --eg-type text_assignment \
      --use-fap-weight \
      --output "outputs/robust_tac/converge_ep${epochs}_eg_text_fapweight"
  done
}

run_weight_controls() {
  run_exp "weight_ep10_eg_text_lam0p5" \
    --epochs 10 \
    --lambda-ca 1.0 \
    --lambda-eg 0.5 \
    --lambda-fap 0.0 \
    --eg-type text_assignment \
    --output "outputs/robust_tac/weight_ep10_eg_text_lam0p5"

  run_exp "weight_ep10_eg_text_lam1p0" \
    --epochs 10 \
    --lambda-ca 1.0 \
    --lambda-eg 1.0 \
    --lambda-fap 0.0 \
    --eg-type text_assignment \
    --output "outputs/robust_tac/weight_ep10_eg_text_lam1p0"

  run_exp "weight_ep10_eg_text_lam1p5" \
    --epochs 10 \
    --lambda-ca 1.0 \
    --lambda-eg 1.5 \
    --lambda-fap 0.0 \
    --eg-type text_assignment \
    --output "outputs/robust_tac/weight_ep10_eg_text_lam1p5"

  run_exp "weight_ep10_eg_text_lam2p0" \
    --epochs 10 \
    --lambda-ca 1.0 \
    --lambda-eg 2.0 \
    --lambda-fap 0.0 \
    --eg-type text_assignment \
    --output "outputs/robust_tac/weight_ep10_eg_text_lam2p0"

  run_exp "weight_ep10_eg_text_fap_lam0p5_0p5" \
    --epochs 10 \
    --lambda-ca 1.0 \
    --lambda-eg 0.5 \
    --lambda-fap 0.5 \
    --eg-type text_assignment \
    --use-fap-weight \
    --output "outputs/robust_tac/weight_ep10_eg_text_fap_lam0p5_0p5"
}

case "${1:-all}" in
  convergence)
    run_convergence
    ;;
  weight)
    run_weight_controls
    ;;
  all)
    run_convergence
    run_weight_controls
    ;;
  *)
    echo "Usage: $0 [convergence|weight|all]" >&2
    exit 1
    ;;
esac

"${PYTHON_BIN}" experiments/robust_tac/summarize_convergence_weight.py \
  --outputs-root outputs/robust_tac \
  --output outputs/robust_tac/convergence_weight_summary.csv
