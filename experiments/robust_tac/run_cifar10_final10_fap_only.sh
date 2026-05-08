#!/usr/bin/env bash
set -euo pipefail

cd /work1/zixuan/projects/FAP

GPU_ID="${GPU_ID:-8}"
PY="/work1/zixuan/envs/conda_envs/fap/bin/python"
GUIDANCE="experiments/robust_tac/guidance_cache/cifar10_tac_guidance_full.npz"
LOG_DIR="outputs/robust_tac/cifar10_final10_logs"
SUMMARY="outputs/robust_tac/cifar10_final10_summary.csv"
mkdir -p "$LOG_DIR"

COMMON=(
  --dataset CIFAR10
  --root /work1/zixuan/data/fap
  --epochs 10
  --batch-size 32
  --eps 1/255
  --eval-eps 1/255
  --step-size 1/255
  --train-steps 2
  --eval-steps 100
  --lambda-ca 1.0
  --lambda-eg 0.5
  --eg-type text_assignment
  --num-clusters auto
  --guidance-type tac_text
  --guidance-path "$GUIDANCE"
  --tac-root /work1/zixuan/projects/2024-ICML-TAC
)

run_one() {
  local name="$1"
  shift
  local out="outputs/robust_tac/${name}"
  local log="${LOG_DIR}/${name}.log"
  echo "===== $(date '+%F %T') GPU check before ${name} =====" | tee -a "$log"
  nvidia-smi | tee -a "$log"
  echo "===== $(date '+%F %T') start ${name} on GPU ${GPU_ID} =====" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" experiments/robust_tac/train_robust_tac.py \
    "${COMMON[@]}" "$@" --output "$out" 2>&1 | tee -a "$log"
  echo "===== $(date '+%F %T') done ${name} =====" | tee -a "$log"
}

run_one cifar10_final10_tprompt_ca_eg_text_fap \
  --mode tprompt --lambda-fap 0.5 --use-fap-weight --n-ctx 2

run_one cifar10_final10_tprompt_ca_eg_text_fap_sym005 \
  --mode tprompt --lambda-fap 0.5 --use-fap-weight --eg-symmetric --eta-sym 0.05 --n-ctx 2

"$PY" - <<'PY'
import csv
import math
import os

runs = [
    ("Static+CA+EG", "cifar10_final10_static_ca_eg_text"),
    ("VPrompt+CA+EG", "cifar10_final10_vprompt_ca_eg_text"),
    ("TPrompt+CA+EG", "cifar10_final10_tprompt_ca_eg_text"),
    ("TPrompt+CA+EG sym005", "cifar10_final10_tprompt_ca_eg_text_sym005"),
    ("TPrompt+CA+EG sym01", "cifar10_final10_tprompt_ca_eg_text_sym01"),
    ("TPrompt+CA+EG+FAP", "cifar10_final10_tprompt_ca_eg_text_fap"),
    ("TPrompt+CA+EG+FAP sym005", "cifar10_final10_tprompt_ca_eg_text_fap_sym005"),
]
fields = [
    "method", "mode", "epochs", "clean_acc", "clean_nmi", "clean_ari",
    "adv_acc", "adv_nmi", "adv_ari", "cfr", "clean_adv_kl_eval",
    "eg_kl_eval", "mean_cos_factor_eval", "prompt_trainable_params",
    "text_prompt_trainable_params", "use_fap_weight", "eg_symmetric", "eta_sym",
]
rows = []
any_nan = False
for method, run_dir in runs:
    path = f"outputs/robust_tac/{run_dir}/summary.csv"
    with open(path) as f:
        src = next(csv.DictReader(f))
    row = {field: src.get(field, "") for field in fields}
    row["method"] = method
    row["epochs"] = row["epochs"] or "10"
    for value in row.values():
        try:
            any_nan = any_nan or math.isnan(float(value))
        except (TypeError, ValueError):
            pass
    rows.append(row)

out = "outputs/robust_tac/cifar10_final10_summary.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"summary: {out}")
print(f"any_nan: {any_nan}")
PY
