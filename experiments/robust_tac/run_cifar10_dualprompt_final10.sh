#!/usr/bin/env bash
set -euo pipefail

cd /work1/zixuan/projects/FAP

GPU_ID="${GPU_ID:-9}"
PY="/work1/zixuan/envs/conda_envs/fap/bin/python"
GUIDANCE="experiments/robust_tac/guidance_cache/cifar10_tac_guidance_full.npz"
LOG_DIR="outputs/robust_tac/cifar10_dualprompt_logs"
SUMMARY="outputs/robust_tac/cifar10_dualprompt_summary.csv"
mkdir -p "$LOG_DIR"

check_gpu_free() {
  local log="$1"
  echo "===== $(date '+%F %T') GPU check before run =====" | tee -a "$log"
  nvidia-smi | tee -a "$log"
  local busy
  busy="$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits | grep "$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F, -v id="$GPU_ID" '$1+0==id {gsub(/^ /, "", $2); print $2}')" || true)"
  if [[ -n "$busy" ]]; then
    echo "GPU ${GPU_ID} is busy; stop without launching experiments." | tee -a "$log"
    echo "$busy" | tee -a "$log"
    exit 1
  fi
}

COMMON=(
  --dataset CIFAR10
  --root /work1/zixuan/data
  --epochs 10
  --batch-size 32
  --eps 1/255
  --eval-eps 1/255
  --step-size 1/255
  --train-steps 1
  --eval-steps 20
  --lambda-ca 1.0
  --lambda-eg 0.5
  --eg-type text_assignment
  --prompt-depth 1
  --n-ctx 2
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
  check_gpu_free "$log"
  echo "===== $(date '+%F %T') start ${name} =====" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" experiments/robust_tac/train_robust_tac.py \
    "${COMMON[@]}" "$@" --output "$out" 2>&1 | tee -a "$log"
  echo "===== $(date '+%F %T') done ${name} =====" | tee -a "$log"
}

run_one cifar10_final10_dualprompt_ca_eg \
  --mode dualprompt

run_one cifar10_final10_dualprompt_ca_eg_sym005 \
  --mode dualprompt --eg-symmetric --eta-sym 0.05

run_one cifar10_final10_dualprompt_ca_eg_sym01 \
  --mode dualprompt --eg-symmetric --eta-sym 0.1

run_one cifar10_final10_dualprompt_ca_eg_fap \
  --mode dualprompt --lambda-fap 0.5 --use-fap-weight

run_one cifar10_final10_dualprompt_ca_eg_fap_sym005 \
  --mode dualprompt --lambda-fap 0.5 --use-fap-weight --eg-symmetric --eta-sym 0.05

"$PY" - <<'PY'
import csv
import os

baseline_tprompt = [
    ("TPrompt+CA+EG", "cifar10_final10_tprompt_ca_eg_text"),
    ("TPrompt+CA+EG sym005", "cifar10_final10_tprompt_ca_eg_text_sym005"),
    ("TPrompt+CA+EG sym01", "cifar10_final10_tprompt_ca_eg_text_sym01"),
    ("TPrompt+CA+EG+FAP", "cifar10_final10_tprompt_ca_eg_text_fap"),
    ("TPrompt+CA+EG+FAP sym005", "cifar10_final10_tprompt_ca_eg_text_fap_sym005"),
]
fixed_runs = [
    ("Static+CA+EG", "cifar10_final10_static_ca_eg_text"),
    ("VPrompt+CA+EG", "cifar10_final10_vprompt_ca_eg_text"),
]
dual_runs = [
    ("DualPrompt+CA+EG", "cifar10_final10_dualprompt_ca_eg"),
    ("DualPrompt+CA+EG sym005", "cifar10_final10_dualprompt_ca_eg_sym005"),
    ("DualPrompt+CA+EG sym01", "cifar10_final10_dualprompt_ca_eg_sym01"),
    ("DualPrompt+CA+EG+FAP", "cifar10_final10_dualprompt_ca_eg_fap"),
    ("DualPrompt+CA+EG+FAP sym005", "cifar10_final10_dualprompt_ca_eg_fap_sym005"),
]
fields = [
    "method", "clean_acc", "clean_nmi", "clean_ari",
    "adv_acc", "adv_nmi", "adv_ari", "cfr",
    "clean_adv_kl_eval", "eg_kl_eval", "mean_cos_factor_eval",
    "visual_prompt_trainable_params", "text_prompt_trainable_params",
    "total_trainable_params", "use_fap_weight", "eg_symmetric", "eta_sym",
]

def load(method, run_dir):
    path = f"outputs/robust_tac/{run_dir}/summary.csv"
    with open(path) as f:
        row = next(csv.DictReader(f))
    row["method"] = method
    if "visual_prompt_trainable_params" not in row:
        row["visual_prompt_trainable_params"] = row.get("prompt_trainable_params", "")
    return {field: row.get(field, "") for field in fields}

rows = []
missing = []
for item in fixed_runs:
    path = f"outputs/robust_tac/{item[1]}/summary.csv"
    if os.path.exists(path):
        rows.append(load(*item))
    else:
        missing.append(path)

tprompt_rows = []
for item in baseline_tprompt:
    path = f"outputs/robust_tac/{item[1]}/summary.csv"
    if os.path.exists(path):
        tprompt_rows.append(load(*item))
if tprompt_rows:
    best = max(tprompt_rows, key=lambda row: float(row["clean_acc"]))
    best["method"] = "Best TPrompt"
    rows.append(best)
else:
    missing.extend(f"outputs/robust_tac/{run_dir}/summary.csv" for _, run_dir in baseline_tprompt)

for item in dual_runs:
    rows.append(load(*item))

os.makedirs("outputs/robust_tac", exist_ok=True)
with open("outputs/robust_tac/cifar10_dualprompt_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("summary: outputs/robust_tac/cifar10_dualprompt_summary.csv")
if missing:
    print("missing_baseline_summaries:")
    for path in missing:
        print(path)
PY
