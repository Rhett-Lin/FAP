#!/usr/bin/env bash
set -euo pipefail

cd /work1/zixuan/projects/FAP
PY="/work1/zixuan/envs/conda_envs/fap/bin/python"

while true; do
  if [[ -f outputs/robust_tac/cifar10_final10_tprompt_ca_eg_text_fap/summary.csv && \
        -f outputs/robust_tac/cifar10_final10_tprompt_ca_eg_text_fap_sym005/summary.csv ]]; then
    break
  fi
  sleep 60
done

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
    with open(f"outputs/robust_tac/{run_dir}/summary.csv") as f:
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
