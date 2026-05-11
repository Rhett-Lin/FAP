import csv
import os


OUTPUT_ROOT = "outputs/robust_tac"
OUT_PATH = os.path.join(OUTPUT_ROOT, "cifar10_v2t_order_comparison_summary.csv")

FIELDS = [
    "method",
    "order",
    "stage",
    "clean_acc",
    "clean_nmi",
    "clean_ari",
    "adv_acc",
    "adv_nmi",
    "adv_ari",
    "cfr",
    "clean_adv_kl_eval",
    "eg_kl_eval",
    "mean_cos_factor_eval",
    "teacher_type",
    "teacher_path",
    "stage0_checkpoint",
    "stage1_checkpoint",
    "loaded_visual_checkpoint",
    "loaded_stage1_heads",
    "visual_prompt_trainable_params",
    "text_prompt_trainable_params",
    "total_trainable_params",
]

RUNS = [
    (
        "VPrompt+CA+EG",
        "V",
        "stage0",
        os.path.join(OUTPUT_ROOT, "cifar10_stage0_vprompt_ca_eg", "summary.csv"),
    ),
    (
        "V2T frozen visual TPrompt teacher",
        "V2T",
        "stage1",
        os.path.join(OUTPUT_ROOT, "cifar10_v2t_tprompt_teacher", "summary.csv"),
    ),
    (
        "V2T2V fixed teacher",
        "V2T2V",
        "stage2",
        os.path.join(OUTPUT_ROOT, "cifar10_v2t2v_fixed_teacher", "summary.csv"),
    ),
    (
        "V2T2V fixed teacher no Stage1 heads",
        "V2T2V",
        "stage2_ablation",
        os.path.join(OUTPUT_ROOT, "cifar10_v2t2v_fixed_teacher_no_stage1_heads", "summary.csv"),
    ),
]


def load_row(method, order, stage, path):
    with open(path, newline="") as f:
        src = next(csv.DictReader(f))
    row = {field: src.get(field, "") for field in FIELDS}
    row["method"] = method
    row["order"] = order
    row["stage"] = src.get("stage") or stage
    return row


def main():
    rows = []
    missing = []
    for method, order, stage, path in RUNS:
        if not os.path.exists(path):
            if stage == "stage2_ablation":
                continue
            missing.append(path)
            continue
        rows.append(load_row(method, order, stage, path))

    if missing:
        raise FileNotFoundError("Missing required summary files:\n" + "\n".join(missing))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary_csv: {OUT_PATH}")


if __name__ == "__main__":
    main()
