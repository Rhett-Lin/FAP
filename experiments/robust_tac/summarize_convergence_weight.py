import argparse
import csv
import os


FIELDS = [
    "method",
    "epochs",
    "lambda_eg",
    "lambda_fap",
    "use_fap_weight",
    "clean_ari",
    "adv_ari",
    "cfr",
    "clean_adv_kl_eval",
    "eg_kl_eval",
    "mean_cos_factor_eval",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize robust TAC convergence and weight runs")
    parser.add_argument("--outputs-root", default="outputs/robust_tac")
    parser.add_argument("--output", default="outputs/robust_tac/convergence_weight_summary.csv")
    return parser.parse_args()


def read_summary(path):
    with open(path, "r", newline="") as f:
        return next(csv.DictReader(f))


def method_from_dir(dirname, row):
    if "ca_eg_text_fapweight" in dirname:
        return "vprompt_ca_eg_text_fapweight"
    if "ca_eg_text" in dirname:
        return "vprompt_ca_eg_text"
    if "ca" in dirname:
        return "vprompt_ca"
    return row.get("mode", dirname)


def epochs_from_dir(dirname, row):
    if dirname.startswith("converge_ep"):
        rest = dirname[len("converge_ep") :]
        return rest.split("_", 1)[0]
    return row.get("epochs", "")


def collect(outputs_root):
    rows = []
    for dirname in sorted(os.listdir(outputs_root)):
        if not (dirname.startswith("converge_ep") or dirname.startswith("weight_ep")):
            continue
        summary_path = os.path.join(outputs_root, dirname, "summary.csv")
        if not os.path.exists(summary_path):
            continue
        row = read_summary(summary_path)
        out = {
            "method": method_from_dir(dirname, row),
            "epochs": epochs_from_dir(dirname, row),
            "lambda_eg": row.get("lambda_eg", ""),
            "lambda_fap": row.get("lambda_fap", ""),
            "use_fap_weight": row.get("use_fap_weight", ""),
            "clean_ari": row.get("clean_ari", ""),
            "adv_ari": row.get("adv_ari", ""),
            "cfr": row.get("cfr", ""),
            "clean_adv_kl_eval": row.get("clean_adv_kl_eval", ""),
            "eg_kl_eval": row.get("eg_kl_eval", ""),
            "mean_cos_factor_eval": row.get("mean_cos_factor_eval", ""),
        }
        rows.append(out)
    return rows


def main():
    args = parse_args()
    rows = collect(args.outputs_root)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
