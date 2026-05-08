import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from attacks import pgd_cluster_attack
from data import make_robust_tac_datasets
from guidance import GuidanceProvider
from losses import (
    cosine_factor,
    kl_target_to_prediction,
    loss_eg_text,
    loss_eg_text_symmetric,
    per_sample_anchor_kl,
    per_sample_eg_kl_text,
    per_sample_eg_kl_text_symmetric,
    per_sample_kl_target_to_prediction,
    tac_clean_loss,
)
from metrics import compute_cluster_metrics
from models import ClusterHead, FrozenCLIPImageEncoder
from text_prompt import TextPromptEncoder
from utils import (
    AverageMeter,
    count_parameters,
    count_parameters_from_iterable,
    count_trainable_parameters,
    ensure_dir,
    set_seed,
    write_summary_csv,
)


def parse_float_or_fraction(value):
    value = str(value)
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Robust TAC + clean-adversarial consistency smoke test")
    parser.add_argument("--dataset", default="Caltech101")
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", default="static", choices=["static", "vprompt", "tprompt"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eps", type=parse_float_or_fraction, default=1.0 / 255.0)
    parser.add_argument("--eval-eps", type=parse_float_or_fraction, default=None)
    parser.add_argument("--step-size", type=parse_float_or_fraction, default=None)
    parser.add_argument("--lambda-ca", type=float, default=1.0)
    parser.add_argument("--lambda-eg", type=float, default=0.0)
    parser.add_argument("--lambda-fap", type=float, default=0.0)
    parser.add_argument("--eg-type", default="none", choices=["none", "text_assignment", "anchor_kl"])
    parser.add_argument("--use-fap-weight", action="store_true")
    parser.add_argument("--tau-relation", type=float, default=0.01)
    parser.add_argument("--num-clusters", default="auto")
    parser.add_argument("--tac-root", default="/work1/zixuan/projects/2024-ICML-TAC")
    parser.add_argument("--guidance-type", default="placeholder", choices=["auto", "placeholder", "tac_text"])
    parser.add_argument("--guidance-path", default=None)
    parser.add_argument("--prompt-depth", type=int, default=0)
    parser.add_argument("--n-ctx", type=int, default=2)
    parser.add_argument("--eg-symmetric", action="store_true")
    parser.add_argument("--eta-sym", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def move_batch(batch, device):
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
        "index": batch["index"].to(device, non_blocking=True),
    }


def get_text_counterpart(args, text_prompt, guidance, indices, v_clean, split):
    if text_prompt is None:
        return guidance.get(indices, v_clean, split=split), guidance.noun_anchor_bank
    tau = guidance.tau_retrieval if guidance.tau_retrieval is not None else 0.005
    return text_prompt.text_counterpart(v_clean, tau)


def compute_eg_terms(args, v_clean, v_adv, p_adv, q_text, noun_anchor_bank):
    if args.eg_type == "none":
        per_sample_eg_kl = torch.zeros(v_clean.shape[0], device=v_clean.device)
        loss_eg = per_sample_eg_kl.mean()
    elif args.eg_type == "text_assignment":
        if args.eg_symmetric:
            per_sample_eg_kl = per_sample_eg_kl_text_symmetric(q_text, p_adv, eta_sym=args.eta_sym)
            loss_eg = loss_eg_text_symmetric(q_text, p_adv, eta_sym=args.eta_sym)
        else:
            per_sample_eg_kl = per_sample_eg_kl_text(q_text, p_adv)
            loss_eg = loss_eg_text(q_text, p_adv)
    elif args.eg_type == "anchor_kl":
        per_sample_eg_kl, _, _ = per_sample_anchor_kl(
            v_clean,
            v_adv,
            noun_anchor_bank,
            args.tau_relation,
        )
        loss_eg = per_sample_eg_kl.mean()
    else:
        raise ValueError(f"Unsupported eg_type: {args.eg_type}")

    cos = cosine_factor(v_clean, v_adv)
    if args.use_fap_weight and args.eg_type != "none":
        loss_fap_weight = torch.mean(cos * per_sample_eg_kl)
    else:
        loss_fap_weight = torch.zeros((), device=v_clean.device)

    return {
        "loss_eg": loss_eg,
        "loss_fap_weight": loss_fap_weight,
        "mean_cos_factor": cos.mean(),
        "mean_eg_kl": per_sample_eg_kl.mean(),
    }


def train_one_epoch(args, encoder, cluster_head, guidance, text_prompt, loader, optimizer, device, epoch):
    encoder.train()
    encoder.clip_model.eval()
    if text_prompt is not None:
        text_prompt.train()
        text_prompt.clip_model.eval()
    cluster_head.train()
    meters = {
        name: AverageMeter()
        for name in [
            "loss",
            "loss_tac",
            "loss_consist",
            "loss_balance",
            "loss_ca",
            "loss_eg",
            "loss_fap_weight",
            "mean_cos_factor",
            "mean_eg_kl",
        ]
    }

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        images = batch["image"]
        indices = batch["index"]

        v_clean = encoder(images)
        p_clean = cluster_head.image_forward(v_clean)
        text_counterpart, noun_anchor_bank = get_text_counterpart(args, text_prompt, guidance, indices, v_clean, split="train")
        q_text = cluster_head.text_forward(text_counterpart)

        loss_tac, loss_consist, loss_balance = tac_clean_loss(q_text, p_clean)
        adv_images = pgd_cluster_attack(
            encoder,
            cluster_head,
            images,
            p_clean.detach(),
            eps=args.eps,
            steps=args.train_steps,
            step_size=args.step_size,
        )
        v_adv = encoder(adv_images)
        p_adv = cluster_head.image_forward(v_adv)
        loss_ca = kl_target_to_prediction(p_clean.detach(), p_adv)
        eg_terms = compute_eg_terms(args, v_clean, v_adv, p_adv, q_text, noun_anchor_bank)
        loss_eg = eg_terms["loss_eg"]
        loss_fap_weight = eg_terms["loss_fap_weight"]
        loss = (
            loss_tac
            + args.lambda_ca * loss_ca
            + args.lambda_eg * loss_eg
            + args.lambda_fap * loss_fap_weight
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if text_prompt is not None:
            grad = text_prompt.ctx.grad
            if grad is None or not torch.isfinite(grad).all():
                raise AssertionError("TPrompt ctx gradient is missing or non-finite")
            if step == 1:
                print(f"text_prompt_ctx_grad_norm: {grad.float().norm().item():.6f}")
        optimizer.step()

        meters["loss"].update(loss.item())
        meters["loss_tac"].update(loss_tac.item())
        meters["loss_consist"].update(loss_consist.item())
        meters["loss_balance"].update(loss_balance.item())
        meters["loss_ca"].update(loss_ca.item())
        meters["loss_eg"].update(loss_eg.item())
        meters["loss_fap_weight"].update(loss_fap_weight.item())
        meters["mean_cos_factor"].update(eg_terms["mean_cos_factor"].item())
        meters["mean_eg_kl"].update(eg_terms["mean_eg_kl"].item())

        if step == 1 or step == len(loader):
            print(
                f"[train epoch {epoch} step {step}/{len(loader)}] "
                f"loss={loss.item():.6f} tac={loss_tac.item():.6f} "
                f"consist={loss_consist.item():.6f} balance={loss_balance.item():.6f} "
                f"ca={loss_ca.item():.6f} eg={loss_eg.item():.6f} "
                f"fap_weight={loss_fap_weight.item():.6f} "
                f"cos={eg_terms['mean_cos_factor'].item():.6f} "
                f"eg_kl={eg_terms['mean_eg_kl'].item():.6f}"
            )

    return {name: meter.avg for name, meter in meters.items()}


def evaluate(args, encoder, cluster_head, guidance, text_prompt, loader, device):
    cluster_head.eval()
    if text_prompt is not None:
        text_prompt.eval()
    labels_all = []
    clean_preds_all = []
    adv_preds_all = []
    eg_kl_values = []
    clean_adv_kl_values = []
    cos_factor_values = []

    for batch in loader:
        batch = move_batch(batch, device)
        images = batch["image"]
        labels = batch["label"]
        with torch.no_grad():
            v_clean = encoder(images)
            p_clean = cluster_head.image_forward(v_clean)
            clean_preds = p_clean.argmax(dim=1)
            text_counterpart, noun_anchor_bank = get_text_counterpart(args, text_prompt, guidance, batch["index"], v_clean, split="test")
            q_text = cluster_head.text_forward(text_counterpart)

        adv_images = pgd_cluster_attack(
            encoder,
            cluster_head,
            images,
            p_clean.detach(),
            eps=args.eval_eps if args.eval_eps is not None else args.eps,
            steps=args.eval_steps,
            step_size=args.step_size,
        )
        with torch.no_grad():
            v_adv = encoder(adv_images)
            p_adv = cluster_head.image_forward(v_adv)
            adv_preds = p_adv.argmax(dim=1)
            clean_adv_kl = per_sample_kl_target_to_prediction(p_clean, p_adv)
            if args.eg_type == "none":
                eg_kl = torch.zeros_like(clean_adv_kl)
            elif args.eg_type == "text_assignment":
                if args.eg_symmetric:
                    eg_kl = per_sample_eg_kl_text_symmetric(q_text, p_adv, eta_sym=args.eta_sym)
                else:
                    eg_kl = per_sample_eg_kl_text(q_text, p_adv)
            elif args.eg_type == "anchor_kl":
                eg_kl, _, _ = per_sample_anchor_kl(
                    v_clean,
                    v_adv,
                    noun_anchor_bank,
                    args.tau_relation,
                )
            else:
                raise ValueError(f"Unsupported eg_type: {args.eg_type}")
            cos_eval = cosine_factor(v_clean, v_adv)

        labels_all.append(labels.cpu().numpy())
        clean_preds_all.append(clean_preds.cpu().numpy())
        adv_preds_all.append(adv_preds.cpu().numpy())
        eg_kl_values.append(eg_kl.cpu().numpy())
        clean_adv_kl_values.append(clean_adv_kl.cpu().numpy())
        cos_factor_values.append(cos_eval.cpu().numpy())

    labels_np = np.concatenate(labels_all)
    clean_np = np.concatenate(clean_preds_all)
    adv_np = np.concatenate(adv_preds_all)
    metrics = compute_cluster_metrics(labels_np, clean_np, adv_np)
    metrics.update(
        {
            "eg_kl_eval": float(np.concatenate(eg_kl_values).mean()),
            "clean_adv_kl_eval": float(np.concatenate(clean_adv_kl_values).mean()),
            "mean_cos_factor_eval": float(np.concatenate(cos_factor_values).mean()),
        }
    )
    return metrics


def assert_training_contract(args, encoder, cluster_head, text_prompt=None):
    encoder_trainable = [name for name, p in encoder.named_parameters() if p.requires_grad]
    head_trainable = [name for name, p in cluster_head.named_parameters() if p.requires_grad]
    prompt_trainable = [name for name, p in encoder.prompt_named_parameters() if p.requires_grad]
    non_prompt_trainable = [name for name in encoder_trainable if "visual.VPT" not in name]
    if args.mode == "static" and encoder_trainable:
        raise AssertionError(f"CLIP encoder unexpectedly trainable: {encoder_trainable[:5]}")
    if args.mode == "vprompt" and not prompt_trainable:
        raise AssertionError("VPrompt mode has no trainable visual prompt parameters")
    if args.mode == "tprompt":
        if encoder_trainable:
            raise AssertionError(f"TPrompt image encoder unexpectedly trainable: {encoder_trainable[:5]}")
        if text_prompt is None or not any(p.requires_grad for p in text_prompt.parameters()):
            raise AssertionError("TPrompt mode has no trainable text prompt parameters")
    if non_prompt_trainable:
        raise AssertionError(f"Non-prompt CLIP parameters unexpectedly trainable: {non_prompt_trainable[:5]}")
    if not head_trainable:
        raise AssertionError("Cluster heads have no trainable parameters")
    if not encoder.clip_backbone_frozen():
        raise AssertionError("CLIP backbone is not frozen")


def main():
    args = parse_args()
    set_seed(args.seed)

    requested_device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    datasets = make_robust_tac_datasets(
        args.dataset,
        args.root,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    inferred_clusters = datasets["num_clusters"]
    if args.num_clusters == "auto":
        num_clusters = inferred_clusters
    else:
        num_clusters = int(args.num_clusters)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=requested_device.type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=requested_device.type == "cuda",
        drop_last=False,
    )

    prompt_depth = args.prompt_depth if args.mode == "vprompt" else 0
    n_ctx = args.n_ctx if args.mode in {"vprompt", "tprompt"} else 0
    if args.mode == "vprompt":
        if prompt_depth != 1:
            raise ValueError("This smoke-test implementation currently supports --prompt-depth 1 only")
        if n_ctx <= 0:
            raise ValueError("--n-ctx must be positive for mode=vprompt")
    if args.mode == "tprompt" and n_ctx <= 0:
        raise ValueError("--n-ctx must be positive for mode=tprompt")

    encoder = FrozenCLIPImageEncoder(
        "ViT-B/32",
        requested_device,
        mode=args.mode,
        prompt_depth=prompt_depth,
        n_ctx=n_ctx,
    ).to(requested_device)
    cluster_head = ClusterHead(in_dim=512, num_clusters=num_clusters).to(requested_device)
    guidance = GuidanceProvider(
        guidance_type=args.guidance_type,
        guidance_path=args.guidance_path,
        device=requested_device,
    )
    text_prompt = None
    if args.mode == "tprompt":
        if guidance.selected_nouns is None or guidance.selected_noun_indices is None:
            raise KeyError("TPrompt requires selected_nouns and selected_noun_indices in guidance npz")
        text_prompt = TextPromptEncoder(encoder.clip_model, guidance.selected_nouns, n_ctx=n_ctx).to(requested_device)
    if args.eg_type == "anchor_kl" and guidance.noun_anchor_bank is None:
        raise ValueError("--eg-type anchor_kl requires tac_text guidance with noun_anchor_bank")
    assert_training_contract(args, encoder, cluster_head, text_prompt=text_prompt)

    text_prompt_parameters = list(text_prompt.parameters()) if text_prompt is not None else []
    trainable_parameters = [p for p in encoder.parameters() if p.requires_grad] + text_prompt_parameters + list(cluster_head.parameters())
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, betas=(0.9, 0.99))
    _, total_params = count_parameters(encoder, cluster_head)
    prompt_trainable_params = count_parameters_from_iterable([p for _, p in encoder.prompt_named_parameters() if p.requires_grad])
    text_prompt_trainable_params = count_parameters_from_iterable([p for p in text_prompt_parameters if p.requires_grad])
    cluster_head_trainable_params = count_trainable_parameters(cluster_head)
    trainable_params = prompt_trainable_params + text_prompt_trainable_params + cluster_head_trainable_params
    total_params += text_prompt_trainable_params

    print(f"dataset name: {args.dataset}")
    print(f"mode: {args.mode}")
    print(f"prompt depth: {prompt_depth}")
    print(f"n_ctx: {n_ctx}")
    print(f"dataset dir: {datasets['dataset_dir']}")
    print(f"split source: {datasets['split_source']}")
    print(f"train size: {len(datasets['train'])}")
    print(f"test size: {len(datasets['test'])}")
    print(f"inferred num_clusters: {inferred_clusters}")
    print(f"using num_clusters: {num_clusters}")
    print(f"guidance_type: {guidance.guidance_type}")
    print(f"guidance_path: {guidance.loaded_path}")
    print(f"guidance_keys: {guidance.keys}")
    print(f"selected noun count: {0 if guidance.selected_nouns is None else len(guidance.selected_nouns)}")
    print(f"prompt trainable params: {prompt_trainable_params}")
    print(f"text prompt trainable params: {text_prompt_trainable_params}")
    print(f"cluster head trainable params: {cluster_head_trainable_params}")
    print(f"total trainable params: {trainable_params}")
    print(f"total params: {total_params}")
    print(f"CLIP backbone frozen: {encoder.clip_backbone_frozen()}")
    print(f"attack eps: {args.eps}")
    print(f"eval attack eps: {args.eval_eps if args.eval_eps is not None else args.eps}")
    print(f"train steps: {args.train_steps}")
    print(f"eval steps: {args.eval_steps}")
    print(f"lambda_ca: {args.lambda_ca}")
    print(f"lambda_eg: {args.lambda_eg}")
    print(f"lambda_fap: {args.lambda_fap}")
    print(f"eg_type: {args.eg_type}")
    print(f"eg_symmetric: {args.eg_symmetric}")
    print(f"eta_sym: {args.eta_sym}")
    print(f"use_fap_weight: {args.use_fap_weight}")
    print(f"tau_relation: {args.tau_relation}")
    print(f"device: {requested_device}")

    last_losses = {}
    for epoch in range(1, args.epochs + 1):
        last_losses = train_one_epoch(args, encoder, cluster_head, guidance, text_prompt, train_loader, optimizer, requested_device, epoch)

    metrics = evaluate(args, encoder, cluster_head, guidance, text_prompt, test_loader, requested_device)
    print("final metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    row = {
        "dataset": args.dataset,
        "mode": args.mode,
        "prompt_depth": prompt_depth,
        "n_ctx": n_ctx,
        "train_size": len(datasets["train"]),
        "test_size": len(datasets["test"]),
        "inferred_num_clusters": inferred_clusters,
        "num_clusters": num_clusters,
        "guidance_type": guidance.guidance_type,
        "guidance_path": guidance.loaded_path or "",
        "eps": args.eps,
        "eval_eps": args.eval_eps if args.eval_eps is not None else args.eps,
        "train_steps": args.train_steps,
        "eval_steps": args.eval_steps,
        "lambda_ca": args.lambda_ca,
        "lambda_eg": args.lambda_eg,
        "lambda_fap": args.lambda_fap,
        "eg_type": args.eg_type,
        "eg_symmetric": args.eg_symmetric,
        "eta_sym": args.eta_sym,
        "use_fap_weight": args.use_fap_weight,
        "tau_relation": args.tau_relation,
        "prompt_trainable_params": prompt_trainable_params,
        "text_prompt_trainable_params": text_prompt_trainable_params,
        "cluster_head_trainable_params": cluster_head_trainable_params,
        "total_trainable_params": trainable_params,
        "clip_backbone_frozen": encoder.clip_backbone_frozen(),
        "loss_ca": last_losses.get("loss_ca", 0.0),
        "loss_eg": last_losses.get("loss_eg", 0.0),
        "loss_fap_weight": last_losses.get("loss_fap_weight", 0.0),
        "mean_cos_factor": last_losses.get("mean_cos_factor", 0.0),
        "mean_eg_kl": last_losses.get("mean_eg_kl", 0.0),
        **{f"loss_{k}": v for k, v in last_losses.items()},
        **metrics,
    }
    ensure_dir(args.output)
    summary_path = os.path.join(args.output, "summary.csv")
    write_summary_csv(summary_path, row)
    print(f"summary_csv: {summary_path}")


if __name__ == "__main__":
    main()
