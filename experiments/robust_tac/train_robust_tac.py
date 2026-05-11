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
    tac_fixed_teacher_loss,
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
    parser.add_argument("--mode", default="static", choices=["static", "vprompt", "tprompt", "dualprompt"])
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
    parser.add_argument("--save-teacher-path", default=None)
    parser.add_argument("--teacher-path", default=None)
    parser.add_argument("--teacher-type", default="none", choices=["none", "fixed_q"])
    parser.add_argument("--frozen-visual-checkpoint", default=None)
    parser.add_argument("--freeze-loaded-visual-prompt", action="store_true")
    parser.add_argument("--stage0-checkpoint", default=None)
    parser.add_argument("--load-stage1-heads", action="store_true")
    parser.add_argument("--stage1-checkpoint", default=None)
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


class FixedTeacherProvider:
    def __init__(self, teacher_path, device):
        if not teacher_path:
            raise ValueError("--teacher-path is required when --teacher-type fixed_q")
        if not os.path.exists(teacher_path):
            raise FileNotFoundError(f"Teacher file not found: {teacher_path}")

        data = np.load(teacher_path, allow_pickle=False)
        required = ["teacher_q_train", "teacher_q_test"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Teacher file missing required arrays: {missing}")

        self.path = teacher_path
        self.keys = list(data.files)
        self.train_q = self._to_prob_tensor(data["teacher_q_train"], "teacher_q_train", device)
        self.test_q = self._to_prob_tensor(data["teacher_q_test"], "teacher_q_test", device)
        self.train_text_counterpart = self._optional_feature(data, "teacher_text_counterpart_train", device)
        self.test_text_counterpart = self._optional_feature(data, "teacher_text_counterpart_test", device)
        self.noun_anchor_bank = self._optional_feature(data, "prompted_noun_anchor_bank", device)

    @staticmethod
    def _to_prob_tensor(array, name, device):
        if array.ndim != 2:
            raise ValueError(f"{name} must have shape [N, K], got {array.shape}")
        tensor = torch.from_numpy(array).float()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
        if tensor.min().item() < -1e-6:
            raise ValueError(f"{name} contains negative probabilities")
        row_sum = tensor.sum(dim=1, keepdim=True)
        if torch.any(row_sum <= 0):
            raise ValueError(f"{name} contains zero-probability rows")
        tensor = tensor / row_sum
        return tensor.to(device)

    @staticmethod
    def _optional_feature(data, key, device):
        if key not in data:
            return None
        tensor = torch.from_numpy(data[key]).float()
        if tensor.ndim != 2 or tensor.shape[1] != 512:
            raise ValueError(f"{key} must have shape [N, 512], got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key} contains NaN or Inf")
        return torch.nn.functional.normalize(tensor, dim=-1).to(device)

    def get_q(self, indices, split):
        bank = self.train_q if split == "train" else self.test_q
        if indices.numel() == 0:
            return bank[indices].detach()
        max_index = int(indices.max().item())
        if max_index >= bank.shape[0]:
            raise IndexError(
                f"{split} teacher has {bank.shape[0]} rows, but requested index {max_index}"
            )
        return bank[indices].detach()


def load_stage1_heads(cluster_head, checkpoint_path, load_heads):
    if not load_heads:
        return False
    if not checkpoint_path:
        raise ValueError("--stage1-checkpoint is required with --load-stage1-heads")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "cluster_head_image" not in checkpoint:
        raise KeyError("Stage 1 checkpoint missing cluster_head_image")
    cluster_head.cluster_head_image.load_state_dict(checkpoint["cluster_head_image"])
    return True


def visual_prompt_state_dict(encoder):
    return {name: param.detach().cpu() for name, param in encoder.prompt_named_parameters()}


def load_visual_prompt_checkpoint(encoder, checkpoint_path, freeze_visual_prompt=False):
    if not checkpoint_path:
        return False
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Visual prompt checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "visual_prompt" not in checkpoint:
        raise KeyError(f"Checkpoint missing visual_prompt: {checkpoint_path}")
    visual_prompt = checkpoint["visual_prompt"]
    loaded = []
    for name, param in encoder.prompt_named_parameters():
        if name not in visual_prompt:
            raise KeyError(f"Checkpoint {checkpoint_path} missing visual prompt parameter: {name}")
        param.data.copy_(visual_prompt[name].to(device=param.device, dtype=param.dtype))
        param.requires_grad_(not freeze_visual_prompt)
        loaded.append(name)
    if not loaded:
        raise AssertionError("No visual prompt parameters were loaded")
    return True


def save_checkpoint(args, encoder, cluster_head, checkpoint_path, extra=None):
    ensure_dir(os.path.dirname(checkpoint_path))
    payload = {
        "visual_prompt": visual_prompt_state_dict(encoder),
        "cluster_head_image": cluster_head.cluster_head_image.state_dict(),
        "cluster_head_text": cluster_head.cluster_head_text.state_dict(),
        "cluster_head": cluster_head.state_dict(),
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


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


def train_one_epoch(args, encoder, cluster_head, guidance, text_prompt, teacher, loader, optimizer, device, epoch):
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
        if teacher is not None:
            q_text = teacher.get_q(indices, split="train")
            noun_anchor_bank = teacher.noun_anchor_bank if teacher.noun_anchor_bank is not None else guidance.noun_anchor_bank
            loss_tac, loss_consist, loss_balance = tac_fixed_teacher_loss(q_text, p_clean)
        else:
            text_counterpart, noun_anchor_bank = get_text_counterpart(args, text_prompt, guidance, indices, v_clean, split="train")
            q_text = cluster_head.text_forward(text_counterpart)
            loss_tac, loss_consist, loss_balance = tac_clean_loss(q_text, p_clean)
        use_adv_training = args.lambda_ca != 0.0 or args.lambda_eg != 0.0 or args.lambda_fap != 0.0
        if use_adv_training:
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
        else:
            loss_ca = torch.zeros((), device=device)
            loss_eg = torch.zeros((), device=device)
            loss_fap_weight = torch.zeros((), device=device)
            eg_terms = {
                "mean_cos_factor": torch.zeros((), device=device),
                "mean_eg_kl": torch.zeros((), device=device),
            }
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


def evaluate(args, encoder, cluster_head, guidance, text_prompt, teacher, loader, device):
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
            if teacher is not None:
                q_text = teacher.get_q(batch["index"], split="test")
                noun_anchor_bank = teacher.noun_anchor_bank if teacher.noun_anchor_bank is not None else guidance.noun_anchor_bank
            else:
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


def collect_stage1_teacher_split(args, encoder, cluster_head, guidance, text_prompt, loader, device, split):
    if text_prompt is None:
        raise ValueError("--save-teacher-path requires mode=tprompt or mode=dualprompt")

    encoder.eval()
    cluster_head.eval()
    text_prompt.eval()
    q_chunks = []
    counterpart_chunks = []
    index_chunks = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            v_clean = encoder(batch["image"])
            text_counterpart, _ = get_text_counterpart(
                args,
                text_prompt,
                guidance,
                batch["index"],
                v_clean,
                split=split,
            )
            q_text = cluster_head.text_forward(text_counterpart)
            q_chunks.append(q_text.detach().cpu())
            counterpart_chunks.append(text_counterpart.detach().cpu())
            index_chunks.append(batch["index"].detach().cpu())

    indices = torch.cat(index_chunks, dim=0)
    q_values = torch.cat(q_chunks, dim=0)
    counterpart_values = torch.cat(counterpart_chunks, dim=0)
    size = int(indices.max().item()) + 1 if indices.numel() else 0
    q_bank = torch.full((size, q_values.shape[1]), float("nan"), dtype=q_values.dtype)
    counterpart_bank = torch.full((size, counterpart_values.shape[1]), float("nan"), dtype=counterpart_values.dtype)
    q_bank[indices] = q_values
    counterpart_bank[indices] = counterpart_values
    if not torch.isfinite(q_bank).all():
        raise FloatingPointError(f"Non-finite or missing {split} teacher_q rows")
    if not torch.isfinite(counterpart_bank).all():
        raise FloatingPointError(f"Non-finite or missing {split} text counterpart rows")
    row_sum = q_bank.sum(dim=1, keepdim=True).clamp_min(1e-9)
    q_bank = q_bank / row_sum
    return q_bank.numpy(), counterpart_bank.numpy()


def save_stage1_teacher(args, encoder, cluster_head, guidance, text_prompt, train_loader, test_loader, device, num_clusters):
    if args.save_teacher_path is None:
        return None
    if args.mode != "tprompt":
        raise ValueError("--save-teacher-path is intended for Stage 1 mode=tprompt")

    train_q, train_counterpart = collect_stage1_teacher_split(
        args, encoder, cluster_head, guidance, text_prompt, train_loader, device, split="train"
    )
    test_q, test_counterpart = collect_stage1_teacher_split(
        args, encoder, cluster_head, guidance, text_prompt, test_loader, device, split="test"
    )
    with torch.no_grad():
        prompted_noun_anchor_bank = text_prompt().detach().cpu().numpy()

    ensure_dir(os.path.dirname(args.save_teacher_path))
    np.savez(
        args.save_teacher_path,
        teacher_q_train=train_q,
        teacher_q_test=test_q,
        teacher_text_counterpart_train=train_counterpart,
        teacher_text_counterpart_test=test_counterpart,
        prompted_noun_anchor_bank=prompted_noun_anchor_bank,
        dataset=np.array(args.dataset),
        mode=np.array(args.mode),
        num_clusters=np.array(num_clusters, dtype=np.int64),
        n_ctx=np.array(args.n_ctx, dtype=np.int64),
        guidance_path=np.array(args.guidance_path or ""),
    )

    checkpoint_path = os.path.join(args.output, "checkpoint.pt")
    ensure_dir(args.output)
    save_checkpoint(
        args,
        encoder,
        cluster_head,
        checkpoint_path,
        extra={
            "text_prompt": text_prompt.state_dict(),
            "teacher_path": args.save_teacher_path,
            "num_clusters": num_clusters,
        },
    )
    return {
        "teacher_path": args.save_teacher_path,
        "checkpoint_path": checkpoint_path,
        "teacher_q_train_shape": tuple(train_q.shape),
        "teacher_q_test_shape": tuple(test_q.shape),
    }


def assert_training_contract(args, encoder, cluster_head, text_prompt=None):
    encoder_trainable = [name for name, p in encoder.named_parameters() if p.requires_grad]
    head_trainable = [name for name, p in cluster_head.named_parameters() if p.requires_grad]
    prompt_trainable = [name for name, p in encoder.prompt_named_parameters() if p.requires_grad]
    non_prompt_trainable = [name for name in encoder_trainable if "visual.VPT" not in name]
    if text_prompt is not None:
        text_prompt_unexpected_trainable = [
            name for name, p in text_prompt.named_parameters() if p.requires_grad and name != "ctx"
        ]
        if text_prompt_unexpected_trainable:
            raise AssertionError(
                "TextPromptEncoder has non-ctx trainable parameters: "
                f"{text_prompt_unexpected_trainable[:5]}"
            )
    if args.mode == "static" and encoder_trainable:
        raise AssertionError(f"CLIP encoder unexpectedly trainable: {encoder_trainable[:5]}")
    if args.mode in {"vprompt", "dualprompt"} and not prompt_trainable:
        raise AssertionError(f"{args.mode} mode has no trainable visual prompt parameters")
    if args.mode == "tprompt":
        if encoder_trainable:
            raise AssertionError(f"TPrompt image encoder unexpectedly trainable: {encoder_trainable[:5]}")
        if text_prompt is None or not any(p.requires_grad for p in text_prompt.parameters()):
            raise AssertionError("TPrompt mode has no trainable text prompt parameters")
    if args.mode == "dualprompt":
        if text_prompt is None or not any(p.requires_grad for p in text_prompt.parameters()):
            raise AssertionError("DualPrompt mode has no trainable text prompt parameters")
    if non_prompt_trainable:
        raise AssertionError(f"Non-prompt CLIP parameters unexpectedly trainable: {non_prompt_trainable[:5]}")
    if not head_trainable:
        raise AssertionError("Cluster heads have no trainable parameters")
    if not encoder.clip_backbone_frozen():
        raise AssertionError("CLIP backbone is not frozen")


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.save_teacher_path is not None and args.mode != "tprompt":
        raise ValueError("--save-teacher-path requires --mode tprompt")
    if args.teacher_type == "fixed_q" and args.teacher_path is None:
        raise ValueError("--teacher-path is required when --teacher-type fixed_q")
    if args.teacher_type == "fixed_q" and args.mode not in {"vprompt", "static"}:
        raise ValueError("--teacher-type fixed_q is currently intended for Stage 2 mode=vprompt or static ablations")
    if args.teacher_type == "fixed_q" and args.eg_symmetric:
        raise ValueError("--eg-symmetric is not meaningful with --teacher-type fixed_q because the teacher is detached")
    if args.frozen_visual_checkpoint and args.mode != "tprompt":
        raise ValueError("--frozen-visual-checkpoint is only supported with --mode tprompt")
    if args.frozen_visual_checkpoint and not args.freeze_loaded_visual_prompt:
        raise ValueError("--frozen-visual-checkpoint requires --freeze-loaded-visual-prompt")
    if args.stage0_checkpoint and args.mode != "vprompt":
        raise ValueError("--stage0-checkpoint is only supported with --mode vprompt")

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

    encoder_mode = "vprompt" if args.mode == "tprompt" and args.frozen_visual_checkpoint else args.mode
    prompt_depth = args.prompt_depth if encoder_mode in {"vprompt", "dualprompt"} else 0
    n_ctx = args.n_ctx if args.mode in {"vprompt", "tprompt", "dualprompt"} else 0
    if encoder_mode in {"vprompt", "dualprompt"}:
        if prompt_depth != 1:
            raise ValueError("This implementation currently supports --prompt-depth 1 only when using visual prompts")
        if n_ctx <= 0:
            raise ValueError(f"--n-ctx must be positive for mode={args.mode}")
    if args.mode == "tprompt" and n_ctx <= 0:
        raise ValueError("--n-ctx must be positive for mode=tprompt")

    encoder = FrozenCLIPImageEncoder(
        "ViT-B/32",
        requested_device,
        mode=encoder_mode,
        prompt_depth=prompt_depth,
        n_ctx=n_ctx,
    ).to(requested_device)
    cluster_head = ClusterHead(in_dim=512, num_clusters=num_clusters).to(requested_device)
    loaded_visual_checkpoint = load_visual_prompt_checkpoint(
        encoder,
        args.frozen_visual_checkpoint or args.stage0_checkpoint,
        freeze_visual_prompt=args.freeze_loaded_visual_prompt,
    )
    loaded_stage1_heads = load_stage1_heads(cluster_head, args.stage1_checkpoint, args.load_stage1_heads)
    if args.teacher_type == "fixed_q":
        for param in cluster_head.cluster_head_text.parameters():
            param.requires_grad_(False)
    guidance = GuidanceProvider(
        guidance_type=args.guidance_type,
        guidance_path=args.guidance_path,
        device=requested_device,
    )
    teacher = FixedTeacherProvider(args.teacher_path, requested_device) if args.teacher_type == "fixed_q" else None
    if teacher is not None:
        if teacher.train_q.shape[1] != num_clusters or teacher.test_q.shape[1] != num_clusters:
            raise ValueError(
                f"Teacher cluster dim mismatch: train={teacher.train_q.shape}, "
                f"test={teacher.test_q.shape}, num_clusters={num_clusters}"
            )
    text_prompt = None
    if args.mode in {"tprompt", "dualprompt"}:
        if guidance.selected_nouns is None or guidance.selected_noun_indices is None:
            raise KeyError(f"{args.mode} requires selected_nouns and selected_noun_indices in guidance npz")
        text_prompt = TextPromptEncoder(
            encoder.clip_model,
            guidance.selected_nouns,
            n_ctx=n_ctx,
            freeze_clip=True,
        ).to(requested_device)
    if args.eg_type == "anchor_kl" and guidance.noun_anchor_bank is None and (teacher is None or teacher.noun_anchor_bank is None):
        raise ValueError("--eg-type anchor_kl requires tac_text guidance or fixed teacher with noun_anchor_bank")
    assert_training_contract(args, encoder, cluster_head, text_prompt=text_prompt)

    text_prompt_parameters = list(text_prompt.parameters()) if text_prompt is not None else []
    trainable_parameters = [p for p in encoder.parameters() if p.requires_grad] + text_prompt_parameters + list(cluster_head.parameters())
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, betas=(0.9, 0.99))
    _, total_params = count_parameters(encoder, cluster_head)
    visual_prompt_trainable_params = count_parameters_from_iterable([p for _, p in encoder.prompt_named_parameters() if p.requires_grad])
    prompt_trainable_params = visual_prompt_trainable_params
    text_prompt_trainable_params = count_parameters_from_iterable([p for p in text_prompt_parameters if p.requires_grad])
    cluster_head_trainable_params = count_trainable_parameters(cluster_head)
    trainable_params = prompt_trainable_params + text_prompt_trainable_params + cluster_head_trainable_params
    total_params += text_prompt_trainable_params

    print(f"dataset name: {args.dataset}")
    print(f"mode: {args.mode}")
    print(f"encoder mode: {encoder_mode}")
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
    print(f"teacher_type: {args.teacher_type}")
    print(f"teacher_path: {args.teacher_path or ''}")
    print(f"teacher_keys: {[] if teacher is None else teacher.keys}")
    print(f"teacher_q_train_shape: {'' if teacher is None else tuple(teacher.train_q.shape)}")
    print(f"teacher_q_test_shape: {'' if teacher is None else tuple(teacher.test_q.shape)}")
    print(f"stage0_checkpoint: {args.stage0_checkpoint or ''}")
    print(f"frozen_visual_checkpoint: {args.frozen_visual_checkpoint or ''}")
    print(f"loaded_visual_checkpoint: {loaded_visual_checkpoint}")
    print(f"stage1_checkpoint: {args.stage1_checkpoint or ''}")
    print(f"loaded_stage1_heads: {loaded_stage1_heads}")
    print(f"selected noun count: {0 if guidance.selected_nouns is None else len(guidance.selected_nouns)}")
    print(f"visual prompt trainable params: {visual_prompt_trainable_params}")
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
        last_losses = train_one_epoch(args, encoder, cluster_head, guidance, text_prompt, teacher, train_loader, optimizer, requested_device, epoch)

    metrics = evaluate(args, encoder, cluster_head, guidance, text_prompt, teacher, test_loader, requested_device)
    saved_teacher = save_stage1_teacher(
        args,
        encoder,
        cluster_head,
        guidance,
        text_prompt,
        train_loader,
        test_loader,
        requested_device,
        num_clusters,
    )
    if saved_teacher is not None:
        print(f"teacher_npz: {saved_teacher['teacher_path']}")
        print(f"checkpoint_pt: {saved_teacher['checkpoint_path']}")
        print(f"teacher_q_train_shape: {saved_teacher['teacher_q_train_shape']}")
        print(f"teacher_q_test_shape: {saved_teacher['teacher_q_test_shape']}")
    else:
        checkpoint_path = save_checkpoint(args, encoder, cluster_head, os.path.join(args.output, "checkpoint.pt"))
        print(f"checkpoint_pt: {checkpoint_path}")
    print("final metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    row = {
        "dataset": args.dataset,
        "method": args.mode,
        "mode": args.mode,
        "prompt_depth": prompt_depth,
        "n_ctx": n_ctx,
        "train_size": len(datasets["train"]),
        "test_size": len(datasets["test"]),
        "inferred_num_clusters": inferred_clusters,
        "num_clusters": num_clusters,
        "guidance_type": guidance.guidance_type,
        "guidance_path": guidance.loaded_path or "",
        "stage": "stage2" if args.teacher_type == "fixed_q" else ("stage1" if args.save_teacher_path else ""),
        "teacher_type": args.teacher_type,
        "teacher_path": args.teacher_path or "",
        "save_teacher_path": args.save_teacher_path or "",
        "stage0_checkpoint": args.stage0_checkpoint or "",
        "frozen_visual_checkpoint": args.frozen_visual_checkpoint or "",
        "loaded_visual_checkpoint": loaded_visual_checkpoint,
        "stage1_checkpoint": args.stage1_checkpoint or "",
        "loaded_stage1_heads": loaded_stage1_heads,
        "teacher_q_train_shape": "" if teacher is None else str(tuple(teacher.train_q.shape)),
        "teacher_q_test_shape": "" if teacher is None else str(tuple(teacher.test_q.shape)),
        "saved_teacher_q_train_shape": "" if saved_teacher is None else str(saved_teacher["teacher_q_train_shape"]),
        "saved_teacher_q_test_shape": "" if saved_teacher is None else str(saved_teacher["teacher_q_test_shape"]),
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
        "visual_prompt_trainable_params": visual_prompt_trainable_params,
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
