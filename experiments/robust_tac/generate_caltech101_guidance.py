import argparse
import csv
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clip import clip
from data import make_robust_tac_datasets
from models import FrozenCLIPImageEncoder
from utils import ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TAC-style text guidance")
    parser.add_argument("--dataset", default="Caltech101")
    parser.add_argument("--root", required=True)
    parser.add_argument("--tac-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-centers", default="auto_num_clusters")
    parser.add_argument("--top-nouns-per-center", type=int, default=5)
    parser.add_argument("--tau-retrieval", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-nouns", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def extract_image_embeddings(encoder, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    features = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            features.append(encoder(images).cpu())
    return F.normalize(torch.cat(features, dim=0), dim=-1).numpy().astype("float32")


def find_noun_resources(tac_root):
    data_dir = Path(tac_root) / "data"
    embedding_path = data_dir / "nouns_embedding_ensemble.npy"
    noun_list_candidates = [
        data_dir / "WordNetNouns.csv",
        data_dir / "nouns.txt",
        data_dir / "wordnet.txt",
        data_dir / "wordnet_nouns.txt",
    ]
    noun_list_path = next((path for path in noun_list_candidates if path.exists()), None)
    return {
        "embedding_path": str(embedding_path) if embedding_path.exists() else None,
        "noun_list_path": str(noun_list_path) if noun_list_path else None,
    }


def read_noun_list(path, max_nouns=None):
    nouns = []
    if path.endswith(".csv"):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "word" not in reader.fieldnames:
                raise ValueError(f"CSV noun list must contain a 'word' column: {path}")
            for row in reader:
                noun = row["word"].strip()
                if noun:
                    nouns.append(noun)
                if max_nouns is not None and len(nouns) >= max_nouns:
                    break
    else:
        with open(path, "r") as f:
            for line in f:
                noun = line.strip().split()[0] if line.strip() else ""
                if noun:
                    nouns.append(noun)
                if max_nouns is not None and len(nouns) >= max_nouns:
                    break
    return nouns


def encode_nouns(clip_model, nouns, batch_size, device):
    features = []
    with torch.no_grad():
        for start in range(0, len(nouns), batch_size):
            batch_nouns = nouns[start : start + batch_size]
            prompts = [f"a photo of a {noun.replace('_', ' ')}." for noun in batch_nouns]
            tokens = clip.tokenize(prompts).to(device)
            text_features = clip_model.encode_text(tokens)
            text_features = F.normalize(text_features.float(), dim=-1)
            features.append(text_features.cpu())
    return torch.cat(features, dim=0).numpy().astype("float32")


def load_or_encode_candidate_nouns(args, encoder):
    resources = find_noun_resources(args.tac_root)
    if resources["embedding_path"]:
        noun_embeddings = np.load(resources["embedding_path"]).astype("float32")
        noun_embeddings = normalize_np(noun_embeddings)
        noun_count = noun_embeddings.shape[0]
        nouns = [f"noun_{i}" for i in range(noun_count)]
        if args.max_nouns is not None:
            noun_embeddings = noun_embeddings[: args.max_nouns]
            nouns = nouns[: args.max_nouns]
        print(f"candidate noun source: existing embeddings {resources['embedding_path']}")
        return nouns, noun_embeddings, resources

    if not resources["noun_list_path"]:
        raise FileNotFoundError(
            "No TAC noun resources found. Expected nouns_embedding_ensemble.npy "
            "or a noun list such as WordNetNouns.csv."
        )

    nouns = read_noun_list(resources["noun_list_path"], max_nouns=args.max_nouns)
    if not nouns:
        raise ValueError(f"No nouns loaded from {resources['noun_list_path']}")
    print(f"candidate noun source: encoded noun list {resources['noun_list_path']}")
    noun_embeddings = encode_nouns(encoder.clip_model, nouns, args.batch_size, next(encoder.parameters()).device)
    return nouns, noun_embeddings, resources


def normalize_np(array):
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def resolve_semantic_centers(value, n_train, num_clusters):
    if value == "auto_tac":
        return max(1, round(n_train / 300))
    if value == "auto_num_clusters":
        return int(num_clusters)
    centers = int(value)
    if centers <= 0:
        raise ValueError("--semantic-centers must be positive")
    return centers


def sklearn_kmeans(features, n_clusters, seed):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300)
    assignments = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_.astype("float32")
    return assignments, normalize_np(centers)


def torch_kmeans(features, n_clusters, seed, iters=100):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.from_numpy(features).float()
    perm = torch.randperm(x.shape[0], generator=generator)[:n_clusters]
    centers = x[perm].clone()
    for _ in range(iters):
        sims = x @ F.normalize(centers, dim=1).t()
        assignments = sims.argmax(dim=1)
        new_centers = centers.clone()
        for idx in range(n_clusters):
            mask = assignments == idx
            if mask.any():
                new_centers[idx] = x[mask].mean(dim=0)
        if torch.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    return assignments.numpy(), F.normalize(centers, dim=1).numpy().astype("float32")


def compute_semantic_centers(features, n_clusters, seed):
    if n_clusters > features.shape[0]:
        raise ValueError(f"semantic centers ({n_clusters}) cannot exceed N_train ({features.shape[0]})")
    try:
        return sklearn_kmeans(features, n_clusters, seed)
    except ImportError:
        print("sklearn is unavailable; using torch k-means fallback")
        return torch_kmeans(features, n_clusters, seed)


def select_noun_anchors(centers, noun_embeddings, top_nouns_per_center):
    centers = normalize_np(centers)
    noun_embeddings = normalize_np(noun_embeddings)
    similarity = torch.from_numpy(centers) @ torch.from_numpy(noun_embeddings).t()
    softmax_nouns = torch.softmax(similarity, dim=0)
    class_pred = torch.argmax(softmax_nouns, dim=0)
    selected = torch.zeros(noun_embeddings.shape[0], dtype=torch.bool)
    for center_idx in range(centers.shape[0]):
        noun_idx = torch.where(class_pred == center_idx)[0]
        if noun_idx.numel() == 0:
            continue
        confidence = softmax_nouns[:, noun_idx].max(dim=0)[0]
        rank = torch.argsort(confidence, descending=True)
        selected[noun_idx[rank[:top_nouns_per_center]]] = True

    selected_indices = torch.where(selected)[0].numpy().astype("int64")
    if selected_indices.size == 0:
        raise ValueError("No noun anchors selected; try increasing --top-nouns-per-center")
    return selected_indices, noun_embeddings[selected_indices]


def retrieve_text_counterpart(image_embeddings, noun_anchor_bank, tau):
    image = torch.from_numpy(normalize_np(image_embeddings)).float()
    anchors = torch.from_numpy(normalize_np(noun_anchor_bank)).float()
    outputs = []
    batch_size = 2048
    for start in range(0, image.shape[0], batch_size):
        batch = image[start : start + batch_size]
        similarity = batch @ anchors.t()
        weights = torch.softmax(similarity / tau, dim=1)
        counterpart = F.normalize(weights @ anchors, dim=1)
        outputs.append(counterpart)
    return torch.cat(outputs, dim=0).numpy().astype("float32")


def dataset_paths(dataset):
    if hasattr(dataset, "records"):
        return np.array([record["path"] for record in dataset.records], dtype=str)
    if hasattr(dataset, "indices"):
        return np.array([str(index) for index in dataset.indices], dtype=str)
    return np.array([str(index) for index in range(len(dataset))], dtype=str)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    datasets = make_robust_tac_datasets(
        args.dataset,
        args.root,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    encoder = FrozenCLIPImageEncoder("ViT-B/32", device=device, mode="static").to(device)
    encoder.eval()
    if not encoder.clip_backbone_frozen():
        raise AssertionError("CLIP backbone must be frozen while generating guidance")

    train_embeddings = extract_image_embeddings(encoder, datasets["train"], args.batch_size, args.num_workers, device)
    test_embeddings = extract_image_embeddings(encoder, datasets["test"], args.batch_size, args.num_workers, device)
    nouns, noun_embeddings, resources = load_or_encode_candidate_nouns(args, encoder)
    noun_embeddings = normalize_np(noun_embeddings)

    semantic_centers = resolve_semantic_centers(
        args.semantic_centers,
        train_embeddings.shape[0],
        datasets["num_clusters"],
    )
    _, centers = compute_semantic_centers(train_embeddings, semantic_centers, args.seed)
    selected_indices, noun_anchor_bank = select_noun_anchors(
        centers,
        noun_embeddings,
        args.top_nouns_per_center,
    )
    selected_nouns = np.array([nouns[i] for i in selected_indices], dtype=str)

    train_text_counterpart = retrieve_text_counterpart(train_embeddings, noun_anchor_bank, args.tau_retrieval)
    test_text_counterpart = retrieve_text_counterpart(test_embeddings, noun_anchor_bank, args.tau_retrieval)

    has_nan = (
        np.isnan(train_text_counterpart).any()
        or np.isnan(test_text_counterpart).any()
        or np.isnan(noun_anchor_bank).any()
    )
    mean_norm = float(np.linalg.norm(train_text_counterpart, axis=1).mean())

    ensure_dir(os.path.dirname(args.output))
    np.savez_compressed(
        args.output,
        train_text_counterpart=train_text_counterpart,
        test_text_counterpart=test_text_counterpart,
        noun_anchor_bank=noun_anchor_bank.astype("float32"),
        train_image_embedding=train_embeddings,
        test_image_embedding=test_embeddings,
        selected_noun_indices=selected_indices,
        selected_nouns=selected_nouns,
        train_paths=dataset_paths(datasets["train"]),
        test_paths=dataset_paths(datasets["test"]),
        dataset=np.array(args.dataset),
        num_clusters=np.array(datasets["num_clusters"], dtype="int64"),
        semantic_centers=np.array(semantic_centers, dtype="int64"),
        tau_retrieval=np.array(args.tau_retrieval, dtype="float32"),
        guidance_type=np.array("tac_text"),
        noun_source=np.array(resources["embedding_path"] or resources["noun_list_path"] or ""),
    )

    print(f"N_train: {train_embeddings.shape[0]}")
    print(f"N_test: {test_embeddings.shape[0]}")
    print(f"candidate noun count: {noun_embeddings.shape[0]}")
    print(f"selected noun count: {noun_anchor_bank.shape[0]}")
    print(f"noun_anchor_bank shape: {noun_anchor_bank.shape}")
    print(f"train_text_counterpart shape: {train_text_counterpart.shape}")
    print(f"test_text_counterpart shape: {test_text_counterpart.shape}")
    print(f"mean norm of text_counterpart: {mean_norm:.6f}")
    print(f"whether NaN exists: {has_nan}")
    print(f"dataset: {args.dataset}")
    print(f"num_clusters: {datasets['num_clusters']}")
    print(f"semantic_centers: {semantic_centers}")
    print(f"tau_retrieval: {args.tau_retrieval}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
