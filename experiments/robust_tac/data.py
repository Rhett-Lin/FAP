import json
import os
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms


IGNORED = {"BACKGROUND_Google", "Faces_easy"}
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


def _caltech_dir(root):
    root = Path(os.path.expanduser(root)).resolve()
    if root.name == "caltech-101":
        return root
    return root / "caltech-101"


def _read_split(split_path, image_dir):
    with open(split_path, "r") as f:
        split = json.load(f)

    out = {}
    for split_name, items in split.items():
        records = []
        for rel_path, label, classname in items:
            category = Path(rel_path).parts[0]
            if category in IGNORED:
                continue
            records.append(
                {
                    "path": str(Path(image_dir) / rel_path),
                    "label": int(label),
                    "classname": NEW_CNAMES.get(classname, classname),
                    "category": category,
                }
            )
        out[split_name] = records
    return out


def _build_from_folders(image_dir):
    records = []
    label = 0
    for category in sorted(os.listdir(image_dir)):
        category_dir = Path(image_dir) / category
        if not category_dir.is_dir() or category in IGNORED:
            continue
        classname = NEW_CNAMES.get(category, category.replace("_", " "))
        for filename in sorted(os.listdir(category_dir)):
            if filename.startswith("."):
                continue
            records.append(
                {
                    "path": str(category_dir / filename),
                    "label": label,
                    "classname": classname,
                    "category": category,
                }
            )
        label += 1

    # Deterministic 50/20/30 split compatible with the FAP fallback ratios.
    split = {"train": [], "val": [], "test": []}
    by_label = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)
    for items in by_label.values():
        n_total = len(items)
        n_train = round(n_total * 0.5)
        n_val = round(n_total * 0.2)
        split["train"].extend(items[:n_train])
        split["val"].extend(items[n_train : n_train + n_val])
        split["test"].extend(items[n_train + n_val :])
    return split


class IndexedCaltech101(Dataset):
    def __init__(self, records, label_map, max_samples=None, image_size=224):
        self.records = []
        for full_index, record in enumerate(records):
            mapped = dict(record)
            mapped["full_index"] = full_index
            mapped["label"] = label_map[record["label"]]
            self.records.append(mapped)
        if max_samples is not None:
            self.records = self.records[: max(0, int(max_samples))]

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        record = self.records[item]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "label": torch.tensor(record["label"], dtype=torch.long),
            "index": torch.tensor(record["full_index"], dtype=torch.long),
        }


class IndexedTorchvisionDataset(Dataset):
    def __init__(self, dataset, max_samples=None):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        if max_samples is not None:
            self.indices = self.indices[: max(0, int(max_samples))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        full_index = self.indices[item]
        image, label = self.dataset[full_index]
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(full_index, dtype=torch.long),
        }


def make_caltech101_datasets(root, max_train_samples=None, max_test_samples=None):
    dataset_dir = _caltech_dir(root)
    image_dir = dataset_dir / "101_ObjectCategories"
    split_path = dataset_dir / "split_zhou_Caltech101.json"
    if not image_dir.exists():
        raise FileNotFoundError(f"Caltech101 image dir not found: {image_dir}")

    if split_path.exists():
        split = _read_split(split_path, image_dir)
        split_source = str(split_path)
    else:
        split = _build_from_folders(image_dir)
        split_source = "folder-fallback"

    train_records = split["train"]
    test_records = split["test"]
    labels = sorted({r["label"] for r in train_records + test_records})
    label_map = {label: idx for idx, label in enumerate(labels)}
    classnames = {}
    for records in (train_records, test_records):
        for record in records:
            classnames[label_map[record["label"]]] = record["classname"]

    train_set = IndexedCaltech101(train_records, label_map, max_train_samples)
    test_set = IndexedCaltech101(test_records, label_map, max_test_samples)
    return {
        "train": train_set,
        "test": test_set,
        "num_clusters": len(labels),
        "classnames": [classnames[i] for i in range(len(classnames))],
        "split_source": split_source,
        "dataset_dir": str(dataset_dir),
    }


def make_cifar10_datasets(root, max_train_samples=None, max_test_samples=None, download=True):
    root = Path(os.path.expanduser(root)).resolve()
    transform = transforms.Compose([transforms.ToTensor()])
    train_base = CIFAR10(root=str(root), train=True, download=download, transform=transform)
    test_base = CIFAR10(root=str(root), train=False, download=download, transform=transform)
    return {
        "train": IndexedTorchvisionDataset(train_base, max_train_samples),
        "test": IndexedTorchvisionDataset(test_base, max_test_samples),
        "num_clusters": 10,
        "classnames": list(train_base.classes),
        "split_source": "torchvision.CIFAR10",
        "dataset_dir": str(root / "cifar-10-batches-py"),
    }


def make_robust_tac_datasets(dataset, root, max_train_samples=None, max_test_samples=None):
    if dataset == "Caltech101":
        return make_caltech101_datasets(root, max_train_samples, max_test_samples)
    if dataset in {"CIFAR10", "CIFAR-10"}:
        return make_cifar10_datasets(root, max_train_samples, max_test_samples)
    raise ValueError(f"Unsupported dataset: {dataset}")
