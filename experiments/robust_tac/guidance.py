import os

import torch
import torch.nn.functional as F


class GuidanceProvider:
    def __init__(self, guidance_type="placeholder", guidance_path=None, device=None):
        self.requested_type = guidance_type
        self.guidance_path = guidance_path
        self.device = device or torch.device("cpu")
        self.train_text_counterpart = None
        self.test_text_counterpart = None
        self.noun_anchor_bank = None
        self.selected_noun_indices = None
        self.selected_nouns = None
        self.tau_retrieval = None
        self.keys = []
        self.loaded_path = None
        self.guidance_type = "placeholder"

        if guidance_type == "placeholder" or guidance_type == "auto":
            return
        if guidance_type == "tac_text":
            self._load_tac_text(guidance_path)
            return
        raise ValueError(f"Unsupported guidance_type: {guidance_type}")

    def _load_tac_text(self, guidance_path):
        if not guidance_path:
            raise ValueError("--guidance-path is required when --guidance-type tac_text")
        if not os.path.exists(guidance_path):
            raise FileNotFoundError(f"Guidance file not found: {guidance_path}")

        import numpy as np

        data = np.load(guidance_path, allow_pickle=False)
        self.keys = list(data.files)
        required = ["train_text_counterpart", "test_text_counterpart", "noun_anchor_bank"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Guidance file missing required arrays: {missing}")

        self.train_text_counterpart = self._to_feature_tensor(data["train_text_counterpart"], "train_text_counterpart")
        self.test_text_counterpart = self._to_feature_tensor(data["test_text_counterpart"], "test_text_counterpart")
        self.noun_anchor_bank = self._to_feature_tensor(data["noun_anchor_bank"], "noun_anchor_bank")
        if "selected_noun_indices" in data:
            self.selected_noun_indices = data["selected_noun_indices"].astype("int64")
        if "selected_nouns" in data:
            self.selected_nouns = data["selected_nouns"].astype(str).tolist()
        if "tau_retrieval" in data:
            self.tau_retrieval = float(data["tau_retrieval"])
        self.loaded_path = guidance_path
        self.guidance_type = "tac_text"

    def _to_feature_tensor(self, array, name):
        if array.ndim != 2 or array.shape[1] != 512:
            raise ValueError(f"{name} must have shape [N, 512], got {array.shape}")
        tensor = torch.from_numpy(array).float()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return F.normalize(tensor, dim=-1).to(self.device)

    def get(self, indices, fallback_features, split="train"):
        if self.guidance_type == "placeholder":
            return fallback_features.detach()

        if split == "train":
            bank = self.train_text_counterpart
        elif split == "test":
            bank = self.test_text_counterpart
        else:
            raise ValueError(f"Unknown guidance split: {split}")

        if indices.numel() == 0:
            return bank[indices].detach()
        max_index = int(indices.max().item())
        if max_index >= bank.shape[0]:
            raise IndexError(
                f"{split} guidance has {bank.shape[0]} rows, "
                f"but requested index {max_index}"
            )
        return bank[indices].detach()
