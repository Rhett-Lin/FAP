import os

import torch
from torch import nn
import torch.nn.functional as F

from clip import clip


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _design_details(mode, prompt_depth=0, n_ctx=0):
    if mode == "static":
        trainer = "CoOp"
        vision_depth = 0
        vision_ctx = 0
    elif mode in {"vprompt", "dualprompt"}:
        trainer = "VPT"
        vision_depth = prompt_depth
        vision_ctx = n_ctx
    elif mode == "tprompt":
        trainer = "CoOp"
        vision_depth = 0
        vision_ctx = 0
    else:
        raise ValueError(f"Unsupported encoder mode: {mode}")
    return {
        "trainer": trainer,
        "vision_depth": vision_depth,
        "language_depth": 0,
        "vision_ctx": vision_ctx,
        "language_ctx": 0,
        "maple_length": n_ctx,
    }


def load_clip_to_device(model_name, device, mode="static", prompt_depth=0, n_ctx=0):
    url = clip._MODELS[model_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = _design_details(mode, prompt_depth=prompt_depth, n_ctx=n_ctx)
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    if device.type == "cpu":
        model.float()
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if mode in {"vprompt", "dualprompt"}:
        for name, param in model.visual.named_parameters():
            if "VPT" in name:
                param.requires_grad_(True)
    return model


class FrozenCLIPImageEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/32", device=None, mode="static", prompt_depth=0, n_ctx=0):
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.prompt_depth = prompt_depth
        self.n_ctx = n_ctx
        self.clip_model = load_clip_to_device(
            model_name,
            device,
            mode=mode,
            prompt_depth=prompt_depth,
            n_ctx=n_ctx,
        )
        self.image_size = self.clip_model.visual.input_resolution
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    @property
    def dtype(self):
        return self.clip_model.dtype

    def preprocess(self, image):
        image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        return (image - self.mean) / self.std

    def forward(self, image):
        image = self.preprocess(image)
        features = self.clip_model.encode_image(image.type(self.dtype))
        features = features.float()
        return F.normalize(features, dim=-1)

    def prompt_parameters(self):
        for name, param in self.clip_model.visual.named_parameters():
            if "VPT" in name:
                yield param

    def prompt_named_parameters(self):
        for name, param in self.clip_model.visual.named_parameters():
            if "VPT" in name:
                yield f"clip_model.visual.{name}", param

    def clip_backbone_frozen(self):
        for name, param in self.clip_model.named_parameters():
            if "visual.VPT" in name:
                continue
            if param.requires_grad:
                return False
        return True


class ClusterHead(nn.Module):
    def __init__(self, in_dim=512, num_clusters=10):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        self.cluster_head_image = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, text, image):
        return self.text_forward(text), self.image_forward(image)

    def text_forward(self, text):
        return self.cluster_head_text(text)

    def image_forward(self, image):
        return self.cluster_head_image(image)


class GuidanceProvider:
    def __init__(self, guidance_type="auto", guidance_path=None, tac_root=None, device=None):
        self.requested_type = guidance_type
        self.guidance_path = guidance_path
        self.tac_root = tac_root
        self.device = device or torch.device("cpu")
        self.text_counterpart = None
        self.noun_anchor_bank = None
        self.loaded_path = None
        self.guidance_type = "placeholder"
        if guidance_type != "placeholder":
            self._try_load_external()

    def _candidate_paths(self):
        candidates = []
        if self.guidance_path:
            candidates.append(self.guidance_path)
        if self.tac_root:
            data_dir = os.path.join(self.tac_root, "data")
            names = [
                "Caltech101_retrieved_nouns_embedding.npy",
                "Caltech101_text_counterpart.npy",
                "caltech101_retrieved_nouns_embedding.npy",
                "caltech101_text_counterpart.npy",
            ]
            candidates.extend(os.path.join(data_dir, name) for name in names)
        return candidates

    def _try_load_external(self):
        import numpy as np

        for path in self._candidate_paths():
            if not path or not os.path.exists(path):
                continue
            if path.endswith(".npz"):
                data = np.load(path)
                if "text_counterpart" in data:
                    self.text_counterpart = torch.from_numpy(data["text_counterpart"]).float()
                if "noun_anchor_bank" in data:
                    self.noun_anchor_bank = torch.from_numpy(data["noun_anchor_bank"]).float()
            elif path.endswith(".npy"):
                self.text_counterpart = torch.from_numpy(np.load(path)).float()
            else:
                continue
            if self.text_counterpart is not None:
                self.text_counterpart = F.normalize(self.text_counterpart, dim=-1).to(self.device)
                if self.noun_anchor_bank is not None:
                    self.noun_anchor_bank = F.normalize(self.noun_anchor_bank, dim=-1).to(self.device)
                self.loaded_path = path
                self.guidance_type = "retrieved"
                break

    def get(self, indices, fallback_features):
        if self.text_counterpart is None:
            return fallback_features.detach()
        if indices.max().item() >= self.text_counterpart.shape[0]:
            raise IndexError(
                f"Guidance cache has {self.text_counterpart.shape[0]} rows, "
                f"but requested index {indices.max().item()}"
            )
        return self.text_counterpart[indices].detach()
