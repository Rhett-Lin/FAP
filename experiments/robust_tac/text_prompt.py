import torch
from torch import nn
import torch.nn.functional as F

from clip import clip


class TextPromptEncoder(nn.Module):
    def __init__(self, clip_model, nouns, n_ctx=2):
        super().__init__()
        if n_ctx <= 0:
            raise ValueError("n_ctx must be positive for text prompt")
        if not nouns:
            raise ValueError("TextPromptEncoder requires selected nouns")

        object.__setattr__(self, "clip_model", clip_model)
        self.nouns = [str(noun) for noun in nouns]
        self.n_ctx = n_ctx
        width = clip_model.token_embedding.weight.shape[1]
        ctx = torch.empty(n_ctx, width)
        nn.init.normal_(ctx, std=0.02)
        self.ctx = nn.Parameter(ctx)

        prompts = [f"{' '.join(['X'] * n_ctx)} {noun.replace('_', ' ')}." for noun in self.nouns]
        self.register_buffer("tokenized_prompts", clip.tokenize(prompts), persistent=False)

        for param in self.clip_model.parameters():
            param.requires_grad_(False)

    @property
    def dtype(self):
        return self.clip_model.dtype

    def forward(self):
        tokens = self.tokenized_prompts.to(self.ctx.device)
        x = self.clip_model.token_embedding(tokens).type(self.dtype)
        ctx = self.ctx.type(self.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([x[:, :1, :], ctx, x[:, 1 + self.n_ctx :, :]], dim=1)
        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(x.float(), dim=-1)

    def text_counterpart(self, image_features, tau_retrieval):
        noun_anchor_bank = self()
        logits = image_features @ noun_anchor_bank.t()
        weights = torch.softmax(logits / tau_retrieval, dim=1)
        return F.normalize(weights @ noun_anchor_bank, dim=-1), noun_anchor_bank

    def prompt_named_parameters(self):
        yield "text_prompt.ctx", self.ctx
