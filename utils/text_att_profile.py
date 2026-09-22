import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TextProfile(nn.Module):
    def __init__(self, num_entities, text_dim=15, latent_dim=64, dropout=0.2):
        super().__init__()

        self.text_dim = text_dim
        self.latent_dim = latent_dim

        self.W_query = nn.Linear(text_dim, latent_dim)
        self.W_key = nn.Linear(text_dim, latent_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, ids, text, mask=None):
        # ids kept in the signature for call-site compatibility (unused).
        if text.dim() == 2:
            text = text.unsqueeze(1)

        if mask is not None:
            mask_f = mask.to(dtype=text.dtype).unsqueeze(-1)  # (B, L, 1)
            counts = mask_f.sum(dim=1).clamp_min(1.0)          # (B, 1)
            content_mean = (text * mask_f).sum(dim=1) / counts  # (B, text_dim)
        else:
            content_mean = text.mean(dim=1)

        query = self.W_query(content_mean).unsqueeze(1)  # (B, 1, latent_dim)
        keys = self.W_key(text)                            # (B, L, latent_dim)

        scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(self.latent_dim)

        if mask is not None:
            mask_b = mask.to(dtype=torch.bool).unsqueeze(1)
            scores = scores.masked_fill(~mask_b, float("-inf"))

            all_masked = (~mask_b).all(dim=-1)
            if all_masked.any():
                scores[all_masked.squeeze(-1)] = 0.0

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        context = torch.bmm(weights, text).squeeze(1)
        context = self.norm(context)

        return F.softmax(context, dim=-1)