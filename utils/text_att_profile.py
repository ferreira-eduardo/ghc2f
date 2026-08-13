import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TextProfile(nn.Module):
    def __init__(self, num_entities, text_dim=15, latent_dim=64, dropout=0.2):
        super().__init__()

        self.text_dim = text_dim
        self.latent_dim = latent_dim

        self.entity_embedding = nn.Embedding(num_entities, latent_dim)
        self.W_query = nn.Linear(latent_dim, latent_dim)
        self.W_key = nn.Linear(text_dim, latent_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        # LayerNorm now must match text_dim
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, ids, text, mask=None):
        query = self.W_query(self.entity_embedding(ids)).unsqueeze(1)

        if text.dim() == 2:
            text = text.unsqueeze(1)
        keys = self.W_key(text)

        scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(self.latent_dim)

        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            # Ensure mask is (Batch, 1, 1) to match scores
            if mask.dim() == 2:
                # If mask is per-user rather than per-text-feature
                mask = mask.any(dim=1, keepdim=True).unsqueeze(2)

            scores = scores.masked_fill(~mask, float("-inf"))

            # Prevent NaN if everything is masked
            all_masked = (~mask).all(dim=-1)
            if all_masked.any():
                scores[all_masked.squeeze(-1)] = 0.0

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        context = torch.bmm(weights, text).squeeze(1)

        context = self.norm(context)

        return F.softmax(context, dim=-1)