import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.text_att_profile import TextProfile
from model.cf_autoencoder import CFAutoEncoder, AEOutput
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import List


@dataclass
class GatedAEOutput(AEOutput):
    z_fused: torch.Tensor = None
    z_cf: torch.Tensor = None
    z_topic: torch.Tensor = None
    gate_values: List[torch.Tensor] = field(default_factory=list)
    pos_scores: torch.Tensor = None
    neg_scores: torch.Tensor = None


class GatedHybridCFAutoEncoder(CFAutoEncoder):
    def __init__(self, layer_sizes, num_users, num_items, text_dim=15,
                 text_latent_dim=64, topic_gamma=0.5, **kwargs):
        super().__init__(layer_sizes, **kwargs)
        self.name = 'GatedHybrid'

        enc_dims = [out_dim for _, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])]

        self.topic_gamma = topic_gamma

        self.code_dim_cf = layer_sizes[-1]

        self.text_to_enc = nn.ModuleList([
            nn.Linear(self.code_dim_cf, d) for d in enc_dims
        ])
        self.gate_enc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d * 2, d),
                nn.Sigmoid()
            ) for d in enc_dims
        ])

        self.fuse_all_layers = True
        self.enc_norms = nn.ModuleList([nn.LayerNorm(d) for d in enc_dims])

        self.register_buffer("item_global_profiles", None)

        # User profiler (Interactions in trainset)
        self.user_profiler = TextProfile(num_users, text_dim, text_latent_dim)

        # Item profiler (Global view - reviews independent of the current user)
        self.item_profiler = TextProfile(num_items, text_dim, text_latent_dim)

        # Projection layers to align topic space with CF space (code_dim)
        self.user_proj = nn.Linear(text_dim, self.code_dim_cf)
        self.item_proj = nn.Linear(text_dim, self.code_dim_cf)

        self.gate_drop = nn.Dropout(p=0.2)

    def encode_with_text(self, x, z_topic_base):
        """
        x: (B, num_items)
        z_topic_base: (B, code_dim_cf)

        Returns
        -------
        h : torch.Tensor
            Fused code.
        gate_values : List[torch.Tensor]
            Per-layer gate activations g_l, needed for the gate entropy
            regularizer.
        """
        h = x
        last_idx = len(self.encoder) - 1
        gate_values = []

        for li, lin in enumerate(self.encoder):
            h = lin(h)

            if li != last_idx:
                h = F.relu(h)

            do_fuse = self.fuse_all_layers or (li == last_idx)
            if do_fuse:
                t = self.text_to_enc[li](z_topic_base)  # (B, out_dim)
                g = self.gate_enc[li](torch.cat([h, t], dim=-1))
                g = self.gate_drop(g)
                h = g * h + (1.0 - g) * t
                h = self.enc_norms[li](h)
                gate_values.append(g)

        h = self.drop(h)
        return h, gate_values

    def forward(self, batch) -> GatedAEOutput:
        ratings_in = batch["ratings_in"].to(self.device)

        # collaborative signal
        z_cf = self.encode(ratings_in)

        u_ids = batch["user_ids"].to(self.device)
        u_text = batch["user_text"].to(self.device)
        u_mask = batch["user_mask"].to(self.device)

        z_user_topic = self.user_profiler(u_ids, u_text, u_mask)
        topic_user = self.user_proj(z_user_topic)

        item_global = self.item_global_profiles
        hist_mask = (ratings_in != 0).float()
        interaction_counts = hist_mask.sum(dim=1, keepdim=True)
        topic_item_global = (hist_mask @ item_global) / interaction_counts.clamp_min(1.0)
        topic_item = self.item_proj(topic_item_global)

        z_topic = self.topic_gamma * topic_user + (1 - self.topic_gamma) * topic_item

        z_fused, gate_values = self.encode_with_text(ratings_in, z_topic)

        logits = self.decode(z_fused)

        pos_items = batch["pos_item_id"].to(self.device)
        neg_items = batch["neg_item_id"].to(self.device)

        if pos_items.dim() < logits.dim():
            pos_items = pos_items.unsqueeze(-1)
        if neg_items.dim() < logits.dim():
            neg_items = neg_items.unsqueeze(-1)

        pos_scores = torch.gather(logits, 1, pos_items)
        neg_scores = torch.gather(logits, 1, neg_items)

        return GatedAEOutput(
            recon=logits,
            code=z_fused,
            z_fused=z_fused,
            z_cf=z_cf,
            z_topic=z_topic,
            gate_values=gate_values,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
        )

    def calculate_loss(self, batch):
        out = self.forward(batch)

        user_ids = batch["user_ids"].to(self.device)

        loss = -torch.log(torch.sigmoid(out.pos_scores - out.neg_scores) + 1e-10).mean()

        return loss, user_ids.size(0)

    @torch.no_grad()
    def predict_step(self, batch):
        """
        Extracts ratings_tgt and compares it against model predictions.
        """
        ratings_tgt = batch["ratings_tgt"].to(self.device)

        out = self(batch)
        y_hat = out.recon

        mask = ratings_tgt != 0

        y_true_flat = ratings_tgt[mask]
        y_pred_flat = y_hat[mask]

        return y_true_flat, y_pred_flat

    def evaluate(self, test_loader, k=10):
        self.eval()
        hr_list, ndcg_list, mrr_list = [], [], []

        pbar = tqdm(test_loader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for batch in pbar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                out = self(batch)
                logits = out.recon  # was: out[0] if isinstance(out, (tuple, list)) else out

                pos_items = batch["pos_item_id"].unsqueeze(-1)
                neg_items = batch["neg_item_id"]
                target_indices = torch.cat([pos_items, neg_items], dim=1)

                test_scores = torch.gather(logits, 1, target_indices)

                pos_scores = test_scores[:, 0].unsqueeze(1)
                ranks = (test_scores > pos_scores).sum(dim=1) + 1

                ranks_cpu = ranks.cpu().numpy()

                hits = (ranks_cpu <= k).astype(float)
                hr_list.extend(hits)

                ndcgs = np.where(ranks_cpu <= k, 1 / np.log2(ranks_cpu + 1), 0.0)
                ndcg_list.extend(ndcgs)

                mrr_list.extend(1 / ranks_cpu)

        return {
            'hit_rate': float(np.mean(hr_list)),
            'ndcg': float(np.mean(ndcg_list)),
            'mrr': float(np.mean(mrr_list))
        }