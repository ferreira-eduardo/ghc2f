import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model.gated_ae import GatedHybridCFAutoEncoder


class GHC2F(GatedHybridCFAutoEncoder):
    def __init__(self, *ae_args, **ae_kwargs):
        super().__init__(*ae_args, **ae_kwargs)
        self.item_input_dim = self.encoder[0].out_features

        self.bottleneck_dim = ae_kwargs['layer_sizes'][-1]

        self.item_projection = nn.Linear(self.item_input_dim, self.bottleneck_dim).to(self.device)

        if self.item_input_dim != self.bottleneck_dim:
            self.item_aligner = nn.Linear(self.item_input_dim, self.bottleneck_dim)
        else:
            self.item_aligner = nn.Identity()


    def get_item_embeddings(self):
        raw_weights = self.encoder[0].weight.t()
        return self.item_aligner(raw_weights)

    def forward_bpr(self, batch):
        _, z_fused, _ = self.forward(batch, return_code=True)

        raw_item_embeddings = self.get_item_embeddings()

        pos_item_ids = batch["pos_item_id"]
        neg_item_ids = batch["neg_item_id"]

        w_pos = raw_item_embeddings[pos_item_ids]
        w_neg = raw_item_embeddings[neg_item_ids]

        pos_scores = (z_fused * w_pos).sum(dim=-1)
        neg_scores = (z_fused * w_neg).sum(dim=-1)

        return pos_scores, neg_scores

    def calculate_loss(self, batch, reg_weight=1e-5, cl_weight=0.1):
        logits, pos_scores, neg_scores, z_cf, z_fused = self.forward(batch, return_code=True)

        loss_bpr = F.softplus(neg_scores - pos_scores).mean()

        loss_cl = self.contrastive_loss(z_cf, z_fused)

        item_embeddings = self.get_item_embeddings()
        reg_loss = (torch.norm(item_embeddings[batch["pos_item_id"]])**2 +
                    torch.norm(item_embeddings[batch["neg_item_id"]])**2 +
                    torch.norm(batch["ratings_in"])**2)

        total_loss = loss_bpr + (cl_weight * loss_cl) + (reg_weight * reg_loss)

        return total_loss, batch["user_ids"].size(0)

    def contrastive_loss(self, z_fused, z_ae, temperature=0.1):
        """
        Standard InfoNCE loss to align two views of the bottleneck.
        z_fused: The gated/fused representation [B, code_dim]
        z_ae: The raw encoder output before fusion [B, code_dim]
        """
        z_fused = F.normalize(z_fused, dim=-1)
        z_ae = F.normalize(z_ae, dim=-1)

        logits = torch.matmul(z_fused, z_ae.t()) / temperature

        labels = torch.arange(z_fused.size(0)).to(self.device)

        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def predict_step(self, batch):
        """
        Optimized for Leave-One-Out Evaluation.
        Returns:
            - all_scores: The predicted scores for all items (B, num_items)
            - target_item: The ground truth item ID (B,)
        """
        all_scores = self.predict_unseen(batch)

        target_item = batch["target_item"].to(self.device)

        return all_scores, target_item

    @torch.no_grad()
    def predict_unseen(self, batch):
        self.eval()
        _, z_fused, _ = self.forward(batch, return_code=True)

        item_embeddings = self.get_item_embeddings()
        all_scores = torch.matmul(z_fused, item_embeddings.t())

        ratings_in = batch["ratings_in"]
        all_scores[ratings_in > 0] = -1e9

        return all_scores


