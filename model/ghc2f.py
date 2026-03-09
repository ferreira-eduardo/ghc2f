import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from model.autoencoder import CFAutoEncoder
from utils.topic_att_profile import TopicProfile


class GHC2F(CFAutoEncoder):
    def __init__(self,layer_sizes, num_users, num_items, topics_dim=15,
                 topics_latent_dim=64, **ae_kwargs):

        super().__init__(layer_sizes, **ae_kwargs)
        self.item_input_dim = self.encoder[0].out_features
        self.training = True

        enc_dims = [out_dim for _, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.code_dim_cf = layer_sizes[-1]

        self.topic_to_enc = nn.ModuleList([
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
        ################################

        self.register_buffer("item_global_profiles", None)

        self.code_dim_cf = layer_sizes[-1]

        # User profiler (Interactions in trainset)
        self.user_profiler = TopicProfile(num_users, topics_dim, topics_latent_dim)

        # Item profiler (Global view - reviews independent of the current user)
        self.item_profiler = TopicProfile(num_items, topics_dim, topics_latent_dim)

        # Projection layers to align topic space (15) with CF space (code_dim)
        self.user_proj = nn.Linear(topics_dim, self.code_dim_cf)
        self.item_proj = nn.Linear(topics_dim, self.code_dim_cf)

        # Gating Mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(self.code_dim_cf * 2, self.code_dim_cf),
            nn.Sigmoid()
        )

        self.gate_drop = nn.Dropout(p=0.2)


        self.bottleneck_dim = ae_kwargs['layer_sizes'][-1]

        # New projection layer for items
        self.item_projection = nn.Linear(self.item_input_dim, self.bottleneck_dim).to(self.device)

        # Create a Projection Layer
        # This ensures that no matter what 'bottleneck_dim' is, items will match
        if self.item_input_dim != self.bottleneck_dim:
            self.item_aligner = nn.Linear(self.item_input_dim, self.bottleneck_dim)
        else:
            self.item_aligner = nn.Identity()

    def encode_with_topics(self, x, z_topic_base):
        """
        x: (B, num_items)
        z_topic_base: (B, code_dim_cf)
        """
        h = x
        last_idx = len(self.encoder) - 1

        for li, lin in enumerate(self.encoder):
            h = lin(h)

            # activation (skip if your bottleneck is linear)
            if li != last_idx:
                h = F.relu(h)
            # else: keep code linear (common in AEs)

            # decide whether to fuse here
            do_fuse = self.fuse_all_layers or (li == last_idx)
            if do_fuse:
                t = self.topic_to_enc[li](z_topic_base)  # (B, out_dim)
                g = self.gate_enc[li](torch.cat([h, t], dim=-1))
                g = self.gate_drop(g)
                h = g * h + (1.0 - g) * t
                h = self.enc_norms[li](h)

        h = self.drop(h)
        return h

    def get_item_embeddings(self):
        # Pulls from first layer: Shape (num_items, 2048)
        raw_weights = self.encoder[0].weight.t()
        # Projects to bottleneck: Shape (num_items, bottleneck_dim)
        return self.item_aligner(raw_weights)

    def forward_bpr(self, batch):
        # Get user representation (Encoder path)
        _, z_fused, _ = self.forward(batch, return_code=True)

        # Get item embeddings
        raw_item_embeddings = self.get_item_embeddings()

        pos_item_ids = batch["pos_item_ids"]
        neg_item_ids = batch["neg_item_ids"]

        # Pull specific vectors
        w_pos = raw_item_embeddings[pos_item_ids]
        w_neg = raw_item_embeddings[neg_item_ids]

        # Dot product with z_fused (Batch, 2048)
        pos_scores = (z_fused * w_pos).sum(dim=-1)
        neg_scores = (z_fused * w_neg).sum(dim=-1)

        return pos_scores, neg_scores

    def calculate_loss(self, batch, reg_weight=1e-5):
        pos_scores, neg_scores = self.forward_bpr(batch)

        # BPR Loss (Softplus of the difference is more stable than -log(sigmoid))
        # We want pos_scores >> neg_scores, so (neg - pos) should be very negative
        loss = F.softplus(neg_scores - pos_scores).mean()

        # L2 Regularization
        # Only regularize the specific embeddings involved in the batch
        # to prevent 'embedding drift'
        item_embeddings = self.get_item_embeddings()
        reg_loss = (torch.norm(item_embeddings[batch["pos_item_ids"]])**2 +
                    torch.norm(item_embeddings[batch["neg_item_ids"]])**2 +
                    torch.norm(batch["ratings_in"])**2) # Optional: regularize input

        total_loss = loss + (reg_weight * reg_loss)

        return total_loss, batch["user_ids"].size(0)


    @torch.no_grad()
    def predict_step(self, batch):
        """
        Optimized for Leave-One-Out Evaluation.
        Returns:
            - all_scores: The predicted scores for all items (B, num_items)
            - target_item: The ground truth item ID (B,)
        """
        # Get the scores for ALL items (using the logic we defined for ranking)
        # This automatically uses z_fused and masked training items
        all_scores = self.predict_unseen(batch)

        # Get the target item from the batch
        target_item = batch["target_item"].to(self.device)

        return all_scores, target_item

    @torch.no_grad()
    def predict_unseen(self, batch):
        self.eval()
        _, z_fused, _ = self.forward(batch, return_code=True)

        item_embeddings = self.get_item_embeddings()

        # Scores for ALL items: (Batch, Code_Dim) @ (Code_Dim, Num_Items)
        all_scores = torch.matmul(z_fused, item_embeddings.t())

        # This ensures they don't appear in Top-K recommendations
        ratings_in = batch["ratings_in"]
        all_scores[ratings_in > 0] = -1e9

        return all_scores

    def evaluate(self, loader, top_k=10):
        self.eval()
        hr, ndcg, mrr = [], [], []

        with torch.no_grad():
            for batch in loader:
                # Get scores for all items and the target item index
                all_scores, target_items = self.predict_step(batch)

                # Find the rank of the target item among all scores
                # torch.topk returns the indices of the highest scores
                _, top_indices = torch.topk(all_scores, k=top_k, dim=1)

                top_indices = top_indices.cpu().numpy()
                target_items = target_items.cpu().numpy()

                # Check if the target item is in the top K
                for i in range(len(target_items)):
                    target = target_items[i]
                    indices = top_indices[i]

                    if target in indices:
                        # 1. Hit Rate
                        hr.append(1.0)

                        # 2. Find the exact rank (0-indexed, so we add 1)
                        rank = np.where(indices == target)[0][0] + 1

                        # 3. NDCG
                        ndcg.append(1 / math.log2(rank + 1))

                        # 4. MRR
                        mrr.append(1 / rank)
                    else:
                        hr.append(0.0)
                        ndcg.append(0.0)
                        mrr.append(0.0)
        return {
            'hit_rate': np.mean(hr),
            'ndcg': np.mean(ndcg),
            'mrr': np.mean(mrr)
        }
