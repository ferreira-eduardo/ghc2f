import numpy as np
import torch
import torch.nn as nn
# from ae_utils import MSEloss
from utils.topic_att_profile import TopicProfile
from model.autoencoder import CFAutoEncoder
import torch.nn.functional as F


class GatedHybridCFAutoEncoder(CFAutoEncoder):
    def __init__(self, layer_sizes, num_users, num_items, topics_dim=15,
                 topics_latent_dim=64, use_hybrid = True ,**kwargs):
        super().__init__(layer_sizes, **kwargs)

        self.training = True  # bpr loss
        self.use_hybrid = use_hybrid

        ########## add to layer-to-layer
        # encoder hidden dims, including code_dim
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

    def forward(self, batch, return_code=False):
        ratings_in = batch["ratings_in"].to(self.device)

        if not self.use_hybrid:
            z_cf = self.encode(ratings_in)
            z_fused = z_cf
        else:
            # --- Original Topic Logic ---
            u_ids = batch["user_ids"].to(self.device)
            u_topics = batch["user_topics"].to(self.device)
            u_mask = batch["user_mask"].to(self.device)

            z_user_topic = self.user_profiler(u_ids, u_topics, u_mask)
            topic_user = self.user_proj(z_user_topic)

            item_global = self.item_global_profiles
            hist_mask = (ratings_in != 0).float()
            interaction_counts = hist_mask.sum(dim=1, keepdim=True)
            topic_item_global = (hist_mask @ item_global) / interaction_counts.clamp_min(1.0)
            topic_item = self.item_proj(topic_item_global)

            z_topic = 0.5 * (topic_user + topic_item)
            z_fused = self.encode_with_topics(ratings_in, z_topic)

        logits = self.decode(z_fused)

        if self.training:
            device = logits.device
            pos_items = batch["pos_item_id"].to(device)
            neg_items = batch["neg_item_id"].to(device)

            pos_scores = torch.gather(logits, 1, pos_items.unsqueeze(-1))
            neg_scores = torch.gather(logits, 1, neg_items.unsqueeze(-1))

            return logits, pos_scores, neg_scores

        return (logits, z_fused, None) if return_code else logits


    def calculate_loss(self, batch):
        logits, pos_scores, neg_scores = self.forward(batch)

        user_ids = batch["user_ids"].to(self.device)
        pos_item_ids = batch["pos_item_id"].to(self.device)
        neg_item_ids = batch["neg_item_id"].to(self.device)

        pos_scores = torch.gather(logits, 1, pos_item_ids.unsqueeze(-1))
        neg_scores = torch.gather(logits, 1, neg_item_ids.unsqueeze(-1))

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        return loss, user_ids.size(0)

    @torch.no_grad()
    def predict_step(self, batch):
        """
        Extracts ratings_tgt and compares it against model predictions.
        """
        ratings_tgt = batch["ratings_tgt"].to(self.device)

        # This returns the reconstructed rating matrix
        y_hat = self(batch)

        mask = ratings_tgt != 0

        # Flatten for metric calculation (MSE/MAE)
        y_true_flat = ratings_tgt[mask]
        y_pred_flat = y_hat[mask]

        return y_true_flat, y_pred_flat


    def evaluate(self, test_loader, k=10):
        """
        Calculates HR@K, NDCG@K, and MRR for LOOCV.
        Assumes test_loader batch contains 'test_item_ids' (the positive)
        and 'negative_item_ids' (the 99 negatives).
        """
        self.eval()
        hr, ndcg, mrr = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                y_hat = self(batch)

                positive_items = batch["pos_item_id"].to(self.device)
                negative_items = batch["neg_item_id"].to(self.device)

                for i in range(y_hat.size(0)):
                    target_idx = positive_items[i].item()
                    neg_indices = negative_items[i].tolist()

                    item_indices = [target_idx] + list(neg_indices)
                    scores = y_hat[i, item_indices]

                    _, top_indices = torch.topk(scores, k=len(item_indices))
                    rank = (top_indices == 0).nonzero(as_tuple=True)[0].item() + 1

                    # Hit Ratio @ K
                    hr.append(1 if rank <= k else 0)

                    # NDCG @ K
                    if rank <= k:
                        ndcg.append(1 / np.log2(rank + 1))
                    else:
                        ndcg.append(0)

                    # MRR (Mean Reciprocal Rank)
                    mrr.append(1 / rank)

        return {'hit_rate': float(np.mean(hr)), 'ndcg': float(np.mean(ndcg)), 'mrr': float(np.mean(mrr))}