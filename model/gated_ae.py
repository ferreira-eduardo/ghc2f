import numpy as np
import torch
import torch.nn as nn
# from ae_utils import MSEloss
from utils.topic_att_profile import TopicProfile
from model.autoencoder import CFAutoEncoder
import torch.nn.functional as F


class GatedHybridCFAutoEncoder(CFAutoEncoder):
    def __init__(self, layer_sizes, num_users, num_items, topics_dim=15,
                 topics_latent_dim=64, **kwargs):
        super().__init__(layer_sizes, **kwargs)

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

        # Projection layers to align topic space with CF space (code_dim)
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

    def forward(self, batch):
        ratings_in = batch["ratings_in"].to(self.device)

        # collaborative signal
        z_cf = self.encode(ratings_in)

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

        pos_items = batch["pos_item_id"].to(self.device)
        neg_items = batch["neg_item_id"].to(self.device)

        if pos_items.dim() < logits.dim():
            pos_items = pos_items.unsqueeze(-1)
        if neg_items.dim() < logits.dim():
            neg_items = neg_items.unsqueeze(-1)

        pos_scores = torch.gather(logits, 1, pos_items)
        neg_scores = torch.gather(logits, 1, neg_items)

        return logits, z_fused, z_cf, pos_scores, neg_scores

    def calculate_loss(self, batch):
        _, _, _, pos_scores, neg_scores = self.forward(batch)

        user_ids = batch["user_ids"].to(self.device)

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
        self.eval()
        hr, ndcg, mrr = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                out = self(batch)
                logits = out[0] if isinstance(out, (tuple, list)) else out

                pos_items = batch["pos_item_id"].to(self.device)
                neg_items = batch["neg_item_id"].to(self.device)

                target_items = torch.cat([pos_items.unsqueeze(-1), neg_items], dim=1)

                scores = torch.gather(logits, 1, target_items)  # [Batch, 100]

                _, indices = torch.sort(scores, descending=True, dim=1)

                ranks = (indices == 0).nonzero(as_tuple=True)[1] + 1
                ranks = ranks.cpu().numpy()

                for rank in ranks:
                    hr.append(1 if rank <= k else 0)
                    if rank <= k:
                        ndcg.append(1 / np.log2(rank + 1))
                    else:
                        ndcg.append(0)
                    mrr.append(1 / rank)

        return {
            'hit_rate': float(np.mean(hr)),
            'ndcg': float(np.mean(ndcg)),
            'mrr': float(np.mean(mrr))
        }
