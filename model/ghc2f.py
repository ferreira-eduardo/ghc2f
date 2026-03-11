import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.autoencoder import CFAutoEncoder
from utils.topic_att_profile import TopicProfile


class GHC2F(CFAutoEncoder):
    def __init__(self, layer_sizes, num_users, num_items, topics_dim=15,
                 topics_latent_dim=64, **ae_kwargs):

        super().__init__(layer_sizes, **ae_kwargs)

        # Architecture Dimensions
        self.item_input_dim = self.encoder[0].in_features
        self.code_dim_cf = layer_sizes[-1]
        self.bottleneck_dim = self.code_dim_cf
        enc_dims = [out_dim for _, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])]

        # Topic Profilers
        self.user_profiler = TopicProfile(num_users, topics_dim, topics_latent_dim)
        self.item_profiler = TopicProfile(num_items, topics_dim, topics_latent_dim)
        self.register_buffer("item_global_profiles", None)

        # Projection & Gating for Layer-wise Fusion
        self.topic_to_enc = nn.ModuleList([nn.Linear(self.code_dim_cf, d) for d in enc_dims])
        self.gate_enc = nn.ModuleList([
            nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid()) for d in enc_dims
        ])
        self.enc_norms = nn.ModuleList([nn.LayerNorm(d) for d in enc_dims])
        self.fuse_all_layers = True
        self.gate_drop = nn.Dropout(p=0.2)

        # Global Alignment Projections
        self.user_proj = nn.Linear(topics_dim, self.code_dim_cf)
        self.item_proj = nn.Linear(topics_dim, self.code_dim_cf)

        # Item Embedding Alignment (for BPR ranking)
        if self.item_input_dim != self.bottleneck_dim:
            self.item_aligner = nn.Linear(self.item_input_dim, self.bottleneck_dim)
        else:
            self.item_aligner = nn.Identity()

    def encode_with_topics(self, x, z_topic_base):
        """Iterative encoding with gated topic fusion at each layer."""
        h = x
        last_idx = len(self.encoder) - 1

        for li, lin in enumerate(self.encoder):
            h = lin(h)
            if li != last_idx:
                h = F.relu(h)

            if self.fuse_all_layers or (li == last_idx):
                t = self.topic_to_enc[li](z_topic_base)
                g = self.gate_enc[li](torch.cat([h, t], dim=-1))
                g = self.gate_drop(g)
                h = g * h + (1.0 - g) * t
                h = self.enc_norms[li](h)

        return self.drop(h)

    def get_item_embeddings(self):
        raw_weights = self.encoder[0].weight  # Num_items, first_hidden]
        return self.item_aligner(raw_weights)

    def forward(self, batch, return_code=True):
        ratings_in = batch["ratings_in"].to(self.device)
        u_ids = batch["user_ids"].to(self.device)
        u_topics = batch["user_topics"].to(self.device)
        u_mask = batch["user_mask"].to(self.device)

        # Generate Topic Base (Fusion of User and Item Global Context)
        z_user_topic = self.user_profiler(u_ids, u_topics, u_mask)
        topic_user = self.user_proj(z_user_topic)

        hist_mask = (ratings_in != 0).float()
        interaction_counts = hist_mask.sum(dim=1, keepdim=True)
        topic_item_global = (hist_mask @ self.item_global_profiles) / interaction_counts.clamp_min(1.0)
        topic_item = self.item_proj(topic_item_global)

        z_topic_base = 0.5 * (topic_user + topic_item)

        # Encode with Gated Fusion
        z_fused = self.encode_with_topics(ratings_in, z_topic_base)

        #(Optional) Get Raw AE code for Contrastive Loss
        z_cf_raw = self.encode(ratings_in)

        # Decode to Rating Space
        logits = self.decode(z_fused)

        return logits, z_cf_raw, z_fused

    def calculate_loss(self, batch, reg_weight=1e-5, cl_weight=0.1):
        """BPR Loss + Contrastive Loss + L2 Regularization."""
        logits, z_cf_raw, z_fused = self.forward(batch)

        # BPR Loss Logic
        item_embs = self.get_item_embeddings()
        pos_item_ids = batch["pos_item_id"].to(self.device)
        neg_item_ids = batch["neg_item_id"].to(self.device)

        w_pos = item_embs[pos_item_ids]
        w_neg = item_embs[neg_item_ids]

        pos_scores = (z_fused * w_pos).sum(dim=-1)
        neg_scores = (z_fused * w_neg).sum(dim=-1)

        loss_bpr = F.softplus(neg_scores - pos_scores).mean()

        # Contrastive Loss
        loss_cl = self.contrastive_loss(z_fused, z_cf_raw)

        # Regularization
        reg_loss = (torch.norm(w_pos) ** 2 + torch.norm(w_neg) ** 2 + torch.norm(batch["ratings_in"]) ** 2)

        total_loss = loss_bpr + (cl_weight * loss_cl) + (reg_weight * reg_loss)
        return total_loss, batch["user_ids"].size(0)

    def contrastive_loss(self, z1, z2, temperature=0.1):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.matmul(z1, z2.t()) / temperature
        labels = torch.arange(z1.size(0)).to(self.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def predict_unseen(self, batch):
        self.eval()
        _, _, z_fused = self.forward(batch)
        item_embeddings = self.get_item_embeddings()

        all_scores = torch.matmul(z_fused, item_embeddings.t())

        ratings_in = batch["ratings_in"].to(self.device)

        if ratings_in.shape[1] != all_scores.shape[1]:
            mask = ratings_in > 0
            all_scores[:, :ratings_in.shape[1]][mask] = -1e9
        else:
            all_scores[ratings_in > 0] = -1e9

        return all_scores

    @torch.no_grad()
    def predict_step(self, batch):
        all_scores = self.predict_unseen(batch)
        target_item = batch["pos_item_id"].to(self.device)
        return all_scores, target_item


    def evaluate(self, loader, top_k=10):
        self.eval()
        hr, ndcg, mrr = [], [], []

        with torch.no_grad():
            for batch in loader:
                # Get scores for all items and the target item index
                all_scores, target_items = self.predict_step(batch)

                # Find the rank of the target item among all scores
                _, top_indices = torch.topk(all_scores, k=top_k, dim=1)

                top_indices = top_indices.cpu().numpy()
                target_items = target_items.cpu().numpy()

                # Check if the target item is in the top K
                for i in range(len(target_items)):
                    target = target_items[i]
                    indices = top_indices[i]

                    if target in indices:
                        # Hit Rate
                        hr.append(1.0)

                        # Find the exact rank (0-indexed, so we add 1)
                        rank = np.where(indices == target)[0][0] + 1

                        # NDCG
                        ndcg.append(1 / math.log2(rank + 1))

                        # MRR
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
