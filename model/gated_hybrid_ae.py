import torch
import torch.nn as nn

from utils.text_att_profile import TextProfile
from model.cf_autoencoder import CFAutoEncoder, AEOutput
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import List


@dataclass
class GatedAEOutput(AEOutput):
    z_fused: torch.Tensor = None
    z_cf: torch.Tensor = None
    z_text: torch.Tensor = None
    gate_values: List[torch.Tensor] = field(default_factory=list)
    pos_scores: torch.Tensor = None
    neg_scores: torch.Tensor = None


class GatedHybridCFAutoEncoder(CFAutoEncoder):
    def __init__(self, layer_sizes, num_users, num_items, text_dim=15,
                 text_latent_dim=64, text_gamma=0.5, fusion_mode="film", **kwargs):

        super().__init__(layer_sizes, **kwargs)
        self.name = 'GatedHybrid'

        self.fusion_mode = fusion_mode

        enc_dims = [out_dim for _, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])]

        self.text_gamma = text_gamma

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

        if fusion_mode in ("convex", "additive"):
            self.gate_enc = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d * 2, d),
                    nn.Sigmoid()
                ) for d in enc_dims
            ])
        else:  # film -- built only when selected, so convex/additive don't
            # pay for unused parameters
            self.film_scale = nn.ModuleList([nn.Linear(d, d) for d in enc_dims])
            self.film_shift = nn.ModuleList([nn.Linear(d, d) for d in enc_dims])

        self.fuse_all_layers = True
        self.enc_norms = nn.ModuleList([nn.LayerNorm(d) for d in enc_dims])

        # User profiler (Interactions in trainset)
        self.user_profiler = TextProfile(num_users, text_dim, text_latent_dim)

        # Item profiler (Global view - reviews independent of the current user)
        self.item_profiler = TextProfile(num_items, text_dim, text_latent_dim)

        # Projection layers to align text space with CF space (code_dim)
        self.user_proj = nn.Linear(text_dim, self.code_dim_cf)
        self.item_proj = nn.Linear(text_dim, self.code_dim_cf)

        self.register_buffer("item_corpus_ids", None)
        self.register_buffer("item_corpus_text", None)
        self.register_buffer("item_corpus_mask", None)

        self.gate_drop = nn.Dropout(p=0.2)

        learn_rate = self.optimizer.param_groups[0]['lr']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

    def encode_with_text(self, x, z_text_base):
        """
        x: (B, num_items)
        z_text_base: (B, code_dim_cf)

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
                t = self.text_to_enc[li](z_text_base)  # (B, out_dim)

                if self.fusion_mode == "convex":
                    h_d = self.gate_drop(h)
                    t_d = self.gate_drop(t)
                    g = self.gate_enc[li](torch.cat([h_d, t_d], dim=-1))
                    h = g * h + (1.0 - g) * t
                    gate_values.append(g)

                elif self.fusion_mode == "additive":
                    h_d = self.gate_drop(h)
                    t_d = self.gate_drop(t)
                    g = self.gate_enc[li](torch.cat([h_d, t_d], dim=-1))
                    h = h + g * t  # only added to
                    gate_values.append(g)
                else: #film
                    t_d = self.gate_drop(t)
                    gamma = self.film_scale[li](t_d)
                    beta = self.film_shift[li](t_d)
                    h = h * (1 + gamma) + beta
                    gate_values.append(gamma)

                h = self.enc_norms[li](h)

        h = self.drop(h)

        return h, gate_values

    def set_item_corpus(self, item_ids: torch.Tensor, item_text: torch.Tensor, item_mask: torch.Tensor):
        """
        Call once after construction. Stores the RAW item corpus
        """
        self.item_corpus_ids = item_ids.to(self.device)
        self.item_corpus_text = item_text.to(self.device)
        self.item_corpus_mask = item_mask.to(self.device)


    def forward(self, batch) -> GatedAEOutput:
        ratings_in = batch["ratings_in"].to(self.device)

        # collaborative signal
        z_cf = self.encode(ratings_in)

        u_ids = batch["user_ids"].to(self.device)
        u_text = batch["user_text"].to(self.device)
        u_mask = batch["user_mask"].to(self.device)

        z_user_text = self.user_profiler(u_ids, u_text, u_mask)
        text_user = self.user_proj(z_user_text)

        item_profiles = self.item_profiler( # (num_corpus_items, text_dim)
            self.item_corpus_ids, self.item_corpus_text, self.item_corpus_mask
        )

        item_global = torch.zeros((ratings_in.size(1), item_profiles.size(1)), device=self.device)
        item_global = item_global.index_copy(0, self.item_corpus_ids, item_profiles)

        hist_mask = (ratings_in != 0).float()

        interaction_counts = hist_mask.sum(dim=1, keepdim=True)
        text_item_global = (hist_mask @ item_global) / interaction_counts.clamp_min(1.0)
        text_item = self.item_proj(text_item_global)


        z_text = self.text_gamma * text_user + (1 - self.text_gamma) * text_item

        z_fused, gate_values = self.encode_with_text(ratings_in, z_text)

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
            z_text=z_text,
            gate_values=gate_values,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
        )

    def calculate_loss(self, batch):
        out = self.forward(batch)

        user_ids = batch["user_ids"].to(self.device)

        loss = -torch.log(torch.sigmoid(out.pos_scores - out.neg_scores) + 1e-10).mean()

        self.last_loss_components = {"bpr": loss.item()}

        return loss, user_ids.size(0)


    @torch.no_grad()
    def gate_diagnostics(self, batch):
        """
        Cheap, no-retrain diagnostic: mean/std of each layer's gate g_l on a
        single forward pass. Values clustering near 1 across a batch
        indicate the gate is suppressing the text signal (h_fused ~= h_cf);
        values near 0 indicate the opposite. Call with any val/train batch.
        Mirrors GHC2F.gate_diagnostics so both classes report identically.
        """
        self.eval()
        batch = {k_: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k_, v in batch.items()}
        out = self.forward(batch)
        rows = []
        for li, g in enumerate(out.gate_values):
            rows.append({
                "layer": li,
                "mean": g.mean().item(),
                "std": g.std().item(),
                "frac_above_0.9": (g > 0.9).float().mean().item(),
                "frac_below_0.1": (g < 0.1).float().mean().item(),
            })
        return rows