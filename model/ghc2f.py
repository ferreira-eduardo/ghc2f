import torch
import torch.nn.functional as F
from torch import nn
from model.gated_hybrid_ae import GatedHybridCFAutoEncoder
from utils.utils import MSEloss


class GHC2F(GatedHybridCFAutoEncoder):
    def __init__(
        self,
        *ae_args,
        contrastive_target: str = "text",   # "text" (fixed) | "collaborative" (legacy, for ablation)
        stop_grad_cf: bool = True,          # only matters when contrastive_target == "collaborative"
        dual_head_decoder: bool = True,
        cl_weight: float = 0.1,
        reg_weight: float = 1e-5,
        mmse_weight: float = 1.0,      
        mmse_weight_cf : float = 1.0,
        gate_entropy_weight: float = 0.0,   # new: 0.0 = off by default, sweep to enable
        **ae_kwargs,
    ):
        super().__init__(*ae_args, **ae_kwargs)
        self.item_input_dim = self.encoder[0].out_features
        self.name = 'GHC2F'

        self.bottleneck_dim = ae_kwargs['layer_sizes'][-1]

        self.item_projection = nn.Linear(self.item_input_dim, self.bottleneck_dim).to(self.device)

        if self.item_input_dim != self.bottleneck_dim:
            self.item_aligner = nn.Linear(self.item_input_dim, self.bottleneck_dim)
        else:
            self.item_aligner = nn.Identity()

        assert contrastive_target in ("text", "collaborative"), \
            f"contrastive_target must be 'text' or 'collaborative', got {contrastive_target}"
        self.contrastive_target = contrastive_target
        self.stop_grad_cf = stop_grad_cf
        self.dual_head_decoder = dual_head_decoder
        self.cl_weight = cl_weight
        self.reg_weight = reg_weight
        self.mmse_weight = mmse_weight
        self.mmse_weight_cf = mmse_weight_cf
        self.gate_entropy_weight = gate_entropy_weight

    def get_item_embeddings(self):
        raw_weights = self.encoder[0].weight.t()
        return self.item_aligner(raw_weights)

    def forward_bpr(self, batch):
        out = self.forward(batch)  # was: z_fused, _, _, _ = self.forward(batch)

        raw_item_embeddings = self.get_item_embeddings()

        pos_item_ids = batch["pos_item_id"]
        neg_item_ids = batch["neg_item_id"]

        w_pos = raw_item_embeddings[pos_item_ids]
        w_neg = raw_item_embeddings[neg_item_ids]

        pos_scores = (out.z_fused * w_pos).sum(dim=-1)
        neg_scores = (out.z_fused * w_neg).sum(dim=-1)

        return pos_scores, neg_scores

    def contrastive_loss(self, z_fused, target, temperature=0.1):
        """
        Standard InfoNCE loss to align z_fused with `target`.

        target is either:
          - z_topic (T), the projected semantic signal — pulls z_fused toward
            incorporating text, the fix for this ablation; or
          - z_cf, the raw collaborative code — the ORIGINAL, buggy alignment.
            Kept only for contrastive_target="collaborative" ablation runs,
            where it should be paired with stop_grad_cf=True so L_CL cannot
            push the gate toward ignoring text (see calculate_loss).
        """
        z_fused = F.normalize(z_fused, dim=-1)
        target = F.normalize(target, dim=-1)

        logits = torch.matmul(z_fused, target.t()) / temperature
        labels = torch.arange(z_fused.size(0)).to(self.device)

        return F.cross_entropy(logits, labels)

    def gate_entropy_loss(self, gate_values):
        """
        Binary entropy of each gate g_l, averaged over dims/layers/batch.
        Returns the NEGATIVE mean entropy — minimizing this term in the total
        loss maximizes entropy, i.e. discourages the gate from saturating at
        trivial all-0 / all-1 solutions.
        """
        if not gate_values:
            return torch.zeros((), device=self.device)

        entropies = []
        for g in gate_values:
            g = g.clamp(1e-6, 1 - 1e-6)
            h = -(g * torch.log(g) + (1 - g) * torch.log(1 - g))
            entropies.append(h.mean())

        return -torch.stack(entropies).mean()

    def calculate_loss(self, batch):
        out = self.forward(batch)

        loss_bpr = F.softplus(out.neg_scores - out.pos_scores).mean()

        # --- contrastive alignment ---
        if self.contrastive_target == "text":
            cl_target = out.z_topic
        else:
            cl_target = out.z_cf.detach() if self.stop_grad_cf else out.z_cf

        loss_cl = self.contrastive_loss(out.z_fused, cl_target)

        # --- reconstruction (dual-head MMSE) ---
        loss_mmse_fused, _ = MSEloss(out.recon, batch["ratings_in"], size_average=True)

        if self.dual_head_decoder:
            recon_cf = self.decode(out.z_cf)
            loss_mmse_cf, _ = MSEloss(recon_cf, batch["ratings_in"], size_average=True)
            loss_mmse =  (self.mmse_weight_fused * loss_mmse_fused
                         + self.mmse_weight_cf * loss_mmse_cf)

        else:
            loss_mmse = loss_mmse_fused

        # --- gate entropy regularizer ---
        loss_gate = self.gate_entropy_loss(out.gate_values)

        # --- L2 regularization (unchanged) ---
        item_embeddings = self.get_item_embeddings()
        reg_loss = (torch.norm(item_embeddings[batch["pos_item_id"]]) ** 2 +
                    torch.norm(item_embeddings[batch["neg_item_id"]]) ** 2 +
                    torch.norm(batch["ratings_in"]) ** 2)

        total_loss = (
            loss_bpr
            + self.cl_weight * loss_cl
            + self.reg_weight * reg_loss
            + self.mmse_weight * loss_mmse
            + self.gate_entropy_weight * loss_gate
        )

        return total_loss, batch["user_ids"].size(0)

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
        out = self.forward(batch)  # was: _, z_fused, _ = self.forward(batch)

        item_embeddings = self.get_item_embeddings()
        all_scores = torch.matmul(out.z_fused, item_embeddings.t())

        ratings_in = batch["ratings_in"]
        all_scores[ratings_in > 0] = -1e9

        return all_scores