import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from model.gated_hybrid_ae import GatedHybridCFAutoEncoder



class GHC2F(GatedHybridCFAutoEncoder):
    def __init__(
        self,
        *ae_args,
        contrastive_target: str = "text",
        stop_grad_cf: bool = True,
        cl_weight: float = 0.1,
        reg_weight: float = 1e-5,
        gate_entropy_weight: float = 0.0,
        **ae_kwargs,
    ):
        super().__init__(*ae_args, **ae_kwargs)
        self.item_input_dim = self.encoder[0].out_features
        self.name = 'GHC2F'

        self.bottleneck_dim = ae_kwargs['layer_sizes'][-1]


        if self.item_input_dim != self.bottleneck_dim:
            self.item_aligner = nn.Linear(self.item_input_dim, self.bottleneck_dim)
        else:
            self.item_aligner = nn.Identity()

        if contrastive_target not in ("text", "collaborative"):
            raise ValueError(f"Unknown contrastive_target: {contrastive_target!r}")

        self.contrastive_target = contrastive_target
        self.stop_grad_cf = stop_grad_cf
        self.cl_weight = cl_weight
        self.reg_weight = reg_weight
        self.gate_entropy_weight = gate_entropy_weight

        learn_rate = self.optimizer.param_groups[0]['lr']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

    def get_item_embeddings(self):
        raw_weights = self.encoder[0].weight.t()
        return self.item_aligner(raw_weights)

    def _score_items(self, z_fused, item_ids):
        """
        item_ids: (B,)   -> one item per row (e.g. a single pos/neg during
                             training). Returns (B,).
        item_ids: (B, K) -> K candidate items per row (e.g. evaluate()'s
                             [pos, neg_1..neg_99]). Returns (B, K).
        """
        logits = self.decode(z_fused)  # (B, num_items)
        if item_ids.dim() == 1:
            return torch.gather(logits, 1, item_ids.unsqueeze(-1)).squeeze(-1)
        return torch.gather(logits, 1, item_ids)



    def contrastive_loss(self, z_fused, target, temperature=0.1):
        """
        Standard InfoNCE loss to align z_fused with `target`.

        target is either:
          - z_text (T), the projected semantic signal — pulls z_fused toward
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

        loss_bpr = F.softplus(out.neg_scores.squeeze(-1) - out.pos_scores.squeeze(-1)).mean()

        loss_cl = self.contrastive_loss(out.z_fused, out.z_text)

        # --- gate entropy regularizer ---
        loss_gate = self.gate_entropy_loss(out.gate_values)

        # --- L2 regularization  ---
        item_embeddings = self.get_item_embeddings()
        reg_loss = (torch.norm(item_embeddings[batch["pos_item_id"]]) ** 2 +
                    torch.norm(item_embeddings[batch["neg_item_id"]]) ** 2 )

        total_loss = (
            loss_bpr
            + self.cl_weight * loss_cl
            + self.reg_weight * reg_loss
            + self.gate_entropy_weight * loss_gate
        )

        self.last_loss_components = {
            "bpr": loss_bpr.item(),
            "cl": loss_cl.item(),
            "gate": loss_gate.item() if torch.is_tensor(loss_gate) else float(loss_gate),
            "reg": reg_loss.item() if torch.is_tensor(reg_loss) else float(reg_loss),
        }

        return total_loss, batch["user_ids"].size(0)


    def evaluate(self, test_loader, k=10):
        self.eval()
        hr_list, ndcg_list, mrr_list = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k_: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k_, v in batch.items()}

                out = self.forward(batch)

                pos_items = batch["pos_item_id"].unsqueeze(-1)
                neg_items = batch["neg_item_id"]
                target_indices = torch.cat([pos_items, neg_items], dim=1)  # (B, 1+num_neg)

                test_scores = self._score_items(out.z_fused, target_indices)
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
            'mrr': float(np.mean(mrr_list)),
        }