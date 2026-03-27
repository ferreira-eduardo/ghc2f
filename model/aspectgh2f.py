import torch
from model.ghc2f import GHC2F
from model.cross_attention import AspectCrossAttention
import torch.nn.functional as F

class AspectGHC2F(GHC2F):
    def __init__(self, *ae_args, **ae_kwargs):
        super().__init__(*ae_args, **ae_kwargs)

        self.aspect_attention = AspectCrossAttention(
            d_cf=self.bottleneck_dim,
            d_text=ae_kwargs.get('d_text', 768)
        )

    def forward(self, batch, return_code=False):
        z_ae = self.encode(batch["ratings_in"])
        z_semantic, _ = self.aspect_attention(batch["text_seq"], batch.get("text_mask"))

        z_final = self.gate_fusion(z_ae, z_semantic)

        if return_code:
            return None, z_final, None, z_ae, z_semantic
        return z_final

    def forward_bpr(self, batch):
        # Obtemos a representação fundida do usuário
        _, z_user, _, z_ae, _ = self.forward(batch, return_code=True)

        item_embeddings = self.get_item_embeddings()

        # Scores para o item Positivo
        w_pos = item_embeddings[batch["pos_item_id"]]
        # Atenção específica para o texto do item positivo
        z_sem_pos, _ = self.aspect_attention(batch["pos_text_seq"], batch.get("pos_mask"))
        pos_scores = ((z_user + z_sem_pos) * w_pos).sum(dim=-1)

        # Scores para o item Negativo
        w_neg = item_embeddings[batch["neg_item_id"]]
        # Atenção específica para o texto do item negativo
        z_sem_neg, _ = self.aspect_attention(batch["neg_text_seq"], batch.get("neg_mask"))
        neg_scores = ((z_user + z_sem_neg) * w_neg).sum(dim=-1)

        return pos_scores, neg_scores

    def calculate_loss(self, batch, reg_weight=1e-5, cl_weight=0.1, aspect_cl_weight=0.05):
        # 1. BPR Loss com os scores refinados pela atenção
        pos_scores, neg_scores = self.forward_bpr(batch)
        loss_bpr = F.softplus(neg_scores - pos_scores).mean()

        # 2. Contrastive Loss original (z_cf vs z_fused)
        # E uma nova Contrastive Loss (z_ae vs z_semantic)
        _, z_fused, _, z_ae, z_semantic = self.forward(batch, return_code=True)

        loss_cl_orig = self.contrastive_loss(z_ae, z_fused)
        loss_cl_aspect = self.contrastive_loss(z_ae, z_semantic)

        # 3. Regularização L2 (mantendo sua lógica original)
        item_embeddings = self.get_item_embeddings()
        reg_loss = (torch.norm(item_embeddings[batch["pos_item_id"]]) ** 2 +
                    torch.norm(batch["ratings_in"]) ** 2)

        total_loss = loss_bpr + (cl_weight * loss_cl_orig) + \
                     (aspect_cl_weight * loss_cl_aspect) + (reg_weight * reg_loss)

        return total_loss, batch["user_ids"].size(0)