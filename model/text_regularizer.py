import torch
from torch import nn, optim

from model.cf_autoencoder import CFAutoEncoder


class TextRegCFAutoEncoder(CFAutoEncoder):
    def __init__(self, layer_sizes, user_text, item_text,
                 lambda_u=0.1, lambda_i=0.1, learn_rate=1e-4,
                 user_has_text=None, item_has_text=None, **kw):
        super().__init__(layer_sizes, learn_rate=learn_rate, **kw)
        self.name = "AE_BPR_TextReg"

        # perfis fixos (construídos só com reviews de treino), shape (N, K)
        self.register_buffer("user_text", user_text.float())
        self.register_buffer("item_text", item_text.float())
        # máscaras explícitas: perfis de aspectos de entidades sem reviews não são
        # zero (s_bar = prior global), então abs().sum() > 0 não serve para eles

        if user_has_text is None:
            user_has_text = user_text.abs().sum(1) > 0
        if item_has_text is None:
            item_has_text = item_text.abs().sum(1) > 0
        self.register_buffer("user_has_text", user_has_text.float())
        self.register_buffer("item_has_text", item_has_text.float())

        K = user_text.size(1)
        self.P = nn.Linear(K, self.code_dim)       # texto -> espaço do código
        self.Q = nn.Linear(K, layer_sizes[1])      # texto -> espaço dos embeddings de item
        self.lambda_u, self.lambda_i = lambda_u, lambda_i

        # o pai criou o otimizador antes de P e Q existirem: recriar
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)

    def item_embeddings(self, idx):
        W = self.encoder[0].weight.t() if self.tied_weights else self.decoder[-1].weight
        return W[idx]                               # (n, hidden_1)

    @staticmethod
    def _masked_mse(a, b, mask):
        d = ((a - b) ** 2).mean(dim=1)              # média nas dimensões: λ independe da largura
        return (d * mask).sum() / mask.sum().clamp(min=1)

    def calculate_loss(self, batch):
        out = self(batch)
        pos_scores, neg_scores = self._pos_neg_scores(out.recon, batch)
        loss_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        users = batch["user_ids"].to(self.device)
        reg_u = self._masked_mse(out.code, self.P(self.user_text[users]),
                                 self.user_has_text[users])

        items = torch.unique(torch.cat([batch["pos_item_id"], batch["neg_item_id"]])).to(self.device)
        reg_i = self._masked_mse(self.item_embeddings(items), self.Q(self.item_text[items]),
                                 self.item_has_text[items])

        loss = loss_bpr + self.lambda_u * reg_u + self.lambda_i * reg_i
        self.last_loss_components = {"bpr": loss_bpr.item(),
                                     "reg_u": reg_u.item(), "reg_i": reg_i.item()}
        return loss, batch["ratings_in"].size(0)