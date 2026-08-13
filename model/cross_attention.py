import torch
import torch.nn as nn


class AspectCrossAttention(nn.Module):
    def __init__(self, d_cf, d_text=768, n_aspects=11, n_heads=4):
        super().__init__()
        self.d_cf = d_cf
        self.n_aspects = n_aspects

        # Representação aprendível dos 11 aspectos
        # Inicializados aleatoriamente, mas serão otimizados pelo BPR
        self.aspect_queries = nn.Parameter(torch.randn(n_aspects, d_cf))

        # Multi-Head Attention para capturar nuances diferentes em cada aspecto
        self.mha = nn.MultiheadAttention(embed_dim=d_cf, num_heads=n_heads, batch_first=True)

        # Projeções para alinhar o texto das reviews ao espaço do Autoencoder
        self.key_proj = nn.Linear(d_text, d_cf)
        self.value_proj = nn.Linear(d_text, d_cf)
        self.user_proj = nn.Linear(d_text, d_cf)

        self.norm = nn.LayerNorm(d_cf)

    def forward(self, text_seq, user_history_text):
        batch_size = text_seq.size(0)

        u_hist = self.user_proj(user_history_text)

        queries = self.aspect_queries.unsqueeze(0).expand(batch_size, -1, -1) + u_hist

        # Projeta o texto
        keys = self.key_proj(text_seq)
        values = self.value_proj(text_seq)


        # context: [B, 11, d_cf]
        context, _ = self.mha(queries, keys, values)
        z_semantic = context.mean(dim=1)

        return z_semantic