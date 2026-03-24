import torch
import torch.nn as nn
import torch.nn.functional as F


class AspectCrossAttention(nn.Module):
    def __init__(self, d_cf, d_text=768, n_aspects=11, n_heads=4):
        super().__init__()
        self.d_cf = d_cf
        self.n_aspects = n_aspects

        # Representação aprendível dos 11 aspectos
        # Inicializados aleatoriamente, mas serão otimizados pelo BPR
        self.aspect_queries = nn.Parameter(torch.randn(n_aspects, d_cf))

        # Projeções para alinhar o texto das reviews ao espaço do Autoencoder
        self.key_proj = nn.Linear(d_text, d_cf)
        self.value_proj = nn.Linear(d_text, d_cf)

        # Multi-Head Attention para capturar nuances diferentes em cada aspecto
        self.mha = nn.MultiheadAttention(embed_dim=d_cf, num_heads=n_heads, batch_first=True)

        # Fusão dos 11 sinais em um único vetor semântico
        self.aspect_fusion = nn.Linear(n_aspects * d_cf, d_cf)
        self.norm = nn.LayerNorm(d_cf)

    def forward(self, text_seq, mask=None):
        """
        text_seq: [B, K, d_text] (Embeddings das reviews)
        mask: [B, K] (True para tokens válidos)
        """
        batch_size = text_seq.size(0)

        # Prepara as queries para o batch: [B, 11, d_cf]
        queries = self.aspect_queries.unsqueeze(0).repeat(batch_size, 1, 1)

        # Projeta o texto
        keys = self.key_proj(text_seq)
        values = self.value_proj(text_seq)

        # Cross-Attention: Aspectos (Q) buscam no Texto (K, V)
        # Invertemos a máscara para o padrão do PyTorch (True = ignorar)
        attn_mask = ~mask if mask is not None else None

        # context: [B, 11, d_cf]
        context, attn_weights = self.mha(queries, keys, values, key_padding_mask=attn_mask)

        # Concatena os 11 aspectos e projeta para a dimensão do bottleneck
        combined = context.reshape(batch_size, -1)
        semantic_out = self.aspect_fusion(combined)

        return self.norm(semantic_out), attn_weights