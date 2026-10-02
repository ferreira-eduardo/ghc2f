from typing import Iterable

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


class TextEncoder(nn.Module):
    def __init__(
            self,
            model_name: str = "intfloat/e5-base-v2",
            revision: str = 'main',
            pooling: str = 'cls',
            prefix: str = 'query',
            max_length: int = 256,
            chunk_size: int = 256
    ):
        super().__init__()
        self.cache: dict[str, Tensor] = {}
        self.model_name = model_name
        self.revision = revision
        self.prefix = prefix
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision)
        self.n_truncated = 0
        self.model.requires_grad_(False)

        self.model.eval()

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True) -> "TextEncoder":
        # model.train() no ReviewAware propagaria para cá e ligaria o dropout,
        # gerando vetores diferentes para o mesmo texto. Congelado = sempre eval.
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def warm_up(self, texts: Iterable[str]) -> None:
        """Codifica e guarda no cache os textos que ainda não estão nele."""
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if any(not t.strip() for t in missing):
            raise ValueError("textos vazios devem ser filtrados antes do encoder")

        for start in range(0, len(missing), self.chunk_size):
            chunk = missing[start: start + self.chunk_size]
            vectors = self._encode(chunk).to("cpu", torch.float16)
            self.cache.update(zip(chunk, vectors))

    def forward(self, texts: list[str]) -> Tensor:
        self.warm_up(texts)
        return torch.stack([self.cache[t] for t in texts]).to(self.device, torch.float32)

    def _encode(self, texts: list[str]) -> Tensor:
        tokens = self.tokenizer(
            [self.prefix + t for t in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        lengths = tokens["attention_mask"].sum(dim=1)
        self.n_truncated += int((lengths == self.max_length).sum())

        hidden = self.model(**tokens).last_hidden_state  # (N, T, D)
        return F.normalize(self._pool(hidden, tokens["attention_mask"]), dim=-1)

    def _pool(self, hidden: Tensor, mask: Tensor) -> Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = mask.unsqueeze(-1).to(hidden.dtype)  # ignora tokens de padding na média
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


if __name__ == "__main__":
    """ Class test """
    text_test = TextEncoder()

    text_test.train()

    sentences = [
        "I really like this guitar",  # 0: referência
        "Rich warm tone",  # 1: paráfrase (mesmo sentido)
        "Solid build quality",  # 2: negação (sentido oposto)
    ]

    tok = text_test(sentences)

    print('Similarity')

    print(cosine_similarity(tok).round(3))



