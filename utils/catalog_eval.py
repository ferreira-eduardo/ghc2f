"""Full-catalog evaluation of the GHC2F-family models under utils/irev_protocol.py.

Shared by grid_search/optimizer_hyper.py (validation) and run_experiment.py (val + test),
so tuning and the final runs score users the same way.
"""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def user_text_arrays(df_text, text_cols, max_reviews):
    """{userId: (n_reviews, text_dim)} with each user's last `max_reviews` reviews,
    the same text input RankingTrainDataset builds."""
    values = df_text[text_cols].to_numpy(dtype=np.float32)
    return {int(u): values[pos[-max_reviews:] if max_reviews else pos]
            for u, pos in df_text.groupby("userId").indices.items()}


def to_protocol_frame(df):
    return df.rename(columns={"userId": "user", "itemId": "item"})


class CatalogScorer:
    """score_fn(users) -> (b, N) scores against the protocol catalog, in catalog order.

    Each user's input is their row of `matrix` (train for val, train + val for test)
    plus their review texts; the model scores every item and the catalog columns are kept.
    """

    def __init__(self, model, matrix, user_text, text_dim, catalog_items, batch_size):
        self.model = model
        self.matrix = matrix
        self.user_text = user_text
        self.text_dim = text_dim
        self.catalog = torch.as_tensor(catalog_items, dtype=torch.long, device=model.device)
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, users):
        self.model.eval()
        out = []
        for start in range(0, len(users), self.batch_size):
            u = [int(x) for x in users[start:start + self.batch_size]]
            texts = [torch.from_numpy(self.user_text[x]) if x in self.user_text
                     else torch.zeros((1, self.text_dim)) for x in u]
            lengths = torch.tensor([len(t) for t in texts])
            user_text = pad_sequence(texts, batch_first=True)
            batch = {
                "user_ids": torch.tensor(u, dtype=torch.long),
                "ratings_in": torch.from_numpy(self.matrix[u].toarray()).float(),
                "user_text": user_text,
                "user_mask": torch.arange(user_text.size(1)).expand(len(u), -1) < lengths.unsqueeze(1),
                # forward() gathers pos/neg scores; unused here
                "pos_item_id": torch.zeros(len(u), dtype=torch.long),
                "neg_item_id": torch.zeros(len(u), dtype=torch.long),
            }
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            logits = self.model(batch).recon  # (b, total_items)
            out.append(logits.index_select(1, self.catalog))
        return torch.cat(out)
