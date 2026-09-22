from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as weight_init
from dataclasses import dataclass
from tqdm import tqdm

from model.metrics import popularity_topk_from_train, novelty_at_k, diversity_at_k, serendipity_at_k


@dataclass
class AEOutput:
    recon: torch.Tensor
    code: torch.Tensor


from utils.utils import MSEloss

_ACTIVATIONS = {
    "selu": F.selu,
    "relu": F.relu,
    "relu6": F.relu6,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "elu": F.elu,
    "lrelu": F.leaky_relu,
    "swish": lambda x: x * torch.sigmoid(x),
    "none": lambda x: x,
}


def activation(input: torch.Tensor, kind: str) -> torch.Tensor:
    try:
        return _ACTIVATIONS[kind](input)
    except KeyError:
        raise ValueError(f"Unknown non-linearity type: {kind}")


class CFAutoEncoder(nn.Module):

    def __init__(
            self,
            layer_sizes: List[int],
            nl_type: str = "selu",
            tied_weights: bool = True,
            learn_rate: float = 1e-4,
            dp_drop_prob: float = 0.0,
            last_layer_activations: bool = False,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "AE_BPR"
        self._nl_type = nl_type
        self._dp_drop_prob = dp_drop_prob
        self._last_layer_activations = last_layer_activations
        self.tied_weights = tied_weights
        self._last = len(layer_sizes) - 2

        ######## ENCODE ########
        self.encoder = nn.ModuleList()
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            lin = nn.Linear(in_dim, out_dim)
            weight_init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
            self.encoder.append(lin)

        # Dropout on code
        self.drop = nn.Dropout(p=self._dp_drop_prob) if self._dp_drop_prob > 0 else nn.Identity()

        ######## DECODE ########
        self.decoder = nn.ModuleList()
        if tied_weights:
            # Tied weights: reuse encoder weights (transposed) in decode.
            # Need only separate biases.
            reversed_enc_layers = list(reversed(layer_sizes))
            self.decode_b = nn.ParameterList()
            for in_dim, out_dim in zip(reversed_enc_layers[:-1], reversed_enc_layers[1:]):
                # out_dim = next in reversed list, which matches original input side
                b = nn.Parameter(torch.zeros(out_dim))
                self.decode_b.append(b)
            self.decoder = None
        else:
            # Unconstrained decoder with its own Linear layers
            reversed_enc_layers = list(reversed(layer_sizes))
            for in_dim, out_dim in zip(reversed_enc_layers[:-1], reversed_enc_layers[1:]):
                lin = nn.Linear(in_dim, out_dim)
                weight_init.xavier_uniform_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
                self.decoder.append(lin)
            self.decode_b = None

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)

    @property
    def code_dim(self) -> int:
        """Dimensionality of the latent representation (code)."""
        return self.encoder[-1].out_features

    @property
    def item_embeddings_(self) -> torch.Tensor:
        """
        (num_items, d) per-item vector, sourced from whichever weight matrix
        actually produces item scores in decode()'s last layer -- so it's
        the model's own notion of "item space", not a separately-trained
        embedding. Correct for BOTH tied_weights branches:
          - tied_weights=True: decode's last step reuses encoder[0].weight
            transposed (see decode()) -> item embedding = encoder[0].weight.t()
          - tied_weights=False: decode's last step is self.decoder[-1], whose
            weight is already (num_items, hidden) -> used directly.
        """
        if self.tied_weights:
            return self.encoder[0].weight.t().detach()  # (num_items, hidden1)
        else:
            return self.decoder[-1].weight.detach()  # (num_items, hidden1)


    def encode(self, x: torch.Tensor) -> torch.Tensor:

        for lin in self.encoder:
            x = activation(lin(x), self._nl_type)
        x = self.drop(x)
        return x

    def decode(self, z: torch.Tensor, item_indices=None) -> torch.Tensor:

        if self.tied_weights:
            out = z
            num_layers = len(self.encoder)
            for dec_idx in range(num_layers):
                enc_layer = self.encoder[num_layers - 1 - dec_idx]
                W, b = enc_layer.weight, self.decode_b[dec_idx]
                is_last = dec_idx == num_layers - 1

                if is_last and item_indices is not None:
                    W_sel = W.t()[item_indices]  # (B, K, hidden_in) — gather only needed columns
                    b_sel = b[item_indices]  # (B, K)
                    out = torch.einsum('bh,bkh->bk', out, W_sel) + b_sel
                else:
                    out = F.linear(out, W.t(), b)
                    if dec_idx != self._last or self._last_layer_activations:
                        out = activation(out, self._nl_type)
            return out
        else:
            out = z
            for dec_idx, lin in enumerate(self.decoder):
                is_last = dec_idx == len(self.decoder) - 1
                if is_last and item_indices is not None:
                    W_sel = lin.weight[item_indices]
                    b_sel = lin.bias[item_indices]
                    out = torch.einsum('bh,bkh->bk', out, W_sel) + b_sel
                else:
                    out = lin(out)
                    if dec_idx != self._last or self._last_layer_activations:
                        out = activation(out, self._nl_type)
            return out

    def forward(self, batch, item_indices=None):
        """
        batch : dict
            Ranking-style batch (see utils/dataset_utils.py: RankingTrainDataset /
            train_collate_fn / loocv_collate_fn) with a "ratings_in" tensor
            [B, num_items]. Shared with GatedHybridCFAutoEncoder/GHC2F so all
            three models can train through the same loaders in grid_search/optimizer_hyper.py.

        Returns
        -------
        AEOutput(recon, code)
        """
        ratings_in = batch["ratings_in"].to(self.device)
        code = self.encode(ratings_in)
        recon = self.decode(code, item_indices=item_indices)

        return AEOutput(recon=recon, code=code)

    def _pos_neg_scores(self, recon, batch):
        pos_ids = batch["pos_item_id"].to(self.device)
        neg_ids = batch["neg_item_id"].to(self.device)
        pos_scores = torch.gather(recon, 1, pos_ids.unsqueeze(-1)).squeeze(-1)
        neg_scores = torch.gather(recon, 1, neg_ids.unsqueeze(-1)).squeeze(-1)

        return pos_scores, neg_scores

    def calculate_loss(self, batch):
        out = self(batch)
        pos_scores, neg_scores = self._pos_neg_scores(out.recon, batch)
        loss_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        self.last_loss_components = {"bpr": loss_bpr.item()}
        return loss_bpr, batch["ratings_in"].size(0)

    def evaluate(self, test_loader, k=10, item_popularity: torch.Tensor = None,
                 k_pop: int = None, verbose_tqdm: bool = True):
        """
        item_popularity : optional (num_items,) tensor of TRAIN-set interaction
            counts per item. If given, also computes novelty/diversity/
            serendipity on a genuine full-catalog top-k (NOT the sampled
            hit_rate/ndcg/mrr candidate set -- see class docstring/comments).
            If omitted, those three keys are simply absent from the result
            (existing callers keep working unchanged).
        k_pop : size of the "obvious" popularity baseline used by
            serendipity. Defaults to k (same cutoff as the eval metric,
            for direct comparability) if item_popularity is given.
        """
        self.eval()
        hits_sum = ndcg_sum = mrr_sum = 0.0
        n = 0

        compute_beyond_acc = item_popularity is not None
        if compute_beyond_acc:
            item_popularity = item_popularity.to(self.device).float()
            k_pop = k_pop or k
            popularity_topk_ids = popularity_topk_from_train(item_popularity, k_pop)
            item_emb = self.item_embeddings_
            novelty_sum = diversity_sum = serendipity_sum = 0.0
            diversity_n = 0  # separate counter: diversity excludes NaN (k=1) users

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Evaluating", leave=not verbose_tqdm is False,
                              disable=not verbose_tqdm):
                batch = {k_: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k_, v in batch.items()}

                pos_items = batch["pos_item_id"].unsqueeze(-1)
                neg_items = batch["neg_item_id"]
                target_indices = torch.cat([pos_items, neg_items], dim=1)

                out = self(batch, item_indices=target_indices)
                test_scores = out.recon  # (B, K) -- sampled-candidate scores only

                pos_scores = test_scores[:, :1]
                ranks = (test_scores > pos_scores).sum(dim=1) + 1

                hits_sum += (ranks <= k).sum()
                ndcg_sum += torch.where(ranks <= k, 1.0 / torch.log2(ranks.float() + 1),
                                        torch.zeros_like(ranks, dtype=torch.float)).sum()
                mrr_sum += (1.0 / ranks.float()).sum()
                n += ranks.numel()

                if compute_beyond_acc:
                    # Full-catalog scoring pass (item_indices=None -> decode()
                    # returns (B, num_items)), separate from the sampled-
                    # candidate scores above. This is the only way to get a
                    # genuine top-k recommendation list -- novelty/diversity/
                    # serendipity computed on the tiny {1 pos + N neg} set
                    # used for hit_rate/ndcg/mrr would be meaningless.
                    full_out = self(batch, item_indices=None)
                    full_scores = full_out.recon  # (B, num_items)

                    # Mask out items already seen in train -- standard
                    # practice: don't "recommend" what the user already has.
                    seen_mask = batch["ratings_in"] > 0
                    full_scores = full_scores.masked_fill(seen_mask, float("-inf"))

                    topk_ids = torch.topk(full_scores, k, dim=1).indices  # (B, k)

                    # hit_mask for serendipity: does this genuine top-k contain
                    # the user's true positive? (independent check from the
                    # sampled-candidate hit_rate above -- different candidate
                    # pool, so don't reuse `ranks` here.)
                    pos_id = batch["pos_item_id"].unsqueeze(-1)  # (B, 1)
                    hit_mask = (topk_ids == pos_id)  # (B, k) bool

                    novelty_sum += novelty_at_k(topk_ids, item_popularity).sum().item()
                    div = diversity_at_k(topk_ids, item_emb)
                    valid_div = ~torch.isnan(div)
                    diversity_sum += div[valid_div].sum().item()
                    diversity_n += valid_div.sum().item()
                    serendipity_sum += serendipity_at_k(topk_ids, hit_mask, popularity_topk_ids).sum().item()

        result = {
            'hit_rate': (hits_sum.float() / n).item(),
            'ndcg': (ndcg_sum / n).item(),
            'mrr': (mrr_sum / n).item(),
        }

        if compute_beyond_acc:
            result['novelty'] = novelty_sum / n
            result['diversity'] = (diversity_sum / diversity_n) if diversity_n > 0 else float("nan")
            result['serendipity'] = serendipity_sum / n

        return result
