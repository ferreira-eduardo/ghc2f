import torch
import torch.nn.functional as F
NOVELTY_EPS = 1e-10



def popularity_topk_from_train(item_popularity: torch.Tensor, k_pop: int) -> torch.Tensor:
    """The k_pop most popular items in the train set -- the 'obvious baseline' for serendipity."""
    return torch.topk(item_popularity, k_pop).indices



def novelty_at_k(topk_item_ids: torch.Tensor, item_popularity: torch.Tensor,
                 eps: float = NOVELTY_EPS) -> torch.Tensor:
    """
    Self-information novelty: novelty(u) = -(1/k) * sum_{i in top_k(u)} log2(pop(i)/N).

    topk_item_ids : (B, k) long -- item ids in each user's top-k.
    item_popularity : (num_items,) float -- TRAIN-set interaction counts per item.
    Returns (B,) novelty per user. Higher = rarer items recommended.
    """
    N = item_popularity.sum().clamp_min(eps)
    pop = item_popularity[topk_item_ids]  # (B, k)
    pop_ratio = (pop / N).clamp_min(eps)
    return -torch.log2(pop_ratio).mean(dim=1)


def diversity_at_k(topk_item_ids: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Intra-list diversity: diversity(u) = 1 - mean_{i<j in top_k(u)} cos_sim(emb_i, emb_j).

    topk_item_ids : (B, k) long.
    item_embeddings : (num_items, d) float -- one row per item.
    Returns (B,) diversity per user. NaN for k=1 (no pairs to compare).
    """
    B, k = topk_item_ids.shape
    if k < 2:
        return torch.full((B,), float("nan"), device=topk_item_ids.device)

    emb = item_embeddings[topk_item_ids]  # (B, k, d)
    emb_n = F.normalize(emb, dim=-1)
    sim = torch.matmul(emb_n, emb_n.transpose(1, 2))  # (B, k, k) cosine sim matrix

    iu = torch.triu_indices(k, k, offset=1)  # upper triangle, i<j pairs
    pairwise_sim = sim[:, iu[0], iu[1]]  # (B, num_pairs)
    return 1.0 - pairwise_sim.mean(dim=1)


def serendipity_at_k(topk_item_ids: torch.Tensor, hit_mask: torch.Tensor,
                     popularity_topk_ids: torch.Tensor) -> torch.Tensor:
    """
    Popularity-baseline serendipity:
    serendipity(u) = |{i in top_k(u) : hit(u,i) AND i not in popularity_topk}| / k

    topk_item_ids : (B, k) long -- this model's top-k item ids per user.
    hit_mask : (B, k) bool -- hit_mask[u,j] True iff topk_item_ids[u,j] is a true positive for u.
    popularity_topk_ids : (k_pop,) long -- fixed "obvious" baseline (top-k_pop most popular
        items in the TRAIN set only -- never validation/test, to avoid leaking future info
        into what counts as "expected").
    Returns (B,) serendipity per user, in [0, 1].
    """
    B, k = topk_item_ids.shape
    device = topk_item_ids.device
    max_id = int(torch.max(topk_item_ids.max(), popularity_topk_ids.max()).item()) + 1
    pop_flag = torch.zeros(max_id, dtype=torch.bool, device=device)
    pop_flag[popularity_topk_ids] = True

    is_in_pop_baseline = pop_flag[topk_item_ids]  # (B, k)
    unexpected = ~is_in_pop_baseline
    serendipitous = hit_mask.bool() & unexpected
    return serendipitous.float().sum(dim=1) / k