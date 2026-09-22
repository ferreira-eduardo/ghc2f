"""
Verifies the item_profiler live-gradient fix (set_item_corpus /
GatedHybridCFAutoEncoder.forward() now calling item_profiler live instead
of reading a precomputed, .detach()'d buffer). Directly diffs parameters
(not indirect similarity stats) for the four components that feed
z_text/z_topic:

  item_profiler -> now called LIVE every forward pass on the raw item
                   corpus (item_corpus_ids/text/mask). Should receive real
                   gradient now -- this script verifies that empirically.
  item_proj     -> applied every forward pass; should already have been
                   receiving gradient even under the old caching scheme.
  user_profiler -> called live every forward pass on real batch data.
                   Should receive gradient normally.
  user_proj     -> same as user_profiler.

Also reports z_text (a.k.a z_topic) diversity decomposed into user-only
and item-only halves, before and after training, so you can see whether
the item side is actually diversifying now, independent of hit_rate noise
over a short run.

Usage:
    python check_topic_pathway.py --dataset imdb --train_epochs 5
"""
import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from test_ghc2f import load_fold, build_model, build_loaders, set_seed, run_epochs

# If you've renamed z_topic -> z_text in GatedAEOutput, this is the
# attribute name this script reads. Change to "z_topic" if you haven't.
Z_TEXT_ATTR = "z_text"


def param_change_report(before: dict, after: dict) -> pd.DataFrame:
    rows = []
    for name, p_before in before.items():
        p_after = after[name]
        diff = (p_after - p_before)
        rows.append({
            "param": name,
            "abs_before_norm": p_before.norm().item(),
            "abs_after_norm": p_after.norm().item(),
            "change_norm": diff.norm().item(),
            "max_abs_change": diff.abs().max().item(),
            "unchanged": bool(torch.equal(p_before, p_after)),
        })
    return pd.DataFrame(rows)


def snapshot_params(module: torch.nn.Module) -> dict:
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def pairwise_cosine_mean(z: torch.Tensor) -> float:
    z = F.normalize(z, dim=-1)
    sim = z @ z.t()
    B = sim.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    return sim[mask].mean().item()


def compute_topic_halves(model, batch_dev):
    """
    Recomputes topic_user and topic_item exactly as GatedHybridCFAutoEncoder
    .forward() now does (item side LIVE via item_profiler on the stored
    corpus, no more cached/.detach()'d item_global_profiles buffer).
    """
    with torch.no_grad():
        out = model.forward(batch_dev)

        u_text = batch_dev["user_text"]
        u_mask = batch_dev["user_mask"]
        u_ids = batch_dev["user_ids"]
        z_user_topic = model.user_profiler(u_ids, u_text, u_mask)
        topic_user = model.user_proj(z_user_topic)

        item_profiles = model.item_profiler(
            model.item_corpus_ids, model.item_corpus_text, model.item_corpus_mask
        )
        item_global = torch.zeros(
            (batch_dev["ratings_in"].size(1), item_profiles.size(1)), device=model.device
        )
        item_global = item_global.index_copy(0, model.item_corpus_ids, item_profiles)

        hist_mask = (batch_dev["ratings_in"] != 0).float()
        counts = hist_mask.sum(dim=1, keepdim=True)
        topic_item_global = (hist_mask @ item_global) / counts.clamp_min(1.0)
        topic_item = model.item_proj(topic_item_global)

    z_text = getattr(out, Z_TEXT_ATTR)
    return topic_user, topic_item, z_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_epochs", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print(f"Loading fold for {args.dataset} ...")
    fold = load_fold(args.dataset)
    train_loader, val_loader = build_loaders(fold, batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model("ghc2f_decoder", fold, device)

    before = {
        "item_profiler": snapshot_params(model.item_profiler),
        "item_proj": snapshot_params(model.item_proj),
        "user_profiler": snapshot_params(model.user_profiler),
        "user_proj": snapshot_params(model.user_proj),
    }

    batch = next(iter(train_loader))
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    topic_user_b, topic_item_b, z_text_b = compute_topic_halves(model, batch_dev)

    print(f"\n{'=' * 70}\nz_text decomposition BEFORE training\n{'=' * 70}")
    print(f"topic_user-only mean_cos_sim: {pairwise_cosine_mean(topic_user_b):.4f}")
    print(f"topic_item-only mean_cos_sim: {pairwise_cosine_mean(topic_item_b):.4f}")
    print(f"z_text (0.5/0.5) mean_cos_sim: {pairwise_cosine_mean(z_text_b):.4f}")

    print(f"\n{'=' * 70}\nTraining ghc2f_decoder for {args.train_epochs} epoch(s)...\n{'=' * 70}")
    run_epochs(model, train_loader, val_loader, args.train_epochs)

    after = {
        "item_profiler": snapshot_params(model.item_profiler),
        "item_proj": snapshot_params(model.item_proj),
        "user_profiler": snapshot_params(model.user_profiler),
        "user_proj": snapshot_params(model.user_proj),
    }

    for component in before:
        print(f"\n{'=' * 70}\nParameter change report: {component}\n{'=' * 70}")
        report = param_change_report(before[component], after[component])
        print(report.to_string(index=False))
        if report["unchanged"].all():
            print(f"-> {component}: EVERY parameter bit-identical after training. "
                  f"Still receiving zero gradient -- fix did not take effect for this module.")
        elif report["unchanged"].any():
            print(f"-> {component}: SOME parameters unchanged, others updated. Mixed signal.")
        else:
            print(f"-> {component}: all parameters changed. Receiving gradient normally.")

    batch2 = next(iter(train_loader))
    batch2_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch2.items()}
    topic_user_a, topic_item_a, z_text_a = compute_topic_halves(model, batch2_dev)

    print(f"\n{'=' * 70}\nz_text decomposition AFTER training\n{'=' * 70}")
    print(f"topic_user-only mean_cos_sim: {pairwise_cosine_mean(topic_user_a):.4f}")
    print(f"topic_item-only mean_cos_sim: {pairwise_cosine_mean(topic_item_a):.4f}")
    print(f"z_text (0.5/0.5) mean_cos_sim: {pairwise_cosine_mean(z_text_a):.4f}")

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    print("item_profiler should now show real parameter changes (not bit-identical). "
          "If topic_item-only mean_cos_sim moved noticeably from its BEFORE value, the "
          "fix is working -- give it more epochs and re-check loss_cl's trajectory. If "
          "item_profiler changed but topic_item-only similarity barely moved, the module "
          "may just need more steps/a higher learning rate than the shared 1e-4 to shift a "
          "text-embedding-style component meaningfully.")