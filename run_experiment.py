import argparse
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

############## MODELS ##############
from model.cf_autoencoder import CFAutoEncoder
from model.ghc2f import GHC2F
from model.gated_hybrid_ae import GatedHybridCFAutoEncoder
############## MODELS ##############
from utils.utils import prepare_inputs, EarlyStoppingRanking
from utils.dataset_utils import RankingTrainDataset, train_collate_fn, loocv_collate_fn, create_sparse_matrix
from utils.dataset_utils import build_interacted_by_user
from utils.leave_one_out_cv import get_loocv_fold_normalized
from utils.train_model import train_model

path = '../../dataset/{}.csv'
text_path = '../../embeddings_reviews/{}.npy'
topic_path = '../../topic_dist/{}.csv'
CHECKPOINT = 'checkpoint/{}.pt'
all_results = []
all_losses = []
k_folds = [0, 1, 2, 3, 4]

use_text = True


def main():
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--bottleneck", type=int, default=512)
    parser.add_argument("--layers", nargs='+', type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"\nLoading datasets: {args.dataset}\n")

    df = pd.read_csv(path.format(args.dataset))

    if use_text:
        text_embedding = np.load(text_path.format(args.dataset))
        df_text = pd.DataFrame(text_embedding)

    else:
        df_text = pd.read_csv(topic_path.format('topic_dist_' + args.dataset))

    df["userId"] = df["userId"].astype(int)
    df["itemId"] = df["itemId"].astype(int)
    df_text["userId"] = df["userId"].astype(int)
    df_text["itemId"] = df["itemId"].astype(int)

    TOTAL_ITEMS = df.itemId.max() + 1
    TOTAL_USERS = df.userId.max() + 1
    text_cols = [col for col in df_text.columns if str(col).isdigit()]
    text_dim = len(text_cols)

    for fold in k_folds:
        print("#" * 50)
        print(f"DATASET: {args.dataset}  - FOLD: {fold}")
        print("#" * 50)

        train, val, test = get_loocv_fold_normalized(df, fold)

        model = GatedHybridCFAutoEncoder(
            layer_sizes=[TOTAL_ITEMS, 4096],
            num_users=TOTAL_USERS,
            num_items=TOTAL_ITEMS,
            text_dim=text_dim,
            text_latent_dim=args.embedding_dim,
            nl_type="selu",
            dp_drop_prob=args.dropout,
            learn_rate=args.lr,
        ).to(device)

        df_text_train = df_text[df_text["itemId"].isin(train["itemId"].unique())].copy()

        full_i_global = torch.zeros((TOTAL_ITEMS, text_dim), device=device)
        item_ids, item_text, item_mask = prepare_inputs(df_text_train, "itemId", text_cols)
        with torch.no_grad():
            profiles = model.item_profiler(item_ids.to(device), item_text.to(device), item_mask.to(device)).detach()
        full_i_global[item_ids] = profiles

        model.item_global_profiles = full_i_global
        train_matrix = create_sparse_matrix(train, TOTAL_USERS, TOTAL_ITEMS)

        # Per-user text profiles (RankingTrainDataset.user_text_map) must only
        # ever see reviews the user has actually written by that point in the
        # split. df_text is row-aligned with the *full* dataset (train+val+test),
        # so passing it in unfiltered would leak a user's own review for their
        # held-out val/test item into "user_text" (fed to TextProfile's
        # attention), trivially revealing the item being ranked.
        def text_seen_through(history_df):
            pairs = pd.MultiIndex.from_frame(history_df[["userId", "itemId"]])
            df_text_pairs = pd.MultiIndex.from_frame(df_text[["userId", "itemId"]])
            return df_text[df_text_pairs.isin(pairs)].copy()

        df_text_train_only = text_seen_through(train)

        train_loader = DataLoader(
            RankingTrainDataset(train_matrix, df_text_train_only, train),
            batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn, num_workers=4
        )

        interacted_train = build_interacted_by_user(train)
        val_loader = DataLoader(
            RankingTrainDataset(train_matrix, df_text_train_only, val),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: loocv_collate_fn(x, interacted_train, TOTAL_ITEMS), num_workers=4
        )

        test_relevant = test[test["is_relevant"] == True].copy()

        history_for_test = pd.concat([train, val])
        df_text_train_val = text_seen_through(history_for_test)
        interacted_hist = build_interacted_by_user(history_for_test)
        test_loader_ranking = DataLoader(
            RankingTrainDataset(train_matrix, df_text_train_val, test_relevant),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: loocv_collate_fn(x, interacted_hist, TOTAL_ITEMS),
            num_workers=4
        )

        ########## trainning ##########
        early_stopping = EarlyStoppingRanking(patience=5, verbose=True)
        print('Starting training process (BPR Loss)...')
        best_val_loss, train_losses = train_model(model, args.epochs, train_loader, val_loader, early_stopping)

        all_losses.append({
            "datasets": args.dataset, "fold": fold,
            "best_val_losses": best_val_loss, "train_losses": train_losses
        })
        print("Iniciando Avaliação do Ranking..")
        early_stopping.load_best_into_model(model) # load model
        values_rank = model.evaluate(test_loader_ranking)

        all_results.append({
            "datasets": args.dataset, "fold": fold,
            **values_rank
        })

        del early_stopping
        torch.cuda.empty_cache()
        gc.collect()


        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"{model.name}_bpr_{args.dataset}_text.csv", index=False)

        df_losses = pd.DataFrame(all_losses)
        df_losses.to_csv(f"{model.name}_bpr_losses_{args.dataset}_text.csv", index=False)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f"{model.name}_bpr_{args.dataset}_text.csv", index=False)

    df_losses = pd.DataFrame(all_losses)
    df_losses.to_csv(f"{model.name}_bpr_losses_{args.dataset}_text.csv", index=False)


if __name__ == "__main__":
    main()
