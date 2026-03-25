import argparse
import gc
import pandas as pd
import torch
from torch.utils.data import DataLoader
#models ####
from model.ghc2f import GHC2F
from model.gated_ae import GatedHybridCFAutoEncoder
from model.aspectgh2f import AspectGHC2F
############
from utils.utils import prepare_inputs, AspectDataset
from utils.dataset_utils import RankingTrainDataset, train_collate_fn, loocv_collate_fn, create_sparse_matrix
from utils.leave_one_out_cv import get_loocv_fold_normalized
from utils.train_model import train_model

path = 'dataset/{}.csv'
CHECKPOINT = 'checkpoint/{}.pt'
all_results = []
all_losses = []
k_folds = [4]


def main():
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
    parser.add_argument("--model_name", type=str, default='')
    args = parser.parse_args()

    print(f"\nLoading datasets: {args.dataset}\n")
    df = pd.read_csv(path.format(args.dataset))
    df_topics = pd.read_csv(path.format('topic_dist_' + args.dataset))

    df["userId"] = df["userId"].astype(int)
    df["itemId"] = df["itemId"].astype(int)
    df_topics["userId"] = df_topics["userId"].astype(int)
    df_topics["itemId"] = df_topics["itemId"].astype(int)

    TOTAL_ITEMS = df.itemId.max() + 1
    TOTAL_USERS = df.userId.max() + 1
    topic_cols = [col for col in df_topics.columns if col.isdigit()]
    topics_dim = len(topic_cols)

    for fold in k_folds:
        print("#" * 50)
        print(f"DATASET: {args.dataset}  - FOLD: {fold}")
        print("#" * 50)

        train, val, test = get_loocv_fold_normalized(df, fold)

        model = AspectGHC2F(
            layer_sizes=[TOTAL_ITEMS, 4096],
            num_users=TOTAL_USERS,
            num_items=TOTAL_ITEMS,
            topics_dim=topics_dim,
            topics_latent_dim=args.embedding_dim,
            nl_type="selu",
            dp_drop_prob=args.dropout,
            learn_rate=args.lr,
            use_hybrid=False
        ).to(device)

        df_topics_train = df_topics[df_topics["itemId"].isin(train["itemId"].unique())].copy()
        item_ids, item_topics, item_mask = prepare_inputs(df_topics_train, "itemId", topic_cols)

        with torch.no_grad():
            profiles = model.item_profiler(item_ids.to(device), item_topics.to(device), item_mask.to(device)).detach()

        full_i_global = torch.zeros((TOTAL_ITEMS, topics_dim), device=device)
        full_i_global[item_ids] = profiles
        model.item_global_profiles = full_i_global

        train_matrix = create_sparse_matrix(train, TOTAL_USERS, TOTAL_ITEMS)

        train_loader = DataLoader(
            RankingTrainDataset(train_matrix, df_topics, train),
            batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn, num_workers=4
        )

        val_loader = DataLoader(
            RankingTrainDataset(train_matrix, df_topics, val),
            batch_size=512, shuffle=False, collate_fn=lambda x: loocv_collate_fn(x, train), num_workers=4
        )

        test_relevant = test[test["is_relevant"] == True].copy()

        history_for_test = pd.concat([train, val])
        test_loader_ranking = DataLoader(
            RankingTrainDataset(train_matrix, df_topics, test_relevant),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: loocv_collate_fn(x, history_for_test),
            num_workers=4
        )

        ########## trainning ##########
        print('Starting training process (BPR Loss)...')
        best_val_loss, train_losses = train_model(model, args.epochs, train_loader, val_loader)

        all_losses.append({
            "datasets": args.dataset, "fold": fold,
            "best_val_losses": best_val_loss, "train_losses": train_losses
        })
        print("Iniciando Avaliação do Ranking..")
        values_rank = model.evaluate(test_loader_ranking)

        all_results.append({
            "datasets": args.dataset, "fold": fold,
            **values_rank
        })

        torch.cuda.empty_cache()
        gc.collect()

        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"{args.model_name}_bpr_{args.dataset}_complement.csv", index=False)

        df_losses = pd.DataFrame(all_losses)
        df_losses.to_csv(f"{args.model_name}_bpr_losses_{args.dataset}_complement.csv", index=False)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f"{args.model_name}_bpr_{args.dataset}_complement.csv", index=False)

    df_losses = pd.DataFrame(all_losses)
    df_losses.to_csv(f"{args.model_name}_bpr_losses_{args.dataset}_complement.csv", index=False)


if __name__ == "__main__":
    main()
