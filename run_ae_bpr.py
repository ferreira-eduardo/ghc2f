import argparse
import gc
import ast
import os.path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from utils.ae_utils import prepare_inputs
from utils.dataset_utils import RankingTrainDataset, train_collate_fn, loocv_collate_fn
from model.ghc2f import GHC2F
from utils.leave_one_out_cv import get_loocv_fold_normalized
from utils.train_models import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"

path = 'datasets/{}.csv'
CHECKPOINT = 'checkpoint/{}.pt'
all_results = []
k_folds = [0, 1, 2, 3, 4]
DATASETS = ['amazon', 'imdb', 'rotten_tomatoes']


def loocv_collate_fn(batch, user_history_dict, total_items, num_negatives=99):
    res = {}
    res["user_ids"] = torch.tensor([d["user_ids"] for d in batch], dtype=torch.int32)

    ratings_in = torch.stack([d["ratings_in"].clone().detach() for d in batch])
    for i, d in enumerate(batch):
        ratings_in[i, d["pos_item_id"]] = 0
    res["ratings_in"] = ratings_in

    topics_list = [d["user_topics"].clone().detach() for d in batch]
    res["user_topics"] = torch.nn.utils.rnn.pad_sequence(topics_list, batch_first=True)
    lengths = torch.tensor([t.size(0) for t in topics_list])
    max_len = res["user_topics"].size(1)
    res["user_mask"] = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    test_items = []
    neg_items = []
    all_item_ids = np.arange(total_items)

    for d in batch:
        u_id = d["user_ids"]
        test_items.append(d["pos_item_id"])

        interacted = user_history_dict.get(u_id, set())
        candidates = np.setdiff1d(all_item_ids, list(interacted), assume_unique=True)

        negs = np.random.choice(candidates, num_negatives,
                                replace=len(candidates) < num_negatives)
        neg_items.append(torch.tensor(negs, dtype=torch.long))

    res["pos_item_id"] = torch.tensor(test_items, dtype=torch.long)
    res["neg_item_id"] = torch.stack(neg_items)
    return res


def main():
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

    print(f"\nLoading datasets: {args.dataset}")
    df = pd.read_csv(f'datasets/{args.dataset}.csv',
                     dtype={'userId': np.int32, 'itemId': np.int32, 'rating': np.float32})
    df_topics = pd.read_csv(path.format('topic_dist_' + args.dataset))

    total_items = df.itemId.max() + 1
    total_users = df.userId.max() + 1
    topic_cols = [col for col in df_topics.columns if col.isdigit()]
    topics_dim = len(topic_cols)
    full_user_history = df.groupby('userId')['itemId'].apply(set).to_dict()

    for fold in k_folds:
        print(f"\n########## FOLD {fold} ##########")
        train, val, test = get_loocv_fold_normalized(df, fold)

        model = GHC2F(
            layer_sizes=[total_items] + args.layers,
            num_users=total_users,
            num_items=total_items,
            topics_dim=topics_dim,
            topics_latent_dim=args.embedding_dim,
            nl_type="selu",
            dp_drop_prob=args.dropout,
            learn_rate=args.lr
        ).to(device)

        # Configuração dos perfis de itens globais
        df_topics_train = df_topics[df_topics["itemId"].isin(train["itemId"].unique())].copy()
        item_ids, item_topics, item_mask = prepare_inputs(df_topics_train, "itemId", topic_cols)

        with torch.no_grad():
            profiles = model.item_profiler(item_ids.to(device), item_topics.to(device),
                                           item_mask.to(device)).detach()

            full_i_global = torch.zeros((total_items, topics_dim), device=device)
            full_i_global[item_ids] = profiles
            model.item_global_profiles = full_i_global

        del df_topics_train, profiles

        history_for_test = pd.concat([train, val], axis=0).reset_index(drop=True)

        train_u = train['userId'].values
        train_i = train['itemId'].values
        train_r = train['rating'].values

        train_matrix = csr_matrix(
            (train_r, (train_u, train_i)),
            shape=(total_users, total_items),
            dtype=np.float32
        )

        history_input_matrix = csr_matrix(
            (history_for_test['rating'].values.astype(np.float32),
             (history_for_test['userId'], history_for_test['itemId'])),
            shape=(total_users, total_items)
        )

        # Dataloader de Treino (Contrastive)
        train_loader = DataLoader(
            RankingTrainDataset(train_matrix, df_topics, train, df, is_train=True),
            batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn, num_workers=4
        )

        del train
        gc.collect()

        # Dataloader de Validação (Contrastive)
        val_loader = DataLoader(
            RankingTrainDataset(train_matrix, df_topics, val, df),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: loocv_collate_fn(x, full_user_history, total_items),
            num_workers=4
        )
        test_relevant = test[test["is_relevant"] == True].copy()

        test_loader_ranking = DataLoader(
            RankingTrainDataset(history_input_matrix, df_topics, test_relevant, df),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: loocv_collate_fn(x, full_user_history, total_items),
            num_workers=4
        )

        print('Starting training process (BPR Loss)...')
        _, _ = train_model(model, args.epochs, train_loader, val_loader)

        print("Iniciando Avaliação do Ranking..")
        checkpoint_path = CHECKPOINT.format('best_bpr_model')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
        else:
            print('Checkpoint file not found...')

        values_rank = model.evaluate(test_loader_ranking)

        all_results.append({
            "datasets": args.dataset, "fold": fold,
            **values_rank
        })

        torch.cuda.empty_cache()
        gc.collect()

        # salva cada datasets
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"contrastive_{args.dataset}.csv", index=False)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f"contrastive_{args.dataset}.csv", index=False)

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
