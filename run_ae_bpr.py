import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.optim import Adam
from torch.utils.data import DataLoader
from model.ghc2f import GHC2F
from utils.ae_utils import prepare_inputs
from utils.bpr_utils import BPRDataset, prepare_bpr_data, generate_user_topics, EvalDataset
from utils.leave_one_out_cv import get_loocv_fold_normalized
from utils.train_models import train_model

datasets = ['imdb','rotten_tomatoes']

for dataset in datasets:
    df = pd.read_csv(f'../../../../dataset/{dataset}.csv')
    df_topics = pd.read_csv(f'../../../../dataset/topic_dist_{dataset}.csv', index_col=0)

    df_topics['userId'] = df['userId'].astype(int)
    df_topics['itemId'] = df['itemId'].astype(int)

    # Normalize and
    df['rating_norm'] = df.groupby('userId')['rating'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    df['rating_norm'] = df['rating_norm'].fillna(0)

    df['is_positive'] = df['rating_norm'].apply(lambda x: True if x > 0 else False)

    TOTAL_USERS = df.userId.nunique()
    TOTAL_ITEMS = df.itemId.nunique()

    user_item_matrix = prepare_bpr_data(df, binarize=False)


    fold = 5
    train, val, test = get_loocv_fold_normalized(df, fold)

    user_topics_matrix = generate_user_topics(train, df_topics)

    train_matrix = csr_matrix(
        (train.rating, (train.userId, train.itemId)),
        shape=(TOTAL_USERS, TOTAL_ITEMS)
    )

    train_dataset = BPRDataset(
        train_matrix, user_topics_matrix, user_topics_matrix
    )

    val_dataset = EvalDataset(
        train_matrix, val, user_topics_matrix
    )

    test_dataset = EvalDataset(
        train_matrix, test, user_topics_matrix
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    topic_dim = 15
    layer_sizes = [TOTAL_ITEMS, 4096]
    model = GHC2F(
        layer_sizes=layer_sizes,
        num_users=TOTAL_USERS,
        num_items=TOTAL_ITEMS,
        topics_dim=15,
        topics_latent_dim=64,
        nl_type="selu",
        dp_drop_prob=0.5,
        learn_rate=1e-3
    ).to(device)

    topic_cols = list(map(str, range(topic_dim)))
    item_ids, item_topics, item_mask = prepare_inputs(df_topics, 'itemId', topic_cols)

    user_item_matrix = prepare_bpr_data(df, binarize=True)
    user_topics_matrix = generate_user_topics(df, df_topics)

    train_dataset = BPRDataset(
        user_item_matrix, user_topics_matrix, num_negatives=5
    )

    with torch.no_grad():
        profiles = model.item_profiler(
            item_ids.to(device),
            item_topics.to(device),
            item_mask.to(device)
        ).detach()

    full_i_global = torch.zeros((TOTAL_ITEMS, topic_dim), device=device)
    full_i_global[item_ids] = profiles
    model.item_global_profiles = full_i_global
    print(model)


    def train_bpr_model(model, dataset, epochs=1, batch_size=256, lr=0.0001, weight_decay=1e-5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            total_loss = 0

            for batch in dataloader:
                # Move all batch tensors to the same device as the model
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()

                loss, n_samples = model.calculate_loss(batch)

                #  Backpropagate
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs} | BPR Loss: {avg_loss:.4f}")

        print("Training Complete.")



    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_val_metric, train_losses = train_model(model, 1, train_loader, val_loader)