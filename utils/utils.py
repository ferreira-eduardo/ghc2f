from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


def MSEloss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        size_average: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # mask of observed ratings
    mask = (targets != 0)
    mask_f = mask.float()

    # difference only on observed entries
    diff = (inputs - targets) * mask_f
    squared = diff.pow(2)

    # avoid division by zero
    num_ratings = mask_f.sum().clamp(min=1.0)

    if size_average:
        loss = squared.sum() / num_ratings
        norm = torch.tensor(1.0, device=loss.device)
    else:
        loss = squared.sum()
        norm = num_ratings

    return loss, norm


def prepare_inputs(df, entity_col, col_names, max_reviews=50):
    vectors = df[col_names].values

    # Create a temporary series to group
    temp_df = pd.DataFrame({
        entity_col: df[entity_col],
        'vec': list(vectors)
    })

    # Group by User/Item
    grouped = temp_df.groupby(entity_col)['vec'].apply(lambda x: list(x)[:max_reviews]).reset_index()

    # Extract IDs
    ids = torch.tensor(grouped[entity_col].values, dtype=torch.long)

    # Convert lists to tensors and pad
    topic_tensors = [torch.tensor(np.array(t), dtype=torch.float32) for t in grouped['vec']]
    topics_padded = pad_sequence(topic_tensors, batch_first=True, padding_value=0.0)

    # Create Mask
    lengths = torch.tensor([len(t) for t in topic_tensors])
    max_len = topics_padded.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return ids, topics_padded, mask



class EarlyStoppingRanking:
    def __init__(self, patience=5, delta=0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, current_metric, model):
        score = current_metric

        if self.best_score is None:
            self.best_score = score
            self.stash_best_state(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.stash_best_state(model)
            self.counter = 0

    def stash_best_state(self, model):
        if self.verbose:
            print("New best metric...")
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def load_best_into_model(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            if self.verbose:
                print("Best state restored...")



