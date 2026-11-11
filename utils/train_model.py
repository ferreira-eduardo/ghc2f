import torch
import numpy as np
from tqdm import tqdm

def train_model(
        model,
        num_epochs,
        train_loader,
        val_loader,
        early_stopping,
        epoch_callback=None,
        validate=None,
):
    """
    epoch_callback(epoch, val_metric) is called after each validation
    (e.g. Optuna reporting/pruning; it may raise to stop the run).
    validate(model) -> float replaces model.evaluate(val_loader)['hit_rate'] as the
    early-stopping metric (higher is better), e.g. full-catalog val nDCG@10.
    """
    device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_losses = []
    best_val_metric = 0

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []

        # --- TRAIN LOOP ---
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            # The Adapter Call: Model handles its own data unpacking and loss
            loss, batch_size = model.calculate_loss(batch)

            # Standard Optimization Steps
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            batch_losses.append(loss.item())

            # Update Progress Bar
            pbar.set_postfix({'Train Loss': loss.item()})

        # --- EPOCH SUMMARY ---
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)

        # --- VALIDATION ---
        if validate is not None:
            hit_rate = float(validate(model))
            print(f" --> Val metric: {hit_rate:.5f}")
        else:
            hit_rate = float(model.evaluate(val_loader)['hit_rate'])
            print(f" --> Val HIT RATE: {hit_rate}")

        # --- EARLY STOPPING ---
        early_stopping(hit_rate, model)

        if hit_rate > best_val_metric:
            best_val_metric = hit_rate

        if epoch_callback is not None:
            epoch_callback(epoch, hit_rate)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return best_val_metric, train_losses


