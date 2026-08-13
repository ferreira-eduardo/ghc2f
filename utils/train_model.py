import torch
import numpy as np
from tqdm import tqdm

def train_model(
        model,
        num_epochs,
        train_loader,
        val_loader,
        early_stopping,
):
    """
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
        val_metrics = model.evaluate(val_loader)
        # val_rmse = val_metrics['rmse']
        # val_mae = val_metrics['mae']
        hit_rate = float(val_metrics['hit_rate'])

        print(f" --> Val HIT RATE: {hit_rate}" )

        # --- EARLY STOPPING ---
        early_stopping(hit_rate, model)

        if hit_rate > best_val_metric:
            best_val_metric = hit_rate

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return best_val_metric, train_losses


