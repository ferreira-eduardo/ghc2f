import torch
import numpy as np
from tqdm import tqdm

from utils.ae_utils import EarlyStoppingRanking, evaluate_loocv

def train_model(
        model,
        num_epochs,
        train_loader,
        val_loader,
        patience=7,
):
    """
    """
    device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_losses = []
    best_val_metric = 0


    model.to(device)

    early_stopper = EarlyStoppingRanking(patience=patience, verbose=True)


    for epoch in range(num_epochs):
        model.train()
        batch_losses = []

        # --- TRAIN LOOP ---
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
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
        val_metrics = evaluate_loocv(model, val_loader, device)
        # val_rmse = val_metrics['rmse']
        # val_mae = val_metrics['mae']
        hit_rate = float(val_metrics['hit_rate'])

        print(f" --> Val HIT RATE: {hit_rate}" )

        # --- EARLY STOPPING ---
        early_stopper(hit_rate, model, 'best_bpr_model.pt')

        if hit_rate > best_val_metric:
            best_val_metric = hit_rate

        if early_stopper.early_stop:
            print("Early stopping triggered")
            break

    return best_val_metric, train_losses


# change for the evaluate model inside of the model
def evaluate_model(model, loader):
    """
    Generic evaluation. Relies on 'model.predict_step(batch)'
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            y, y_hat = model.predict_step(batch)

            # Calculate metrics on the CPU to save GPU memory during eval
            y = y.cpu()
            y_hat = y_hat.cpu()

            squared_error = (y - y_hat) ** 2
            abs_error = (y - y_hat).abs()

            total_mse += squared_error.sum().item()
            total_mae += abs_error.sum().item()
            num_samples += y.size(0)

    rmse = np.sqrt(total_mse / num_samples) if num_samples > 0 else 0.0
    mae = total_mae / num_samples if num_samples > 0 else 0.0

    return {'rmse': rmse, 'mae': mae}
