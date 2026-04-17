import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    FEAT_DIR, MODEL_DIR, GT_CONSTANT_C, FUSION_CONFIG, SEED, TRAIN_END
)


# Ground truth weight computation

def compute_ground_truth_weights(
    close_df: pd.DataFrame,
    c: int = GT_CONSTANT_C,
) -> pd.DataFrame:
    """
    Compute ground-truth portfolio weights using paper Eq. (1).
    w_{i,t} = softmax(ρ_{i,t} × c)
    """
    pct_change = close_df.pct_change().fillna(0)
    scaled_np  = (pct_change * c).values
    scaled_np  = scaled_np - scaled_np.max(axis=1, keepdims=True)
    exp_vals   = np.exp(scaled_np)
    weights    = exp_vals / (exp_vals.sum(axis=1, keepdims=True) + 1e-10)

    gt_weights = pd.DataFrame(
        weights, index=close_df.index, columns=close_df.columns
    )
    assert np.allclose(gt_weights.sum(axis=1), 1.0, atol=1e-5)
    return gt_weights


# Pytorch Dataset

class FusionDataset(Dataset):
    """
    Dataset for supervised pre-training.

    Each sample:
        X:            stacked agent actions  → (3, N)
        y:            ground-truth weights   → (N,)
        next_returns: next-day pct changes   → (N,)
    """

    def __init__(
        self,
        stacked_actions: np.ndarray,   # (T, 3, N)
        gt_weights:      np.ndarray,   # (T, N)
        next_returns:    np.ndarray,   # (T, N)  ← next-day percentage returns
    ):
        assert stacked_actions.shape[0] == gt_weights.shape[0]
        assert stacked_actions.shape[0] == next_returns.shape[0]

        self.X = torch.tensor(stacked_actions, dtype=torch.float32)
        self.y = torch.tensor(gt_weights,      dtype=torch.float32)
        self.r = torch.tensor(next_returns,    dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.r[idx]


# Loss functions

def sharpe_approx_loss(
    pred_weights: torch.Tensor,   # (B, N)
    next_returns: torch.Tensor,   # (B, N)
    eps: float = 1e-6,
) -> torch.Tensor:
    
    port_returns = (pred_weights * next_returns).sum(dim=-1)   # (B,)
    mean_r = port_returns.mean()
    std_r  = port_returns.std() + eps
    return -(mean_r / std_r)


def gate_entropy_regulariser(alpha: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    
    a   = alpha.clamp(eps, 1.0 - eps).squeeze(-1)   # (B,)
    ent = -(a * a.log() + (1 - a) * (1 - a).log())  # (B,) — per-sample entropy
    return -ent.mean()                               # negative = loss to minimise


# Supervised Pre Training

def pretrain_fusion_module(
    fusion_model:    nn.Module,
    stacked_actions: np.ndarray,   # (T, 3, N)
    gt_weights:      np.ndarray,   # (T, N)
    next_returns:    np.ndarray,   # (T, N) 
    config:          dict = FUSION_CONFIG,
    save_name:       str  = "fusion_pretrained",
    lambda_mse:      float = 1.0,
    lambda_sharpe:   float = 0.3,
    lambda_gate:     float = 0.1,
) -> nn.Module:
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Pre-training Fusion Module (Fixed Supervised Training)")
    print(f"   Device:        {device}")
    print(f"   Input shape:   (batch, 3, {stacked_actions.shape[2]})")
    print(f"   Epochs:        {config['epochs']}")
    print(f"   Loss weights:  MSE={lambda_mse}, Sharpe={lambda_sharpe}, Gate={lambda_gate}")
    print(f"   >> Gate entropy loss forces alpha to vary (FIXED frozen gate)")

    fusion_model = fusion_model.to(device)

    # Dataset and loaders
    dataset = FusionDataset(stacked_actions, gt_weights, next_returns)
    n_train = int(0.9 * len(dataset))
    n_val   = len(dataset) - n_train

    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=config["batch_size"], shuffle=False)

    optimizer = optim.AdamW(
        fusion_model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-5
    )
    mse_criterion = nn.MSELoss()

    train_losses  = []
    val_losses    = []
    alpha_history = []   # track gate mean + std per epoch
    best_val      = float("inf")
    best_state    = None

    for epoch in range(1, config["epochs"] + 1):

        # Train
        fusion_model.train()
        train_loss  = 0.0
        epoch_alpha = []

        for X_batch, y_batch, r_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            r_batch = r_batch.to(device)

            optimizer.zero_grad()

            # Forward with internals to get alpha
            pred, internals = fusion_model(X_batch, return_internals=True)

            # Three-component loss
            l_mse    = mse_criterion(pred, y_batch)
            l_sharpe = sharpe_approx_loss(pred, r_batch)
            alpha    = internals["gate_alpha"]                    # (B, 1)
            l_gate   = gate_entropy_regulariser(alpha)

            loss = (lambda_mse   * l_mse
                  + lambda_sharpe * l_sharpe
                  + lambda_gate   * l_gate)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            epoch_alpha.append(alpha.detach().cpu().squeeze(-1))

        train_loss /= len(train_loader)
        scheduler.step()

        # Track alpha statistics
        all_alphas = torch.cat(epoch_alpha)
        alpha_mean = all_alphas.mean().item()
        alpha_std  = all_alphas.std().item()
        alpha_history.append((alpha_mean, alpha_std))

        # Validate
        fusion_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, r_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                r_batch = r_batch.to(device)
                pred     = fusion_model(X_batch)
                val_loss += mse_criterion(pred, y_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()}

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{config['epochs']} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"α mean={alpha_mean:.3f} std={alpha_std:.3f}")

    # Load best weights
    fusion_model.load_state_dict(best_state)

    # Save
    save_path = os.path.join(MODEL_DIR, f"{save_name}.pt")
    torch.save({
        "model_state_dict": best_state,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "alpha_history":    alpha_history,
        "config":           config,
    }, save_path)
    print(f"\n Best model saved to {save_path}")
    print(f" Best validation loss: {best_val:.6f}")

    # Check gate is no longer frozen
    final_alpha_std = alpha_history[-1][1]
    if final_alpha_std < 0.01:
        print(" WARNING: gate_alpha std still low, consider increasing lambda_gate")
    else:
        print(f" Gate is dynamic: final α std = {final_alpha_std:.4f} (> 0.01 = healthy)")

    _plot_training_curve(train_losses, val_losses, alpha_history, save_name)
    return fusion_model


def _plot_training_curve(train_losses, val_losses, alpha_history, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Loss plot
    axes[0].plot(train_losses, label="Train Loss", color="royalblue")
    axes[0].plot(val_losses,   label="Val Loss",   color="tomato")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Training Loss: {name}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gate alpha evolution
    means = [a[0] for a in alpha_history]
    stds  = [a[1] for a in alpha_history]
    epochs = range(1, len(means) + 1)
    axes[1].plot(epochs, means, color="green",  label="α mean")
    axes[1].fill_between(
        epochs,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2, color="green", label="α ± std"
    )
    axes[1].axhline(0.693, color="red", linestyle="--", alpha=0.5,
                    label="frozen original (0.693)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gate Alpha (α)")
    axes[1].set_title("Dynamic Gate Evolution (should vary)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"{name}_training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f" Training curve (with gate evolution) saved: {path}")


# Main

def main():

    print(" Loading data...")
    close_df = pd.read_csv(
        os.path.join(FEAT_DIR, "aligned_close.csv"), index_col=0, parse_dates=True
    )
    tickers  = pd.read_csv(
        os.path.join(FEAT_DIR, "tickers.csv"), header=0
    ).iloc[:, 0].tolist()

    n_assets = len(tickers)
    print(f"  Assets: {n_assets}")

    # Ground truth weights (train period only)
    print("\n Computing ground truth weights...")
    train_close = close_df[close_df.index <= TRAIN_END]
    gt_weights  = compute_ground_truth_weights(train_close, c=GT_CONSTANT_C)

    # Next-day percentage returns (for Sharpe loss) — shift by 1 day
    # next_returns[t] = pct_change on day t+1 (what you earn if you hold weights from day t)
    pct_change   = train_close.pct_change().fillna(0)
    next_returns = pct_change.shift(-1).fillna(0)   # (T, N)

    # Stacked agent actions (train set)
    print("\n Loading agent actions...")
    actions_path    = os.path.join(FEAT_DIR, "agent_actions_train")
    stacked_actions = np.load(os.path.join(actions_path, "stacked_actions.npy"))
    action_dates    = pd.read_csv(
        os.path.join(actions_path, "dates.csv"), parse_dates=["date"]
    )["date"].tolist()

    print(f"  Stacked actions shape: {stacked_actions.shape}")

    # Align GT weights, next_returns, and actions on common dates
    gt_aligned   = gt_weights.loc[gt_weights.index.isin(action_dates)]
    nr_aligned   = next_returns.loc[next_returns.index.isin(action_dates)]
    common_dates = [d for d in action_dates if d in gt_aligned.index]
    act_mask     = [d in gt_aligned.index for d in action_dates]

    stacked_aligned  = stacked_actions[[i for i, m in enumerate(act_mask) if m]]
    gt_aligned_vals  = gt_aligned.loc[common_dates].values
    nr_aligned_vals  = nr_aligned.loc[common_dates].values

    print(f"  Aligned — Actions: {stacked_aligned.shape}, GT: {gt_aligned_vals.shape}, "
          f"NextRet: {nr_aligned_vals.shape}")

    # Load Transformer model
    try:
        from fusion.transformer_fusion import TransformerFusionModule
        fusion_model = TransformerFusionModule(
            n_assets=n_assets,
            n_agents=3,
            **FUSION_CONFIG
        )
        print("\n Using Fixed Transformer Fusion Module")
        save_name = "transformer_fusion_pretrained"
    except ImportError:
        print("\n TransformerFusionModule not found — check transformer_fusion.py")
        return

    # Pre-train with fixed three-component loss
    pretrain_fusion_module(
        fusion_model    = fusion_model,
        stacked_actions = stacked_aligned,
        gt_weights      = gt_aligned_vals,
        next_returns    = nr_aligned_vals,
        config          = FUSION_CONFIG,
        save_name       = save_name,
        lambda_mse      = 1.0,
        lambda_sharpe   = 0.3,
        lambda_gate     = 0.1,
    )

    # Save GT weights
    gt_weights.to_csv(os.path.join(FEAT_DIR, "ground_truth_weights.csv"))
    print(f"\n Ground truth weights saved.")
    print("\n Fixed supervised pre-training complete!")
    print("   Next step: run backtest.py")


if __name__ == "__main__":
    main()
