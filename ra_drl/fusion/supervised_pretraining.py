"""
fusion/supervised_pretraining.py
──────────────────────────────────
Supervised pre-training of the fusion module (CNN in paper; Transformer in your version).

WHY SUPERVISED PRE-TRAINING?
────────────────────────────
The Transformer/CNN fusion module needs to learn HOW to weigh the 3 agents' outputs.
Pure RL training of the fusion is unstable (sparse rewards, high variance).

The paper's solution: Pre-train the fusion module in a SUPERVISED fashion using
historical "ground truth" optimal weights.

GROUND TRUTH WEIGHT FORMULA (Paper Equation 1):
  w_{i,t} = exp(ρ_{i,t} × c) / Σ_j exp(ρ_{j,t} × c)

  where:
    ρ_{i,t} = percentage change in price of stock i at time t
              = (P_{i,t} - P_{i,t-1}) / P_{i,t-1}
    c = constant ∈ {1, 2, 3, 4, 5}  [we use c=3 from config]

  INTUITION: Assets with HIGHER recent returns get HIGHER weight.
  This is a RETROSPECTIVE oracle — it knows which stocks did well.
  Used only to initialize the fusion module parameters.

  After pre-training, the fusion module refines itself during RL training.

SUPERVISED TRAINING PROCESS:
  Input:  Stacked agent actions → shape (batch, 3, N)
  Target: Ground-truth weights  → shape (batch, N)
  Loss:   MSE between predicted weights and GT weights
  
  This teaches the fusion module: "given what each specialized agent wants,
  output the historically best allocation."

YOUR PARTNER'S TRANSFORMER (handles the forward pass here):
  Input: (batch, 3, N) — 3 agents × N assets
  The Transformer attends across the 3 agents, weighing their contributions.
  Output: (batch, N) — fused portfolio weights
"""

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


# ─────────────────────────────────────────────────────────
# GROUND TRUTH WEIGHT COMPUTATION
# ─────────────────────────────────────────────────────────

def compute_ground_truth_weights(
    close_df: pd.DataFrame,
    c: int = GT_CONSTANT_C,
) -> pd.DataFrame:
    """
    Compute ground-truth portfolio weights using paper Eq. (1).

    w_{i,t} = softmax(ρ_{i,t} × c)

    where ρ_{i,t} = percentage price change of stock i at time t.

    Args:
        close_df: (T × N) closing prices
        c:        scaling constant (paper uses 1-5, we use 3)

    Returns:
        gt_weights: (T × N) DataFrame of ground-truth weights
                    Note: day t's weight uses day t's percentage change
                    (retrospective oracle — only for pre-training)
    """
    # Daily percentage returns
    pct_change = close_df.pct_change().fillna(0)  # ρ_{i,t}

    # Compute softmax weights per day
    scaled = pct_change * c          # ρ × c
    # Numerical stability: subtract row max before exp
    scaled_np = scaled.values
    scaled_np = scaled_np - scaled_np.max(axis=1, keepdims=True)
    exp_vals  = np.exp(scaled_np)
    weights   = exp_vals / (exp_vals.sum(axis=1, keepdims=True) + 1e-10)

    gt_weights = pd.DataFrame(
        weights,
        index   = close_df.index,
        columns = close_df.columns,
    )

    # Validate: each row sums to 1
    row_sums = gt_weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "GT weights don't sum to 1!"

    return gt_weights


# ─────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────

class FusionDataset(Dataset):
    """
    Dataset for supervised pre-training of fusion module.

    Each sample:
        X: stacked agent actions → shape (3, N) — 3 agents, N assets
        y: ground-truth weights  → shape (N,)
    """

    def __init__(
        self,
        stacked_actions: np.ndarray,   # shape (T, 3, N)
        gt_weights:      np.ndarray,   # shape (T, N)
    ):
        assert stacked_actions.shape[0] == gt_weights.shape[0], \
            f"Length mismatch: actions {stacked_actions.shape[0]} vs GT {gt_weights.shape[0]}"

        self.X = torch.tensor(stacked_actions, dtype=torch.float32)
        self.y = torch.tensor(gt_weights,      dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────
# CNN FUSION MODULE (paper's original — for reference)
# ─────────────────────────────────────────────────────────

class CNNFusionModule(nn.Module):
    """
    Paper's original CNN fusion module.
    Kept here for reference and ablation studies.

    Architecture (from paper Fig. 2 + Fig. 3):
    Input: (batch, 1, 3, N)  — 1 channel, 3 agents, N assets
    Conv2D(kernel=(1,3))     → extracts cross-agent features per asset
    Flatten → FC layers      → output weights (N,)

    The (1,3) kernel means:
      - Height=1: looks at ONE asset at a time
      - Width=3:  looks at ALL 3 agent outputs for that asset
    This lets the network learn: "given all 3 agents' opinions on stock i,
    what weight should we give stock i?"
    """

    def __init__(self, n_assets: int, n_agents: int = 3):
        super().__init__()
        self.n_assets = n_assets
        self.n_agents = n_agents

        # (1,3) kernel: height=1 (one asset), width=3 (three agents)
        self.conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = 32,
            kernel_size  = (1, n_agents),   # (1, 3)
            stride       = 1,
            padding      = 0,
        )

        # After conv: (batch, 32, N, 1) → flatten to (batch, 32*N)
        flat_size = 32 * n_assets

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_assets),
            nn.Softmax(dim=-1),   # weights sum to 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, N) stacked agent actions
        Returns:
            weights: (batch, N) portfolio weights
        """
        # Add channel dim: (batch, 1, 3, N)
        # Then permute for Conv2D: (batch, 1, N, 3)
        # So kernel (1,3) processes all 3 agent outputs per asset
        x = x.unsqueeze(1)              # (batch, 1, 3, N)
        x = x.permute(0, 1, 3, 2)      # (batch, 1, N, 3)
        x = self.conv(x)                # (batch, 32, N, 1)
        return self.mlp(x)


# ─────────────────────────────────────────────────────────
# SUPERVISED PRE-TRAINING
# ─────────────────────────────────────────────────────────

def pretrain_fusion_module(
    fusion_model: nn.Module,
    stacked_actions: np.ndarray,  # (T, 3, N)
    gt_weights: np.ndarray,       # (T, N)
    config: dict = FUSION_CONFIG,
    save_name: str = "fusion_pretrained",
) -> nn.Module:
    """
    Supervised pre-training of any fusion module (CNN or Transformer).

    Loss = MSE(predicted_weights, gt_weights)

    After this, the fusion module knows how to weigh agent contributions
    based on what was historically optimal.

    Args:
        fusion_model:    nn.Module (CNN or Transformer — from Partner 3)
        stacked_actions: (T, 3, N) — agent actions
        gt_weights:      (T, N)    — ground truth from paper Eq.(1)
        config:          training hyperparameters
        save_name:       filename for saved model

    Returns:
        Trained fusion_model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🏋️  Pre-training Fusion Module (Supervised)")
    print(f"   Device: {device}")
    print(f"   Input shape:  (batch, 3, {stacked_actions.shape[2]})")
    print(f"   Target shape: (batch, {gt_weights.shape[1]})")
    print(f"   Epochs:       {config['epochs']}")
    print(f"   Batch size:   {config['batch_size']}")
    print(f"   LR:           {config['lr']}")

    fusion_model = fusion_model.to(device)

    # ── Dataset + DataLoader
    dataset    = FusionDataset(stacked_actions, gt_weights)
    n_train    = int(0.9 * len(dataset))
    n_val      = len(dataset) - n_train

    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=config["batch_size"], shuffle=False)

    # ── Optimizer + Loss
    optimizer  = optim.Adam(fusion_model.parameters(), lr=config["lr"])
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion  = nn.MSELoss()

    # ── Training loop
    train_losses = []
    val_losses   = []
    best_val     = float("inf")
    best_state   = None

    for epoch in range(1, config["epochs"] + 1):
        # -- Train
        fusion_model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred  = fusion_model(X_batch)
            loss  = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -- Validate
        fusion_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred     = fusion_model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # Save best
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()}

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{config['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # ── Load best weights
    fusion_model.load_state_dict(best_state)

    # ── Save
    save_path = os.path.join(MODEL_DIR, f"{save_name}.pt")
    torch.save({
        "model_state_dict": best_state,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "config":           config,
    }, save_path)
    print(f"\n  💾 Best model saved to {save_path}")
    print(f"  ✅ Best validation loss: {best_val:.6f}")

    # ── Plot training curve
    _plot_training_curve(train_losses, val_losses, save_name)

    return fusion_model


def _plot_training_curve(train_losses, val_losses, name):
    """Save training loss plot."""
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss", color="royalblue")
    plt.plot(val_losses,   label="Val Loss",   color="tomato")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Fusion Pre-training: {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"{name}_training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Training curve saved: {path}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    """Run supervised pre-training pipeline."""

    # ── Load features and close prices
    print("📂 Loading data...")
    close_df  = pd.read_csv(
        os.path.join(FEAT_DIR, "aligned_close.csv"), index_col=0, parse_dates=True
    )
    tickers   = pd.read_csv(
        os.path.join(FEAT_DIR, "tickers.csv"), header=0
    ).iloc[:, 0].tolist()

    n_assets = len(tickers)
    print(f"  Assets: {n_assets}")

    # ── Compute ground truth weights
    print("\n🎯 Computing ground truth weights...")
    # Use only TRAINING period to prevent lookahead
    train_close = close_df[close_df.index <= TRAIN_END]
    gt_weights  = compute_ground_truth_weights(train_close, c=GT_CONSTANT_C)
    print(f"  GT weights shape: {gt_weights.shape}")
    print(f"  Row sum check:    {gt_weights.sum(axis=1).describe()}")

    # ── Load stacked agent actions (train set)
    print("\n📋 Loading agent actions...")
    actions_path = os.path.join(FEAT_DIR, "agent_actions_train")
    stacked_actions = np.load(os.path.join(actions_path, "stacked_actions.npy"))
    action_dates = pd.read_csv(
        os.path.join(actions_path, "dates.csv"), parse_dates=["date"]
    )["date"].tolist()

    print(f"  Stacked actions shape: {stacked_actions.shape}")  # (T, 3, N)

    # ── Align GT weights with agent action dates
    gt_aligned = gt_weights.loc[gt_weights.index.isin(action_dates)]
    common_idx = [d for d in action_dates if d in gt_aligned.index]

    act_mask   = [d in gt_aligned.index for d in action_dates]
    stacked_aligned = stacked_actions[[i for i, m in enumerate(act_mask) if m]]
    gt_aligned_vals = gt_aligned.loc[common_idx].values

    print(f"  Aligned shapes — Actions: {stacked_aligned.shape}, GT: {gt_aligned_vals.shape}")

    # ── IMPORT PARTNER'S TRANSFORMER (with fallback to CNN)
    try:
        from fusion.transformer_fusion import TransformerFusionModule
        fusion_model = TransformerFusionModule(
            n_assets  = n_assets,
            n_agents  = 3,
            **FUSION_CONFIG
        )
        print("\n✅ Using Transformer Fusion Module (Partner 3's implementation)")
        save_name = "transformer_fusion_pretrained"
    except ImportError:
        print("\n⚠️  TransformerFusionModule not found — using CNN (paper original)")
        fusion_model = CNNFusionModule(n_assets=n_assets, n_agents=3)
        save_name = "cnn_fusion_pretrained"

    # ── Pre-train
    pretrain_fusion_module(
        fusion_model    = fusion_model,
        stacked_actions = stacked_aligned,
        gt_weights      = gt_aligned_vals,
        config          = FUSION_CONFIG,
        save_name       = save_name,
    )

    # ── Also save GT weights for reference
    gt_weights.to_csv(os.path.join(FEAT_DIR, "ground_truth_weights.csv"))
    print(f"\n💾 Ground truth weights saved.")
    print("\n🎉 Supervised pre-training complete!")
    print(f"   Next step: run backtest.py")


if __name__ == "__main__":
    main()
