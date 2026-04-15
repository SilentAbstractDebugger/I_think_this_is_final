"""
fusion/transformer_fusion.py
─────────────────────────────
⚡ THIS FILE IS OWNED BY PARTNER 3 ⚡

Transformer-based Fusion Module — replaces the paper's CNN (1×3 kernel).

WHY TRANSFORMER INSTEAD OF CNN?
────────────────────────────────
The paper uses a CNN with a (1,3) kernel to aggregate 3 agent outputs per asset.
This means it assigns FIXED local weights to the 3 agents — same aggregation rules
for all assets.

Transformer is BETTER because:
1. SELF-ATTENTION: Can model cross-agent AND cross-asset relationships
2. DYNAMIC weighting: Attention scores change per input (not fixed like conv weights)
3. GLOBAL context: Attends to all 3 agents and all N assets simultaneously
4. More expressive: Can learn "when log_return agent is confident, trust it more"

ARCHITECTURE OVERVIEW:
─────────────────────
Input: (batch, 3, N)  — 3 agents, N=29 assets each

Step 1: Linear embedding
  Each agent's weight vector (N,) → embedded to (d_model,) = (64,)
  Now: (batch, 3, d_model)   ← sequence of 3 "tokens" (one per agent)

Step 2: Positional Encoding
  Add learned positional encoding to distinguish Agent 1, 2, 3
  Still: (batch, 3, d_model)

Step 3: Transformer Encoder
  Multi-head self-attention across the 3 agent tokens
  Feed-forward network
  Layer norm
  Stack 2 layers → (batch, 3, d_model)

Step 4: Aggregate
  Average pool across the 3 agent tokens → (batch, d_model)
  OR use a learnable [CLS] token

Step 5: MLP head
  (batch, d_model) → (batch, 128) → (batch, 64) → (batch, N)
  Softmax → final portfolio weights

WHAT PARTNER 3 NEEDS TO IMPLEMENT:
────────────────────────────────────
The class TransformerFusionModule(nn.Module) with:
  - __init__(n_assets, n_agents, d_model, nhead, num_layers, dim_feedforward, dropout)
  - forward(x: Tensor[batch, 3, N]) → Tensor[batch, N]

The supervised_pretraining.py will import and use this class.
"""

import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    With only 3 positions (3 agents), simple learned embeddings also work well.
    """

    def __init__(self, d_model: int, max_len: int = 10, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Sinusoidal encoding
        position = torch.arange(max_len).unsqueeze(1)                          # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional_encoding: same shape
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TransformerFusionModule(nn.Module):
    """
    Transformer-based fusion of 3 PPO agent outputs.

    INPUT:
        x: (batch, 3, N)
           3 = one row per agent (log_return, dsr, mdd)
           N = number of assets (29 for Dow)

    OUTPUT:
        weights: (batch, N) — final portfolio allocation
                 Softmax applied → sums to 1, all ≥ 0

    PARTNER 3: This is the full implementation. Verify the forward pass and
    adjust d_model / nhead / num_layers if needed. The defaults match FUSION_CONFIG.
    """

    def __init__(
        self,
        n_assets:        int,
        n_agents:        int   = 3,
        d_model:         int   = 64,
        nhead:           int   = 4,
        num_layers:      int   = 2,
        dim_feedforward: int   = 128,
        dropout:         float = 0.1,
        **kwargs,   # absorb extra config keys safely
    ):
        super().__init__()

        self.n_assets = n_assets
        self.n_agents = n_agents
        self.d_model  = d_model

        # ── Step 1: Project each agent's N-dim action vector to d_model
        # Each agent produces a vector of N weights → embed to d_model
        self.input_proj = nn.Linear(n_assets, d_model)

        # ── Step 2: Positional encoding (distinguishes Agent 1, 2, 3)
        self.pos_enc = PositionalEncoding(d_model, max_len=n_agents + 1, dropout=dropout)

        # ── Step 3: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,    # (batch, seq, features) — more intuitive
            norm_first      = True,    # Pre-norm (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
        )

        # ── Step 4: Learnable CLS token for aggregation
        # Alternative to mean pooling — lets model focus on what matters
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ── Step 5: MLP output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_assets),
        )

        # ── Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, N) — stacked agent portfolio weight proposals

        Returns:
            weights: (batch, N) — fused portfolio weights (sum=1, all≥0)

        Step-by-step shapes:
            Input:              (B, 3, N)
            After input_proj:   (B, 3, d_model)
            Prepend CLS token:  (B, 4, d_model)
            After pos_enc:      (B, 4, d_model)
            After transformer:  (B, 4, d_model)
            Extract CLS:        (B, d_model)
            After MLP:          (B, N)
            After softmax:      (B, N)
        """
        B = x.size(0)

        # Step 1: Embed each agent's action
        # x: (B, 3, N) → (B, 3, d_model)
        tokens = self.input_proj(x)   # (B, 3, d_model)

        # Step 2: Prepend CLS token for aggregation
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)   # (B, 4, d_model) — CLS + 3 agents

        # Step 3: Positional encoding
        tokens = self.pos_enc(tokens)   # (B, 4, d_model)

        # Step 4: Transformer encoder (self-attention across 4 tokens)
        encoded = self.transformer_encoder(tokens)   # (B, 4, d_model)

        # Step 5: Extract CLS token representation
        cls_out = encoded[:, 0, :]   # (B, d_model)

        # Step 6: MLP → logits
        logits = self.output_head(cls_out)   # (B, N)

        # Step 7: Softmax → valid portfolio weights
        weights = torch.softmax(logits, dim=-1)   # (B, N)

        return weights

    def get_attention_weights(self, x: torch.Tensor):
        """
        Extract attention weights for interpretability.
        Shows how much weight each agent gets for each asset.

        Returns dict with attention maps from each layer.
        """
        B = x.size(0)
        tokens = self.input_proj(x)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_enc(tokens)

        attention_maps = []
        for layer in self.transformer_encoder.layers:
            # Access multi-head attention
            attn_out, attn_weights = layer.self_attn(
                tokens, tokens, tokens,
                need_weights=True,
                average_attn_weights=False
            )
            attention_maps.append(attn_weights.detach().cpu().numpy())
            tokens = layer(tokens)

        return attention_maps


# ─────────────────────────────────────────────────────────
# INFERENCE UTILITIES (used by backtest.py)
# ─────────────────────────────────────────────────────────

class FusionInference:
    """
    Wrapper for running the trained fusion model in inference mode.
    Used during backtesting to get final portfolio weights.
    """

    def __init__(self, model: nn.Module, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model  = model.to(device).eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, n_assets: int, **model_kwargs):
        """Load fusion model from saved checkpoint."""
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = TransformerFusionModule(n_assets=n_assets, **model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  ✅ Fusion model loaded from {checkpoint_path}")
        return cls(model)

    @torch.no_grad()
    def predict(self, stacked_actions: np.ndarray) -> np.ndarray:
        """
        Get final portfolio weights for a batch of agent actions.

        Args:
            stacked_actions: numpy array of shape (T, 3, N) or (3, N)

        Returns:
            weights: numpy array of shape (T, N) or (N,)
        """
        squeeze = False
        if stacked_actions.ndim == 2:
            stacked_actions = stacked_actions[np.newaxis]   # add batch dim
            squeeze = True

        x = torch.tensor(stacked_actions, dtype=torch.float32).to(self.device)
        w = self.model(x).cpu().numpy()

        if squeeze:
            w = w[0]

        return w

    @torch.no_grad()
    def predict_single_date(self, agent_actions: dict, tickers: list) -> dict:
        """
        Get weights for a single date given agent action dict.

        Args:
            agent_actions: {reward_type: np.array(N,)}
            tickers:       list of N ticker symbols

        Returns:
            {ticker: weight} dict
        """
        # Stack: (3, N)
        stacked = np.stack([
            agent_actions["log_return"],
            agent_actions["dsr"],
            agent_actions["mdd"],
        ], axis=0)

        weights = self.predict(stacked)   # (N,)
        return dict(zip(tickers, weights))


# ─────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing TransformerFusionModule ===\n")

    n_assets = 29
    n_agents = 3
    batch    = 16

    model = TransformerFusionModule(
        n_assets=n_assets,
        n_agents=n_agents,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward pass
    x = torch.randn(batch, n_agents, n_assets)
    w = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {w.shape}")
    print(f"Weight sum (should be 1.0): {w.sum(dim=-1)}")
    print(f"All weights ≥ 0: {(w >= 0).all()}")
    print(f"All weights ≤ 1: {(w <= 1).all()}")

    print("\n✅ TransformerFusionModule test passed!")
