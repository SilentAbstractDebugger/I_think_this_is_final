"""
config.py — Central Configuration for RA-DRL Project
All hyperparameters, paths, and settings live here.
Modify this file instead of hunting through multiple scripts.
"""

import os

# PATHS

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
FEAT_DIR   = os.path.join(BASE_DIR, "data", "features")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

for d in [DATA_DIR, FEAT_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)


# DATASET — DOW 30
# Paper uses top 30 by market cap; Dow 30 official constituents (as of 2024)
DOW30_TICKERS = [
    "AAPL", "AMGN", "AXP",  "BA",   "CAT",
    "CRM",  "CSCO", "CVX",  "DIS",  "DOW",
    "GS",   "HD",   "HON",  "IBM",  "INTC",
    "JNJ",  "JPM",  "KO",   "MCD",  "MMM",
    "MRK",  "MSFT", "NKE",  "PG",   "TRV",
    "UNH",  "V",    "VZ",   "WBA",  "WMT",
]
N_ASSETS = len(DOW30_TICKERS)   # 30 (paper uses 29 for Dow)

TRAIN_START = "2011-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2024-03-31"


# TECHNICAL INDICATORS
SMA_SHORT   = 30
SMA_LONG    = 60
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
RSI_PERIOD  = 14
CCI_PERIOD  = 20
ADX_PERIOD  = 14
BB_PERIOD   = 20
BB_STD      = 2


# RL ENVIRONMENT
INITIAL_CAPITAL     = 1_000_000   # $1M as in paper
TRANSACTION_COST    = 0.0005      # 0.05% as in paper
LOOKBACK_WINDOW     = 60          # days of history in covariance matrix


# PPO AGENT HYPERPARAMETERS
# (Best found by Bayesian Optimization — tune via hyperopt)
PPO_CONFIG = {
    "learning_rate":       3e-4,
    "n_steps":             256,      # steps before each update
    "batch_size":          64,
    "n_epochs":            10,       # PPO epochs
    "gamma":               0.99,     # discount factor
    "gae_lambda":          0.95,     # GAE lambda
    "clip_range":          0.2,      # ε-clip (paper uses 0.2)
    "ent_coef":            0.01,     # entropy coefficient
    "vf_coef":             0.5,      # value function coefficient
    "max_grad_norm":       0.5,
    "policy_kwargs": {
        "net_arch": [256, 128],      # Actor + Critic hidden layers
        "activation_fn": "relu",
    },
    "total_timesteps":     500_000,  # 500 episodes equivalent
    "verbose":             1,
}

# DSR REWARD PARAMETERS
DSR_ETA = 1.0 / 252   # adaptation rate ≈ 1/trading_days_per_year


# GROUND TRUTH (Supervised Pre-training)
GT_CONSTANT_C = 3     # c ∈ {1,2,3,4,5} from paper eq (1)

# FUSION MODULE (Transformer — handled by Partner 3)

FUSION_CONFIG = {
    "d_model":           128,   # larger — now making own market predictions
    "nhead":             4,     # attention heads
    "num_layers":        3,     # deeper = more reasoning capacity
    "dim_feedforward":   256,
    "dropout":           0.1,
    "top_k":             5,     # top-K assets for concentration signal
    "use_residual_gate": True,  # blend transformer + agents during early training
    "lr":                5e-4,  # slightly lower LR for complex model
    "epochs":            150,   # more epochs for richer architecture
    "batch_size":        32,
    "weight_decay":      1e-4,  # L2 regularization
}

# BAYESIAN OPTIMIZATION SEARCH SPACE
from hyperopt import hp
HYPEROPT_SPACE = {
    "learning_rate": hp.loguniform("learning_rate", -8 * 2.303, -1 * 2.303),
    "gamma":         hp.uniform("gamma", 0.8, 0.999),
    "n_epochs":      hp.quniform("n_epochs", 5, 50, 1),
    "ent_coef":      hp.uniform("ent_coef", 0.01, 0.1),
    "vf_coef":       hp.uniform("vf_coef", 0.5, 1.0),
    "net_arch_size": hp.choice("net_arch_size", [[64, 64], [128, 128], [256, 128], [256, 256]]),
}
HYPEROPT_MAX_EVALS = 50


# BENCHMARKS

RISK_FREE_RATE  = 0.0525   # approximate US risk-free rate for Sharpe calc
MARKET_INDEX    = "^DJI"   # Dow Jones Industrial Average
OMEGA_THRESHOLD = 0.0      # minimum acceptable return for Omega ratio


# RANDOM SEED
SEED = 42
