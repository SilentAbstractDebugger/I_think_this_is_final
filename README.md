# RA-DRL: Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization
### Full Implementation — Dow 30 Dataset

> Based on: *Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization: A Multi-reward Approach (2025)*
> Modified: CNN Fusion → **Transformer Fusion** (your team's enhancement)

---

## 🏗️ Project Architecture

```
ra_drl/
├── data/
│   ├── download_data.py       ← Downloads Dow 30 from Yahoo Finance
│   └── feature_engineering.py ← Computes 8 technical indicators + covariance matrix
│
├── envs/
│   └── portfolio_env.py       ← Custom RL Environment (MDP: State, Action, Reward)
│
├── agents/
│   ├── ppo_agent.py           ← PPO Actor-Critic base agent
│   └── train_agents.py        ← Trains 3 PPO agents (LogReturn, DSR, MDD)
│
├── fusion/
│   ├── supervised_pretraining.py ← Ground-truth weight generation + pre-training
│   └── transformer_fusion.py  ← (Partner handles) Transformer-based fusion module
│
├── utils/
│   ├── rewards.py             ← Log Return, DSR, MDD reward functions
│   ├── metrics.py             ← Sharpe, Sortino, Calmar, Omega, Stability
│   └── visualization.py      ← Cumulative wealth plots, metric tables
│
├── benchmarks/
│   └── baselines.py           ← MVO, 1/N, Market Index, Single-Objective
│
├── train.py                   ← MAIN TRAINING SCRIPT (YOU run this)
├── backtest.py                ← MAIN BACKTESTING SCRIPT
├── config.py                  ← All hyperparameters in one place
└── requirements.txt
```

---

## 👥 Team Division

| Task | Owner |
|---|---|
| Data Download + Feature Engineering | **You two** |
| RL Environment (MDP) | **You two** |
| PPO Agent Implementation | **You two** |
| Reward Functions (LR, DSR, MDD) | **You two** |
| Supervised Pre-training (Ground Truth) | **You two** |
| Benchmarks (MVO, 1/N) | **You two** |
| Metrics + Visualization | **You two** |
| Transformer Fusion Module | **Partner 3** |
| Hyperparameter Tuning (Bayesian Opt) | **All together** |
| Backtesting + Final Report | **All together** |

---

## 🔁 Full Pipeline Flow

```
1. Download Dow 30 daily OHLCV data (Yahoo Finance)
         ↓
2. Compute 8 Technical Indicators + Covariance Matrix → STATE SPACE
         ↓
3. Build RL Environment (gym-compatible)
         ↓
4. Train PPO Agent 1 → Reward = Log Return
   Train PPO Agent 2 → Reward = Differential Sharpe Ratio
   Train PPO Agent 3 → Reward = Maximum Drawdown penalty
         ↓
5. Compute Ground-Truth weights from historical % changes → Supervised Pre-training
         ↓
6. Stack actions of 3 agents → Feed to Transformer Fusion (Partner 3)
         ↓
7. Transformer outputs final portfolio weights
         ↓
8. Backtest on 2021–2024 with 0.05% transaction cost
         ↓
9. Compute Sharpe, Sortino, Calmar, Omega, Stability, Annual Return, Cumulative Return
         ↓
10. Compare vs MVO, 1/N, Market Index, Single-Objective agent
```

---

## ⚡ Quick Start

```bash
pip install -r requirements.txt

# Step 1: Download data
python data/download_data.py

# Step 2: Train 3 PPO agents
python agents/train_agents.py

# Step 3: Pre-train fusion module (supervised)
python fusion/supervised_pretraining.py

# Step 4: Full backtest
python backtest.py

# Step 5: Generate all plots and metrics
python utils/visualization.py
```