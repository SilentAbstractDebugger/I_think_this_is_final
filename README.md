# FusionRL-X
**Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization

Multi-Reward Fusion with Transformer-Based Decision Aggregation

# Overview
This is a course projesct that implements a Risk Adjusted Deep Reinforcement Learning (RA-DRL) framework for portfolio optimization using a multi-reward, multi-agent architecture.
Traditional RL-based portfolio strategies optimize a single objective, often leading to unstable or impractical investment behavior. This project addresses that limitation by:

1. Training three specialized PPO agents, each optimized for a distinct financial objective and
2. Combining their decisions using a Transformer-based fusion module and
3. Producing a single, risk-aware portfolio allocation strategy

---

## The Problem

Triaditional portfolio optimization methods, such as Mean-Variance Optimization (MVO), rely on strong assumptions like normally distributed returns, and linear relationships between assets. The  real-world financial markets, are highly dynamic, non-stationary, and influenced by complex, nonlinear interactions due to which these assumptions often fails.
Furthermore, single-agent DRL models are less robust in practical deployment.
Real markets demand a balance between growth, stability, and downside protection.

The goal is to develop a system that can:
1. Simultaneously optimize multiple financial objectives  
2. Adapt to changing market regimes  
3. Generate stable and risk-aware portfolio allocations  
4. Outperform traditional and single-agent baselines 

---

## Our Approach

Recent advances in Deep Reinforcement Learning (DRL) have enabled adaptive portfolio strategies by learning directly from market data. This leads to suboptimal behavior, as financial decision-making inherently involves balancing multiple conflicting objectives, including return maximization, risk control, and drawdown minimization.
We decompose the problem into three specialized learning objectives:

| Agent | Reward Function | Goal |
|:------|:----------------|:-----|
| Agent 1 |	Log Returns | Maximize growth |
| Agent 2 | Differential Sharpe Ratio (DSR) | Optimize risk-adjusted returns |
|Agent 3 | Maximum Drawdown (MDD) |	Minimize large losses |


Instead of choosing one, we are fusing all three decisions using a Transformer model, which enables:
1. Context aware weighting of strategies
2. Temporal dependency modeling
3. Adaptive risk balancing

This approach aims to bridge the gap between theoretical portfolio optimization and real-world financial decision-making.

---

## 📊 Workflow

![Workflow Diagram](RA_DRL_workflow.jpeg)

---

## Project Structure

```
ra_drl/
│
├-- data/
│   ├-- download_data.py
│   └-- feature_engineering.py
│
├-- envs/
│   └-- portfolio_env.py
│
├-- agents/
│   ├-- ppo_agent.py
│   └-- train_agents.py
│
├-- fusion/
│   ├-- supervised_pretraining.py
│   └-- transformer_fusion.py  
│
├-- utils/
│   ├-- rewards.py
│   ├-- metrics.py
│   └-- visualization.py
│
├-- benchmarks/
│   └-- baselines.py
│
├-- config.py
├-- train.py
├-- backtest.py
requirements.txt
```

---

## Dataset

**Dataset:** Dow 30 (Dow Jones Industrial Average constituents)  
**Source:** Yahoo Finance  
**Frequency:** Daily (OHLCV)  

###  Features (State Space)

**Covariance Matrix**  
- Captures relationships between asset returns  

**Technical Indicators**  
- SMA (30, 60)  
- MACD  
- RSI  
- ADX  
- CCI  
- Bollinger Bands  

---
      
## Model Components

### 1. Base RL Agent
- **Algorithm:** Proximal Policy Optimization (PPO)  
- **Architecture:** Actor-Critic Networks  

### 2. Reward Functions
- Log Return  
- Differential Sharpe Ratio (DSR)  
- Maximum Drawdown (MDD)  

### 3. Fusion Module
- **Architecture:** Transformer Encoder  
- **Input:** Actions from 3 PPO agents  
- **Output:** Final portfolio weights  

---
  
## Evaluation Metrics

| Metric | Description |
|:-------|:------------|
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk focus |
| Calmar Ratio | Return vs drawdown |
| Omega Ratio |	Gain/loss distribution |
| Annual Return	| Yearly performance |
| Max Drawdown | Worst loss |
| Volatility | Risk level |

---
   
## How to Run

Run everything from project root.

### Install dependencies
```
    pip install -r requirements.txt
```

### Download Data
```
    python data/download_data.py
```

### Train PPO Agents
```
    python agents/train_agents.py
```

### Supervised Pre-training
```
    python fusion/supervised_pretraining.py
```

### Train Transformer Fusion
```
    python train.py
```

### Backtesting
```
    python backtest.py
```

### Visualization
```
    python utils/visualization.py
```

---

##  Training Details

- **Transaction Cost:** 0.05% per trade  
- **Train/Test Split:** Time-based split to preserve temporal dependencies  
- **Hyperparameter Optimization:** Bayesian Optimization using Hyperopt
  
---

## Contributors
- Diksha Agrawal
- Aastha Sharma
- Gaurav Kumar

---

## Expected Outcomes

- Robust portfolio performance with enhanced risk-adjusted returns  
- Reduced drawdowns compared to single-objective reinforcement learning approaches  
- Consistent performance across different market regimes:
  - Bull markets  
  - Bear markets  
  - Sideways markets
 
--- 
      
# References
 1. Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization: A Multi-reward Approach (2025)
 2. Deep Reinforcement Learning for Portfolio Selection

---

## Results 

| Model            | Sharpe Ratio | Return | Max Drawdown |
|------------------|-------------|--------|--------------|
| RA-DRL (Ours)    | TBD         | TBD    | TBD          |
| PPO (Single)     | TBD         | TBD    | TBD          |
| MVO              | TBD         | TBD    | TBD          |
| Equal Weight (1/N) | TBD       | TBD    | TBD          |

---

##  Future Work

- This model can be extended to multiple markets (e.g., Sensex, NASDAQ)  
- This enables real time trading and deployment  
- Incorporation of risk-aware transformer architectures with attention constraints  

---

##  License

MIT License.

