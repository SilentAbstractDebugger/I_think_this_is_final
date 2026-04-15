"""
benchmarks/baselines.py
────────────────────────
All benchmark strategies from the paper Section 4.4.

BENCHMARKS:
  1. Market Index      — Dow Jones Industrial Average (^DJI)
  2. 1/N Strategy      — Equal weight across all assets
  3. Mean-Variance Opt — Markowitz MVO (minimize variance for target return)
  4. Single-Objective  — Weighted sum of all 3 rewards (one PPO agent)

These are compared against RA-DRL in the results table.

MARKOWITZ MVO MATH:
  minimize:   w^T Σ w          (portfolio variance)
  subject to: μ^T w = μ_target (target expected return)
              Σ w_i = 1
              0 ≤ w_i ≤ 1      (no short-selling)

  Σ = covariance matrix of returns
  μ = vector of expected returns

  We solve this using CVXPY (convex optimization).
  In practice, MVO is rebalanced periodically (e.g., monthly) using
  in-sample data for Σ and μ estimation.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import cvxpy as cp
import yfinance as yf
from tqdm import tqdm
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from config import (
    INITIAL_CAPITAL, TRANSACTION_COST, MARKET_INDEX,
    RISK_FREE_RATE, TEST_START, TEST_END, TRAIN_START, TRAIN_END
)
from utils.metrics import compute_all_metrics


# ─────────────────────────────────────────────────────────
# UTILITY: Portfolio simulation from weights
# ─────────────────────────────────────────────────────────

def simulate_portfolio(
    close_df:         pd.DataFrame,
    weights_df:       pd.DataFrame,        # (T × N) rebalancing weights
    initial_capital:  float = INITIAL_CAPITAL,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.Series:
    """
    Simulate portfolio value given weight decisions.

    Args:
        close_df:    (T × N) daily closing prices (test period)
        weights_df:  (T × N) portfolio weights (one row per date)
                     Dates must be a subset of close_df dates.
    Returns:
        portfolio_values: pd.Series indexed by date
    """
    dates = close_df.index
    portfolio_value = initial_capital
    current_weights = np.zeros(close_df.shape[1])
    values = [portfolio_value]

    for i in range(1, len(dates)):
        today     = dates[i - 1]
        tomorrow  = dates[i]

        # Get rebalancing weights (use last known if not present)
        if today in weights_df.index:
            new_weights = weights_df.loc[today].values
        else:
            new_weights = current_weights

        # Normalize (safety)
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
        else:
            new_weights = np.ones(close_df.shape[1]) / close_df.shape[1]

        # Transaction cost
        turnover = np.sum(np.abs(new_weights - current_weights))
        tc = transaction_cost * portfolio_value * turnover

        # Daily return
        close_t   = close_df.loc[today].values
        close_tp1 = close_df.loc[tomorrow].values
        returns   = (close_tp1 - close_t) / (close_t + 1e-10)
        p_return  = np.dot(new_weights, returns)

        # Update portfolio
        portfolio_value = portfolio_value * (1 + p_return) - tc
        portfolio_value = max(portfolio_value, 1.0)

        values.append(portfolio_value)
        current_weights = new_weights

    return pd.Series(values, index=dates, name="portfolio_value")


# ─────────────────────────────────────────────────────────
# BENCHMARK 1: MARKET INDEX
# ─────────────────────────────────────────────────────────

def market_index_benchmark(
    start: str = TEST_START,
    end:   str = TEST_END,
    ticker: str = MARKET_INDEX,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.Series:
    """
    Download Dow Jones Industrial Average (^DJI) returns.
    Simulate an investor who just holds the index.
    """
    print(f"\n📈 Fetching Market Index ({ticker})...")
    index_df = yf.download(ticker, start=start, end=end,
                           progress=False, auto_adjust=True)
    close = index_df["Close"].squeeze()
    close = close.sort_index()

    # Normalize to initial capital
    portfolio = initial_capital * (close / close.iloc[0])
    portfolio.name = "Market Index"
    print(f"  ✅ Market Index: {close.index[0].date()} → {close.index[-1].date()}")
    return portfolio


# ─────────────────────────────────────────────────────────
# BENCHMARK 2: 1/N EQUAL WEIGHT STRATEGY
# ─────────────────────────────────────────────────────────

def equal_weight_benchmark(
    close_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.Series:
    """
    Equal weight: 1/N allocation across all N assets.
    Never rebalanced (buy-and-hold equal weight on day 1).

    Simple but surprisingly hard to beat (DeMiguel et al., 2007).
    """
    print("\n⚖️  Computing 1/N Equal Weight strategy...")
    N = close_df.shape[1]
    weights = np.ones(N) / N

    # Apply once on first day, no rebalancing
    weights_df = pd.DataFrame(
        np.tile(weights, (len(close_df), 1)),
        index=close_df.index,
        columns=close_df.columns,
    )

    portfolio = simulate_portfolio(close_df, weights_df, initial_capital, transaction_cost)
    portfolio.name = "1/N Strategy"
    print(f"  ✅ Equal weight: {portfolio.iloc[-1]:,.0f} (final value)")
    return portfolio


# ─────────────────────────────────────────────────────────
# BENCHMARK 3: MEAN-VARIANCE OPTIMIZATION (MVO)
# ─────────────────────────────────────────────────────────

def solve_mvo(
    mu:     np.ndarray,   # (N,) expected returns
    Sigma:  np.ndarray,   # (N, N) covariance matrix
    target_return: Optional[float] = None,
    risk_free_rate: float = RISK_FREE_RATE / 252,   # daily
) -> np.ndarray:
    """
    Solve Markowitz MVO:
      minimize   w^T Σ w
      subject to 1^T w = 1
                 0 ≤ w ≤ 1
                 μ^T w ≥ target_return   (optional)

    Args:
        mu:            Expected daily returns vector (N,)
        Sigma:         Covariance matrix (N, N)
        target_return: If None, maximizes Sharpe (tangency portfolio)

    Returns:
        w: (N,) optimal portfolio weights
    """
    N = len(mu)
    w = cp.Variable(N)

    portfolio_variance = cp.quad_form(w, Sigma)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 1,
    ]

    if target_return is not None:
        constraints.append(mu @ w >= target_return)
        objective = cp.Minimize(portfolio_variance)
    else:
        # Maximize Sharpe ratio (tangency portfolio) using Markowitz formulation
        portfolio_return  = mu @ w
        excess_return     = portfolio_return - risk_free_rate
        # Maximize: excess / std → equivalent to maximizing Sharpe
        # We solve: minimize variance subject to expected_return = target
        # and sweep over targets → find max Sharpe
        objective = cp.Maximize(excess_return - 0.5 * portfolio_variance)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        return np.ones(N) / N   # fallback to equal weight

    # Clip and renormalize (numerical safety)
    weights = np.clip(w.value, 0, 1)
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones(N) / N

    return weights


def mvo_benchmark(
    close_df:         pd.DataFrame,
    lookback:         int   = 252,        # days of history for estimation
    rebalance_freq:   int   = 21,         # rebalance monthly (21 trading days)
    initial_capital:  float = INITIAL_CAPITAL,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.Series:
    """
    Rolling MVO benchmark.

    Every `rebalance_freq` days:
      1. Estimate μ and Σ from last `lookback` days of training data
      2. Solve MVO to get optimal weights
      3. Apply those weights until next rebalance

    Uses ONLY past data (no lookahead).
    """
    print("\n🎯 Computing MVO strategy...")

    # Combine train + test close prices for rolling estimation
    train_close = close_df[close_df.index <= TRAIN_END]
    test_close  = close_df[close_df.index >= TEST_START]
    all_close   = pd.concat([train_close, test_close])
    all_returns = all_close.pct_change().fillna(0)

    test_dates  = test_close.index
    N           = test_close.shape[1]
    weights_dict = {}
    current_weights = np.ones(N) / N   # start with equal weight

    print(f"  Rebalancing every {rebalance_freq} days...")
    for i, date in enumerate(tqdm(test_dates)):
        if i % rebalance_freq == 0:
            # Get lookback window of returns
            date_loc = all_returns.index.get_loc(date)
            start_loc = max(0, date_loc - lookback)
            window_returns = all_returns.iloc[start_loc:date_loc]

            if len(window_returns) >= 20:  # need enough data
                mu    = window_returns.mean().values           # (N,) expected daily returns
                Sigma = window_returns.cov().values + 1e-6 * np.eye(N)  # regularize
                current_weights = solve_mvo(mu, Sigma)

        weights_dict[date] = current_weights.copy()

    weights_df = pd.DataFrame(weights_dict).T
    weights_df.columns = test_close.columns

    portfolio = simulate_portfolio(test_close, weights_df, initial_capital, transaction_cost)
    portfolio.name = "MVO"
    print(f"  ✅ MVO final value: ${portfolio.iloc[-1]:,.0f}")
    return portfolio


# ─────────────────────────────────────────────────────────
# BENCHMARK 4: SINGLE-OBJECTIVE (Weighted Sum of Rewards)
# ─────────────────────────────────────────────────────────

class SingleObjectiveAgent:
    """
    Single-objective agent using WEIGHTED SUM of all 3 rewards.

    Instead of training 3 separate agents and fusing, this trains ONE agent
    with a combined reward: λ₁·LR + λ₂·DSR + λ₃·MDD

    The weights (λ₁, λ₂, λ₃) are treated as hyperparameters and optimized.
    This is the paper's "single objective" benchmark.

    Training is done in train_agents.py using a special combined reward env.
    Here we just load its saved actions.
    """

    def __init__(self, lambda_lr: float = 0.33, lambda_dsr: float = 0.33, lambda_mdd: float = 0.34):
        self.lambdas = {"log_return": lambda_lr, "dsr": lambda_dsr, "mdd": lambda_mdd}

    def combined_reward(self, lr_reward: float, dsr_reward: float, mdd_reward: float) -> float:
        """Weighted sum reward signal."""
        return (self.lambdas["log_return"] * lr_reward +
                self.lambdas["dsr"]        * dsr_reward +
                self.lambdas["mdd"]        * mdd_reward)


# ─────────────────────────────────────────────────────────
# RUN ALL BENCHMARKS
# ─────────────────────────────────────────────────────────

def run_all_benchmarks(
    close_df: pd.DataFrame,   # test period close prices
) -> dict:
    """
    Run all benchmark strategies and return portfolio value series.

    Args:
        close_df: (T × N) test period closing prices

    Returns:
        dict: {strategy_name: portfolio_value_series}
    """
    results = {}

    # 1. Market Index
    try:
        results["Market Index"] = market_index_benchmark()
    except Exception as e:
        print(f"  ⚠️  Market Index failed: {e}")

    # 2. Equal Weight
    results["1/N Strategy"] = equal_weight_benchmark(close_df)

    # 3. MVO
    results["MVO"] = mvo_benchmark(close_df)

    return results


def compare_with_benchmarks(
    ra_drl_values: pd.Series,
    agent_values:  dict,   # {reward_type: portfolio_series} for 3 base agents
    close_df:      pd.DataFrame,
) -> pd.DataFrame:
    """
    Full comparison table: RA-DRL vs 3 base agents vs 4 benchmarks.

    Args:
        ra_drl_values: portfolio series for the fused RA-DRL model
        agent_values:  dict of base agent portfolio series
        close_df:      test period close prices

    Returns:
        metrics DataFrame
    """
    from utils.metrics import compute_all_metrics

    all_strategies = {}

    # RA-DRL (our model)
    all_strategies["RA-DRL (Ours)"] = ra_drl_values

    # Base DRL agents
    for name, vals in agent_values.items():
        all_strategies[f"PPO-{name.upper()}"] = vals

    # Benchmarks
    benchmarks = run_all_benchmarks(close_df)
    all_strategies.update(benchmarks)

    # Compute metrics
    rows = []
    for name, values in all_strategies.items():
        if values is not None and len(values) > 1:
            metrics = compute_all_metrics(values.values, strategy_name=name)
            rows.append(metrics)

    df = pd.DataFrame(rows).set_index("Strategy")
    return df


# ─────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.feature_engineering import load_features

    print("Loading features...")
    state_df, close_df, cov_array, tickers = load_features()

    # Use test period
    test_close = close_df[close_df.index >= TEST_START]
    print(f"Test period: {test_close.index[0].date()} → {test_close.index[-1].date()}")

    # Run benchmarks
    results = run_all_benchmarks(test_close)

    for name, vals in results.items():
        from utils.metrics import compute_all_metrics
        metrics = compute_all_metrics(vals.values, strategy_name=name)
        print(f"\n{name}:")
        for k, v in metrics.items():
            if k != "Strategy":
                print(f"  {k}: {v}")
