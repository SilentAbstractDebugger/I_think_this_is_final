"""
This is full backtesting pipeline for RA-DRL.

This code does : 
  1. Loads trained PPO agents (log_return, dsr, mdd)
  2. Loads trained Transformer fusion module
  3. Run all agents on test data (2021-01-01 to 2024-03-31)
  4. Get fusion module's final portfolio weights for each day
  5. Simulate trading with 0.05% transaction cost
  6. Run benchmark strategies (MVO, 1/N, Market Index)
  7. Compute all 8 metrics for every strategy
  8. Prints results table and generate plots
  9. Runs paired t-test for statistical significance
  10. Saves all results to results/

To RUN do: python backtest.py
"""

import os
import sys
sys.path.append(os.path.abspath("."))

import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from data.feature_engineering import load_features
from agents.ppo_agent import PPOPortfolioAgent
from envs.portfolio_env import PortfolioEnv
from fusion.transformer_fusion import TransformerFusionModule, FusionInference
from benchmarks.baselines import run_all_benchmarks, simulate_portfolio
from utils.metrics import compare_strategies, paired_t_test
from utils.visualization import plot_all
from config import (
    MODEL_DIR, RESULT_DIR, FEAT_DIR,
    INITIAL_CAPITAL, TRANSACTION_COST,
    TEST_START, TEST_END, FUSION_CONFIG
)


def load_all_agents(state_df, close_df, tickers):
    """Load all 3 trained PPO agents from disk."""
    agents = {}
    reward_types = ["log_return", "dsr", "mdd"]

    for rt in reward_types:
        env = PortfolioEnv(
            state_df=state_df,
            close_df=close_df,
            reward_type=rt,
            mode="test",
        )
        agent = PPOPortfolioAgent(rt, env)
        agent.load()
        agents[rt] = agent
        print(f"Successfully Loaded agent: {rt}")

    return agents


def generate_test_actions(agents, state_df, close_df):
    """
    Run all trained agents on the test set and collect their daily weight decisions.
    Returns:
        stacked: (T, 3, N) numpy array —> T days, 3 agents, N assets
        dates:   list of T dates
    """
    print("\n Generating test period agent actions...")

    # Checking if already saved
    save_path = os.path.join(FEAT_DIR, "agent_actions_test", "stacked_actions.npy")
    dates_path = os.path.join(FEAT_DIR, "agent_actions_test", "dates.csv")

    if os.path.exists(save_path):
        print(" Loading pre-saved test actions...")
        stacked = np.load(save_path)
        dates   = pd.read_csv(dates_path, parse_dates=["date"])["date"].tolist()
        print(f"  Shape: {stacked.shape}")
        return stacked, dates

    actions_per_agent = {}
    for reward_type, agent in agents.items():
        env = PortfolioEnv(
            state_df=state_df,
            close_df=close_df,
            reward_type=reward_type,
            mode="test",
        )

        obs, _ = env.reset()
        done = False
        step = 0
        agent_actions = []
        dates_agent   = []

        while not done:
            date    = env.dates[min(step, len(env.dates) - 1)]
            weights = agent.get_action(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(weights)
            done    = terminated or truncated

            agent_actions.append(weights)
            dates_agent.append(date)
            step += 1

        actions_per_agent[reward_type] = np.array(agent_actions)   # (T, N)
        print(f" {reward_type}: {actions_per_agent[reward_type].shape}")

    # Stack: (T, 3, N)
    agent_order = ["log_return", "dsr", "mdd"]
    min_T   = min(len(actions_per_agent[r]) for r in agent_order)
    stacked = np.stack([actions_per_agent[r][:min_T] for r in agent_order], axis=1)
    dates   = dates_agent[:min_T]

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, stacked)
    pd.Series(dates, name="date").to_csv(dates_path, index=False)

    print(f"\n Stacked actions: {stacked.shape} (T, 3 agents, N assets)")
    return stacked, dates


def run_fusion_inference(stacked_actions, dates, tickers, n_assets):
    """
    Run Market-Predictive Transformer on test set agent actions.
    Also saves diagnostic info which is: gate_alpha, disagreement patterns.
    """
    print("\n Running Market-Predictive Transformer Fusion...")

    ckpt_path = os.path.join(MODEL_DIR, "transformer_fusion_pretrained.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(MODEL_DIR, "cnn_fusion_pretrained.pt")

    if not os.path.exists(ckpt_path):
        print(" No fusion model found! Run fusion/supervised_pretraining.py first.")
        N = n_assets
        weights = np.tile(np.ones(N) / N, (len(dates), 1))
        return pd.DataFrame(weights, index=dates, columns=tickers)

    # Load model
    checkpoint = torch.load(ckpt_path, map_location="cpu")
  
    # Local import to ensure the class is available
    from fusion.transformer_fusion import TransformerFusionModule, FusionInference
    
    model = TransformerFusionModule(n_assets=n_assets, **FUSION_CONFIG)
    model.load_state_dict(checkpoint["model_state_dict"])
    fusion = FusionInference(model)

    # Getting weights and diagnostics for every date
    print(f"  Running inference on {len(dates)} test dates...")
    weights_list    = []
    gate_alphas     = []
    disagree_norms  = []

    batch_size = 64
    for i in range(0, len(dates), batch_size):
        batch = stacked_actions[i:i+batch_size]
        diag  = fusion.predict_with_diagnostics(batch)
        
        # 1. Collect Portfolio Weights
        weights_list.append(diag["final_weights"])
        
        # 2. Collect Gate Alphas (ensuring they are flat to avoid shape errors)
        alpha_val = diag.get("gate_alpha", 1.0)
        if isinstance(alpha_val, (np.ndarray, torch.Tensor)):
            gate_alphas.extend(alpha_val.flatten())
        elif isinstance(alpha_val, list):
            gate_alphas.extend(alpha_val)
        else:
            # If it's a single scalar for the whole batch, repeat it
            gate_alphas.extend([alpha_val] * len(batch))

        # 3. Disagreement magnitude
        if "disagree_emb" in diag:
            d_norm = np.linalg.norm(diag["disagree_emb"], axis=-1)
            disagree_norms.extend(d_norm.flatten())

    # Concatenating all batches into one (T, N) matrix
    weights = np.concatenate(weights_list, axis=0)
    weights_df = pd.DataFrame(weights, index=dates, columns=tickers)

    # Calculate the means safely
    avg_alpha = np.mean(gate_alphas) if gate_alphas else 1.0
    avg_disagree = np.mean(disagree_norms) if disagree_norms else 0.0

    print(f" Fusion complete")
    print(f"      Learned gate α = {avg_alpha:.4f}  "
          f"→ {avg_alpha*100:.1f}% Transformer, {(1-avg_alpha)*100:.1f}% agent blend")
    print(f"      Weight sum check (should be ~1): {weights_df.sum(axis=1).mean():.6f}")

    # Saving diagnostics
    diag_df = pd.DataFrame({
        "date":          dates,
        "gate_alpha":    avg_alpha,
        "disagree_norm": avg_disagree,
    })
    diag_df.to_csv(os.path.join(RESULT_DIR, "fusion_diagnostics.csv"), index=False)

    return weights_df


def run_base_agent_portfolios(agents, state_df, close_df, tickers):
    """
    Simulate portfolio performance of each individual PPO agent.
    This is used for comparison in results table.
    """
    print("\n Simulating base agent portfolios...")

    test_close = close_df[close_df.index >= TEST_START]
    agent_portfolios = {}

    for reward_type, agent in agents.items():
        env = PortfolioEnv(
            state_df=state_df,
            close_df=close_df,
            reward_type=reward_type,
            mode="test",
        )

        obs, _ = env.reset()
        done   = False
        step   = 0
        weights_dict = {}

        while not done:
            date    = env.dates[min(step, len(env.dates) - 1)]
            weights = agent.get_action(obs, deterministic=True)
            weights_dict[date] = weights
            obs, _, terminated, truncated, _ = env.step(weights)
            done = terminated or truncated
            step += 1

        # Simulate with transaction cost
        wdf = pd.DataFrame(weights_dict).T
        wdf.columns = tickers

        portfolio = simulate_portfolio(
            close_df=test_close,
            weights_df=wdf,
            initial_capital=INITIAL_CAPITAL,
            transaction_cost=TRANSACTION_COST,
        )
        portfolio.name = f"PPO-{reward_type}"
        agent_portfolios[reward_type] = portfolio
        print(f" {reward_type}: final ${portfolio.iloc[-1]:,.0f}")

    return agent_portfolios


def run_backtest():
    """Master Backtest Function -> This runs everything"""

    print("=" * 70)
    print("  RA-DRL FULL BACKTEST")
    print(f" The test period is: {TEST_START} → {TEST_END}")
    print("=" * 70)

    # Load features
    print("\n Loading features...")
    state_df, close_df, cov_array, tickers = load_features()
    n_assets  = len(tickers)
    test_close = close_df[close_df.index >= TEST_START]
    print(f"  Assets: {n_assets}")
    print(f"  Test dates: {test_close.index[0].date()} → {test_close.index[-1].date()}")

    # Load trained PPO agents
    print("\n Loading PPO agents...")
    agents = load_all_agents(state_df, close_df, tickers)

    # Generating test actions for all agents
    stacked_actions, action_dates = generate_test_actions(agents, state_df, close_df)

    # Fusion inference → RA-DRL weights
    ra_drl_weights = run_fusion_inference(stacked_actions, action_dates, tickers, n_assets)

    # Simulating RA-DRL portfolio
    print("\n Simulating RA-DRL portfolio...")
    ra_drl_portfolio = simulate_portfolio(
        close_df=test_close,
        weights_df=ra_drl_weights,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
    )
    ra_drl_portfolio.name = "RA-DRL"
    print(f" RA-DRL final value: ${ra_drl_portfolio.iloc[-1]:,.0f}")

    # Run base agent portfolios
    agent_portfolios = run_base_agent_portfolios(agents, state_df, close_df, tickers)

    # Run benchmarks
    print("\n Running benchmarks...")
    benchmarks = run_all_benchmarks(test_close)

    # Combine all strategies
    all_strategies = {"RA-DRL": ra_drl_portfolio}
    for rt, p in agent_portfolios.items():
        all_strategies[f"PPO-{rt}"] = p
    all_strategies.update(benchmarks)

    # Computing all metrics
    print("\n Computing performance metrics...")
    metrics_df = compare_strategies(all_strategies)

    print("\n" + "=" * 70)
    print("  RESULTS TABLE")
    print("=" * 70)
    print(metrics_df.to_string())

    # Proper statistical significance tests
    print("\n Running proper statistical significance tests...")
    from utils.statistical_tests import run_all_significance_tests
    sig_results = run_all_significance_tests(
        strategies    = all_strategies,
        ra_drl_name   = "RA-DRL",
        rf            = 0.0525,
        print_results = True,
    )
    sig_results.to_csv(os.path.join(RESULT_DIR, "significance_tests.csv"), index=False)

    # Save results
    print(f"\n Saving the results to {RESULT_DIR}/")
    metrics_df.to_csv(os.path.join(RESULT_DIR, "metrics_comparison.csv"))

    portfolio_df = pd.DataFrame({k: v for k, v in all_strategies.items() if v is not None})
    portfolio_df.to_csv(os.path.join(RESULT_DIR, "portfolio_values.csv"))

    ra_drl_weights.to_csv(os.path.join(RESULT_DIR, "ra_drl_weights.csv"))

    # Generating plots
    print("\n Generating visualization plots...")
    plot_all(all_strategies, metrics_df)

    print("\n" + "=" * 70)
    print(" SUCCESSFULLY COMPLETED BACKTEST!")
    print(f"  Results saved to: {RESULT_DIR}/")
    print("=" * 70)

    return metrics_df, all_strategies


if __name__ == "__main__":
    metrics_df, all_strategies = run_backtest()
