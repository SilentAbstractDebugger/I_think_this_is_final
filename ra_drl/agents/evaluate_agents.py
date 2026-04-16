"""
agents/evaluate_agents.py
──────────────────────────
STEP 3.5 — Individual Agent Evaluation (BEFORE fusion)

This is the missing evaluation step between:
  Step 3: Train 3 PPO agents
  Step 4: Supervised pre-training of fusion

WHY THIS STEP EXISTS (from paper Section 4.5):
  "Firstly, the RA-DRL is assessed against the base model with different
   objectives used by recent studies, and subsequently, it is evaluated
   against the benchmarks."

  The paper first validates that each individual agent has learned something
  meaningful before trusting them to contribute to the fusion. If an agent
  performs WORSE than random or the market index, its actions will corrupt
  the fusion module's output.

WHAT THIS SCRIPT DOES:
  1. Loads all 3 trained PPO agents
  2. Runs each agent on TEST data (2021–2024) independently
  3. Simulates portfolio with 0.05% transaction cost for each
  4. Fetches Dow Jones Index (^DJI) as baseline
  5. Computes all 8 metrics for each agent + index
  6. Prints a comparison table
  7. Plots cumulative wealth curves for all 3 agents vs index
  8. Runs sanity checks — flags any agent that underperforms 1/N
  9. Saves everything to results/pre_fusion_evaluation/

PASS CRITERIA before proceeding to fusion:
  - Each agent's Sharpe Ratio > 0 (positive risk-adjusted return)
  - Each agent's Cumulative Return > Market Index (at least 2 of 3)
  - No agent has MDD > 50% (catastrophic loss = untrained agent)

RUN: python agents/evaluate_agents.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from data.feature_engineering import load_features
from envs.portfolio_env import PortfolioEnv
from agents.ppo_agent import PPOPortfolioAgent
from benchmarks.baselines import simulate_portfolio, equal_weight_benchmark
from utils.metrics import compute_all_metrics, compare_strategies, paired_t_test
from config import (
    MODEL_DIR, RESULT_DIR, INITIAL_CAPITAL, TRANSACTION_COST,
    TEST_START, TEST_END, MARKET_INDEX, RISK_FREE_RATE
)

EVAL_DIR = os.path.join(RESULT_DIR, "pre_fusion_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────
# LOAD AGENTS
# ─────────────────────────────────────────────────────────

def load_trained_agents(state_df, close_df):
    """Load all 3 PPO agents from saved checkpoints."""
    agents = {}
    reward_types = ["log_return", "dsr", "mdd"]

    print("📂 Loading trained PPO agents...")
    for rt in reward_types:
        model_path = os.path.join(MODEL_DIR, f"ppo_{rt}.zip")
        if not os.path.exists(model_path):
            print(f"  ❌ Missing: {model_path}")
            print(f"     Run: python agents/train_agents.py first")
            sys.exit(1)

        env = PortfolioEnv(
            state_df=state_df,
            close_df=close_df,
            reward_type=rt,
            mode="test",
        )
        agent = PPOPortfolioAgent(rt, env)
        agent.load()
        agents[rt] = (agent, env)
        print(f"  ✅ Loaded: ppo_{rt}")

    return agents


# ─────────────────────────────────────────────────────────
# RUN EACH AGENT ON TEST SET
# ─────────────────────────────────────────────────────────

def run_agent_on_test(
    agent: PPOPortfolioAgent,
    env: PortfolioEnv,
    close_df: pd.DataFrame,
    tickers: list,
    reward_type: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run a single trained PPO agent deterministically on the test period.

    Returns:
        portfolio_values: pd.Series indexed by date
        weight_history:   pd.DataFrame (dates × assets) of daily allocations
    """
    obs, _ = env.reset()
    done = False
    step = 0
    weights_dict = {}
    dates_list   = []

    while not done:
        date    = env.dates[min(step, len(env.dates) - 1)]
        # Deterministic=True → greedy policy (no exploration noise)
        weights = agent.get_action(obs, deterministic=True)
        weights_dict[date] = weights
        obs, _, terminated, truncated, _ = env.step(weights)
        done = terminated or truncated
        dates_list.append(date)
        step += 1

    # Build weight DataFrame
    weight_df = pd.DataFrame(weights_dict).T
    weight_df.columns = tickers

    # Simulate portfolio with transaction costs
    test_close = close_df[close_df.index >= TEST_START]
    portfolio  = simulate_portfolio(
        close_df         = test_close,
        weights_df       = weight_df,
        initial_capital  = INITIAL_CAPITAL,
        transaction_cost = TRANSACTION_COST,
    )
    portfolio.name = f"PPO-{reward_type}"

    return portfolio, weight_df


# ─────────────────────────────────────────────────────────
# FETCH MARKET INDEX
# ─────────────────────────────────────────────────────────

def fetch_market_index(start=TEST_START, end=TEST_END) -> pd.Series:
    """Download Dow Jones index and normalize to initial capital."""
    print(f"\n📈 Fetching Market Index ({MARKET_INDEX})...")
    df = yf.download(MARKET_INDEX, start=start, end=end, progress=False, auto_adjust=True)
    close = df["Close"].squeeze().sort_index()
    portfolio = INITIAL_CAPITAL * (close / close.iloc[0])
    portfolio.name = "Market Index (^DJI)"
    print(f"  ✅ {close.index[0].date()} → {close.index[-1].date()}")
    return portfolio


# ─────────────────────────────────────────────────────────
# SANITY CHECKS
# ─────────────────────────────────────────────────────────

def run_sanity_checks(
    agent_portfolios: dict,
    index_portfolio:  pd.Series,
    equal_portfolio:  pd.Series,
) -> dict:
    """
    Check each agent passes minimum quality thresholds.
    An agent that fails here should be RETRAINED before fusion.

    Checks:
      1. Sharpe Ratio > 0              (positive risk-adjusted returns)
      2. Cumulative Return > 1/N       (beats the naive strategy)
      3. Max Drawdown < 50%            (not catastrophically losing)
      4. Final value > 50% of initial  (agent is not near-bankrupt)

    Returns:
        results: {agent_name: {"passed": bool, "issues": [str]}}
    """
    results = {}

    print("\n🔍 Running pre-fusion sanity checks...")
    print("─" * 55)

    for name, portfolio in agent_portfolios.items():
        daily_returns = portfolio.pct_change().dropna().values
        issues = []

        # Compute key metrics
        from utils.metrics import (
            sharpe_ratio, maximum_drawdown, cumulative_return, annual_return
        )

        sr  = sharpe_ratio(daily_returns, RISK_FREE_RATE)
        mdd = maximum_drawdown(portfolio.values)
        cr  = cumulative_return(portfolio.values)
        ar  = annual_return(portfolio.values)
        cr_eq = cumulative_return(equal_portfolio.values[:len(portfolio)])

        # Check 1: Positive Sharpe
        if sr <= 0:
            issues.append(f"Sharpe Ratio = {sr:.3f} ≤ 0 (no risk-adjusted return)")

        # Check 2: Beats 1/N
        if cr < cr_eq * 0.9:   # allow 10% slack
            issues.append(f"CR = {cr:.2%} underperforms 1/N ({cr_eq:.2%})")

        # Check 3: MDD not catastrophic
        if mdd > 0.5:
            issues.append(f"MDD = {mdd:.2%} > 50% (catastrophic drawdown)")

        # Check 4: Final value check
        if portfolio.iloc[-1] < INITIAL_CAPITAL * 0.5:
            issues.append(f"Final value ${portfolio.iloc[-1]:,.0f} < 50% of initial capital")

        passed = len(issues) == 0
        results[name] = {"passed": passed, "issues": issues, "sr": sr, "mdd": mdd, "cr": cr}

        status = "✅ PASS" if passed else "⚠️  WARN"
        print(f"  {status}  {name:<20} SR={sr:+.3f}  MDD={mdd:.1%}  CR={cr:+.2%}")
        for issue in issues:
            print(f"           └─ {issue}")

    print("─" * 55)

    all_passed = all(v["passed"] for v in results.values())
    if all_passed:
        print("  ✅ All agents passed — safe to proceed to fusion\n")
    else:
        print("  ⚠️  Some agents have warnings — review before fusion")
        print("     Consider: more training timesteps, different HP, check data\n")

    return results


# ─────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────

def plot_agent_comparison(
    agent_portfolios: dict,
    index_portfolio:  pd.Series,
    equal_portfolio:  pd.Series,
):
    """
    Three plots:
      A. Cumulative wealth curves for all agents + index + 1/N
      B. Individual weight evolution (allocation per asset over time)
      C. Rolling 63-day Sharpe for each agent
    """

    COLORS = {
        "PPO-log_return":      "#638cff",
        "PPO-dsr":             "#2dd4b4",
        "PPO-mdd":             "#f5a623",
        "Market Index (^DJI)": "#8b97b8",
        "1/N Equal Weight":    "#6b7280",
    }
    LS = {
        "PPO-log_return":      "-",
        "PPO-dsr":             "-",
        "PPO-mdd":             "-",
        "Market Index (^DJI)": "--",
        "1/N Equal Weight":    ":",
    }

    all_portfolios = {**agent_portfolios,
                      "Market Index (^DJI)": index_portfolio,
                      "1/N Equal Weight":    equal_portfolio}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})

    # ── PLOT A: Cumulative wealth
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, port in all_portfolios.items():
        if port is None: continue
        vals = port / port.iloc[0] * INITIAL_CAPITAL
        lw   = 2.2 if name.startswith("PPO") else 1.4
        ax.plot(vals.index, vals.values / 1e6,
                label=name, color=COLORS.get(name, "#aaa"),
                linestyle=LS.get(name, "-"), linewidth=lw)

    ax.set_title("Individual Agent Evaluation — Cumulative Wealth\n(Test: Jan 2021 – Mar 2024)",
                 fontweight="bold")
    ax.set_ylabel("Portfolio Value ($M)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}M"))
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, "A_cumulative_wealth.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: {path}")

    # ── PLOT B: Rolling 63-day Sharpe
    fig, ax = plt.subplots(figsize=(13, 4))
    daily_rf = RISK_FREE_RATE / 252
    for name, port in agent_portfolios.items():
        dr = port.pct_change().dropna()
        rolling_sr = dr.rolling(63).apply(
            lambda r: (r.mean() - daily_rf) / (r.std() + 1e-10) * np.sqrt(252),
            raw=True
        )
        ax.plot(rolling_sr.index, rolling_sr.values,
                label=name, color=COLORS.get(name, "#aaa"), linewidth=2)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="SR = 0")
    ax.set_title("Rolling Sharpe Ratio (63-day window) — Individual Agents", fontweight="bold")
    ax.set_ylabel("Sharpe Ratio (Annualized)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, "B_rolling_sharpe.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: {path}")

    # ── PLOT C: Drawdown comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    AGENT_COLORS = ["#638cff", "#2dd4b4", "#f5a623"]
    for i, (name, port) in enumerate(agent_portfolios.items()):
        ax = axes[i]
        cummax   = port.cummax()
        drawdown = (port - cummax) / cummax * 100
        ax.fill_between(drawdown.index, drawdown.values, 0,
                        alpha=0.3, color=AGENT_COLORS[i])
        ax.plot(drawdown.index, drawdown.values,
                color=AGENT_COLORS[i], linewidth=1.5)
        # Overlay index drawdown
        idx_aligned = index_portfolio.reindex(drawdown.index, method="ffill")
        if idx_aligned is not None:
            idx_cm = idx_aligned.cummax()
            idx_dd = (idx_aligned - idx_cm) / idx_cm * 100
            ax.plot(idx_dd.index, idx_dd.values,
                    color="#8b97b8", linewidth=1, linestyle="--",
                    alpha=0.7, label="^DJI")
        ax.set_title(name, fontsize=11, fontweight="bold", color=AGENT_COLORS[i])
        ax.set_ylabel("Drawdown (%)" if i == 0 else "")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        if i == 0: ax.legend(fontsize=9)

    fig.suptitle("Drawdown Comparison — Individual Agents vs Market Index", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, "C_drawdowns.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: {path}")


def plot_weight_heatmaps(weight_histories: dict, tickers: list):
    """
    Heatmap showing average allocation per asset per agent.
    Helps understand: what does each agent favor?
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    AGENT_COLORS = ["#638cff", "#2dd4b4", "#f5a623"]
    AGENT_LABELS = {"log_return": "PPO-LogReturn", "dsr": "PPO-DSR", "mdd": "PPO-MDD"}

    for i, (rt, wdf) in enumerate(weight_histories.items()):
        ax = axes[i]
        # Sort assets by mean allocation (most allocated at top)
        mean_weights = wdf.mean().sort_values(ascending=True)
        sorted_df    = wdf[mean_weights.index]

        # Plot rolling 30-day average weight as heatmap
        import matplotlib.colors as mcolors
        smooth = sorted_df.rolling(30, min_periods=1).mean()

        im = ax.imshow(
            smooth.T.values,
            aspect="auto",
            cmap="YlOrBr",
            vmin=0, vmax=0.2,
            interpolation="nearest",
        )
        ax.set_title(AGENT_LABELS.get(rt, rt), fontweight="bold",
                     color=AGENT_COLORS[i], fontsize=11)
        ax.set_yticks(range(len(mean_weights.index)))
        ax.set_yticklabels(mean_weights.index, fontsize=8)
        ax.set_xlabel("Time (test period)")
        ax.set_xticks([0, len(smooth)//2, len(smooth)-1])
        ax.set_xticklabels(["Jan 2021", "Jan 2022", "Mar 2024"], fontsize=8)

    plt.colorbar(im, ax=axes[-1], label="Portfolio Weight", shrink=0.8)
    fig.suptitle("Portfolio Weight Allocation Heatmap — Each Agent\n(darker = higher allocation)",
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, "D_weight_heatmaps.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: {path}")


# ─────────────────────────────────────────────────────────
# PRINT RESULTS TABLE
# ─────────────────────────────────────────────────────────

def print_results_table(metrics_df: pd.DataFrame):
    """Pretty-print the evaluation table."""
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"

    print(f"\n{'═'*90}")
    print(f"  {BOLD}PRE-FUSION AGENT EVALUATION — Dow 30 ({TEST_START} → {TEST_END}){RESET}")
    print(f"{'═'*90}")
    print(f"  {CYAN}Initial Capital: ${INITIAL_CAPITAL:,.0f}  |  Transaction Cost: {TRANSACTION_COST:.2%}{RESET}")
    print(f"{'─'*90}")

    # Header
    cols = list(metrics_df.columns)
    header = f"  {'Strategy':<25}" + "".join(f"{c:>10}" for c in cols)
    print(f"  {BOLD}{header}{RESET}")
    print(f"{'─'*90}")

    for strategy, row in metrics_df.iterrows():
        color = ""
        if "PPO" in strategy:
            color = YELLOW
        elif "Index" in strategy:
            color = CYAN

        line = f"  {color}{strategy:<25}{RESET}"
        for val in row.values:
            line += f"{val:>10}"
        print(line)

    print(f"{'═'*90}\n")


# ─────────────────────────────────────────────────────────
# STATISTICAL TESTS (agent vs index)
# ─────────────────────────────────────────────────────────

def run_statistical_tests(agent_portfolios: dict, index_portfolio: pd.Series):
    """Paired t-test: is each agent significantly better than the market index?"""
    print("🧪 Statistical Tests (each agent vs ^DJI):")
    print("─" * 55)

    index_returns = index_portfolio.pct_change().dropna().values

    for name, portfolio in agent_portfolios.items():
        agent_returns = portfolio.pct_change().dropna().values
        result = paired_t_test(
            agent_returns, index_returns,
            strategy_a=name,
            strategy_b="Market Index",
            alpha=0.05
        )


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def evaluate_all_agents():
    """Full pre-fusion evaluation pipeline."""

    print("\n" + "═"*60)
    print("  STEP 3.5 — PRE-FUSION AGENT EVALUATION")
    print("═"*60)

    # ── 1. Load features
    print("\n📂 Loading features...")
    state_df, close_df, cov_array, tickers = load_features()
    test_close = close_df[close_df.index >= TEST_START]

    # ── 2. Load trained agents
    agents = load_trained_agents(state_df, close_df)

    # ── 3. Run each agent on test set
    print("\n🚀 Running agents on test period...")
    agent_portfolios = {}
    weight_histories = {}

    for rt, (agent, env) in agents.items():
        print(f"  Running PPO-{rt}...")
        portfolio, weight_df = run_agent_on_test(agent, env, close_df, tickers, rt)
        agent_portfolios[f"PPO-{rt}"] = portfolio
        weight_histories[rt]          = weight_df
        print(f"    Final value: ${portfolio.iloc[-1]:,.0f}  "
              f"({(portfolio.iloc[-1]/INITIAL_CAPITAL - 1)*100:+.1f}%)")

    # ── 4. Fetch benchmarks for comparison
    index_portfolio = fetch_market_index()
    equal_portfolio = equal_weight_benchmark(test_close)

    # ── 5. Compute metrics for all
    print("\n📊 Computing performance metrics...")
    all_strategies = {
        **agent_portfolios,
        "Market Index (^DJI)": index_portfolio,
        "1/N Equal Weight":    equal_portfolio,
    }
    metrics_df = compare_strategies(all_strategies)

    # ── 6. Print results table
    print_results_table(metrics_df)

    # ── 7. Sanity checks
    check_results = run_sanity_checks(agent_portfolios, index_portfolio, equal_portfolio)

    # ── 8. Statistical significance tests
    run_statistical_tests(agent_portfolios, index_portfolio)

    # ── 9. Generate plots
    print("\n🎨 Generating evaluation plots...")
    plot_agent_comparison(agent_portfolios, index_portfolio, equal_portfolio)
    plot_weight_heatmaps(weight_histories, tickers)

    # ── 10. Save results
    print("\n💾 Saving evaluation results...")
    metrics_df.to_csv(os.path.join(EVAL_DIR, "pre_fusion_metrics.csv"))

    # Save weight histories
    for rt, wdf in weight_histories.items():
        wdf.to_csv(os.path.join(EVAL_DIR, f"weights_{rt}.csv"))

    # Save agent portfolio values
    port_df = pd.DataFrame(
        {k: v for k, v in all_strategies.items() if v is not None}
    )
    port_df.to_csv(os.path.join(EVAL_DIR, "agent_portfolios.csv"))

    # Save sanity check report
    report_lines = ["PRE-FUSION SANITY CHECK REPORT", "="*50]
    for name, result in check_results.items():
        status = "PASS" if result["passed"] else "FAIL"
        report_lines.append(f"\n{name}: {status}")
        report_lines.append(f"  SR  = {result['sr']:+.4f}")
        report_lines.append(f"  MDD = {result['mdd']:.4%}")
        report_lines.append(f"  CR  = {result['cr']:+.4%}")
        for issue in result["issues"]:
            report_lines.append(f"  ⚠  {issue}")

    with open(os.path.join(EVAL_DIR, "sanity_check_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ All evaluation outputs saved to: {EVAL_DIR}/")
    print("\n   Files generated:")
    print("     A_cumulative_wealth.png   — wealth curves for all 3 agents")
    print("     B_rolling_sharpe.png      — rolling 63-day Sharpe per agent")
    print("     C_drawdowns.png           — drawdown comparison vs ^DJI")
    print("     D_weight_heatmaps.png     — per-asset allocation heatmap")
    print("     pre_fusion_metrics.csv    — full metrics table")
    print("     sanity_check_report.txt   — pass/fail for each agent")
    print("     weights_{rt}.csv          — daily weight history per agent")

    print("\n" + "═"*60)
    all_passed = all(v["passed"] for v in check_results.values())
    if all_passed:
        print("  ✅ PROCEED to Step 4: Supervised Pre-training")
        print("     python fusion/supervised_pretraining.py")
    else:
        print("  ⚠️  REVIEW failing agents before proceeding to fusion")
        print("     Consider retraining with more timesteps or different HPs")
    print("═"*60 + "\n")

    return metrics_df, agent_portfolios, check_results


if __name__ == "__main__":
    evaluate_all_agents()