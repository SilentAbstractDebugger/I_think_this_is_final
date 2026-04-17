import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional
from config import RISK_FREE_RATE, OMEGA_THRESHOLD

# INDIVIDUAL METRICS
def cumulative_return(portfolio_values: np.ndarray) -> float:
    """
    CR = (Final Value / Initial Value) - 1
    Expressed as a percentage.
    """
    if len(portfolio_values) < 2 or portfolio_values[0] <= 0:
        return 0.0
    return (portfolio_values[-1] / portfolio_values[0]) - 1


def annual_return(
    portfolio_values: np.ndarray,
    trading_days_per_year: int = 252
) -> float:
    """
    AR = (1 + CR)^(252/T) - 1

    T = number of trading days in the test period.
    Annualizes the total return to make strategies comparable.
    """
    cr = cumulative_return(portfolio_values)
    T  = len(portfolio_values) - 1
    if T <= 0:
        return 0.0
    return (1 + cr) ** (trading_days_per_year / T) - 1


def annual_volatility(
    daily_returns: np.ndarray,
    trading_days_per_year: int = 252
) -> float:
    # AV = std(daily_returns) × √252
    if len(daily_returns) < 2:
        return 0.0
    return float(np.std(daily_returns, ddof=1)) * np.sqrt(trading_days_per_year)


def maximum_drawdown(portfolio_values: np.ndarray) -> float:
    # MDD = max over all t of [(Peak_up_to_t - Value_at_t) / Peak_up_to_t]
    if len(portfolio_values) < 2:
        return 0.0
    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (cummax - portfolio_values) / (cummax + 1e-10)
    return float(np.max(drawdowns))


def sharpe_ratio(
    daily_returns:          np.ndarray,
    risk_free_rate:         float = RISK_FREE_RATE,
    trading_days_per_year:  int   = 252,
) -> float:
    """
    SR = (r_p - r_f) / σ_p

    r_p = annualized portfolio return
    r_f = risk-free rate (daily: r_f / 252)
    σ_p = annualized portfolio volatility

    Measures excess return per unit of total risk.
    Higher = better risk-adjusted return.
    """
    if len(daily_returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / trading_days_per_year
    excess   = daily_returns - daily_rf
    mean_ex  = np.mean(excess)
    std_ex   = np.std(excess, ddof=1)

    if std_ex < 1e-10:
        return 0.0

    # Annualize
    return float((mean_ex / std_ex) * np.sqrt(trading_days_per_year))


def calmar_ratio(
    daily_returns:  np.ndarray,
    portfolio_values: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days_per_year: int = 252,
) -> float:
    """
    CAR = (AR - r_f) / MDD

    Measures return relative to the worst drawdown experienced.
    High CAR = good returns with low drawdowns (important for risk-averse investors).
    """
    ar  = annual_return(portfolio_values, trading_days_per_year)
    mdd = maximum_drawdown(portfolio_values)

    if mdd < 1e-10:
        return 0.0 if ar <= risk_free_rate else float("inf")

    return float((ar - risk_free_rate) / mdd)


def sortino_ratio(
    daily_returns:  np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days_per_year: int = 252,
) -> float:
    """
    SOR = (r_p - r_f) / DR

    DR = downside deviation = std of NEGATIVE returns only.

    Unlike Sharpe, Sortino only penalizes DOWNSIDE volatility.
    Upward volatility is not penalized (it's a good thing!).
    """
    if len(daily_returns) < 2:
        return 0.0

    daily_rf   = risk_free_rate / trading_days_per_year
    excess     = daily_returns - daily_rf
    downside   = excess[excess < 0]

    if len(downside) < 2:
        return 0.0

    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-10:
        return 0.0

    mean_excess = np.mean(excess)
    return float((mean_excess / downside_std) * np.sqrt(trading_days_per_year))


def omega_ratio(
    daily_returns:  np.ndarray,
    threshold:      float = OMEGA_THRESHOLD,
) -> float:
    if len(daily_returns) < 2:
        return 0.0

    gains  = np.sum(np.maximum(daily_returns - threshold, 0))
    losses = np.sum(np.maximum(threshold - daily_returns, 0))

    if losses < 1e-10:
        return float("inf") if gains > 0 else 0.0

    return float(gains / losses)


def stability(portfolio_values: np.ndarray) -> float:
    """
    Stability = R² of linear regression fit to cumulative log returns.
    A perfectly linear (monotonically growing) portfolio has R² = 1.
    """
    if len(portfolio_values) < 3:
        return 0.0

    # Cumulative log returns
    cum_log_ret = np.log(portfolio_values / portfolio_values[0])
    x = np.arange(len(cum_log_ret))

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, cum_log_ret)
    return float(r_value ** 2)


def compute_all_metrics(
    portfolio_values: Union[np.ndarray, pd.Series],
    strategy_name:   str   = "Strategy",
    risk_free_rate:  float = RISK_FREE_RATE,
    trading_days_per_year: int = 252,
) -> dict:
    
    if isinstance(portfolio_values, pd.Series):
        portfolio_values = portfolio_values.values

    portfolio_values = portfolio_values.astype(float)

    # Daily returns from portfolio values
    daily_returns = np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-10)

    metrics = {
        "Strategy":         strategy_name,
        "CR (%)":           round(cumulative_return(portfolio_values) * 100, 2),
        "AR (%)":           round(annual_return(portfolio_values, trading_days_per_year) * 100, 2),
        "AV (%)":           round(annual_volatility(daily_returns, trading_days_per_year) * 100, 2),
        "MDD (%)":          round(maximum_drawdown(portfolio_values) * 100, 2),
        "SR":               round(sharpe_ratio(daily_returns, risk_free_rate, trading_days_per_year), 3),
        "SOR":              round(sortino_ratio(daily_returns, risk_free_rate, trading_days_per_year), 3),
        "CAR":              round(calmar_ratio(daily_returns, portfolio_values, risk_free_rate, trading_days_per_year), 3),
        "OR":               round(omega_ratio(daily_returns, OMEGA_THRESHOLD), 3),
        "Stability (R²)":   round(stability(portfolio_values), 4),
    }

    return metrics


def compare_strategies(
    strategies: dict,   # {strategy_name: portfolio_value_array_or_series}
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    rows = []
    for name, values in strategies.items():
        metrics = compute_all_metrics(values, strategy_name=name, risk_free_rate=risk_free_rate)
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("Strategy")
    return df

def paired_t_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    strategy_a: str = "RA-DRL",
    strategy_b: str = "Baseline",
    alpha: float = 0.05,
) -> dict:
   
    # Align lengths
    min_len = min(len(returns_a), len(returns_b))
    a = returns_a[:min_len]
    b = returns_b[:min_len]

    t_stat, p_value = stats.ttest_rel(a, b)

    result = {
        "t_statistic":    round(t_stat,  4),
        "p_value":        round(p_value, 6),
        "alpha":          alpha,
        "reject_H0":      p_value < alpha,
        "significant":    p_value < alpha,
        "conclusion": (
            f" {strategy_a} is significantly different from {strategy_b} "
            f"(p={p_value:.4f} < α={alpha})"
            if p_value < alpha else
            f" No significant difference between {strategy_a} and {strategy_b} "
            f"(p={p_value:.4f} ≥ α={alpha})"
        )
    }

    print(f"\n Paired t-test: {strategy_a} vs {strategy_b}")
    print(f"   t-statistic: {result['t_statistic']}")
    print(f"   p-value:     {result['p_value']}")
    print(f"   {result['conclusion']}")

    return result

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    print("=== Testing Metrics ===\n")

    # Simulate a strategy that grows from $1M to $1.5M over 3 years
    np.random.seed(42)
    T = 252 * 3  # 3 years of trading days
    daily_r = np.random.normal(0.0003, 0.01, T)  # small positive drift

    # Build portfolio value series
    portfolio = np.ones(T + 1) * 1_000_000
    for t in range(1, T + 1):
        portfolio[t] = portfolio[t-1] * (1 + daily_r[t-1])

    # Compare two strategies
    strategies = {
        "RA-DRL":    portfolio,
        "Buy&Hold":  portfolio * np.random.uniform(0.8, 1.2, len(portfolio)),
        "Equal":     np.linspace(1_000_000, 1_250_000, len(portfolio)),
    }

    df = compare_strategies(strategies)
    print("\n Strategy Comparison:")
    print(df.to_string())

    # Statistical test
    dr1 = np.diff(portfolio) / portfolio[:-1]
    dr2 = np.diff(strategies["Buy&Hold"]) / strategies["Buy&Hold"][:-1]
    paired_t_test(dr1, dr2, "RA-DRL", "Buy&Hold")
