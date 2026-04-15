import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from config import RESULT_DIR, INITIAL_CAPITAL

# Paper-quality styling
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

# Color scheme
COLORS = {
    "RA-DRL":       "#2196F3",   # blue (our model — most prominent)
    "PPO-log_return": "#4CAF50", # green
    "PPO-dsr":      "#FF9800",   # orange
    "PPO-mdd":      "#9C27B0",   # purple
    "Market Index": "#607D8B",   # gray
    "1/N Strategy": "#795548",   # brown
    "MVO":          "#F44336",   # red
}

LINESTYLES = {
    "RA-DRL":       "-",
    "PPO-log_return": "--",
    "PPO-dsr":      "--",
    "PPO-mdd":      "--",
    "Market Index": ":",
    "1/N Strategy": ":",
    "MVO":          "-.",
}


def _get_color(name: str) -> str:
    for key in COLORS:
        if key in name:
            return COLORS[key]
    return "#333333"


def _get_ls(name: str) -> str:
    for key in LINESTYLES:
        if key in name:
            return LINESTYLES[key]
    return "-"

def plot_cumulative_wealth(
    strategies:   dict,          # {name: pd.Series of portfolio values}
    title:        str = "Cumulative Portfolio Wealth — Dow 30 (Jan 2021 – Mar 2024)",
    figsize:      tuple = (14, 6),
    save:         bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    for name, values in strategies.items():
        if values is None or len(values) < 2:
            continue

        # Convert to pd.Series if needed
        if isinstance(values, np.ndarray):
            values = pd.Series(values)

        # Normalize to $1M initial (in case different starting values)
        normalized = values / values.iloc[0] * INITIAL_CAPITAL

        lw = 2.5 if "RA-DRL" in name else 1.5
        ax.plot(
            normalized.index,
            normalized.values / 1e6,   # in millions
            label     = name,
            color     = _get_color(name),
            linestyle = _get_ls(name),
            linewidth = lw,
            zorder    = 10 if "RA-DRL" in name else 5,
        )

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($ Millions)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)

    # Shade market crash periods (COVID recovery → Ukraine war → Fed hikes)
    ax.axvspan(
        pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
        alpha=0.05, color="red", label="Bear Market 2022"
    )

    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:.2f}M"))

    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, "01_cumulative_wealth.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

    return fig

def plot_drawdowns(
    strategies:  dict,
    top_n:       int   = 4,    # plot only top strategies for clarity
    figsize:     tuple = (14, 5),
    save:        bool  = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    # Pick top strategies (include RA-DRL + top benchmarks)
    keys = list(strategies.keys())
    priority = [k for k in keys if "RA-DRL" in k] + [k for k in keys if "RA-DRL" not in k]
    keys = priority[:top_n]

    for name in keys:
        values = strategies[name]
        if values is None or len(values) < 2:
            continue

        if isinstance(values, np.ndarray):
            values = pd.Series(values)

        # Compute drawdown series
        cummax    = values.cummax()
        drawdown  = (values - cummax) / cummax * 100   # in %

        ax.fill_between(
            drawdown.index, drawdown.values, 0,
            alpha     = 0.3,
            color     = _get_color(name),
            label     = name,
        )
        ax.plot(
            drawdown.index, drawdown.values,
            color     = _get_color(name),
            linewidth = 1.5,
        )

    ax.set_title("Portfolio Drawdown (% from Peak)", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend(loc="lower left")
    ax.set_ylim(top=5)   # leave space at top
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, "02_drawdowns.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

    return fig

def plot_metrics_heatmap(
    metrics_df:  pd.DataFrame,
    figsize:     tuple = (14, 6),
    save:        bool  = True,
) -> plt.Figure:
    # Select numeric columns
    numeric_cols = [c for c in metrics_df.columns if c not in ["Strategy"]]
    df = metrics_df[numeric_cols].copy()

    # Normalize each column to [0, 1] for coloring
    # For metrics where HIGHER is better
    higher_is_better = {"CR (%)": True, "AR (%)": True, "SR": True,
                        "SOR": True, "CAR": True, "OR": True, "Stability (R²)": True}
    # For AV and MDD, LOWER is better → invert
    lower_is_better  = {"AV (%)": True, "MDD (%)": True}

    norm_df = df.copy()
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min < 1e-8:
            norm_df[col] = 0.5
        elif col in lower_is_better:
            norm_df[col] = 1 - (df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = (df[col] - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        norm_df,
        annot     = df.round(2),
        fmt       = "",
        cmap      = "RdYlGn",    # Red (bad) → Yellow → Green (good)
        linewidths = 0.5,
        ax        = ax,
        cbar_kws  = {"label": "Relative Performance (normalized)"},
        annot_kws = {"size": 9},
    )

    ax.set_title("Performance Metrics Comparison\n(Green = Better)", fontweight="bold", pad=15)
    ax.set_ylabel("")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, "03_metrics_heatmap.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

    return fig

def plot_weight_evolution(
    weights_df:  pd.DataFrame,   # (T × N) — RA-DRL daily weights
    top_n:       int   = 10,     # show top 10 most-traded assets
    figsize:     tuple = (14, 6),
    save:        bool  = True,
) -> plt.Figure:
    # Pick top N most actively allocated assets (by mean weight)
    top_assets = weights_df.mean().nlargest(top_n).index.tolist()
    df = weights_df[top_assets]

    # Smooth with 5-day rolling mean for readability
    df_smooth = df.rolling(5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab20.colors[:top_n]
    ax.stackplot(
        df_smooth.index,
        df_smooth.T.values,
        labels  = top_assets,
        colors  = colors,
        alpha   = 0.85,
    )

    ax.set_title(f"RA-DRL Portfolio Allocation (Top {top_n} Assets)", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Weight")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)

    # Legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, "04_weight_evolution.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f" Saved: {path}")

    return fig

def plot_rolling_sharpe(
    strategies:  dict,
    window:      int   = 63,    # ~3 months rolling window
    figsize:     tuple = (14, 5),
    save:        bool  = True,
) -> plt.Figure:
    from config import RISK_FREE_RATE

    fig, ax = plt.subplots(figsize=figsize)
    daily_rf = RISK_FREE_RATE / 252

    priority = [k for k in strategies if "RA-DRL" in k] + \
               [k for k in strategies if "RA-DRL" not in k]

    for name in priority[:5]:   # top 5 for clarity
        values = strategies[name]
        if values is None or len(values) < window + 5:
            continue
        if isinstance(values, np.ndarray):
            values = pd.Series(values)

        daily_returns = values.pct_change().dropna()
        rolling_sharpe = daily_returns.rolling(window).apply(
            lambda r: (r.mean() - daily_rf) / (r.std() + 1e-10) * np.sqrt(252),
            raw=True
        )

        ax.plot(
            rolling_sharpe.index,
            rolling_sharpe.values,
            label     = name,
            color     = _get_color(name),
            linestyle = _get_ls(name),
            linewidth = 2 if "RA-DRL" in name else 1.2,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Rolling Sharpe Ratio ({window}-day window)", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio (Annualized)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend()
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, "05_rolling_sharpe.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

    return fig

def plot_metric_bars(
    metrics_df:  pd.DataFrame,
    metrics:     list = ["CR (%)", "SR", "SOR", "CAR"],
    figsize:     tuple = (16, 10),
    save:        bool  = True,
) -> plt.Figure:
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)

    if n == 1:
        axes = [axes]

    strategies = metrics_df.index.tolist()
    colors = [_get_color(s) for s in strategies]

    for ax, metric in zip(axes, metrics):
        values = metrics_df[metric].values
        bars = ax.bar(range(len(strategies)), values, color=colors, edgecolor="white", linewidth=0.5)

        # Highlight RA-DRL
        for i, s in enumerate(strategies):
            if "RA-DRL" in s:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2)

        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.suptitle("Performance Metrics Comparison — RA-DRL vs Baselines",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, "06_metric_bars.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

    return fig

def plot_all(strategies: dict, metrics_df: pd.DataFrame):
    """Generate all plots at once."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"\n Generating plots → {RESULT_DIR}/")

    plot_cumulative_wealth(strategies)
    plot_drawdowns(strategies)
    plot_metrics_heatmap(metrics_df)
    plot_rolling_sharpe(strategies)
    plot_metric_bars(metrics_df)

    # Weight evolution (only if RA-DRL weights available)
    weights_path = os.path.join(RESULT_DIR, "ra_drl_weights.csv")
    if os.path.exists(weights_path):
        weights_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        plot_weight_evolution(weights_df)

    print(f"\n All plots saved to {RESULT_DIR}/")


if __name__ == "__main__":
    print("Testing visualization with dummy data...")

    # Create synthetic portfolio series
    np.random.seed(42)
    T = 252 * 3 + 65   # ~3 years + Q1 2024
    dates = pd.date_range("2021-01-04", periods=T, freq="B")

    def make_portfolio(drift, vol, seed=42):
        np.random.seed(seed)
        r = np.random.normal(drift, vol, T)
        p = INITIAL_CAPITAL * np.cumprod(1 + r)
        return pd.Series(p, index=dates)

    strategies = {
        "RA-DRL":       make_portfolio(0.0006, 0.009, 42),
        "PPO-log_return": make_portfolio(0.0005, 0.011, 1),
        "PPO-dsr":      make_portfolio(0.0004, 0.010, 2),
        "PPO-mdd":      make_portfolio(0.0003, 0.008, 3),
        "Market Index": make_portfolio(0.0004, 0.012, 4),
        "1/N Strategy": make_portfolio(0.0003, 0.010, 5),
        "MVO":          make_portfolio(0.0003, 0.009, 6),
    }

    from utils.metrics import compare_strategies
    metrics_df = compare_strategies(strategies)

    plot_all(strategies, metrics_df)
    print("Visualization test complete!")
