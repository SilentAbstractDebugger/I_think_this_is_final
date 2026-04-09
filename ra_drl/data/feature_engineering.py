
"""
Computes the STATE SPACE for the RL environment as described in the paper:
  - 8 Technical Indicators per asset:
      1. SMA-30  (Simple Moving Average, 30 days)
      2. SMA-60  (Simple Moving Average, 60 days)
      3. MACD    (Moving Average Convergence Divergence)
      4. RSI-14  (Relative Strength Index)
      5. CCI     (Commodity Channel Index)
      6. ADX     (Average Directional Index)
      7. BB_upper (Bollinger Band Upper)
      8. BB_lower (Bollinger Band Lower)
  - Covariance Matrix of closing prices (rolling window)

Maths :

SMA(n) = mean of last n closing prices

MACD = EMA(12) - EMA(26)
  EMA(n) = Close * (2/(n+1)) + EMA_prev * (1 - 2/(n+1))

RSI:
  RS = Avg_Gain / Avg_Loss over 14 days
  RSI = 100 - (100 / (1 + RS))

CCI = (Typical Price - SMA(20)) / (0.015 × Mean Absolute Deviation)
  Typical Price = (High + Low + Close) / 3

ADX = smoothed directional movement index (measures trend strength)

Bollinger Bands:
  Middle = SMA(20)
  Upper  = Middle + 2 × std(20)
  Lower  = Middle - 2 × std(20)

Covariance Matrix:
  60-day rolling covariance between all asset pairs (n × n matrix)
  Used to capture cross-asset risk relationships

"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import ta  # Technical Analysis library
from tqdm import tqdm
from config import (
    DATA_DIR, FEAT_DIR,
    SMA_SHORT, SMA_LONG,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    RSI_PERIOD, CCI_PERIOD, ADX_PERIOD,
    BB_PERIOD, BB_STD,
    LOOKBACK_WINDOW, TRAIN_START, TRAIN_END, TEST_START, TEST_END
)
import warnings
warnings.filterwarnings("ignore")


#Technical Indicator Functions

def compute_sma(close: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(window=window, min_periods=window).mean()


def compute_macd(close: pd.Series,
                 fast: int = MACD_FAST,
                 slow: int = MACD_SLOW) -> pd.Series:
    """
    MACD Line = EMA(fast) - EMA(slow)
    We return MACD line (not signal or histogram).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    RSI = 100 - 100/(1+RS)
    RS  = Avg Gain / Avg Loss (Wilder's smoothing)
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    # Wilder's smoothing (exponential with alpha=1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_cci(high: pd.Series, low: pd.Series,
                close: pd.Series, period: int = CCI_PERIOD) -> pd.Series:
    """
    CCI = (Typical Price - SMA(TP)) / (0.015 × Mean Abs Deviation)
    Typical Price = (High + Low + Close) / 3
    """
    tp  = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad + 1e-10)


def compute_adx(high: pd.Series, low: pd.Series,
                close: pd.Series, period: int = ADX_PERIOD) -> pd.Series:
    """
    Average Directional Index — measures trend strength.
    Uses ta library for correctness.
    """
    adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=period)
    return adx_indicator.adx()


def compute_bollinger(close: pd.Series,
                      period: int = BB_PERIOD,
                      std_dev: int = BB_STD):
    """
    Bollinger Bands
    Returns: (upper_band, lower_band)
    """
    indicator = ta.volatility.BollingerBands(
        close=close, window=period, window_dev=std_dev
    )
    return indicator.bollinger_hband(), indicator.bollinger_lband()


def compute_all_indicators(close: pd.Series, high: pd.Series,
                           low: pd.Series) -> pd.DataFrame:
    """
    Compute all 8 technical indicators for a single asset.
    Returns a DataFrame with columns:
      [sma_30, sma_60, macd, rsi, cci, adx, bb_upper, bb_lower]
    """
    sma30    = compute_sma(close, SMA_SHORT)
    sma60    = compute_sma(close, SMA_LONG)
    macd     = compute_macd(close)
    rsi      = compute_rsi(close)
    cci      = compute_cci(high, low, close)
    adx      = compute_adx(high, low, close)
    bb_up, bb_lo = compute_bollinger(close)

    df = pd.DataFrame({
        "sma_30":    sma30,
        "sma_60":    sma60,
        "macd":      macd,
        "rsi":       rsi,
        "cci":       cci,
        "adx":       adx,
        "bb_upper":  bb_up,
        "bb_lower":  bb_lo,
    }, index=close.index)

    return df


# Covariance Matrix

def compute_rolling_covariance(close_matrix: pd.DataFrame,
                                window: int = LOOKBACK_WINDOW) -> dict:
    """
    Compute rolling covariance matrix for each date.

    Args:
        close_matrix: (T × N) DataFrame of closing prices
        window: lookback window in trading days (paper uses 60)

    Returns:
        dict {date: np.array(N×N)} — covariance matrix per date
    """
    # Daily log returns
    log_returns = np.log(close_matrix / close_matrix.shift(1)).dropna()

    cov_matrices = {}
    dates = log_returns.index[window:]  # start after we have enough history

    for date in dates:
        # Get last `window` days of returns
        end_idx   = log_returns.index.get_loc(date)
        start_idx = max(0, end_idx - window)
        window_returns = log_returns.iloc[start_idx:end_idx]
        cov = window_returns.cov().values  # (N × N) numpy array
        cov_matrices[date] = cov

    return cov_matrices

#Normalization

def normalize_features(features_df: pd.DataFrame,
                        train_end: str = TRAIN_END) -> pd.DataFrame:
    """
    Z-score normalization using ONLY training data statistics.
    This prevents lookahead bias.

    Train stats → applied to both train and test.
    """
    train_mask = features_df.index <= train_end
    mean = features_df[train_mask].mean()
    std  = features_df[train_mask].std().replace(0, 1)  # avoid division by zero
    normalized = (features_df - mean) / std
    return normalized


# Main feature builder

class FeatureBuilder:
    """
    Builds the complete state space for the RL environment.

    State at time t = {
        covariance_matrix: shape (N, N),
        technical_indicators: shape (N, 8)
    }

    Flattened state dimension: N*N + N*8
    For N=29 (Dow): 29*29 + 29*8 = 841 + 232 = 1073 features
    """

    def __init__(self):
        self.close   = None
        self.high    = None
        self.low     = None
        self.tickers = None
        self.indicators_dict = {}   # {ticker: DataFrame of indicators}
        self.cov_matrices    = {}   # {date: np.array(N, N)}
        self.state_df        = None # final normalized state per date

    def load_data(self):
        """Load raw price data."""
        print("📂 Loading raw price data...")
        self.close = pd.read_csv(
            os.path.join(DATA_DIR, "close_prices.csv"), index_col=0, parse_dates=True
        )
        self.high  = pd.read_csv(
            os.path.join(DATA_DIR, "high_prices.csv"), index_col=0, parse_dates=True
        )
        self.low   = pd.read_csv(
            os.path.join(DATA_DIR, "low_prices.csv"), index_col=0, parse_dates=True
        )

        self.tickers = list(self.close.columns)
        print(f" Loaded {len(self.tickers)} tickers, {len(self.close)} dates")

    def compute_indicators(self):
        """Compute 8 technical indicators for each asset."""
        print("\n Computing technical indicators for each asset...")
        for ticker in tqdm(self.tickers):
            close_s = self.close[ticker]
            high_s  = self.high[ticker]
            low_s   = self.low[ticker]

            indicators = compute_all_indicators(close_s, high_s, low_s)
            self.indicators_dict[ticker] = indicators

    def compute_covariances(self):
        """Compute rolling covariance matrices."""
        print("\n Computing rolling covariance matrices...")
        self.cov_matrices = compute_rolling_covariance(self.close, window=LOOKBACK_WINDOW)
        print(f"  Covariance matrices computed for {len(self.cov_matrices)} dates")

    def build_state_dataframe(self):
        """
        Build a flat DataFrame where each row = state at that date.
        Columns = [cov_0_0, cov_0_1, ..., cov_N_N, sma30_AAPL, sma60_AAPL, ...]

        This flat representation is what the RL environment uses.
        """
        print("\n🏗️  Building flat state DataFrame...")

        # Align all indicators to same dates
        all_dates = list(self.cov_matrices.keys())

        rows = []
        for date in tqdm(all_dates):
            row = {}

            # Covariance matrix (upper triangle or full flattened)
            cov = self.cov_matrices[date]
            for i in range(len(self.tickers)):
                for j in range(len(self.tickers)):
                    row[f"cov_{i}_{j}"] = cov[i, j]

            # Technical indicators
            for ticker in self.tickers:
                if date in self.indicators_dict[ticker].index:
                    ind_row = self.indicators_dict[ticker].loc[date]
                    for col in ind_row.index:
                        row[f"{col}_{ticker}"] = ind_row[col]
                else:
                    # Fill NaN for missing dates
                    for col in ["sma_30","sma_60","macd","rsi","cci","adx","bb_upper","bb_lower"]:
                        row[f"{col}_{ticker}"] = np.nan

            rows.append(row)

        self.state_df = pd.DataFrame(rows, index=all_dates)
        self.state_df.index = pd.to_datetime(self.state_df.index)
        self.state_df = self.state_df.sort_index()

        # Forward fill then backward fill remaining NaNs
        self.state_df = self.state_df.fillna(method="ffill").fillna(method="bfill")

        print(f"  State DataFrame: {self.state_df.shape}")
        print(f"     {self.state_df.shape[1]} features per state")

    def normalize(self):
        """Normalize using training data statistics only (prevent lookahead bias)."""
        print("\n Normalizing features (train stats only)...")
        self.state_df = normalize_features(self.state_df, train_end=TRAIN_END)
        print(f"  Normalization done")

    def save(self):
        """Save features to disk."""
        print("\n  Saving features...")

        # Full state DataFrame
        self.state_df.to_csv(os.path.join(FEAT_DIR, "states.csv"))

        # Save covariance matrices as numpy array
        dates = list(self.cov_matrices.keys())
        n = len(self.tickers)
        cov_array = np.stack([self.cov_matrices[d] for d in dates], axis=0)  # (T, N, N)
        np.save(os.path.join(FEAT_DIR, "cov_matrices.npy"), cov_array)
        pd.Series(dates, name="date").to_csv(
            os.path.join(FEAT_DIR, "cov_dates.csv"), index=False
        )

        # Save close prices aligned to cov dates
        aligned_close = self.close.loc[self.close.index.isin(dates)]
        aligned_close.to_csv(os.path.join(FEAT_DIR, "aligned_close.csv"))

        # Save ticker list
        pd.Series(self.tickers).to_csv(
            os.path.join(FEAT_DIR, "tickers.csv"), index=False
        )

        print(f"  All features saved to {FEAT_DIR}/")

    def run(self):
        """Execute the full feature engineering pipeline."""
        self.load_data()
        self.compute_indicators()
        self.compute_covariances()
        self.build_state_dataframe()
        self.normalize()
        self.save()
        print("\n Feature engineering complete!")
        return self.state_df


def load_features():
    """
    Utility to load pre-computed features.
    Used by the RL environment and training scripts.

    Returns:
        state_df:    (T × features) DataFrame
        close_df:    (T × N) closing prices aligned to state dates
        cov_array:   (T, N, N) covariance matrices
        tickers:     list of N ticker symbols
    """
    state_df  = pd.read_csv(
        os.path.join(FEAT_DIR, "states.csv"), index_col=0, parse_dates=True
    )
    close_df  = pd.read_csv(
        os.path.join(FEAT_DIR, "aligned_close.csv"), index_col=0, parse_dates=True
    )
    cov_array = np.load(os.path.join(FEAT_DIR, "cov_matrices.npy"))
    tickers   = pd.read_csv(
        os.path.join(FEAT_DIR, "tickers.csv"), header=0
    ).iloc[:, 0].tolist()

    return state_df, close_df, cov_array, tickers


if __name__ == "__main__":
    builder = FeatureBuilder()
    builder.run()
