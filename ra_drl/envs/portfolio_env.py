
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from utils.rewards import LogReturnReward, DSRReward, MDDReward, get_reward_fn
from config import (
    INITIAL_CAPITAL, TRANSACTION_COST, TRAIN_START, TRAIN_END, TEST_START, TEST_END
)

class PortfolioEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        state_df:         pd.DataFrame,
        close_df:         pd.DataFrame,
        reward_type:      str  = "log_return",
        mode:             str  = "train",
        initial_capital:  float = INITIAL_CAPITAL,
        transaction_cost: float = TRANSACTION_COST,
    ):
        super().__init__()

        if mode == "train":  # filtering data by modes
            mask = (state_df.index >= TRAIN_START) & (state_df.index <= TRAIN_END)
        elif mode == "test":
            mask = (state_df.index >= TEST_START) & (state_df.index <= TEST_END)
        else:
            raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")

        self.state_df  = state_df[mask].copy() # features (covariance + indicators)
        self.close_df  = close_df[close_df.index.isin(self.state_df.index)].copy() # stock prices
        self.dates     = self.state_df.index.tolist()
        self.n_assets  = self.close_df.shape[1]
        self.n_features = self.state_df.shape[1]

        # Reward function based on selected reward type
        self.reward_type = reward_type
        self.reward_fn   = get_reward_fn(reward_type)

        # Financial parameters
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost

        # Observation space which is same as feature vector for current date
        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (self.n_features,),
            dtype = np.float32,
        )

        # Action space = portfolio weigts for each asset before softmax
        # Using logits in [-∞, +∞]; softmax normalizes them to valid weights
        self.action_space = spaces.Box(
            low  = -5.0,
            high =  5.0,
            shape = (self.n_assets,),
            dtype = np.float32,
        )

        # Environment State variables
        self.current_step:     int   = 0
        self.portfolio_value:  float = initial_capital
        self.current_weights:  np.ndarray = np.ones(self.n_assets) / self.n_assets  # equal weight start
        self.portfolio_history: list = []
        self.weight_history:    list = []
        self.return_history:    list = []

    # Reset env to start dataset with equal portfolio allocation
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step    = 0
        self.portfolio_value = self.initial_capital
        self.current_weights = np.ones(self.n_assets) / self.n_assets

        self.portfolio_history = [self.initial_capital]
        self.weight_history    = [self.current_weights.copy()]
        self.return_history    = []

        # Reset reward function internal state
        if self.reward_type == "log_return":
            self.reward_fn.reset(self.initial_capital)
        elif self.reward_type == "dsr":
            self.reward_fn.reset()
        elif self.reward_type == "mdd":
            self.reward_fn.reset(self.initial_capital)

        obs = self._get_observation()
        return obs.astype(np.float32), {}

    # Simulating one trading day
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Converting raw action logits → portfolio weights via softmax
        new_weights = self._softmax(action)

        # 2. Compute transaction cost for this rebalancing
        turnover = np.sum(np.abs(new_weights - self.current_weights))
        tc = self.transaction_cost * self.portfolio_value * turnover

        # 3. Get today's and tomorrow's closing prices
        today     = self.dates[self.current_step]
        next_step = self.current_step + 1

        if next_step >= len(self.dates):
            obs = self._get_observation()
            return obs.astype(np.float32), 0.0, True, False, self._get_info()

        tomorrow  = self.dates[next_step]

        # 4. Compute daily asset returns
        close_today    = self.close_df.loc[today].values.astype(float)
        close_tomorrow = self.close_df.loc[tomorrow].values.astype(float)
        asset_returns  = (close_tomorrow - close_today) / (close_today + 1e-10)  # daily % returns

        # 5. Portfolio return = weighted sum of asset returns
        portfolio_return = np.dot(new_weights, asset_returns)

        # 6. Update portfolio value
        prev_value       = self.portfolio_value
        self.portfolio_value = prev_value * (1 + portfolio_return) - tc
        self.portfolio_value = max(self.portfolio_value, 1.0)  # prevent negative/zero

        # 7. Compute reward based on reward type
        if self.reward_type == "log_return":
            reward = self.reward_fn.compute(self.portfolio_value)
        elif self.reward_type == "dsr":
            log_return = np.log(self.portfolio_value / max(prev_value, 1.0))
            reward = self.reward_fn.compute(log_return)
        elif self.reward_type == "mdd":
            reward = self.reward_fn.compute(self.portfolio_value)
        else:
            reward = 0.0

        # 8. Update state
        self.current_weights = new_weights
        self.current_step    = next_step

        # 9. Record history
        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(new_weights.copy())
        self.return_history.append(portfolio_return)

        # 10. Check if episode done
        terminated = (self.current_step >= len(self.dates) - 1)

        obs = self._get_observation()
        return obs.astype(np.float32), float(reward), terminated, False, self._get_info()

    def render(self, mode="human"):
        """Print current portfolio status."""
        if len(self.dates) > self.current_step:
            date = self.dates[self.current_step]
            print(f"  [{date.date()}] Portfolio: ${self.portfolio_value:,.2f} | "
                  f"Step: {self.current_step}/{len(self.dates)}")

    # Returns feature vector for current timestep
    def _get_observation(self) -> np.ndarray:
        step = min(self.current_step, len(self.dates) - 1)
        date = self.dates[step]
        obs  = self.state_df.loc[date].values
        obs  = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        return obs.astype(np.float32)

    # Convert raw actions values to normalized portfolio weights
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)   # subtract max for numerical stability
        e_x = np.exp(x)
        return e_x / (e_x.sum() + 1e-10)

    # Retrun diagnostic
    def _get_info(self) -> Dict[str, Any]:
        return {
            "portfolio_value": self.portfolio_value,
            "current_step":    self.current_step,
            "n_steps":         len(self.dates),
            "weights":         self.current_weights.copy(),
        }

    # Return portfolio value history as a dated Series.
    def get_portfolio_history(self) -> pd.Series:
        n = min(len(self.portfolio_history), len(self.dates))
        return pd.Series(
            self.portfolio_history[:n],
            index=self.dates[:n],
            name="portfolio_value"
        )

    # Return weight history as a DataFrame (dates × assets)
    def get_weight_history(self) -> pd.DataFrame:
        n = min(len(self.weight_history), len(self.dates))
        tickers = list(self.close_df.columns)
        return pd.DataFrame(
            self.weight_history[:n],
            index=self.dates[:n],
            columns=tickers
        )

    # Return daily portfolio returns
    def get_daily_returns(self) -> pd.Series:
        ph = np.array(self.portfolio_history)
        returns = pd.Series(
            np.diff(ph) / ph[:-1],
            index=self.dates[1:len(self.portfolio_history)],
            name="daily_return"
        )
        return returns

class PortfolioEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        state_df:         pd.DataFrame,
        close_df:         pd.DataFrame,
        reward_type:      str  = "log_return",
        mode:             str  = "train",
        initial_capital:  float = INITIAL_CAPITAL,
        transaction_cost: float = TRANSACTION_COST,
    ):
        super().__init__()

        if mode == "train":  # filtering data by modes
            mask = (state_df.index >= TRAIN_START) & (state_df.index <= TRAIN_END)
        elif mode == "test":
            mask = (state_df.index >= TEST_START) & (state_df.index <= TEST_END)
        else:
            raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")

        self.state_df  = state_df[mask].copy() # features (covariance + indicators)
        self.close_df  = close_df[close_df.index.isin(self.state_df.index)].copy() # stock prices
        self.dates     = self.state_df.index.tolist()
        self.n_assets  = self.close_df.shape[1]
        self.n_features = self.state_df.shape[1]

        # Reward function based on selected reward type
        self.reward_type = reward_type
        self.reward_fn   = get_reward_fn(reward_type)

        # Financial parameters
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost

        # Observation space which is same as feature vector for current date
        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (self.n_features,),
            dtype = np.float32,
        )

        # Action space = portfolio weigts for each asset before softmax
        # Using logits in [-∞, +∞]; softmax normalizes them to valid weights
        self.action_space = spaces.Box(
            low  = -5.0,
            high =  5.0,
            shape = (self.n_assets,),
            dtype = np.float32,
        )

        # Environment State variables
        self.current_step:     int   = 0
        self.portfolio_value:  float = initial_capital
        self.current_weights:  np.ndarray = np.ones(self.n_assets) / self.n_assets  # equal weight start
        self.portfolio_history: list = []
        self.weight_history:    list = []
        self.return_history:    list = []

    # Reset env to start dataset with equal portfolio allocation
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step    = 0
        self.portfolio_value = self.initial_capital
        self.current_weights = np.ones(self.n_assets) / self.n_assets

        self.portfolio_history = [self.initial_capital]
        self.weight_history    = [self.current_weights.copy()]
        self.return_history    = []

        # Reset reward function internal state
        if self.reward_type == "log_return":
            self.reward_fn.reset(self.initial_capital)
        elif self.reward_type == "dsr":
            self.reward_fn.reset()
        elif self.reward_type == "mdd":
            self.reward_fn.reset(self.initial_capital)

        obs = self._get_observation()
        return obs.astype(np.float32), {}

    # Simulating one trading day
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Converting raw action logits → portfolio weights via softmax
        new_weights = self._softmax(action)

        # 2. Compute transaction cost for this rebalancing
        turnover = np.sum(np.abs(new_weights - self.current_weights))
        tc = self.transaction_cost * self.portfolio_value * turnover

        # 3. Get today's and tomorrow's closing prices
        today     = self.dates[self.current_step]
        next_step = self.current_step + 1

        if next_step >= len(self.dates):
            obs = self._get_observation()
            return obs.astype(np.float32), 0.0, True, False, self._get_info()

        tomorrow  = self.dates[next_step]

        # 4. Compute daily asset returns
        close_today    = self.close_df.loc[today].values.astype(float)
        close_tomorrow = self.close_df.loc[tomorrow].values.astype(float)
        asset_returns  = (close_tomorrow - close_today) / (close_today + 1e-10)  # daily % returns

        # 5. Portfolio return = weighted sum of asset returns
        portfolio_return = np.dot(new_weights, asset_returns)

        # 6. Update portfolio value
        prev_value       = self.portfolio_value
        self.portfolio_value = prev_value * (1 + portfolio_return) - tc
        self.portfolio_value = max(self.portfolio_value, 1.0)  # prevent negative/zero

        # 7. Compute reward based on reward type
        if self.reward_type == "log_return":
            reward = self.reward_fn.compute(self.portfolio_value)
        elif self.reward_type == "dsr":
            log_return = np.log(self.portfolio_value / max(prev_value, 1.0))
            reward = self.reward_fn.compute(log_return)
        elif self.reward_type == "mdd":
            reward = self.reward_fn.compute(self.portfolio_value)
        else:
            reward = 0.0

        # 8. Update state
        self.current_weights = new_weights
        self.current_step    = next_step

        # 9. Record history
        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(new_weights.copy())
        self.return_history.append(portfolio_return)

        # 10. Check if episode done
        terminated = (self.current_step >= len(self.dates) - 1)

        obs = self._get_observation()
        return obs.astype(np.float32), float(reward), terminated, False, self._get_info()

    def render(self, mode="human"):
        """Print current portfolio status."""
        if len(self.dates) > self.current_step:
            date = self.dates[self.current_step]
            print(f"  [{date.date()}] Portfolio: ${self.portfolio_value:,.2f} | "
                  f"Step: {self.current_step}/{len(self.dates)}")

    # Returns feature vector for current timestep
    def _get_observation(self) -> np.ndarray:
        step = min(self.current_step, len(self.dates) - 1)
        date = self.dates[step]
        obs  = self.state_df.loc[date].values
        obs  = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        return obs.astype(np.float32)

    # Convert raw actions values to normalized portfolio weights
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)   # subtract max for numerical stability
        e_x = np.exp(x)
        return e_x / (e_x.sum() + 1e-10)

    # Retrun diagnostic
    def _get_info(self) -> Dict[str, Any]:
        return {
            "portfolio_value": self.portfolio_value,
            "current_step":    self.current_step,
            "n_steps":         len(self.dates),
            "weights":         self.current_weights.copy(),
        }

    # Return portfolio value history as a dated Series.
    def get_portfolio_history(self) -> pd.Series:
        n = min(len(self.portfolio_history), len(self.dates))
        return pd.Series(
            self.portfolio_history[:n],
            index=self.dates[:n],
            name="portfolio_value"
        )

    # Return weight history as a DataFrame (dates × assets)
    def get_weight_history(self) -> pd.DataFrame:
        n = min(len(self.weight_history), len(self.dates))
        tickers = list(self.close_df.columns)
        return pd.DataFrame(
            self.weight_history[:n],
            index=self.dates[:n],
            columns=tickers
        )

    # Return daily portfolio returns
    def get_daily_returns(self) -> pd.Series:
        ph = np.array(self.portfolio_history)
        returns = pd.Series(
            np.diff(ph) / ph[:-1],
            index=self.dates[1:len(self.portfolio_history)],
            name="daily_return"
        )
        return returns

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.feature_engineering import load_features

    print("Loading features...")
    state_df, close_df, cov_array, tickers = load_features()

    print(f"Creating environment with {len(tickers)} assets...")
    env = PortfolioEnv(
        state_df=state_df,
        close_df=close_df,
        reward_type="log_return",
        mode="train",
    )

    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space:      {env.action_space.shape}")
    print(f"  Training dates:    {env.dates[0].date()} → {env.dates[-1].date()}")
    print(f"  N assets:          {env.n_assets}")

    # Run one episode
    obs, _ = env.reset()
    total_reward = 0
    n_steps      = 0
    done = False

    print("\nRunning one episode (first 5 steps)...")
    while not done and n_steps < 5:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        n_steps      += 1
        env.render()

    print(f"\n Episode test passed. Reward over {n_steps} steps: {total_reward:.4f}")
    print(f"   Final portfolio: ${info['portfolio_value']:,.2f}")
