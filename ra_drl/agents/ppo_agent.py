import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
from typing import Optional
from config import PPO_CONFIG, MODEL_DIR, SEED

class PortfolioLoggingCallback(BaseCallback):
    def __init__(self, agent_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.agent_name = agent_name
        self.episode_rewards = []
        self.episode_count   = 0

    def _on_step(self) -> bool:
        # Collect episode reward info from infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)
                self.episode_count += 1

                if self.episode_count % 50 == 0:
                    mean_r = np.mean(self.episode_rewards[-50:])
                    print(f"  [{self.agent_name}] Episode {self.episode_count:4d} | "
                          f"Mean Reward (last 50): {mean_r:+.4f}")
        return True

class PPOPortfolioAgent:
    def __init__(
        self,
        agent_name: str,
        env,
        eval_env=None,
        config: dict = PPO_CONFIG,
    ):
        self.agent_name = agent_name
        self.config     = config
        self.model_path = os.path.join(MODEL_DIR, f"ppo_{agent_name}")

        # Wrap env for stable-baselines3 (adds episode monitoring)
        self.env = Monitor(env)
        self.vec_env = DummyVecEnv([lambda: self.env])

        # Eval env (optional, used by EvalCallback)
        self.eval_env = None
        if eval_env is not None:
            self.eval_env = DummyVecEnv([lambda: Monitor(eval_env)])

        # FIX: Dynamically read activation from config rather than hardcoding ReLU
        activation_str = config.get("policy_kwargs", {}).get("activation_fn", "relu").lower()
        activation_fn = nn.ReLU if activation_str == "relu" else nn.Tanh
        
        # Build PPO model
        policy_kwargs = {
            "net_arch": config.get("policy_kwargs", {}).get("net_arch", [128, 128]),
            "activation_fn": activation_fn, 
        }

        # FIX: Ensure fallbacks (.get) exist for parameters like max_grad_norm 
        # to strictly protect against gradient explosions
        self.model = PPO(
            policy          = "MlpPolicy",   # standard Multi-Layer Perceptron policy
            env             = self.vec_env,
            learning_rate   = config.get("learning_rate", 3e-4),
            n_steps         = config.get("n_steps", 2048),
            batch_size      = config.get("batch_size", 64),
            n_epochs        = config.get("n_epochs", 10),
            gamma           = config.get("gamma", 0.99),
            gae_lambda      = config.get("gae_lambda", 0.95),
            clip_range      = config.get("clip_range", 0.2),   # ε = 0.2
            ent_coef        = config.get("ent_coef", 0.0),     # entropy bonus coefficient
            vf_coef         = config.get("vf_coef", 0.5),      # value function loss weight
            max_grad_norm   = config.get("max_grad_norm", 0.5), 
            policy_kwargs   = policy_kwargs,
            verbose         = config.get("verbose", 1),
            seed            = SEED,
            device          = "cuda" if torch.cuda.is_available() else "cpu",
        )

        print(f"\n PPO Agent [{agent_name}] initialized")
        print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"   Params: {sum(p.numel() for p in self.model.policy.parameters()):,}")

    def train(self, total_timesteps: Optional[int] = None) -> "PPOPortfolioAgent":
        timesteps = total_timesteps or self.config.get("total_timesteps", 10_000)

        print(f"\n Training PPO [{self.agent_name}] for {timesteps:,} timesteps...")

        # Callbacks
        callbacks = [PortfolioLoggingCallback(self.agent_name)]

        if self.eval_env is not None:
            eval_callback = EvalCallback(
                eval_env          = self.eval_env,
                best_model_save_path = self.model_path + "_best/",
                log_path          = self.model_path + "_logs/",
                eval_freq         = 5000,
                n_eval_episodes   = 1,
                deterministic     = True,
                verbose           = 0,
            )
            callbacks.append(eval_callback)

        checkpoint_callback = CheckpointCallback(
            save_freq  = 50_000,
            save_path  = self.model_path + "_checkpoints/",
            name_prefix = f"ppo_{self.agent_name}",
        )
        callbacks.append(checkpoint_callback)

        # Train
        self.model.learn(
            total_timesteps = timesteps,
            callback        = callbacks,
            progress_bar    = False,
            reset_num_timesteps = True,
        )

        # Save final model
        self.save()
        print(f"\n Training complete! Model saved to {self.model_path}")

        return self

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(self.model_path)
        print(f"   Saved: {self.model_path}.zip")

    def load(self):
        path = self.model_path + ".zip"
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model at {path}. Train first!")
        self.model = PPO.load(path, env=self.vec_env, device="auto")
        print(f" Loaded: {path}")
        return self

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_tensor = obs.reshape(1, -1)
        action, _ = self.model.predict(obs_tensor, deterministic=deterministic)
        action = action.flatten()
        
        # FIX: Protect against NaN actions breaking inference scripts
        if np.any(np.isnan(action)):
            # If network degraded, fallback to uniform allocation to prevent crash
            weights = np.ones_like(action) / len(action)
            return weights

        # Apply softmax to get valid portfolio weights
        weights = np.exp(action - action.max())
        weights = weights / (weights.sum() + 1e-10)
        return weights

    def generate_actions_for_period(
        self,
        env,
        mode: str = "test",
    ) -> pd.DataFrame:
        obs, _ = env.reset()
        done    = False
        actions = []
        dates   = []
        step    = 0

        while not done:
            date = env.dates[min(step, len(env.dates) - 1)]
            weights = self.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(weights)
            done = terminated or truncated

            actions.append(weights)
            dates.append(date)
            step += 1

        tickers = list(env.close_df.columns)
        df = pd.DataFrame(actions, index=dates, columns=tickers)
        return df