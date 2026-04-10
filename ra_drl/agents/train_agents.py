import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from hyperopt import fmin, tpe, Trials, STATUS_OK, hp
import warnings
warnings.filterwarnings("ignore")

from data.feature_engineering import load_features
from envs.portfolio_env import PortfolioEnv
from agents.ppo_agent import PPOPortfolioAgent
from config import (
    PPO_CONFIG, HYPEROPT_SPACE, HYPEROPT_MAX_EVALS,
    TRAIN_START, TRAIN_END, MODEL_DIR, FEAT_DIR, SEED
)

def run_hyperopt(
    state_df: pd.DataFrame,
    close_df: pd.DataFrame,
    reward_type: str,
    max_evals: int = HYPEROPT_MAX_EVALS,
) -> dict:
    print(f"\n Running Bayesian Optimization for [{reward_type}] ({max_evals} trials)...")

    # Use only last 10% of training data for hyperopt (as in paper)
    train_dates = state_df[
        (state_df.index >= TRAIN_START) & (state_df.index <= TRAIN_END)
    ].index
    cutoff_idx = int(len(train_dates) * 0.9)
    hp_start   = train_dates[cutoff_idx]
    hp_end     = train_dates[-1]

    state_hp = state_df[(state_df.index >= hp_start) & (state_df.index <= hp_end)]
    close_hp = close_df[close_df.index.isin(state_hp.index)]

    def objective(params):
        """Single hyperopt trial: train briefly, return negative reward."""
        try:
            # Build config from sampled params
            net_arch = params["net_arch_size"]
            config = {
                **PPO_CONFIG,
                "learning_rate": float(params["learning_rate"]),
                "gamma":         float(params["gamma"]),
                "n_epochs":      int(params["n_epochs"]),
                "ent_coef":      float(params["ent_coef"]),
                "vf_coef":       float(params["vf_coef"]),
                "policy_kwargs": {"net_arch": net_arch, "activation_fn": "relu"},
                "total_timesteps": 10_000,  # short training for HP search
                "verbose": 0,
            }

            # Create mini-environment
            env = PortfolioEnv(
                state_df=state_hp,
                close_df=close_hp,
                reward_type=reward_type,
                mode="train",
            )
            # Monkey-patch dates to use hp_start/hp_end
            env.state_df = state_hp
            env.dates    = state_hp.index.tolist()

            agent = PPOPortfolioAgent(f"{reward_type}_hp", env, config=config)
            agent.train(total_timesteps=10_000)

            # Evaluate: run one episode, get total reward
            obs, _ = env.reset()
            done = False
            total_r = 0.0
            while not done:
                action, _ = agent.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_r += r

            # Return negative (hyperopt minimizes)
            return {"loss": -total_r, "status": STATUS_OK}

        except Exception as e:
            print(f"  Trial failed: {e}")
            return {"loss": 1e6, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn        = objective,
        space     = HYPEROPT_SPACE,
        algo      = tpe.suggest,
        max_evals = max_evals,
        trials    = trials,
        rstate    = np.random.default_rng(SEED),
        verbose   = False,
    )

    # Map choice index back to actual net_arch
    arch_choices = [[64, 64], [128, 128], [256, 128], [256, 256]]
    best_arch    = arch_choices[int(best.get("net_arch_size", 2))]

    best_params = {
        "learning_rate": float(best["learning_rate"]),
        "gamma":         float(best["gamma"]),
        "n_epochs":      int(best["n_epochs"]),
        "ent_coef":      float(best["ent_coef"]),
        "vf_coef":       float(best["vf_coef"]),
        "net_arch":      best_arch,
    }

    print(f" Best params: {best_params}")
    return best_params

def train_all_agents(
    use_hyperopt: bool = True,
    skip_if_exists: bool = True,
) -> dict:
    # Load pre-computed features
    print("Loading features...")
    state_df, close_df, cov_array, tickers = load_features()
    print(f"  State shape: {state_df.shape}")
    print(f"  Close shape: {close_df.shape}")
    print(f"  Assets: {len(tickers)}")

    reward_types = ["log_return", "dsr", "mdd"]
    agents = {}

    for reward_type in reward_types:
        print(f"\n{'='*60}")
        print(f"  AGENT: {reward_type.upper()}")
        print(f"{'='*60}")

        model_path = os.path.join(MODEL_DIR, f"ppo_{reward_type}.zip")

        # Skip if already trained
        if skip_if_exists and os.path.exists(model_path):
            print(f" Model exists at {model_path}, loading...")
            env = PortfolioEnv(
                state_df=state_df,
                close_df=close_df,
                reward_type=reward_type,
                mode="train",
            )
            agent = PPOPortfolioAgent(reward_type, env)
            agent.load()
            agents[reward_type] = agent
            continue

        # Create training environment
        train_env = PortfolioEnv(
            state_df=state_df,
            close_df=close_df,
            reward_type=reward_type,
            mode="train",
        )

        # Bayesian HP Optimization
        config = PPO_CONFIG.copy()
        if use_hyperopt:
            best_params = run_hyperopt(state_df, close_df, reward_type)
            config.update({
                "learning_rate": best_params["learning_rate"],
                "gamma":         best_params["gamma"],
                "n_epochs":      best_params["n_epochs"],
                "ent_coef":      best_params["ent_coef"],
                "vf_coef":       best_params["vf_coef"],
                "policy_kwargs": {
                    "net_arch":       best_params["net_arch"],
                    "activation_fn":  "relu",
                },
            })
            # Save best HP config
            import json
            hp_path = os.path.join(MODEL_DIR, f"best_hp_{reward_type}.json")
            with open(hp_path, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f" Best HPs saved to {hp_path}")

        # Train agent
        agent = PPOPortfolioAgent(
            agent_name=reward_type,
            env=train_env,
            config=config,
        )
        agent.train()
        agents[reward_type] = agent

    print(f"\n{'='*60}")
    print("  ALL 3 AGENTS TRAINED SUCCESSFULLY!")
    print(f"{'='*60}\n")

    return agents, state_df, close_df, tickers

def generate_agent_actions(
    agents: dict,
    state_df: pd.DataFrame,
    close_df: pd.DataFrame,
    mode: str = "train",
) -> dict:
    print(f"\n Generating agent actions ({mode} mode)...")

    actions_dict = {}

    for reward_type, agent in agents.items():
        env = PortfolioEnv(
            state_df=state_df,
            close_df=close_df,
            reward_type=reward_type,
            mode=mode,
        )

        obs, _ = env.reset()
        done   = False
        actions = []
        dates   = []
        step    = 0

        while not done:
            date    = env.dates[min(step, len(env.dates) - 1)]
            weights = agent.get_action(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(weights)
            done    = terminated or truncated

            actions.append(weights)
            dates.append(date)
            step += 1

        tickers = list(close_df.columns)
        actions_df = pd.DataFrame(actions, index=dates, columns=tickers)
        actions_dict[reward_type] = actions_df
        print(f" {reward_type}: {actions_df.shape}")

    # Save stacked actions for fusion module
    save_path = os.path.join(FEAT_DIR, f"agent_actions_{mode}")
    os.makedirs(save_path, exist_ok=True)

    for reward_type, df in actions_dict.items():
        df.to_csv(os.path.join(save_path, f"{reward_type}.csv"))

    # Also save as a 3D numpy array: (T, 3, N)
    agent_order = ["log_return", "dsr", "mdd"]
    min_len = min(len(actions_dict[r]) for r in agent_order)

    stacked = np.stack([
        actions_dict[r].values[:min_len]
        for r in agent_order
    ], axis=1)  # shape (T, 3, N)

    np.save(os.path.join(save_path, "stacked_actions.npy"), stacked)

    # Save dates for alignment
    common_dates = actions_dict["log_return"].index[:min_len]
    pd.Series(common_dates, name="date").to_csv(
        os.path.join(save_path, "dates.csv"), index=False
    )

    print(f"\n Stacked actions saved: {stacked.shape} (T, 3=agents, N=assets)")
    print(f"     Shape for fusion module: {stacked.shape}")

    return actions_dict, stacked, common_dates

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hyperopt",    action="store_true", help="Skip Bayesian optimization")
    parser.add_argument("--no-skip",        action="store_true", help="Retrain even if model exists")
    parser.add_argument("--timesteps",      type=int, default=None, help="Override training timesteps")
    args = parser.parse_args()

    if args.timesteps:
        PPO_CONFIG["total_timesteps"] = args.timesteps

    # Train
    agents, state_df, close_df, tickers = train_all_agents(
        use_hyperopt   = not args.no_hyperopt,
        skip_if_exists = not args.no_skip,
    )

    # Generate actions for both train and test
    print("\n Generating actions for TRAIN set (for supervised pre-training)...")
    generate_agent_actions(agents, state_df, close_df, mode="train")

    print("\n Generating actions for TEST set (for backtesting)...")
    generate_agent_actions(agents, state_df, close_df, mode="test")
