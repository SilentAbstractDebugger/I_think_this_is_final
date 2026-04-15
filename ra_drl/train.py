"""
train.py
─────────
MASTER TRAINING SCRIPT — Run this to train everything.

This is the single entry point for the complete training pipeline.
Run it in order:

  python train.py --step 1   # Download data
  python train.py --step 2   # Feature engineering
  python train.py --step 3   # Train 3 PPO agents
  python train.py --step 4   # Supervised pre-training of Transformer fusion
  python train.py --all      # Run all steps sequentially

TEAM RESPONSIBILITIES:
  Steps 1-4: You two
  Step 4 (fusion forward pass): Partner 3 implements transformer_fusion.py
"""

import os
import sys
import argparse
import time

sys.path.append(os.path.abspath("."))


def step1_download_data():
    """Download Dow 30 OHLCV data from Yahoo Finance."""
    print("\n" + "═"*60)
    print("  STEP 1: DOWNLOAD DATA")
    print("═"*60)
    from data.download_data import main
    main()


def step2_feature_engineering():
    """Compute technical indicators and covariance matrices."""
    print("\n" + "═"*60)
    print("  STEP 2: FEATURE ENGINEERING")
    print("═"*60)
    from data.feature_engineering import FeatureBuilder
    builder = FeatureBuilder()
    builder.run()


def step3_train_agents(use_hyperopt=True, timesteps=None):
    """Train the 3 PPO agents with their respective reward functions."""
    print("\n" + "═"*60)
    print("  STEP 3: TRAIN PPO AGENTS")
    print("═"*60)
    from agents.train_agents import train_all_agents, generate_agent_actions
    from config import PPO_CONFIG

    if timesteps:
        PPO_CONFIG["total_timesteps"] = timesteps

    agents, state_df, close_df, tickers = train_all_agents(
        use_hyperopt=use_hyperopt,
        skip_if_exists=True,
    )

    # Generate actions for both train and test
    print("\n  Generating actions for TRAIN set...")
    generate_agent_actions(agents, state_df, close_df, mode="train")

    print("\n  Generating actions for TEST set...")
    generate_agent_actions(agents, state_df, close_df, mode="test")


def step35_evaluate_agents():
    """Evaluate each trained agent individually before fusion."""
    print("\n" + "═"*60)
    print("  STEP 3.5: INDIVIDUAL AGENT EVALUATION")
    print("═"*60)
    from agents.evaluate_agents import evaluate_all_agents
    metrics_df, agent_portfolios, check_results = evaluate_all_agents()

    # Block pipeline if any agent critically fails
    critical_fail = any(
        result["mdd"] > 0.5 or result["cr"] < -0.3
        for result in check_results.values()
    )
    if critical_fail:
        print("\n❌ PIPELINE HALTED — one or more agents critically failed.")
        print("   Retrain with: python train.py --step 3 --no-skip --timesteps 750000")
        import sys; sys.exit(1)

    return metrics_df


def step4_supervised_pretrain():
    """Pre-train Transformer fusion module with ground-truth weights."""
    print("\n" + "═"*60)
    print("  STEP 4: SUPERVISED PRE-TRAINING (FUSION MODULE)")
    print("═"*60)
    from fusion.supervised_pretraining import main
    main()


def run_all(args):
    """Run all steps sequentially."""
    start = time.time()

    step1_download_data()
    step2_feature_engineering()
    step3_train_agents(
        use_hyperopt=not args.no_hyperopt,
        timesteps=args.timesteps,
    )
    step35_evaluate_agents()
    step4_supervised_pretrain()

    elapsed = time.time() - start
    print(f"\n\n🎉 FULL TRAINING PIPELINE COMPLETE!")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print(f"\n   Next: python backtest.py")


def main():
    parser = argparse.ArgumentParser(
        description="RA-DRL Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --all                   # Full pipeline
  python train.py --step 1               # Just download data
  python train.py --step 3 --no-hyperopt # Train without HP tuning
  python train.py --step 3 --timesteps 100000  # Quick test run
        """
    )

    parser.add_argument("--all",           action="store_true", help="Run all steps")
    parser.add_argument("--step",          type=int, choices=[1, 2, 3, 35, 4],
                        help="Run a specific step only (35 = agent evaluation)")
    parser.add_argument("--no-hyperopt",   action="store_true",
                        help="Skip Bayesian hyperparameter optimization")
    parser.add_argument("--timesteps",     type=int, default=None,
                        help="Override PPO training timesteps (e.g. 50000 for quick test)")

    args = parser.parse_args()

    if not args.all and args.step is None:
        parser.print_help()
        print("\n⚠️  Please specify --all or --step N")
        sys.exit(1)

    if args.all:
        run_all(args)
    elif args.step == 1:
        step1_download_data()
    elif args.step == 2:
        step2_feature_engineering()
    elif args.step == 3:
        step3_train_agents(
            use_hyperopt=not args.no_hyperopt,
            timesteps=args.timesteps,
        )
    elif args.step == 35:
        step35_evaluate_agents()
    elif args.step == 4:
        step4_supervised_pretrain()


if __name__ == "__main__":
    main()
