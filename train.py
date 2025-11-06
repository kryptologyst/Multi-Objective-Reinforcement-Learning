#!/usr/bin/env python3
"""
Main training script for multi-objective reinforcement learning.

This script provides a command-line interface for training multi-objective RL agents
on various environments with different algorithms.
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from envs.multi_objective_envs import make_multi_objective_env
from agents.multi_objective_agents import create_multi_objective_agent
from src.training import MultiObjectiveTrainer, TrainingLogger, Visualizer, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multi-objective reinforcement learning agents"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--env",
        type=str,
        choices=["line", "cartpole", "mountaincar"],
        help="Environment to use (overrides config)"
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        choices=["qlearning", "dqn", "ppo"],
        help="Agent type to use (overrides config)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of training episodes (overrides config)"
    )
    
    parser.add_argument(
        "--weights",
        type=float,
        nargs=2,
        help="Weight vector for combining objectives (overrides config)"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate a pre-trained model"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file for evaluation"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup the device for training."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("Using CPU device")
    else:
        device = device_arg
        print(f"Using {device} device")
    
    return device


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.env:
        config["environment"]["name"] = args.env
    if args.agent:
        config["agent"]["type"] = args.agent
    if args.episodes:
        config["training"]["max_episodes"] = args.episodes
    if args.weights:
        config["agent"]["weights"] = args.weights
    if args.wandb:
        config["logging"]["use_wandb"] = True
    if args.no_tensorboard:
        config["logging"]["use_tensorboard"] = False
    
    # Setup device
    device = setup_device(args.device)
    
    # Create environment
    print(f"Creating environment: {config['environment']['name']}")
    env = make_multi_objective_env(
        config["environment"]["name"],
        **{k: v for k, v in config["environment"].items() if k != "name"}
    )
    
    # Create agent
    print(f"Creating agent: {config['agent']['type']}")
    agent = create_multi_objective_agent(
        config["agent"]["type"],
        env,
        tuple(config["agent"]["weights"]),
        **{k: v for k, v in config["agent"].items() 
           if k not in ["type", "weights"]}
    )
    
    # Create logger
    logger = TrainingLogger(
        log_dir=config["logging"]["log_dir"],
        use_wandb=config["logging"]["use_wandb"],
        use_tensorboard=config["logging"]["use_tensorboard"],
        project_name=config["logging"]["project_name"]
    )
    
    # Create trainer
    trainer = MultiObjectiveTrainer(agent, env, logger, config["training"])
    
    if args.eval_only:
        # Evaluation only mode
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation mode")
            sys.exit(1)
        
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
        
        print("Running evaluation...")
        eval_reward, eval_length = trainer._evaluate()
        print(f"Evaluation Results:")
        print(f"  Average Reward: {eval_reward:.2f}")
        print(f"  Average Length: {eval_length:.1f}")
        
    else:
        # Training mode
        print("Starting training...")
        print(f"Environment: {config['environment']['name']}")
        print(f"Agent: {config['agent']['type']}")
        print(f"Episodes: {config['training']['max_episodes']}")
        print(f"Weights: {config['agent']['weights']}")
        print(f"Device: {device}")
        
        # Train agent
        stats = trainer.train()
        
        # Create visualizations
        if config.get("visualization", {}).get("plot_training_curves", True):
            print("Creating training curves...")
            visualizer = Visualizer()
            
            plot_dir = Path(config.get("visualization", {}).get("plot_dir", "plots"))
            plot_dir.mkdir(exist_ok=True)
            
            visualizer.plot_training_curves(
                stats,
                save_path=str(plot_dir / "training_curves.png"),
                show=False
            )
            
            if config.get("visualization", {}).get("plot_reward_components", True):
                visualizer.plot_reward_components(
                    stats["reward_components"],
                    save_path=str(plot_dir / "reward_components.png"),
                    show=False
                )
            
            if (config.get("visualization", {}).get("plot_policy_heatmap", True) and 
                hasattr(agent, 'q_table')):
                visualizer.plot_policy_heatmap(
                    agent.q_table,
                    save_path=str(plot_dir / "policy_heatmap.png"),
                    show=False
                )
            
            print(f"Plots saved to {plot_dir}")
    
    # Clean up
    logger.close()
    env.close()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
