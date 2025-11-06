"""
Training utilities for multi-objective reinforcement learning.

This module provides training loops, logging, and visualization tools.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from datetime import datetime
import logging
import wandb
from tensorboard import SummaryWriter

from agents.multi_objective_agents import MultiObjectiveAgent
from envs.multi_objective_envs import make_multi_objective_env


class TrainingLogger:
    """Logger for training statistics and metrics."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        project_name: str = "multi-objective-rl"
    ):
        """
        Initialize the training logger.
        
        Args:
            log_dir: Directory to save logs
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            project_name: Name of the project for wandb
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # Initialize TensorBoard
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(self.log_dir / "tensorboard")
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(project=project_name)
        
        # Initialize file logger
        self.setup_file_logger()
        
        # Statistics storage
        self.stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "epsilon_history": [],
            "loss_history": [],
            "reward_components": []
        }
    
    def setup_file_logger(self) -> None:
        """Setup file logging."""
        log_file = self.log_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        epsilon: float = None,
        loss: float = None,
        reward_components: Tuple[float, ...] = None,
        additional_metrics: Dict[str, float] = None
    ) -> None:
        """Log episode statistics."""
        self.stats["episode_rewards"].append(reward)
        self.stats["episode_lengths"].append(length)
        
        if epsilon is not None:
            self.stats["epsilon_history"].append(epsilon)
        
        if loss is not None:
            self.stats["loss_history"].append(loss)
        
        if reward_components is not None:
            self.stats["reward_components"].append(reward_components)
        
        # Log to console
        self.logger.info(
            f"Episode {episode}: Reward={reward:.2f}, Length={length}, "
            f"Epsilon={epsilon:.3f if epsilon else 'N/A'}, Loss={loss:.4f if loss else 'N/A'}"
        )
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_scalar("Episode/Reward", reward, episode)
            self.tb_writer.add_scalar("Episode/Length", length, episode)
            if epsilon is not None:
                self.tb_writer.add_scalar("Episode/Epsilon", epsilon, episode)
            if loss is not None:
                self.tb_writer.add_scalar("Episode/Loss", loss, episode)
            
            if reward_components is not None:
                for i, component in enumerate(reward_components):
                    self.tb_writer.add_scalar(f"Reward/Component_{i}", component, episode)
        
        # Log to wandb
        if self.use_wandb:
            log_dict = {
                "episode": episode,
                "reward": reward,
                "length": length
            }
            if epsilon is not None:
                log_dict["epsilon"] = epsilon
            if loss is not None:
                log_dict["loss"] = loss
            if reward_components is not None:
                for i, component in enumerate(reward_components):
                    log_dict[f"reward_component_{i}"] = component
            if additional_metrics:
                log_dict.update(additional_metrics)
            
            wandb.log(log_dict)
    
    def log_training_step(
        self,
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log training step metrics."""
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"Training/{key}", value, step)
        
        if self.use_wandb:
            wandb.log({f"training_{key}": value for key, value in metrics.items()})
    
    def save_stats(self, filepath: str) -> None:
        """Save training statistics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def close(self) -> None:
        """Close loggers."""
        if self.use_tensorboard:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()


class MultiObjectiveTrainer:
    """Trainer for multi-objective RL agents."""
    
    def __init__(
        self,
        agent: MultiObjectiveAgent,
        env: Any,
        logger: TrainingLogger,
        config: Dict[str, Any]
    ):
        """
        Initialize the trainer.
        
        Args:
            agent: Multi-objective RL agent
            env: Environment
            logger: Training logger
            config: Training configuration
        """
        self.agent = agent
        self.env = env
        self.logger = logger
        self.config = config
        
        # Training parameters
        self.max_episodes = config.get("max_episodes", 1000)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 1000)
        self.eval_frequency = config.get("eval_frequency", 100)
        self.save_frequency = config.get("save_frequency", 500)
        
        # Evaluation parameters
        self.eval_episodes = config.get("eval_episodes", 10)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train(self) -> Dict[str, List[float]]:
        """Train the agent."""
        self.logger.logger.info("Starting training...")
        
        best_reward = float('-inf')
        
        for episode in range(self.max_episodes):
            # Training episode
            episode_reward, episode_length, episode_stats = self._run_episode(training=True)
            
            # Log episode
            self.logger.log_episode(
                episode=episode,
                reward=episode_reward,
                length=episode_length,
                epsilon=episode_stats.get("epsilon"),
                loss=episode_stats.get("loss"),
                reward_components=episode_stats.get("reward_components")
            )
            
            # Evaluation
            if episode % self.eval_frequency == 0:
                eval_reward, eval_length = self._evaluate()
                self.logger.logger.info(f"Evaluation - Episode {episode}: Reward={eval_reward:.2f}, Length={eval_length:.1f}")
                
                # Save best model
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self._save_checkpoint(episode, is_best=True)
            
            # Regular checkpointing
            if episode % self.save_frequency == 0:
                self._save_checkpoint(episode)
        
        self.logger.logger.info("Training completed!")
        return self.logger.stats
    
    def _run_episode(self, training: bool = True) -> Tuple[float, int, Dict[str, Any]]:
        """Run a single episode."""
        obs, info = self.env.reset()
        total_reward = 0
        episode_length = 0
        episode_stats = {}
        
        for step in range(self.max_steps_per_episode):
            # Select action
            action = self.agent.select_action(obs, training=training)
            
            # Take step
            next_obs, reward_vector, terminated, truncated, info = self.env.step(action)
            
            # Update agent (if training)
            if training:
                if hasattr(self.agent, 'update'):
                    stats = self.agent.update(obs, action, reward_vector, next_obs, terminated or truncated)
                    episode_stats.update(stats)
                elif hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(obs, action, reward_vector, next_obs, terminated or truncated)
                    stats = self.agent.update()
                    episode_stats.update(stats)
            
            # Accumulate reward
            total_reward += self.agent.combine_rewards(reward_vector)
            episode_length += 1
            
            # Store reward components
            if "reward_components" not in episode_stats:
                episode_stats["reward_components"] = reward_vector
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        return total_reward, episode_length, episode_stats
    
    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate the agent."""
        total_rewards = []
        total_lengths = []
        
        for _ in range(self.eval_episodes):
            reward, length, _ = self._run_episode(training=False)
            total_rewards.append(reward)
            total_lengths.append(length)
        
        return np.mean(total_rewards), np.mean(total_lengths)
    
    def _save_checkpoint(self, episode: int, is_best: bool = False) -> None:
        """Save agent checkpoint."""
        suffix = "_best" if is_best else f"_episode_{episode}"
        checkpoint_path = self.checkpoint_dir / f"agent{suffix}.pth"
        
        if hasattr(self.agent, 'save'):
            self.agent.save(str(checkpoint_path))
        
        # Save training stats
        stats_path = self.checkpoint_dir / f"stats{suffix}.json"
        self.logger.save_stats(str(stats_path))


class Visualizer:
    """Visualization tools for multi-objective RL."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize the visualizer."""
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_training_curves(
        self,
        stats: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(stats["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(stats["episode_lengths"])
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Length")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epsilon decay
        if stats["epsilon_history"]:
            axes[1, 0].plot(stats["epsilon_history"])
            axes[1, 0].set_title("Epsilon Decay")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Epsilon")
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss
        if stats["loss_history"]:
            axes[1, 1].plot(stats["loss_history"])
            axes[1, 1].set_title("Training Loss")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_reward_components(
        self,
        reward_components: List[Tuple[float, ...]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Plot reward components over time."""
        if not reward_components:
            return
        
        components = np.array(reward_components)
        n_components = components.shape[1]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i in range(n_components):
            ax.plot(components[:, i], label=f"Component {i+1}", alpha=0.7)
        
        ax.set_title("Reward Components Over Time")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_policy_heatmap(
        self,
        q_table: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Plot Q-table as a heatmap."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(
            q_table.T,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            ax=ax,
            cbar_kws={'label': 'Q-value'}
        )
        
        ax.set_title("Q-Table Heatmap")
        ax.set_xlabel("State")
        ax.set_ylabel("Action")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


if __name__ == "__main__":
    # Example usage
    from agents.multi_objective_agents import MultiObjectiveQAgent
    from envs.multi_objective_envs import MultiObjectiveLineEnv
    
    # Create environment and agent
    env = MultiObjectiveLineEnv()
    agent = MultiObjectiveQAgent(n_states=11, n_actions=5)
    
    # Create logger
    logger = TrainingLogger()
    
    # Create trainer
    config = {
        "max_episodes": 100,
        "max_steps_per_episode": 100,
        "eval_frequency": 20,
        "save_frequency": 50
    }
    
    trainer = MultiObjectiveTrainer(agent, env, logger, config)
    
    # Train agent
    stats = trainer.train()
    
    # Visualize results
    visualizer = Visualizer()
    visualizer.plot_training_curves(stats)
    
    # Clean up
    logger.close()
    env.close()
