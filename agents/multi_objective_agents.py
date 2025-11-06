"""
Multi-objective reinforcement learning agents.

This module provides various agents for multi-objective RL, including
traditional Q-learning and modern deep RL algorithms.
"""

from typing import Tuple, Dict, Any, Optional, List, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
from abc import ABC, abstractmethod
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import wandb


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class MultiObjectiveAgent(ABC):
    """Abstract base class for multi-objective RL agents."""
    
    def __init__(self, weights: Tuple[float, ...]):
        """
        Initialize the agent.
        
        Args:
            weights: Weight vector for combining multiple objectives
        """
        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)  # Normalize weights
        
    @abstractmethod
    def select_action(self, state: Any, training: bool = True) -> Any:
        """Select an action given a state."""
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """Update the agent's policy."""
        pass
    
    def combine_rewards(self, reward_vector: Tuple[float, ...]) -> float:
        """Combine multiple rewards into a scalar using weighted sum."""
        return np.dot(self.weights, reward_vector)


class MultiObjectiveQAgent(MultiObjectiveAgent):
    """
    Multi-objective Q-learning agent with epsilon-greedy exploration.
    
    This is the modernized version of the original MultiObjectiveAgent.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        weights: Tuple[float, float] = (1.0, 0.5),
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize the multi-objective Q-learning agent.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            weights: Weight vector for combining objectives
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        super().__init__(weights)
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Training statistics
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "epsilon_history": [],
            "loss_history": []
        }
    
    def select_action(self, state: int, training: bool = True) -> int:
        """Select an action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(
        self, 
        state: int, 
        action: int, 
        reward_vector: Tuple[float, ...], 
        next_state: int, 
        done: bool
    ) -> Dict[str, float]:
        """Update Q-values using the Q-learning algorithm."""
        # Combine rewards
        scalar_reward = self.combine_rewards(reward_vector)
        
        # Q-learning update
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state]) if not done else 0
        target_q = scalar_reward + self.discount_factor * max_next_q
        
        # Update Q-value
        td_error = target_q - current_q
        self.q_table[state, action] += self.learning_rate * td_error
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            "td_error": abs(td_error),
            "epsilon": self.epsilon,
            "scalar_reward": scalar_reward
        }
    
    def get_policy(self) -> np.ndarray:
        """Get the current policy (greedy)."""
        return np.argmax(self.q_table, axis=1)
    
    def save(self, filepath: str) -> None:
        """Save the agent's Q-table and parameters."""
        np.savez(
            filepath,
            q_table=self.q_table,
            weights=self.weights,
            epsilon=self.epsilon,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor
        )
    
    def load(self, filepath: str) -> None:
        """Load the agent's Q-table and parameters."""
        data = np.load(filepath)
        self.q_table = data['q_table']
        self.weights = data['weights']
        self.epsilon = float(data['epsilon'])
        self.learning_rate = float(data['learning_rate'])
        self.discount_factor = float(data['discount_factor'])


class MultiObjectiveDQNAgent(MultiObjectiveAgent):
    """
    Multi-objective Deep Q-Network agent with experience replay.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        weights: Tuple[float, float] = (1.0, 0.5),
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100
    ):
        """Initialize the multi-objective DQN agent."""
        super().__init__(weights)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self._build_network(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = self._build_network(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training statistics
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "epsilon_history": [],
            "loss_history": []
        }
        
        self.update_count = 0
    
    def _build_network(self, state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
        """Build the Q-network architecture."""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: int, 
        reward_vector: Tuple[float, ...], 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Store a transition in the replay buffer."""
        scalar_reward = self.combine_rewards(reward_vector)
        transition = Transition(state, action, scalar_reward, next_state, done)
        self.replay_buffer.append(transition)
    
    def update(self) -> Dict[str, float]:
        """Update the Q-network using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        batch = Transition(*zip(*batch))
        
        # Convert to tensors
        states = torch.FloatTensor(batch.state).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)
        dones = torch.BoolTensor(batch.done).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon
        }
    
    def save(self, filepath: str) -> None:
        """Save the agent's networks and parameters."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'weights': self.weights,
            'epsilon': self.epsilon,
            'training_stats': self.training_stats
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's networks and parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.weights = checkpoint['weights']
        self.epsilon = checkpoint['epsilon']
        self.training_stats = checkpoint['training_stats']


class MultiObjectivePPOAgent(MultiObjectiveAgent):
    """
    Multi-objective PPO agent using stable-baselines3.
    """
    
    def __init__(
        self,
        env: gym.Env,
        weights: Tuple[float, float] = (1.0, 0.5),
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 1
    ):
        """Initialize the multi-objective PPO agent."""
        super().__init__(weights)
        
        self.env = env
        self.weights = weights
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose
        )
        
        # Training statistics
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_loss": [],
            "value_loss": []
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using the PPO policy."""
        action, _ = self.model.predict(state, deterministic=not training)
        return action
    
    def update(self, total_timesteps: int = 10000) -> Dict[str, float]:
        """Train the PPO agent."""
        self.model.learn(total_timesteps=total_timesteps)
        
        # Get training statistics
        if hasattr(self.model, 'logger') and self.model.logger:
            stats = {
                "policy_loss": self.model.logger.name_to_value.get("train/policy_loss", 0.0),
                "value_loss": self.model.logger.name_to_value.get("train/value_loss", 0.0),
                "explained_variance": self.model.logger.name_to_value.get("train/explained_variance", 0.0)
            }
        else:
            stats = {"policy_loss": 0.0, "value_loss": 0.0, "explained_variance": 0.0}
        
        return stats
    
    def save(self, filepath: str) -> None:
        """Save the PPO model."""
        self.model.save(filepath)
    
    def load(self, filepath: str) -> None:
        """Load the PPO model."""
        self.model = PPO.load(filepath, env=self.env)


class MultiObjectiveRewardWrapper(gym.Wrapper):
    """
    Wrapper to convert multi-objective rewards to scalar rewards for stable-baselines3.
    """
    
    def __init__(self, env: gym.Env, weights: Tuple[float, ...]):
        """
        Initialize the reward wrapper.
        
        Args:
            env: Environment to wrap
            weights: Weight vector for combining multiple objectives
        """
        super().__init__(env)
        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)  # Normalize weights
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one step and combine multi-objective rewards."""
        observation, reward_vector, terminated, truncated, info = self.env.step(action)
        
        # Combine rewards
        scalar_reward = np.dot(self.weights, reward_vector)
        
        # Add original reward vector to info
        info["reward_vector"] = reward_vector
        info["scalar_reward"] = scalar_reward
        
        return observation, scalar_reward, terminated, truncated, info


def create_multi_objective_agent(
    agent_type: str,
    env: gym.Env,
    weights: Tuple[float, ...] = (1.0, 0.5),
    **kwargs
) -> MultiObjectiveAgent:
    """
    Factory function to create multi-objective agents.
    
    Args:
        agent_type: Type of agent ('qlearning', 'dqn', 'ppo')
        env: Environment
        weights: Weight vector for combining objectives
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        Multi-objective agent instance
    """
    if agent_type == "qlearning":
        return MultiObjectiveQAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            weights=weights,
            **kwargs
        )
    elif agent_type == "dqn":
        return MultiObjectiveDQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            weights=weights,
            **kwargs
        )
    elif agent_type == "ppo":
        # Wrap environment for PPO
        wrapped_env = MultiObjectiveRewardWrapper(env, weights)
        return MultiObjectivePPOAgent(wrapped_env, weights, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    # Test the agents
    from envs.multi_objective_envs import MultiObjectiveLineEnv
    
    print("Testing Multi-Objective Q-Learning Agent...")
    env = MultiObjectiveLineEnv()
    agent = MultiObjectiveQAgent(n_states=11, n_actions=5)
    
    obs, info = env.reset()
    for step in range(10):
        action = agent.select_action(obs)
        next_obs, reward_vec, terminated, truncated, info = env.step(action)
        stats = agent.update(obs, action, reward_vec, next_obs, terminated or truncated)
        obs = next_obs
        
        if terminated or truncated:
            break
    
    print("Agent test completed!")
