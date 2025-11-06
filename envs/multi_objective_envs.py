"""
Multi-objective reinforcement learning environments.

This module provides various environments for multi-objective RL, including
the original 1D line environment and additional gymnasium-based environments.
"""

from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class MultiObjectiveLineEnv(gym.Env):
    """
    A 1D line environment where an agent must reach a goal while balancing
    speed and energy consumption.
    
    This is the modernized version of the original MultiObjectiveEnv.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        goal_position: int = 10,
        max_position: int = 10,
        action_space_size: int = 5,
        energy_penalty_factor: float = 0.1,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the multi-objective line environment.
        
        Args:
            goal_position: Target position to reach
            max_position: Maximum position allowed
            action_space_size: Number of discrete actions (should be odd)
            energy_penalty_factor: Penalty factor for energy consumption
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.goal_position = goal_position
        self.max_position = max_position
        self.energy_penalty_factor = energy_penalty_factor
        self.render_mode = render_mode
        
        # Action space: step sizes from -2 to +2
        self.action_space = spaces.Discrete(action_space_size)
        self.actions = np.linspace(-2, 2, action_space_size, dtype=int)
        
        # State space: position from 0 to max_position
        self.observation_space = spaces.Discrete(max_position + 1)
        
        # Initialize state
        self.position = 0
        self.step_count = 0
        self.max_steps = 100
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.position = 0
        self.step_count = 0
        
        observation = self.position
        info = {
            "position": self.position,
            "step_count": self.step_count,
            "goal_position": self.goal_position
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[int, Tuple[float, float], bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Get actual step size from action index
        step_size = self.actions[action]
        
        # Update position
        self.position += step_size
        self.position = np.clip(self.position, 0, self.max_position)
        self.step_count += 1
        
        # Calculate multi-objective reward
        progress_reward = 1.0 if self.position == self.goal_position else 0.0
        energy_penalty = -abs(step_size) * self.energy_penalty_factor
        
        reward_vector = (progress_reward, energy_penalty)
        
        # Check termination conditions
        terminated = self.position == self.goal_position
        truncated = self.step_count >= self.max_steps
        
        info = {
            "position": self.position,
            "step_count": self.step_count,
            "step_size": step_size,
            "progress_reward": progress_reward,
            "energy_penalty": energy_penalty,
            "reward_vector": reward_vector
        }
        
        return self.position, reward_vector, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Position: {self.position}/{self.goal_position}, Step: {self.step_count}")
        elif self.render_mode == "rgb_array":
            # Create a simple visualization
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.set_xlim(0, self.max_position)
            ax.set_ylim(-0.5, 0.5)
            
            # Draw line
            ax.plot([0, self.max_position], [0, 0], 'k-', linewidth=2)
            
            # Draw goal
            ax.plot(self.goal_position, 0, 'go', markersize=15, label='Goal')
            
            # Draw agent
            ax.plot(self.position, 0, 'ro', markersize=10, label='Agent')
            
            ax.set_title(f"Multi-Objective Line Environment (Step {self.step_count})")
            ax.set_xlabel("Position")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Convert plot to numpy array
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            return image
        
        return None
    
    def close(self) -> None:
        """Close the environment."""
        pass


class MultiObjectiveCartPoleEnv(gym.Env):
    """
    Multi-objective version of CartPole where the agent must balance the pole
    while minimizing energy consumption.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        energy_penalty_factor: float = 0.01,
        render_mode: Optional[str] = None
    ):
        """Initialize the multi-objective CartPole environment."""
        super().__init__()
        
        self.energy_penalty_factor = energy_penalty_factor
        self.render_mode = render_mode
        
        # Use the original CartPole environment as base
        self.base_env = gym.make("CartPole-v1", render_mode=render_mode)
        
        # Copy spaces
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        observation, info = self.base_env.reset(seed=seed, options=options)
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, Tuple[float, float], bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        observation, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Calculate multi-objective reward
        progress_reward = reward  # Original CartPole reward
        energy_penalty = -abs(action) * self.energy_penalty_factor  # Penalty for action magnitude
        
        reward_vector = (progress_reward, energy_penalty)
        
        # Add additional info
        info.update({
            "progress_reward": progress_reward,
            "energy_penalty": energy_penalty,
            "reward_vector": reward_vector
        })
        
        return observation, reward_vector, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        return self.base_env.render()
    
    def close(self) -> None:
        """Close the environment."""
        self.base_env.close()


class MultiObjectiveMountainCarEnv(gym.Env):
    """
    Multi-objective version of MountainCar where the agent must reach the goal
    while minimizing energy consumption.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        energy_penalty_factor: float = 0.1,
        render_mode: Optional[str] = None
    ):
        """Initialize the multi-objective MountainCar environment."""
        super().__init__()
        
        self.energy_penalty_factor = energy_penalty_factor
        self.render_mode = render_mode
        
        # Use the original MountainCar environment as base
        self.base_env = gym.make("MountainCar-v0", render_mode=render_mode)
        
        # Copy spaces
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        observation, info = self.base_env.reset(seed=seed, options=options)
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, Tuple[float, float], bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        observation, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Calculate multi-objective reward
        progress_reward = reward  # Original MountainCar reward
        energy_penalty = -abs(action) * self.energy_penalty_factor  # Penalty for action magnitude
        
        reward_vector = (progress_reward, energy_penalty)
        
        # Add additional info
        info.update({
            "progress_reward": progress_reward,
            "energy_penalty": energy_penalty,
            "reward_vector": reward_vector
        })
        
        return observation, reward_vector, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        return self.base_env.render()
    
    def close(self) -> None:
        """Close the environment."""
        self.base_env.close()


def make_multi_objective_env(
    env_name: str,
    **kwargs
) -> Union[MultiObjectiveLineEnv, MultiObjectiveCartPoleEnv, MultiObjectiveMountainCarEnv]:
    """
    Factory function to create multi-objective environments.
    
    Args:
        env_name: Name of the environment ('line', 'cartpole', 'mountaincar')
        **kwargs: Additional arguments for environment initialization
        
    Returns:
        Multi-objective environment instance
    """
    env_map = {
        "line": MultiObjectiveLineEnv,
        "cartpole": MultiObjectiveCartPoleEnv,
        "mountaincar": MultiObjectiveMountainCarEnv
    }
    
    if env_name not in env_map:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(env_map.keys())}")
    
    return env_map[env_name](**kwargs)


if __name__ == "__main__":
    # Test the environments
    print("Testing Multi-Objective Line Environment...")
    env = MultiObjectiveLineEnv(render_mode="human")
    obs, info = env.reset()
    print(f"Initial observation: {obs}, Info: {info}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward_vec, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Action={action}, Obs={obs}, Reward={reward_vec}, Done={terminated or truncated}")
        env.render()
        
        if terminated or truncated:
            break
    
    env.close()
    print("Environment test completed!")
