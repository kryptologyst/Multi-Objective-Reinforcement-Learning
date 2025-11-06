# Multi-Objective Reinforcement Learning

A comprehensive implementation of multi-objective reinforcement learning (MORL) that allows agents to balance multiple competing objectives such as speed vs energy efficiency, performance vs cost, or risk vs return.

## Features

- **Multiple Environments**: 1D Line, CartPole, and MountainCar environments with multi-objective rewards
- **State-of-the-Art Algorithms**: Q-Learning, Deep Q-Network (DQN), and Proximal Policy Optimization (PPO)
- **Modern Libraries**: Built with Gymnasium, Stable-Baselines3, PyTorch, and NumPy
- **Interactive Web Interface**: Streamlit-based UI for training and visualization
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Visualization Tools**: Training curves, reward components, and policy heatmaps
- **Configuration System**: YAML-based configuration management
- **Unit Tests**: Comprehensive test suite for all components
- **Type Hints**: Full type annotations for better code quality

## 📁 Project Structure

```
├── agents/                    # RL agent implementations
│   └── multi_objective_agents.py
├── envs/                      # Environment implementations
│   └── multi_objective_envs.py
├── src/                       # Core training utilities
│   └── training.py
├── config/                    # Configuration files
│   └── default.yaml
├── tests/                     # Unit tests
│   └── test_multi_objective_rl.py
├── notebooks/                 # Jupyter notebooks (optional)
├── logs/                      # Training logs
├── checkpoints/               # Model checkpoints
├── plots/                     # Generated plots
├── train.py                   # Main training script
├── app.py                     # Streamlit web interface
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Multi-Objective-Reinforcement-Learning.git
   cd Multi-Objective-Reinforcement-Learning
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Command Line Training

Train a Q-learning agent on the line environment:

```bash
python train.py --env line --agent qlearning --episodes 500
```

Train a DQN agent on CartPole:

```bash
python train.py --env cartpole --agent dqn --episodes 1000 --weights 1.0 0.5
```

Train a PPO agent with custom configuration:

```bash
python train.py --config config/default.yaml --env mountaincar --agent ppo
```

### Web Interface

Launch the interactive Streamlit interface:

```bash
streamlit run app.py
```

This opens a web interface where you can:
- Configure training parameters
- Visualize training progress in real-time
- Download training results
- Explore different environments and algorithms

### Python API

```python
from envs.multi_objective_envs import make_multi_objective_env
from agents.multi_objective_agents import create_multi_objective_agent
from src.training import MultiObjectiveTrainer, TrainingLogger

# Create environment and agent
env = make_multi_objective_env("line")
agent = create_multi_objective_agent("qlearning", env, weights=(1.0, 0.5))

# Create trainer
logger = TrainingLogger()
config = {"max_episodes": 100, "max_steps_per_episode": 100}
trainer = MultiObjectiveTrainer(agent, env, logger, config)

# Train agent
stats = trainer.train()
```

## Architecture

### Environments

- **MultiObjectiveLineEnv**: 1D line environment where agents move from position 0 to 10
- **MultiObjectiveCartPoleEnv**: CartPole with energy consumption penalty
- **MultiObjectiveMountainCarEnv**: MountainCar with energy efficiency objective

Each environment returns a reward vector with multiple components that can be weighted differently.

### Agents

- **MultiObjectiveQAgent**: Traditional Q-learning with epsilon-greedy exploration
- **MultiObjectiveDQNAgent**: Deep Q-Network with experience replay
- **MultiObjectivePPOAgent**: Proximal Policy Optimization using Stable-Baselines3

All agents support multi-objective rewards through weighted combination.

### Training System

- **MultiObjectiveTrainer**: Handles training loops, evaluation, and checkpointing
- **TrainingLogger**: Manages logging to console, files, TensorBoard, and W&B
- **Visualizer**: Creates training curves, reward component plots, and policy visualizations

## Configuration

Configuration is managed through YAML files. See `config/default.yaml` for all available options:

```yaml
# Environment settings
environment:
  name: "line"
  goal_position: 10
  energy_penalty_factor: 0.1

# Agent settings
agent:
  type: "qlearning"
  weights: [1.0, 0.5]
  learning_rate: 0.1
  epsilon: 0.2

# Training settings
training:
  max_episodes: 1000
  eval_frequency: 100
  save_frequency: 500
```

## Visualization

The project includes comprehensive visualization tools:

- **Training Curves**: Episode rewards, lengths, epsilon decay, and loss
- **Reward Components**: Individual objective performance over time
- **Policy Heatmaps**: Q-table visualization for Q-learning agents
- **Real-time Monitoring**: Live training progress in Streamlit interface

## Testing

Run the comprehensive test suite:

```bash
python tests/test_multi_objective_rl.py
```

Or use pytest:

```bash
pytest tests/ -v
```

Tests cover:
- Environment functionality
- Agent behavior
- Training utilities
- Integration scenarios

## Logging and Monitoring

### TensorBoard

View training metrics in TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

### Weights & Biases

Enable W&B logging by setting `use_wandb: true` in config or using `--wandb` flag.

### File Logging

Training logs are saved to `logs/training.log` with detailed episode information.

## 🔧 Advanced Usage

### Custom Environments

Create custom multi-objective environments by inheriting from `gym.Env`:

```python
class CustomMultiObjectiveEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action and observation spaces
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(...)
    
    def step(self, action):
        # Return (observation, reward_vector, terminated, truncated, info)
        return obs, (reward1, reward2), done, truncated, info
```

### Custom Agents

Implement custom agents by inheriting from `MultiObjectiveAgent`:

```python
class CustomAgent(MultiObjectiveAgent):
    def select_action(self, state, training=True):
        # Implement action selection
        pass
    
    def update(self, *args, **kwargs):
        # Implement policy update
        pass
```

### Hyperparameter Tuning

Use the configuration system to experiment with different hyperparameters:

```bash
# Different weight combinations
python train.py --weights 1.0 0.0  # Only progress
python train.py --weights 0.5 0.5  # Balanced
python train.py --weights 0.0 1.0  # Only energy efficiency
```

## Examples

### Example 1: Basic Training

```python
from envs.multi_objective_envs import MultiObjectiveLineEnv
from agents.multi_objective_agents import MultiObjectiveQAgent

# Create environment and agent
env = MultiObjectiveLineEnv()
agent = MultiObjectiveQAgent(n_states=11, n_actions=5)

# Training loop
for episode in range(100):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(obs)
        next_obs, reward_vec, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward_vec, next_obs, terminated or truncated)
        obs = next_obs
        done = terminated or truncated
```

### Example 2: Evaluation

```python
# Evaluate trained agent
eval_rewards = []
for _ in range(10):
    obs, info = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(obs, training=False)
        obs, reward_vec, terminated, truncated, info = env.step(action)
        episode_reward += agent.combine_rewards(reward_vec)
        done = terminated or truncated
    
    eval_rewards.append(episode_reward)

print(f"Average evaluation reward: {np.mean(eval_rewards):.2f}")
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for the environment framework
- Stable-Baselines3 for modern RL algorithms
- PyTorch for deep learning capabilities
- Streamlit for the web interface
- The RL community for inspiration and resources

## Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the documentation
- Review the test cases for usage examples
 
 
# Multi-Objective-Reinforcement-Learning
