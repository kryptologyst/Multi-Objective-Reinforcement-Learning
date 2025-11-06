"""
Unit tests for multi-objective reinforcement learning components.

This module provides comprehensive tests for environments, agents, and training utilities.
"""

import unittest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from envs.multi_objective_envs import (
    MultiObjectiveLineEnv, 
    MultiObjectiveCartPoleEnv, 
    MultiObjectiveMountainCarEnv,
    make_multi_objective_env
)
from agents.multi_objective_agents import (
    MultiObjectiveQAgent,
    MultiObjectiveDQNAgent,
    MultiObjectivePPOAgent,
    MultiObjectiveRewardWrapper,
    create_multi_objective_agent
)
from src.training import TrainingLogger, MultiObjectiveTrainer, Visualizer


class TestMultiObjectiveEnvironments(unittest.TestCase):
    """Test cases for multi-objective environments."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.line_env = MultiObjectiveLineEnv()
        self.cartpole_env = MultiObjectiveCartPoleEnv()
        self.mountaincar_env = MultiObjectiveMountainCarEnv()
    
    def test_line_env_initialization(self):
        """Test line environment initialization."""
        self.assertEqual(self.line_env.goal_position, 10)
        self.assertEqual(self.line_env.max_position, 10)
        self.assertEqual(self.line_env.position, 0)
        self.assertEqual(self.line_env.step_count, 0)
    
    def test_line_env_reset(self):
        """Test line environment reset."""
        obs, info = self.line_env.reset()
        self.assertEqual(obs, 0)
        self.assertEqual(self.line_env.position, 0)
        self.assertEqual(self.line_env.step_count, 0)
        self.assertIn("position", info)
        self.assertIn("step_count", info)
    
    def test_line_env_step(self):
        """Test line environment step."""
        obs, info = self.line_env.reset()
        
        # Test valid action
        action = 2  # Move right by 2
        next_obs, reward_vec, terminated, truncated, info = self.line_env.step(action)
        
        self.assertEqual(next_obs, 2)
        self.assertEqual(len(reward_vec), 2)
        self.assertIsInstance(reward_vec[0], float)  # Progress reward
        self.assertIsInstance(reward_vec[1], float)  # Energy penalty
        self.assertFalse(terminated)
        self.assertFalse(truncated)
    
    def test_line_env_boundaries(self):
        """Test line environment boundary conditions."""
        obs, info = self.line_env.reset()
        
        # Test moving beyond boundaries
        action = 0  # Stay in place
        next_obs, reward_vec, terminated, truncated, info = self.line_env.step(action)
        self.assertEqual(next_obs, 0)
        
        # Test reaching goal
        self.line_env.position = 9
        action = 1  # Move right by 1
        next_obs, reward_vec, terminated, truncated, info = self.line_env.step(action)
        self.assertEqual(next_obs, 10)
        self.assertTrue(terminated)
        self.assertEqual(reward_vec[0], 1.0)  # Progress reward
    
    def test_cartpole_env_initialization(self):
        """Test CartPole environment initialization."""
        self.assertIsNotNone(self.cartpole_env.base_env)
        self.assertEqual(self.cartpole_env.action_space.n, 2)
        self.assertEqual(len(self.cartpole_env.observation_space.shape), 1)
    
    def test_mountaincar_env_initialization(self):
        """Test MountainCar environment initialization."""
        self.assertIsNotNone(self.mountaincar_env.base_env)
        self.assertEqual(self.mountaincar_env.action_space.n, 3)
        self.assertEqual(len(self.mountaincar_env.observation_space.shape), 1)
    
    def test_make_multi_objective_env(self):
        """Test environment factory function."""
        env = make_multi_objective_env("line")
        self.assertIsInstance(env, MultiObjectiveLineEnv)
        
        env = make_multi_objective_env("cartpole")
        self.assertIsInstance(env, MultiObjectiveCartPoleEnv)
        
        env = make_multi_objective_env("mountaincar")
        self.assertIsInstance(env, MultiObjectiveMountainCarEnv)
        
        with self.assertRaises(ValueError):
            make_multi_objective_env("invalid")


class TestMultiObjectiveAgents(unittest.TestCase):
    """Test cases for multi-objective agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MultiObjectiveLineEnv()
        self.q_agent = MultiObjectiveQAgent(n_states=11, n_actions=5)
        self.dqn_agent = MultiObjectiveDQNAgent(state_dim=1, action_dim=5)
    
    def test_q_agent_initialization(self):
        """Test Q-learning agent initialization."""
        self.assertEqual(self.q_agent.n_states, 11)
        self.assertEqual(self.q_agent.n_actions, 5)
        self.assertEqual(self.q_agent.q_table.shape, (11, 5))
        self.assertEqual(len(self.q_agent.weights), 2)
    
    def test_q_agent_action_selection(self):
        """Test Q-learning agent action selection."""
        state = 5
        action = self.q_agent.select_action(state, training=True)
        self.assertIn(action, range(5))
        
        # Test greedy action selection
        action = self.q_agent.select_action(state, training=False)
        self.assertIn(action, range(5))
    
    def test_q_agent_update(self):
        """Test Q-learning agent update."""
        state = 5
        action = 2
        reward_vector = (0.5, -0.1)
        next_state = 7
        done = False
        
        initial_q = self.q_agent.q_table[state, action].copy()
        stats = self.q_agent.update(state, action, reward_vector, next_state, done)
        
        # Check that Q-value was updated
        self.assertNotEqual(self.q_agent.q_table[state, action], initial_q)
        
        # Check returned statistics
        self.assertIn("td_error", stats)
        self.assertIn("epsilon", stats)
        self.assertIn("scalar_reward", stats)
    
    def test_q_agent_reward_combination(self):
        """Test Q-learning agent reward combination."""
        reward_vector = (1.0, -0.5)
        combined_reward = self.q_agent.combine_rewards(reward_vector)
        
        expected_reward = np.dot(self.q_agent.weights, reward_vector)
        self.assertAlmostEqual(combined_reward, expected_reward)
    
    def test_dqn_agent_initialization(self):
        """Test DQN agent initialization."""
        self.assertEqual(self.dqn_agent.state_dim, 1)
        self.assertEqual(self.dqn_agent.action_dim, 5)
        self.assertIsNotNone(self.dqn_agent.q_network)
        self.assertIsNotNone(self.dqn_agent.target_network)
        self.assertIsNotNone(self.dqn_agent.optimizer)
    
    def test_dqn_agent_action_selection(self):
        """Test DQN agent action selection."""
        state = np.array([0.5])
        action = self.dqn_agent.select_action(state, training=True)
        self.assertIn(action, range(5))
    
    def test_dqn_agent_experience_replay(self):
        """Test DQN agent experience replay."""
        state = np.array([0.5])
        action = 2
        reward_vector = (0.5, -0.1)
        next_state = np.array([0.7])
        done = False
        
        # Store transition
        self.dqn_agent.store_transition(state, action, reward_vector, next_state, done)
        
        # Check that transition was stored
        self.assertEqual(len(self.dqn_agent.replay_buffer), 1)
        
        # Test update (should not fail with empty buffer)
        stats = self.dqn_agent.update()
        self.assertIn("loss", stats)
    
    def test_reward_wrapper(self):
        """Test reward wrapper."""
        weights = (0.7, 0.3)
        wrapped_env = MultiObjectiveRewardWrapper(self.env, weights)
        
        obs, info = wrapped_env.reset()
        action = 1
        next_obs, scalar_reward, terminated, truncated, info = wrapped_env.step(action)
        
        # Check that reward was combined
        self.assertIsInstance(scalar_reward, float)
        self.assertIn("reward_vector", info)
        self.assertIn("scalar_reward", info)
    
    def test_create_multi_objective_agent(self):
        """Test agent factory function."""
        agent = create_multi_objective_agent("qlearning", self.env)
        self.assertIsInstance(agent, MultiObjectiveQAgent)
        
        agent = create_multi_objective_agent("dqn", self.env)
        self.assertIsInstance(agent, MultiObjectiveDQNAgent)
        
        with self.assertRaises(ValueError):
            create_multi_objective_agent("invalid", self.env)


class TestTrainingUtilities(unittest.TestCase):
    """Test cases for training utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MultiObjectiveLineEnv()
        self.agent = MultiObjectiveQAgent(n_states=11, n_actions=5)
        self.logger = TrainingLogger(use_wandb=False, use_tensorboard=False)
    
    def test_training_logger_initialization(self):
        """Test training logger initialization."""
        self.assertIsNotNone(self.logger.log_dir)
        self.assertFalse(self.logger.use_wandb)
        self.assertFalse(self.logger.use_tensorboard)
        self.assertIsNotNone(self.logger.stats)
    
    def test_training_logger_log_episode(self):
        """Test training logger episode logging."""
        episode = 1
        reward = 10.5
        length = 25
        epsilon = 0.1
        loss = 0.05
        reward_components = (1.0, -0.5)
        
        self.logger.log_episode(episode, reward, length, epsilon, loss, reward_components)
        
        self.assertEqual(len(self.logger.stats["episode_rewards"]), 1)
        self.assertEqual(len(self.logger.stats["episode_lengths"]), 1)
        self.assertEqual(len(self.logger.stats["epsilon_history"]), 1)
        self.assertEqual(len(self.logger.stats["loss_history"]), 1)
        self.assertEqual(len(self.logger.stats["reward_components"]), 1)
    
    def test_training_logger_save_stats(self):
        """Test training logger stats saving."""
        # Add some dummy stats
        self.logger.stats["episode_rewards"] = [1.0, 2.0, 3.0]
        self.logger.stats["episode_lengths"] = [10, 20, 30]
        
        # Save stats
        stats_path = "test_stats.json"
        self.logger.save_stats(stats_path)
        
        # Check that file was created
        self.assertTrue(Path(stats_path).exists())
        
        # Clean up
        Path(stats_path).unlink()
    
    def test_multi_objective_trainer_initialization(self):
        """Test multi-objective trainer initialization."""
        config = {
            "max_episodes": 100,
            "max_steps_per_episode": 50,
            "eval_frequency": 20,
            "save_frequency": 50,
            "eval_episodes": 5,
            "checkpoint_dir": "test_checkpoints"
        }
        
        trainer = MultiObjectiveTrainer(self.agent, self.env, self.logger, config)
        
        self.assertEqual(trainer.max_episodes, 100)
        self.assertEqual(trainer.max_steps_per_episode, 50)
        self.assertEqual(trainer.eval_frequency, 20)
        self.assertEqual(trainer.save_frequency, 50)
        self.assertEqual(trainer.eval_episodes, 5)
    
    def test_trainer_run_episode(self):
        """Test trainer episode execution."""
        config = {
            "max_episodes": 10,
            "max_steps_per_episode": 10,
            "eval_frequency": 5,
            "save_frequency": 10,
            "eval_episodes": 3,
            "checkpoint_dir": "test_checkpoints"
        }
        
        trainer = MultiObjectiveTrainer(self.agent, self.env, self.logger, config)
        
        # Run a single episode
        reward, length, stats = trainer._run_episode(training=True)
        
        self.assertIsInstance(reward, float)
        self.assertIsInstance(length, int)
        self.assertIsInstance(stats, dict)
        self.assertGreaterEqual(length, 1)
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        visualizer = Visualizer()
        self.assertIsNotNone(visualizer)
    
    def test_visualizer_plot_training_curves(self):
        """Test visualizer training curves plotting."""
        visualizer = Visualizer()
        
        stats = {
            "episode_rewards": [1.0, 2.0, 3.0],
            "episode_lengths": [10, 20, 30],
            "epsilon_history": [0.2, 0.1, 0.05],
            "loss_history": [0.5, 0.3, 0.1]
        }
        
        # Should not raise an exception
        visualizer.plot_training_curves(stats, show=False)
    
    def test_visualizer_plot_policy_heatmap(self):
        """Test visualizer policy heatmap plotting."""
        visualizer = Visualizer()
        
        q_table = np.random.rand(11, 5)
        
        # Should not raise an exception
        visualizer.plot_policy_heatmap(q_table, show=False)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training process."""
        # Create environment and agent
        env = MultiObjectiveLineEnv()
        agent = MultiObjectiveQAgent(n_states=11, n_actions=5)
        logger = TrainingLogger(use_wandb=False, use_tensorboard=False)
        
        # Training configuration
        config = {
            "max_episodes": 10,
            "max_steps_per_episode": 20,
            "eval_frequency": 5,
            "save_frequency": 10,
            "eval_episodes": 3,
            "checkpoint_dir": "test_checkpoints"
        }
        
        # Create trainer
        trainer = MultiObjectiveTrainer(agent, env, logger, config)
        
        # Run training
        stats = trainer.train()
        
        # Check that training completed successfully
        self.assertIn("episode_rewards", stats)
        self.assertIn("episode_lengths", stats)
        self.assertEqual(len(stats["episode_rewards"]), 10)
        self.assertEqual(len(stats["episode_lengths"]), 10)
        
        # Clean up
        env.close()
        logger.close()
    
    def test_agent_save_load(self):
        """Test agent checkpointing."""
        # Create agent
        agent = MultiObjectiveQAgent(n_states=11, n_actions=5)
        
        # Modify Q-table
        agent.q_table[0, 0] = 1.0
        
        # Save agent
        checkpoint_path = "test_agent.npz"
        agent.save(checkpoint_path)
        
        # Create new agent and load
        new_agent = MultiObjectiveQAgent(n_states=11, n_actions=5)
        new_agent.load(checkpoint_path)
        
        # Check that Q-table was loaded correctly
        self.assertEqual(new_agent.q_table[0, 0], 1.0)
        
        # Clean up
        Path(checkpoint_path).unlink()


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestMultiObjectiveEnvironments))
    test_suite.addTest(unittest.makeSuite(TestMultiObjectiveAgents))
    test_suite.addTest(unittest.makeSuite(TestTrainingUtilities))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
