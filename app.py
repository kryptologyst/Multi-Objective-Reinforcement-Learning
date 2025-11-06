"""
Streamlit web interface for multi-objective reinforcement learning.

This module provides an interactive web interface for training and visualizing
multi-objective RL agents.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import yaml
import sys
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from envs.multi_objective_envs import make_multi_objective_env
from agents.multi_objective_agents import create_multi_objective_agent
from src.training import MultiObjectiveTrainer, TrainingLogger, Visualizer, load_config


def setup_page():
    """Setup the Streamlit page configuration."""
    st.set_page_config(
        page_title="Multi-Objective RL",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Multi-Objective Reinforcement Learning")
    st.markdown("""
    This interface allows you to train and visualize multi-objective RL agents
    that balance multiple competing objectives (e.g., speed vs energy efficiency).
    """)


def create_sidebar():
    """Create the sidebar with configuration options."""
    st.sidebar.header("Configuration")
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["line", "cartpole", "mountaincar"],
        index=0,
        help="Choose the environment to train on"
    )
    
    # Agent selection
    agent_type = st.sidebar.selectbox(
        "Agent Type",
        ["qlearning", "dqn", "ppo"],
        index=0,
        help="Choose the RL algorithm"
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    max_episodes = st.sidebar.slider(
        "Max Episodes",
        min_value=10,
        max_value=2000,
        value=500,
        step=10
    )
    
    # Weight configuration
    st.sidebar.subheader("Objective Weights")
    weight1 = st.sidebar.slider(
        "Progress Weight",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1
    )
    weight2 = st.sidebar.slider(
        "Energy Weight",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1
    )
    
    # Agent-specific parameters
    st.sidebar.subheader("Agent Parameters")
    if agent_type == "qlearning":
        learning_rate = st.sidebar.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01
        )
        epsilon = st.sidebar.slider(
            "Epsilon",
            min_value=0.01,
            max_value=1.0,
            value=0.2,
            step=0.01
        )
        agent_params = {
            "learning_rate": learning_rate,
            "epsilon": epsilon
        }
    elif agent_type == "dqn":
        learning_rate = st.sidebar.slider(
            "Learning Rate",
            min_value=1e-5,
            max_value=1e-2,
            value=1e-3,
            step=1e-5,
            format="%.5f"
        )
        batch_size = st.sidebar.slider(
            "Batch Size",
            min_value=16,
            max_value=128,
            value=32,
            step=16
        )
        agent_params = {
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }
    else:  # ppo
        learning_rate = st.sidebar.slider(
            "Learning Rate",
            min_value=1e-5,
            max_value=1e-2,
            value=3e-4,
            step=1e-5,
            format="%.5f"
        )
        n_steps = st.sidebar.slider(
            "Steps per Update",
            min_value=512,
            max_value=4096,
            value=2048,
            step=512
        )
        agent_params = {
            "learning_rate": learning_rate,
            "n_steps": n_steps
        }
    
    return {
        "env_name": env_name,
        "agent_type": agent_type,
        "max_episodes": max_episodes,
        "weights": [weight1, weight2],
        "agent_params": agent_params
    }


def create_environment_info(env_name: str):
    """Create environment information display."""
    st.subheader("Environment Information")
    
    env_info = {
        "line": {
            "name": "1D Line Environment",
            "description": "Agent moves on a 1D line from position 0 to goal position 10. Must balance speed and energy consumption.",
            "objectives": ["Progress towards goal", "Energy efficiency"],
            "actions": "Step sizes: -2, -1, 0, 1, 2"
        },
        "cartpole": {
            "name": "CartPole Environment",
            "description": "Agent balances a pole on a cart while minimizing energy consumption.",
            "objectives": ["Pole balancing", "Energy efficiency"],
            "actions": "Left/Right force on cart"
        },
        "mountaincar": {
            "name": "MountainCar Environment",
            "description": "Agent drives a car up a mountain while minimizing energy consumption.",
            "objectives": ["Reaching the goal", "Energy efficiency"],
            "actions": "Left/Right/No acceleration"
        }
    }
    
    info = env_info[env_name]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Name:** {info['name']}")
        st.write(f"**Description:** {info['description']}")
    
    with col2:
        st.write("**Objectives:**")
        for obj in info["objectives"]:
            st.write(f"- {obj}")
        st.write(f"**Actions:** {info['actions']}")


def train_agent(config: dict, progress_bar, status_text):
    """Train the agent with progress tracking."""
    # Create environment
    env = make_multi_objective_env(config["env_name"])
    
    # Create agent
    agent = create_multi_objective_agent(
        config["agent_type"],
        env,
        tuple(config["weights"]),
        **config["agent_params"]
    )
    
    # Create logger
    logger = TrainingLogger(use_wandb=False, use_tensorboard=False)
    
    # Create trainer
    trainer_config = {
        "max_episodes": config["max_episodes"],
        "max_steps_per_episode": 1000,
        "eval_frequency": 50,
        "save_frequency": 1000,
        "eval_episodes": 5
    }
    
    trainer = MultiObjectiveTrainer(agent, env, logger, trainer_config)
    
    # Training loop with progress tracking
    stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "epsilon_history": [],
        "loss_history": [],
        "reward_components": []
    }
    
    best_reward = float('-inf')
    
    for episode in range(config["max_episodes"]):
        # Run episode
        episode_reward, episode_length, episode_stats = trainer._run_episode(training=True)
        
        # Update stats
        stats["episode_rewards"].append(episode_reward)
        stats["episode_lengths"].append(episode_length)
        
        if "epsilon" in episode_stats:
            stats["epsilon_history"].append(episode_stats["epsilon"])
        if "loss" in episode_stats:
            stats["loss_history"].append(episode_stats["loss"])
        if "reward_components" in episode_stats:
            stats["reward_components"].append(episode_stats["reward_components"])
        
        # Update progress
        progress = (episode + 1) / config["max_episodes"]
        progress_bar.progress(progress)
        
        status_text.text(f"Episode {episode + 1}/{config['max_episodes']} - Reward: {episode_reward:.2f}")
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Small delay for visualization
        time.sleep(0.01)
    
    env.close()
    return stats, agent


def plot_training_results(stats: dict):
    """Plot training results using Plotly."""
    if not stats["episode_rewards"]:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Episode Rewards", "Episode Lengths", "Epsilon Decay", "Training Loss"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    episodes = list(range(len(stats["episode_rewards"])))
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=stats["episode_rewards"], name="Reward", line=dict(color="blue")),
        row=1, col=1
    )
    
    # Episode lengths
    fig.add_trace(
        go.Scatter(x=episodes, y=stats["episode_lengths"], name="Length", line=dict(color="green")),
        row=1, col=2
    )
    
    # Epsilon decay
    if stats["epsilon_history"]:
        fig.add_trace(
            go.Scatter(x=episodes[:len(stats["epsilon_history"])], y=stats["epsilon_history"], 
                      name="Epsilon", line=dict(color="red")),
            row=2, col=1
        )
    
    # Training loss
    if stats["loss_history"]:
        fig.add_trace(
            go.Scatter(x=episodes[:len(stats["loss_history"])], y=stats["loss_history"], 
                      name="Loss", line=dict(color="orange")),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Training Progress"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Length", row=1, col=2)
    fig.update_yaxes(title_text="Epsilon", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_reward_components(stats: dict):
    """Plot reward components over time."""
    if not stats["reward_components"]:
        return
    
    components = np.array(stats["reward_components"])
    episodes = list(range(len(components)))
    
    fig = go.Figure()
    
    for i in range(components.shape[1]):
        fig.add_trace(go.Scatter(
            x=episodes,
            y=components[:, i],
            name=f"Component {i+1}",
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Reward Components Over Time",
        xaxis_title="Episode",
        yaxis_title="Reward",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_policy_heatmap(agent):
    """Plot policy heatmap for Q-learning agents."""
    if not hasattr(agent, 'q_table'):
        return
    
    q_table = agent.q_table
    
    fig = go.Figure(data=go.Heatmap(
        z=q_table.T,
        colorscale='RdYlBu_r',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Q-Table Heatmap",
        xaxis_title="State",
        yaxis_title="Action",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    setup_page()
    
    # Create sidebar
    config = create_sidebar()
    
    # Display environment information
    create_environment_info(config["env_name"])
    
    st.subheader("Training")
    
    # Training controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        train_button = st.button("🚀 Start Training", type="primary")
    
    with col2:
        if st.button("📊 Show Sample Results"):
            # Load sample data for demonstration
            sample_stats = {
                "episode_rewards": np.random.cumsum(np.random.normal(0.1, 0.5, 100)),
                "episode_lengths": np.random.normal(50, 10, 100),
                "epsilon_history": np.exp(-np.linspace(0, 3, 100)) * 0.2,
                "loss_history": np.exp(-np.linspace(0, 2, 100)) * 0.5,
                "reward_components": [(np.random.normal(0.5, 0.2), np.random.normal(-0.1, 0.1)) for _ in range(100)]
            }
            
            st.success("Showing sample training results!")
            plot_training_results(sample_stats)
            plot_reward_components(sample_stats)
    
    # Training section
    if train_button:
        st.info("Training started! This may take a few minutes...")
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train agent
        with st.spinner("Training in progress..."):
            stats, agent = train_agent(config, progress_bar, status_text)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success("Training completed!")
        
        # Display results
        st.subheader("Training Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Reward", f"{stats['episode_rewards'][-1]:.2f}")
        
        with col2:
            st.metric("Best Reward", f"{max(stats['episode_rewards']):.2f}")
        
        with col3:
            st.metric("Avg Episode Length", f"{np.mean(stats['episode_lengths']):.1f}")
        
        with col4:
            st.metric("Total Episodes", len(stats['episode_rewards']))
        
        # Plot results
        plot_training_results(stats)
        plot_reward_components(stats)
        plot_policy_heatmap(agent)
        
        # Download results
        st.subheader("Download Results")
        
        # Create downloadable data
        results_df = pd.DataFrame({
            "Episode": range(len(stats["episode_rewards"])),
            "Reward": stats["episode_rewards"],
            "Length": stats["episode_lengths"]
        })
        
        if stats["epsilon_history"]:
            results_df["Epsilon"] = stats["epsilon_history"]
        if stats["loss_history"]:
            results_df["Loss"] = stats["loss_history"]
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Training Data (CSV)",
            data=csv,
            file_name="training_results.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About Multi-Objective RL
    
    Multi-objective reinforcement learning addresses scenarios where agents must balance
    multiple competing objectives. This is common in real-world applications such as:
    
    - **Robotics**: Speed vs energy efficiency
    - **Finance**: Risk vs return
    - **Healthcare**: Effectiveness vs cost
    - **Transportation**: Time vs fuel consumption
    
    The key challenge is finding policies that achieve good performance across all objectives
    rather than optimizing for a single goal.
    """)


if __name__ == "__main__":
    main()
