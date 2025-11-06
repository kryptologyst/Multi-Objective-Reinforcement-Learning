#!/usr/bin/env python3
"""
Setup script for multi-objective reinforcement learning project.

This script helps set up the project environment and run basic tests.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Multi-Objective RL Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = ["logs", "checkpoints", "plots", "notebooks"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Run tests
    print("\n🧪 Running tests...")
    if not run_command("python tests/test_multi_objective_rl.py", "Running unit tests"):
        print("⚠️  Some tests failed, but setup continues")
    
    # Test basic functionality
    print("\n🔍 Testing basic functionality...")
    test_script = """
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

try:
    from envs.multi_objective_envs import make_multi_objective_env
    from agents.multi_objective_agents import create_multi_objective_agent
    
    # Test environment creation
    env = make_multi_objective_env("line")
    print("✓ Environment creation successful")
    
    # Test agent creation
    agent = create_multi_objective_agent("qlearning", env, weights=(1.0, 0.5))
    print("✓ Agent creation successful")
    
    # Test basic interaction
    obs, info = env.reset()
    action = agent.select_action(obs)
    next_obs, reward_vec, terminated, truncated, info = env.step(action)
    print("✓ Basic interaction successful")
    
    env.close()
    print("✓ All basic tests passed!")
    
except Exception as e:
    print(f"✗ Basic test failed: {e}")
    sys.exit(1)
"""
    
    if not run_command(f'python -c "{test_script}"', "Basic functionality test"):
        print("❌ Basic functionality test failed")
        sys.exit(1)
    
    # Success message
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run training: python train.py --env line --agent qlearning --episodes 100")
    print("2. Launch web interface: streamlit run app.py")
    print("3. Explore the demo notebook: jupyter notebook notebooks/demo.ipynb")
    print("4. Check the README.md for more examples")
    print("\nHappy learning! 🤖🎓")


if __name__ == "__main__":
    main()
