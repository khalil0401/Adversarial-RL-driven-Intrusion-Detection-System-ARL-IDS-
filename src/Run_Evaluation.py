# 3. Configuration & Imports
import sys
import os
import argparse

# Add source to python path
sys.path.append(os.getcwd())

from src.train import train
from src.evaluate import evaluate
from src.visualization.plot_results import parse_logs, plot_training

class Config:
    def __init__(self):
        # Dataset Parameters
        # On Kaggle, this is likely "/kaggle/input/ton-iot-network-dataset/train_test_network.csv"
        self.data_path = "src/train_test_network.csv"
        
        # Training Hyperparameters
        self.seed = 42
        self.episodes = 200000        # Recommended: 80,000 for competitive convergence
        self.encoder_epochs = 100     # Pre-training epochs for Autoencoder
        self.lr = 1e-3                # Learning Rate
        self.epsilon_decay = 0.99995  # Slow decay for long exploration
        self.target_update_freq = 1000
        self.weight_update_freq = 1000
        
        # System Flags
        self.skip_encoder_train = False
        self.max_steps_per_episode = 1
        self.no_reward_shaping = False # Enable F1-based Dynamic Reward Shaping
        self.no_adversary = False      # Enable RL Attacker
        self.no_curriculum = False
        
        # Evaluation Paths
        self.defender_path = "results/checkpoints/policy_net.pth"
        self.attacker_path = "results/checkpoints/attacker_net.pth"

args = Config()

print(f"Configuration Loaded. Target Episodes: {args.episodes}")
evaluate(args)