# 3. Configuration & Imports
import sys
import os
import argparse

# Add source to path
sys.path.append(os.getcwd())

from src.train import train
from src.evaluate import evaluate

# Define arguments class to simulate argparse
class Args:
    def __init__(self):
        self.data_path = "src\IoTtrain_test_network.csv" # CHANGE THIS PATH to your dataset location
        self.model_path = "results\checkpoints\policy_net.pth"
        self.seed = 42
        self.episodes = 200000 # Increased for better convergence
        self.encoder_epochs = 100
        self.lr = 1e-3
        self.epsilon_decay = 0.99995 # Slower decay for longer training
        self.target_update_freq = 100
        self.weight_update_freq = 1000
        self.skip_encoder_train = False
        self.max_steps_per_episode = 1
        self.no_reward_shaping = False
        self.no_adversary = False
        self.no_curriculum = False

args = Args()

print(f"Data Path configured to: {args.data_path}")
print("Ensure you have added the ToN-IoT dataset to your Kaggle environment at this path (or update it).")
evaluate(args)