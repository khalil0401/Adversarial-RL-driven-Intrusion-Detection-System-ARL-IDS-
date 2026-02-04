# Run Training Script
# This is a standalone entry point for training the ARL-IDS model
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train
from src.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ARL-IDS Model")
    parser.add_argument("--data_path", type=str, default=Config.get_data_path(),
                       help="Path to the ToN_IoT dataset CSV file")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--episodes", type=int, default=Config.EPISODES,
                       help="Number of training episodes")
    parser.add_argument("--encoder_epochs", type=int, default=Config.ENCODER_EPOCHS,
                       help="Number of epochs for encoder training")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--epsilon_decay", type=float, default=Config.EPSILON_DECAY,
                       help="Epsilon decay rate")
    parser.add_argument("--target_update_freq", type=int, default=Config.TARGET_UPDATE_FREQ,
                       help="Frequency of target network updates")
    parser.add_argument("--weight_update_freq", type=int, default=Config.WEIGHT_UPDATE_FREQ,
                       help="Frequency of class weight updates")
    parser.add_argument("--skip_encoder_train", action="store_true",
                       help="Skip encoder training (use existing checkpoint)")
    parser.add_argument("--max_steps_per_episode", type=int, default=1,
                       help="Max steps per episode")
    
    # Ablation Flags
    parser.add_argument("--no_reward_shaping", action="store_true",
                       help="Disable dynamic reward weighting")
    parser.add_argument("--no_adversary", action="store_true",
                       help="Disable adversarial generation (static dataset)")
    parser.add_argument("--no_curriculum", action="store_true",
                       help="Disable curriculum learning (random sampling)")
    
    args = parser.parse_args()
    
    # Print configuration
    Config.print_config()
    
    try:
        print(f"\nStarting training for {args.episodes} episodes with decay {args.epsilon_decay}...")
        print(f"Data path: {args.data_path}")
        print("=" * 60 + "\n")
        train(args)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Checkpoints saved to: results/checkpoints/")
    except FileNotFoundError as e:
        print("\n[ERROR] Dataset not found!")
        print(f"Expected location: {args.data_path}")
        print("\nTo set a custom data path, use one of these methods:")
        print("  1. Command line: python src/run_Training.py --data_path /path/to/data.csv")
        print("  2. Environment variable: set ARL_IDS_DATA_PATH=/path/to/data.csv")
        print(f"\nError details: {e}")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()