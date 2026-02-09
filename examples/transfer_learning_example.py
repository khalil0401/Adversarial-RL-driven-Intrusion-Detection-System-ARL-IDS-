"""
Example: Using Transfer Learning for New Dataset

This script demonstrates how to use the transfer learning utilities
to adapt a pre-trained ARL-IDS model to a new dataset.
"""

import numpy as np
import torch
from src.data.ton_iot_loader import TonIoTLoader
from src.representation.encoder import StateEncoder
from src.agents.ddqn_agent import DDQNAgent
from src.agents.attacker_agent import AttackerAgent
from src.utils.transfer_learning import TransferLearningManager, check_dataset_compatibility
from src.config import Config

# ============================================================================
# 1. Load Pre-trained Model
# ============================================================================
print("="*60)
print("Transfer Learning Example")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load source dataset (for compatibility check)
source_loader = TonIoTLoader("src/train_test_network.csv", seed=42)
X_source, _, y_source, _ = source_loader.load_and_process()

# Initialize components
input_dim = X_source.shape[1]
state_dim = 64
n_classes = len(np.unique(y_source))

encoder = StateEncoder(input_dim, latent_dim=state_dim, n_classes=n_classes, device=device)
defender = DDQNAgent(state_dim, n_classes, device=device)
attacker = AttackerAgent(input_dim=input_dim, action_dim=input_dim*2, device=device)

# Initialize transfer learning manager
tl_manager = TransferLearningManager(encoder=encoder, defender=defender, attacker=attacker)

print("\n📦 Loading pre-trained model...")
tl_manager.load_full_model(
    encoder_path="results/checkpoints/encoder.pth",
    defender_path="results/checkpoints/policy_net.pth",
    attacker_path="results/checkpoints/attacker_net.pth",
    device=device
)

# ============================================================================
# 2. Load New Dataset
# ============================================================================
print("\n📊 Loading new dataset...")
# In this example, we use the same dataset for demonstration
# In practice, replace with your new dataset path
new_data_path = "path/to/new/dataset.csv"
# target_loader = TonIoTLoader(new_data_path, seed=42)
# X_target, _, y_target, _ = target_loader.load_and_process()

# For demonstration, we'll simulate a new dataset
X_target = X_source + np.random.normal(0, 0.1, X_source.shape)  # Slight distribution shift
y_target = y_source

# ============================================================================
# 3. Check Compatibility
# ============================================================================
print("\n🔍 Checking dataset compatibility...")
compatibility = check_dataset_compatibility(
    source_data=(X_source, y_source),
    target_data=(X_target, y_target)
)

print(f"\n{'✅' if compatibility['compatible'] else '❌'} Compatible: {compatibility['compatible']}")
for warning in compatibility['warnings']:
    print(warning)
for rec in compatibility['recommendations']:
    print(rec)

# ============================================================================
# 4. Choose Fine-tuning Strategy
# ============================================================================
print("\n🎯 Selecting fine-tuning strategy...")

# Option 1: Freeze encoder, fine-tune RL agents only (fastest)
# config = tl_manager.get_fine_tuning_mode('agents_only')

# Option 2: Fine-tune encoder, freeze agents (for feature adaptation)
# config = tl_manager.get_fine_tuning_mode('encoder_only')

# Option 3: Fine-tune classifier only (for new classes)
# config = tl_manager.get_fine_tuning_mode('classifier_only')

# Option 4: Fine-tune everything (most thorough, slowest)
config = tl_manager.get_fine_tuning_mode('full')

print(f"\nSelected mode: {config['description']}")
print(f"  Train Encoder: {config['train_encoder']}")
print(f"  Train Defender: {config['train_defender']}")
print(f"  Train Attacker: {config['train_attacker']}")

# ============================================================================
# 5. Fine-tune on New Dataset
# ============================================================================
print("\n🏋️  Starting fine-tuning...")
print("(In a real scenario, you would call train() with reduced episodes)")
print("Example: python src/run_Training.py --data_path path/to/new/data.csv --episodes 10000")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("Transfer Learning Workflow Summary")
print("="*60)
print("""
1. ✅ Loaded pre-trained model
2. ✅ Checked dataset compatibility  
3. ✅ Selected fine-tuning strategy
4. ⏭️  Next: Fine-tune with reduced episodes

Recommended Training Command:
  python src/run_Training.py \\
    --data_path path/to/new/dataset.csv \\
    --episodes 10000 \\
    --skip_encoder_train  # If using 'agents_only' mode

Expected Time Savings:
  - agents_only: ~50% reduction (skip encoder training)
  - encoder_only: ~30% reduction (RL agents converge faster)
  - full: Minimal savings, but better adaptation
""")
