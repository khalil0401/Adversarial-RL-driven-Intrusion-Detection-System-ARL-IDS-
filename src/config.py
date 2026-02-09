"""
Centralized Configuration for ARL-IDS
Handles paths and hyperparameters with environment variable support
"""
import os
from pathlib import Path

# =============================================================================
# Project Root and Directories
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
ABLATION_DIR = RESULTS_DIR / "ablation"

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)
ABLATION_DIR.mkdir(exist_ok=True)

# =============================================================================
# Data Paths
# =============================================================================
# Default data path - can be overridden by environment variable
DEFAULT_DATA_PATH = PROJECT_ROOT / "src" / "train_test_network.csv"
DATA_PATH = Path(os.getenv("ARL_IDS_DATA_PATH", str(DEFAULT_DATA_PATH)))

# =============================================================================
# Model Checkpoints
# =============================================================================
ENCODER_CHECKPOINT = CHECKPOINTS_DIR / "encoder.pth"
POLICY_NET_CHECKPOINT = CHECKPOINTS_DIR / "policy_net.pth"
ATTACKER_NET_CHECKPOINT = CHECKPOINTS_DIR / "attacker_net.pth"

# =============================================================================
# Hyperparameters (Default Values)
# =============================================================================
class Config:
    """Configuration class with default hyperparameters"""
    
    # Data
    DATA_PATH = str(DATA_PATH)
    TEST_SIZE = 0.2
    SEED = 42
    
    # Training
    EPISODES = 50000
    ENCODER_EPOCHS = 200  # Increased for better feature representation
    LEARNING_RATE = 5e-4  # Slightly faster LR (was 1e-4)
    EPSILON_DECAY = 0.9999 
    TARGET_UPDATE_FREQ = 1000
    WEIGHT_UPDATE_FREQ = 1000
    
    # Architecture
    LATENT_DIM = 64
    HIDDEN_DIM = 256
    
    # Device
    DEVICE = "cuda" if os.getenv("FORCE_CPU") != "1" else "cpu"
    
    # Ablation Flags
    SKIP_ENCODER_TRAIN = False
    NO_REWARD_SHAPING = False
    NO_ADVERSARY = False
    NO_CURRICULUM = False
    
    @classmethod
    def get_data_path(cls):
        """Get data path, checking if file exists"""
        if not Path(cls.DATA_PATH).exists():
            print(f"[WARNING] Data file not found at {cls.DATA_PATH}")
            print(f"   Set environment variable ARL_IDS_DATA_PATH to specify custom location")
            print(f"   Example: set ARL_IDS_DATA_PATH=C:\\path\\to\\train_test_network.csv")
        return cls.DATA_PATH
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("ARL-IDS Configuration")
        print("=" * 60)
        print(f"Data Path:        {cls.DATA_PATH}")
        print(f"Results Dir:      {RESULTS_DIR}")
        print(f"Checkpoints Dir:  {CHECKPOINTS_DIR}")
        print(f"Episodes:         {cls.EPISODES}")
        print(f"Encoder Epochs:   {cls.ENCODER_EPOCHS}")
        print(f"Learning Rate:    {cls.LEARNING_RATE}")
        print(f"Device:           {cls.DEVICE}")
        print("=" * 60)


# =============================================================================
# Helper Functions
# =============================================================================
def get_checkpoint_path(name):
    """Get path to a specific checkpoint file"""
    return str(CHECKPOINTS_DIR / f"{name}.pth")

def ensure_dirs():
    """Ensure all required directories exist"""
    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    ABLATION_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    # Print configuration when run directly
    Config.print_config()
    print(f"\nData file exists: {Path(Config.DATA_PATH).exists()}")
