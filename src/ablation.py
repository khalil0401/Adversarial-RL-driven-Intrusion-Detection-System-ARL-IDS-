import subprocess
import logging
import sys
from src.config import Config, ABLATION_DIR

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Ablation_Study")

def run_ablation():
    # Define ablation scenarios
    scenarios = [
        {"name": "baseline_full", "flags": []},
        {"name": "no_reward_shaping", "flags": ["--no_reward_shaping"]},
        {"name": "no_adversary", "flags": ["--no_adversary"]},
        {"name": "no_curriculum", "flags": ["--no_curriculum"]},
    ]
    
    # Ensure output directory exists
    ABLATION_DIR.mkdir(exist_ok=True)
    
    logger.info(f"Starting Ablation Study with {len(scenarios)} scenarios...")
    
    for scenario in scenarios:
        name = scenario["name"]
        flags = scenario["flags"]
        
        logger.info(f"Running Scenario: {name}")
        
        # Construct command
        # Use sys.executable to ensure we use the same python environment
        cmd = [sys.executable, "src/train.py", "--episodes", "50000"] + flags
        
        # Add data path explicitly from config
        cmd.extend(["--data_path", Config.get_data_path()])
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            # Capture output
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            
            # Save logs
            log_file = ABLATION_DIR / f"{name}.log"
            with open(log_file, "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n=== STDERR ===\n")
                    f.write(result.stderr)
            
            logger.info(f"Scenario {name} completed. Logs saved to {log_file}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Trial {name} failed: {e}")

if __name__ == "__main__":
    run_ablation()
