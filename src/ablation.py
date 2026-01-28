import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AblationRunner")

TRIALS = {
    "baseline": [], # Full model
    "no_reward_shaping": ["--no_reward_shaping"],
    "no_adversary": ["--no_adversary"],
    "no_curriculum": ["--no_curriculum"]
}

def run_ablation():
    os.makedirs("results/ablation", exist_ok=True)
    
    for name, flags in TRIALS.items():
        logger.info(f"Running Ablation Trial: {name}")
        
        cmd = ["python", "-m", "src.train", "--episodes", "500"] + flags
        
        # In a real scenario, we might want to change the output directory for checkpoints
        # But train.py currently hardcodes "results/checkpoints".
        # We should modify train.py to accept --output_dir or handle it here by moving files after run.
        # For simplicity in this research reconstruction, we will just run them. 
        # Ideally, we modify train.py to support --output_dir.
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Trial {name} completed successfully.")
            
            # Move results to valid folder
            # Assuming train.py saves to results/checkpoints
            target_dir = f"results/ablation/{name}"
            os.makedirs(target_dir, exist_ok=True)
            if os.path.exists("results/checkpoints/policy_net.pth"):
                os.rename("results/checkpoints/policy_net.pth", f"{target_dir}/policy_net.pth")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Trial {name} failed: {e}")

if __name__ == "__main__":
    run_ablation()
