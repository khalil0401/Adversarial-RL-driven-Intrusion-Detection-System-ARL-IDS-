import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from src.data.ton_iot_loader import TonIoTLoader
from src.representation.encoder import StateEncoder
from src.agents.ddqn_agent import DDQNAgent
from src.agents.attacker_agent import AttackerAgent # New Import

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Joint_Evaluate")

def evaluate(args):
    # Set Seed for fairness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load Data
    logger.info("Loading Test Data...")
    loader = TonIoTLoader(args.data_path, seed=42)
    
    if not os.path.exists(args.data_path):
        logger.warning("Data path not found. Generating dummy test data.")
        X_test = np.random.rand(50, 38)
        y_test = np.random.randint(0, 10, 50)
        loader.classes_ = list(range(10))
    else:
        # We assume load_and_process returns the split, we just want X_test, y_test
        _, X_test, _, y_test = loader.load_and_process()

    input_dim = X_test.shape[1]
    n_classes = len(np.unique(y_test))
    state_dim = 64 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Models
    logger.info("Loading Models (Encoder, Defender, Attacker)...")
    
    # Encoder
    encoder = StateEncoder(input_dim, latent_dim=state_dim, n_classes=n_classes, device=device)
    try:
        encoder.load("results/checkpoints/encoder.pth")
    except FileNotFoundError:
        logger.error("Encoder checkpoint missing.")
        return

    # Defender
    defender = DDQNAgent(state_dim, n_classes, device=device)
    try:
        defender.policy_net.load_state_dict(torch.load(args.defender_path))
        defender.policy_net.eval()
    except FileNotFoundError:
        logger.error(f"Defender model missing at {args.defender_path}")
        return

    # Attacker
    attacker = AttackerAgent(input_dim=input_dim, action_dim=input_dim*2, device=device)
    try:
        attacker.policy_net.load_state_dict(torch.load(args.attacker_path))
        attacker.policy_net.eval()
    except FileNotFoundError:
        logger.error(f"Attacker model missing at {args.attacker_path}")
        return

    logger.info("Starting Joint Evaluation Loop...")
    
    batch_size = 128
    n_test = X_test.shape[0]
    
    # Store Predictions and Probabilities
    clean_preds = []
    adv_preds = []
    clean_probs = [] # Confidence on correct class
    adv_probs = []   # Confidence on correct class
    perturbation_norms = []
    
    for i in range(0, n_test, batch_size):
        # Raw Data Batch
        X_batch_raw = X_test[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]
        
        # --- Clean Pass (Baseline) ---
        states_clean = encoder.get_latent(X_batch_raw)
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states_clean).to(device)
            q_values = defender.policy_net(states_tensor)
            probs = F.softmax(q_values, dim=1)
            preds = q_values.argmax(1).cpu().numpy()
            
            # Store confidence for the TRUE class
            batch_probs = probs.cpu().numpy()
            clean_preds.extend(preds)
            clean_probs.extend([batch_probs[j, y_batch[j]] for j in range(len(y_batch))])

        # --- Adversarial Pass (Attacker Generation) ---
        # 1. Attacker observes RAW data
        with torch.no_grad():
            raw_tensor = torch.FloatTensor(X_batch_raw).to(device)
            atk_q_values = attacker.policy_net(raw_tensor)
            atk_actions = atk_q_values.argmax(1).cpu().numpy()
        
        # 2. Apply Perturbations
        X_batch_adv = X_batch_raw.copy()
        for idx, action in enumerate(atk_actions):
            perturbation = attacker.get_perturbation(action)
            X_batch_adv[idx] = np.clip(X_batch_raw[idx] + perturbation, 0.0, 1.0)
            
            # Track Magnitude
            diff = X_batch_adv[idx] - X_batch_raw[idx]
            perturbation_norms.append(np.linalg.norm(diff))

        # 3. Defender observes MUTATED data
        states_adv = encoder.get_latent(X_batch_adv)
        with torch.no_grad():
            states_tensor_adv = torch.FloatTensor(states_adv).to(device)
            q_values_adv = defender.policy_net(states_tensor_adv)
            probs_adv = F.softmax(q_values_adv, dim=1)
            preds_adv = q_values_adv.argmax(1).cpu().numpy()
            
            batch_probs_adv = probs_adv.cpu().numpy()
            adv_preds.extend(preds_adv)
            adv_probs.extend([batch_probs_adv[j, y_batch[j]] for j in range(len(y_batch))])

    # Convert to Arrays
    clean_preds = np.array(clean_preds)
    adv_preds = np.array(adv_preds)
    clean_probs = np.array(clean_probs)
    adv_probs = np.array(adv_probs)
    perturbation_norms = np.array(perturbation_norms)
    
    # --- Step 2: Defender Metrics ---
    acc_clean = accuracy_score(y_test, clean_preds)
    f1_clean = f1_score(y_test, clean_preds, average='weighted')
    
    acc_adv = accuracy_score(y_test, adv_preds)
    f1_adv = f1_score(y_test, adv_preds, average='weighted')
    
    robustness_gap = acc_clean - acc_adv
    
    # --- Step 3: Attacker Metrics ---
    # Attack Success Rate (ASR):
    # Fraction of correctly classified samples that became incorrect after attack
    correct_indices = np.where(clean_preds == y_test)[0]
    flipped_indices = np.where(adv_preds[correct_indices] != y_test[correct_indices])[0]
    
    if len(correct_indices) > 0:
        asr = len(flipped_indices) / len(correct_indices)
    else:
        asr = 0.0
        
    # Confidence Degradation
    conf_drop = np.mean(clean_probs - adv_probs)
    
    # --- Logging & Reporting ---
    logger.info("=== Joint Evaluation Results ===")
    logger.info(f"Clean Accuracy: {acc_clean:.4f} | F1: {f1_clean:.4f}")
    logger.info(f"Adv.  Accuracy: {acc_adv:.4f}   | F1: {f1_adv:.4f}")
    logger.info(f"Robustness Gap: {robustness_gap:.4f} (Lower is better)")
    logger.info("-" * 30)
    logger.info(f"Attack Success Rate (ASR): {asr:.4f}")
    logger.info(f"Confidence Drop: {conf_drop:.4f}")
    logger.info(f"Avg Perturbation Norm (L2): {np.mean(perturbation_norms):.6f}")

    # Save Detailed Report
    with open("results/joint_evaluation_results.txt", "w") as f:
        f.write("=== JOINT ADVERSARIAL EVALUATION ===\n\n")
        f.write("--- Defender Performance ---\n")
        f.write(f"Accuracy (Clean): {acc_clean:.4f}\n")
        f.write(f"Accuracy (Adv):   {acc_adv:.4f}\n")
        f.write(f"Robustness Gap:   {robustness_gap:.4f}\n")
        f.write(f"F1 Score (Clean): {f1_clean:.4f}\n")
        f.write(f"F1 Score (Adv):   {f1_adv:.4f}\n\n")
        
        f.write("--- Attacker Performance ---\n")
        f.write(f"Attack Success Rate (ASR): {asr:.4f}\n")
        f.write(f"Confidence Degradation:    {conf_drop:.4f}\n")
        f.write(f"Avg L2 Perturbation:       {np.mean(perturbation_norms):.6f}\n\n")
        
        f.write("--- Classification Report (Adversarial) ---\n")
        f.write(classification_report(y_test, adv_preds))

    logger.info("Results saved to results/joint_evaluation_results.txt")

if __name__ == "__main__":
    from src.config import Config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=Config.get_data_path())
    parser.add_argument("--defender_path", type=str, default="results/checkpoints/policy_net.pth")
    parser.add_argument("--attacker_path", type=str, default="results/checkpoints/attacker_net.pth")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    evaluate(args)
