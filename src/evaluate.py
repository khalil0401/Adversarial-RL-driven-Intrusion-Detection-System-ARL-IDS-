import argparse
import logging
import numpy as np
import torch
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from src.data.ton_iot_loader import TonIoTLoader
from src.representation.encoder import StateEncoder
from src.agents.ddqn_agent import DDQNAgent

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Evaluate")

def evaluate(args):
    # Load Data
    loader = TonIoTLoader(args.data_path, seed=42)
    
    # Mock for testing capability if file missing
    if not os.path.exists(args.data_path):
        logger.warning(f"Data path {args.data_path} not found. Creating dummy data.")
        X_train = np.random.rand(100, 38)
        y_train = np.random.randint(0, 10, 100)
        X_test = np.random.rand(50, 38)
        y_test = np.random.randint(0, 10, 50)
        loader.classes_ = list(range(10))
    else:
        # We need the full split to respect the original training scaler fit?
        # Ideally we should save the scaler. For this implementation, we re-fit on train and transform test.
        X_train, X_test, y_train, y_test = loader.load_and_process()

    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    state_dim = 16 

    # Load Encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = StateEncoder(input_dim, latent_dim=state_dim, device=device)
    try:
        encoder.load("results/checkpoints/encoder.pth")
    except FileNotFoundError:
        logger.error("Encoder checkpoint not found. Run training first.")
        return

    # Load Agent
    agent = DDQNAgent(state_dim, n_classes, device=device)
    try:
        agent.policy_net.load_state_dict(torch.load(args.model_path))
        agent.policy_net.eval()
    except FileNotFoundError:
        logger.error(f"Agent checkpoint not found at {args.model_path}. Run training first.")
        return
        
    logger.info("Starting Evaluation...")
    
    preds = []
    
    # Batch prediction
    # We can process test set in batches using the Agent
    # But Agent expects state. So we need to encode X_test first.
    
    batch_size = 128
    n_test = X_test.shape[0]
    
    for i in range(0, n_test, batch_size):
        X_batch = X_test[i:i+batch_size]
        
        # Encode
        states = encoder.get_latent(X_batch)
        
        # Predict
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(device)
            q_values = agent.policy_net(states_tensor)
            batch_preds = q_values.argmax(1).cpu().numpy()
            
        preds.extend(batch_preds)
        
    preds = np.array(preds)
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1 Score (Weighted): {f1:.4f}")
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))
    
    # Save results to file
    with open("results/evaluation_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score (Weighted): {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, preds))
    
    logger.info("Results saved to results/evaluation_results.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="D:/An adversarial environment reinforcement/train_test_network.csv")
    parser.add_argument("--model_path", type=str, default="results/checkpoints/policy_net.pth")
    # Allow running as script or imported
    if __name__ == "__main__":
        args = parser.parse_args()
        evaluate(args)
