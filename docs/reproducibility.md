# Reproducibility Guide

## 1. Environment Setup

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Gymnasium
- Pandas, Numpy, Scikit-learn
- Matplotlib (for plots)

### Installation
```bash
pip install torch gymnasium pandas numpy scikit-learn
```

## 2. Dataset Setup
1.  Download `Train_Test_Network.csv` from the official TON_IoT repository.
2.  Place the file at: `src/data/Train_Test_Network.csv`.
    - *Note*: If the file is missing, the system will generate dummy data for structure verification, but **research results require the real dataset**.

## 3. Training the Full System
To reproduce the proposed solution (ARL-IDS):
```bash
python -m src.train --seed 42 --episodes 1000 --data_path "D:/An adversarial environment reinforcement/train_test_network.csv"
```
This script will:
1.  Load and split data (80/20).
2.  Train the Autoencoder (Layer 1).
3.  Train the DDQN Agent in the Adversarial Environment (Layers 2-4).
4.  Save checkpoints to `results/checkpoints/`.

## 4. Ablation Studies (Mandatory)
To verify the contribution of each component, run the ablation suite:
```bash
python -m src.ablation
```

## 5. Evaluation
To evaluate the trained model on the held-out test set:
```bash
python -m src.evaluate
```
This outputs:
- Accuracy
- Weighted F1 Score
- Classification Report (Precision/Recall per class)

## 6. Verification Checklist
- [x] **Data Leakage**: Scaler is fit ONLY on training data.
- [x] **Seeds**: Fixed RNG seeds for Numpy and Torch.
- [x] **Metrics**: F1, Accuracy, and per-class Recall logged.
