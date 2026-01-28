# Adversarial RL-driven Intrusion Detection System (ARL-IDS)

This project implements a **Research-Grade Adversarial Reinforcement Learning** system for Intrusion Detection (IDS) on the **TON_IoT** dataset. 

It replaces traditional heavy Large Language Models (LLMs) with a **Lightweight Adversarial Generation** layer (Layer 4) to ensure high-speed, valid, and reproducible adversarial training.

## 🏗️ 4-Layer Architecture

1.  **Layer 1: Data & Representation**
    *   **Loader**: `src/data/ton_iot_loader.py` handles the TON_IoT dataset, performing rigorous preprocessing (MinMax scaling, label encoding) without data leakage.
    *   **Encoder**: `src/representation/encoder.py` uses an Autoencoder to compress high-dimensional traffic features into a robust 16-dim latent state for the agent.

2.  **Layer 2: Adversarial Environment**
    *   **Env**: `src/envs/adversarial_ids_env.py` is a Gymnasium-compatible environment.
    *   **Curriculum Learning**: Dynamically adjusts difficulty based on agent performance ($d = 1 - P(correct)$).
    *   **Active Sampling**: Replays hard samples from an Adversarial Buffer to prevent forgetting.

3.  **Layer 3: RL Agent (IDS)**
    *   **Agent**: `src/agents/ddqn_agent.py` implements **Double DQN** (DDQN) with Experience Replay and Target Networks.
    *   **Dynamic Reward Shaping**: Solves class imbalance by dynamically weighting rewards: $R = W_c \times (\pm 1)$, where $W_c$ adapts based on class recall.

4.  **Layer 4: Lightweight Adversary**
    *   **Generator**: `src/adversary/lightweight_generator.py` uses bounded feature mutation (constraints $\pm 10\%$) to generate valid but challenging adversarial traffic samples 1000x faster than LLMs.

---

## 🚀 Installation

1.  **Prerequisites**: Python 3.8+
2.  **Install Dependencies**:
    ```bash
    pip install torch gymnasium pandas numpy scikit-learn
    ```
3.  **Dataset**:
    *   Download `Train_Test_Network.csv` from the [TON_IoT Dataset](https://research.unsw.edu.au/projects/toniot-datasets).
    *   Place it in: `D:/An adversarial environment reinforcement/train_test_network.csv` (or configure via `--data_path`).

---

## 🏃 Usage

### 1. Training (Full Model)
To train the system from scratch (Layer 1 Autoencoder + Layer 3 Agent):
```bash
python -m src.train --seed 42 --episodes 10000 --encoder_epochs 5
```
*   `--episodes`: Number of training steps/samples (Recommend 50,000+ for convergence).
*   `--encoder_epochs`: Epochs for pre-training the state encoder.

### 2. Ablation Studies (Mandatory)
To verify the contribution of specific components (as per research rigor):
```bash
python -m src.ablation
```
This runs 3 variations automatically:
*   **Baseline**: Full methodology.
*   **No Reward Shaping**: Disables dynamic class weighting.
*   **No Adversary**: Disables adversarial generation.
*   **No Curriculum**: Disables difficulty-based sampling.

### 3. Evaluation
To evaluate the trained model on the held-out test set:
```bash
python -m src.evaluate --model_path results/checkpoints/policy_net.pth
```
Results (Accuracy, F1, Classification Report) are saved to `results/evaluation_results.txt`.

---

## 📂 Project Structure

```
d:/An adversarial environment reinforcement/
├── src/
│   ├── adversary/         # Lightweight generator
│   ├── agents/            # DDQN Agent
│   ├── data/              # TON_IoT Loader
│   ├── envs/              # Adversarial Gym Env
│   ├── representation/    # Autoencoder
│   ├── ablation.py        # Ablation study runner
│   ├── evaluate.py        # Evaluation script
│   └── train.py           # Main training loop
├── docs/
│   ├── analysis.md        # Research analysis
│   └── reproducibility.md # Detailed reproduction guide
├── results/
│   ├── checkpoints/       # Saved models (.pth)
│   └── evaluation_results.txt
└── README.md
```

## ⚠️ Note on Results
If running a short demo (e.g., 1,000 episodes), accuracy may be low (~35%) due to undersampling. **Full training (50k+ episodes) is required** to reach >95% performance.
