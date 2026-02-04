# Adaptive Adversarial Reinforcement Learning for IoT Intrusion Detection

## Abstract
This repository implements a **robust, adversarial Intrusion Detection System (IDS)** for IoT networks, utilizing a **Double Deep Q-Network (DDQN)** agent operating within a non-stationary environment. The system addresses two critical challenges in modern IDS research: (1) **Class Imbalance**, tackled via a novel **F1-based Dynamic Reward Shaping** mechanism, and (2) **Adversarial Robustness**, ensured through continuous injection of mutated samples via a lightweight stochastic adversary. The architecture decouples representation learning (Autoencoder) from policy optimization (RL), achieving **91.90% accuracy** and **0.92 weighted F1-score** on the **ToN_IoT** dataset, with verifiable separation of semantically similar attack classes (e.g., DoS vs. Backdoor).

## 1. Key Contributions
*   **Two-Stage Learning Protocol**: Decouples feature extraction (supervised Autoencoder) from decision-making (RL), preventing policy collapse during early training phases.
*   **F1-Based Dynamic Reward Shaping**: A closed-loop feedback mechanism that scales class-specific rewards $R_c$ based on the rolling F1-score, prioritizing under-performing minority classes without manual tuning.
*   **Stochastic Adversarial Curriculum**: Replaces computationally expensive GANs/LLMs with a high-throughput, constraint-aware perturbation module that induces distributional shift, forcing the agent to learn robust decision boundaries.
*   **Verifiable Reproducibility**: Includes a strict seed-locking protocol and mandatory ablation pipeline to isolate component contributions.

---

## 2. Problem Formulation
We formulate the IDS problem as a **Markov Decision Process (MDP)** defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P}, \gamma \rangle$:

*   **State Space ($\mathcal{S}$)**: A continuous latent vector $z \in \mathbb{R}^{64}$, derived from the raw traffic feature vector $x \in \mathbb{R}^{38}$ via a pre-trained Autoencoder $\phi: \mathcal{X} \to \mathcal{Z}$. The encoder uses an **expansion-compression architecture** (38 $\to$ 128 $\to$ 64) to disentangle non-linear feature interactions.
*   **Action Space ($\mathcal{A}$)**: A discrete space $\mathcal{A} = \{0, 1, \dots, N-1\}$ representing the classification decision (Normal traffic + 9 attack categories).
*   **Reward Function ($\mathcal{R}$)**: A dynamic function dependent on the classification correctness and the current class difficulty weight $W_c$:
    $$ R(s_t, a_t) = \begin{cases} +W_{c} & \text{if } a_t = y_{true} \\ -W_{c} & \text{if } a_t \neq y_{true} \end{cases} $$
    where $W_c$ is updated episodically: $W_c \leftarrow W_c \cdot (1 + \alpha \cdot (1 - \text{F1}_c))$.
*   **Transition ($\mathcal{P}$)**: Deterministic state transitions in a single-step episode setting, modified stochastically by the Adversarial Buffer sampling probability $P_{adv}$.

---

## 3. System Architecture
The framework is composed of four distinct, modular layers:

### Layer 1: Representation Learning (The 'Eye')
*   **Component**: `src/representation/encoder.py`
*   **Objective**: Minimize Reconstruction Loss ($\mathcal{L}_{MSE}$) and Classification Loss ($\mathcal{L}_{CE}$).
*   **Justification**: Raw IoT traffic features are noisy and often sparse. The Autoencoder projects these into a dense, 64-dimensional latent manifold, reducing the "curse of dimensionality" for the RL agent.

### Layer 2: Adversarial Environment (The 'Arena')
*   **Component**: `src/envs/adversarial_ids_env.py`
*   **Mechanism**: Implements an **Adversarial Buffer** that stores high-loss samples ("hard negatives").
*   **Curriculum**: The probability of sampling from this buffer increases as the agent's accuracy improves, creating an auto-curriculum that prevents plateauing.

### Layer 3: Reinforcement Learning Agent (The 'Brain')
*   **Component**: `src/agents/ddqn_agent.py`
*   **Algorithm**: **Double DQN (DDQN)**.
*   **Justification**: Standard DQN suffers from Q-value overestimation, which is detrimental in high-stakes security environments. DDQN decouples action selection from value estimation, providing stable convergence.

### Layer 4: RL Attacker Agent (The 'Adversary')
*   **Component**: `src/agents/attacker_agent.py`
*   **Architecture**: Double Deep Q-Network (Input: 38, Output: 76).
*   **Mechanism**: A learning agent that observes raw traffic and selects specific features to perturb ($\epsilon$-bounded). It optimizes a **Zero-Sum** reward function (Reward = -IDS Score).
*   **Advantage**: Unlike static noise generators, the RL Attacker actively hunts for the IDS's blind spots, creating a dynamic "arms race" during training.

---

## 4. Experimental Results
The system was evaluated on the **ToN_IoT** dataset (Train: 168k, Test: 42k) after 200,000 episodes of training.

| Metric | Baseline (Static Reward) | **ARL-IDS (F1-Based)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 75.78% | **91.90%** | +16.12% |
| **Weighted F1** | 0.7607 | **0.9235** | +0.1628 |
| **DoS Recall** | 0.28 | **0.93** | **Solved** |
| **Backdoor Recall** | 0.98 | **1.00** | +0.02 |

**Critical Finding**: The initial confusion between **DoS** and **Backdoor** attacks (due to feature similarity) was resolved by increasing the Autoencoder's capacity (Latent Dim $32 \to 64$), enabling the RL agent to discern subtle distributional differences.

---

## 5. Usage and Reproducibility

### Dependencies
*   Python 3.8+
*   PyTorch, Gymnasium, NumPy, Pandas, Scikit-learn

### Data Preparation
1.  Download `Train_Test_Network.csv` from the official [ToN_IoT repository](https://research.unsw.edu.au/projects/toniot-datasets).
2.  Place in: `src/IoTtrain_test_network.csv` (or configure path in `src/start.py`).

### Full Training Pipeline (Competitive Self-Play)
The system employs a **Competitive Reinforcement Learning** loop where the Defender and Attacker train simultaneously.

See [SETUP.md](SETUP.md) for detailed installation and configuration instructions.

To train the full system:
```bash
python src/run_Training.py --episodes 200000 --encoder_epochs 100
```
*Outputs*: `results/checkpoints/policy_net.pth` (Defender) and `results/checkpoints/attacker_net.pth` (Attacker).

### Full Evaluation (Joint Attacker–Defender Assessment)
We implement a rigorous **Joint Policy Assessment** protocol where both agents are frozen and evaluated.
To run the joint evaluation:
```bash
python src/evaluate.py
```
*Outputs*: `results/joint_evaluation_results.txt` containing full comparative metrics.

## 6. Scientific Positioning
**Why Joint Evaluation?**
Evaluating an IDS against static noise (e.g., Gaussian) provides a false sense of security. By training a dedicated RL Attacker and then evaluating the IDS against this *optimized* adversary, we measure the **Worst-Case Robustness**. This methodology prevents "Gradient Masking" and overfitting to specific static attack patterns, ensuring the IDS is resilient against adaptive threats—a critical requirement for real-world IoT deployment.

### Ablation Studies
To verify component efficacy, run the ablation suite:
```bash
python src/ablation.py
```
This runs trials for `no_reward_shaping`, `no_adversary`, and `no_curriculum` variations.

---

## 6. Limitations and Threats to Validity
1.  **Computational Cost**: While the adversary is lightweight, the RL training requires $\approx 200,000$ sequential steps for convergence, which is computationally more intensive than supervised XGBoost/Random Forest.
2.  **Feature Drift**: The static mapping of the Autoencoder means significant shifts in network protocol distributions (e.g., new protocols) would require retraining Layer 1.
3.  **Single-Step Assumption**: The environment treats IDS as a sequence of independent classification tasks (contextual bandit formulation) rather than a temporally correlated sequence, which simplifies the state but may miss long-term attack patterns.

## 7. Citation
If you use this code for research, please cite:
```bibtex
@software{arl_ids_2026,
  author = {Khalil BENCHEIKH},
  title = {Adversarial RL-driven Intrusion Detection System (ARL-IDS)},
  year = {2026},
  url = {https://github.com/your-repo/arl-ids}
}
```
