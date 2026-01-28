# Research Analysis: Adversarial RL-driven IDS (TON_IoT)

## 1. Problem Definition & Gap
Current IDS solutions (Static RL, Supervised Learning) fail to generalize against evolving attacks. LLM-based solutions are heavy and prone to hallucination. We propose a **Lightweight Adversarial Reinforcement Learning (LARL)** framework.

## 2. Methodology: The 4-Layer Architecture

### Layer 1: Data & Representation
- **Dataset**: TON_IoT (Network Train/Test).
- **Preprocessing**: MinMax Normalization + Label Encoding.
- **State Space**: 16-dimensional Latent Vector from an Autoencoder.
    - *Why?* Reduces noise, handles high-dimensional traffic features, extracts manifold structure.

### Layer 2: Adversarial Environment
- **Curriculum Learning**: Difficulty varies dynamically based on agent performance ($d = 1 - P(correct)$).
- **Adversarial Buffer**: Stores hard samples that fooled the agent.
- **Sampling Strategy**: 
    - Early training: Random dataset sampling.
    - Late training: High probability of sampling from Buffer (Active Learning).

### Layer 3: RL Agent (Double DQN)
- **Algorithm**: Double DQN to reduce maximization bias.
- **Dynamic Reward Shaping**:
    - $R = W_c \times (\pm 1)$
    - $W_c$ updates every $N$ episodes based on Recall/FNR.
    - *Novelty*: Provable convergence stability vs. arbitrary LLM rewards.

### Layer 4: Lightweight Adversary
- **Mechanism**: Bounded Feature Mutation.
- **Constraints**: Perturbations limited to $\pm 10\%$ (normalized) to preserve validity.
- *Advantage*: 1000x faster than LLM generation, guaranteed valid feature types.

## 3. Experimental Plan & Ablation
We validate the contribution by systematically removing components:
1.  **No Curriculum**: Proves value of ordering difficulty.
2.  **No Adversary**: Proves value of robustness against mutations.
3.  **No Reward Shaping**: Proves value of class balancing mechanism.

## 4. Expected Results
- Higher F1 score on Minority Classes (due to Reward Shaping).
- Higher Robustness (Drop in performance on perturbed test set is lower than baseline).
- Reproducible, stable convergence.
