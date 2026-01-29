6. Learning-Based Adversarial Attacker

## 6.1 Motivation
Heuristic noise injection (e.g., Gaussian or uniform perturbation) fails to capture the worst-case robustness of an IDS because it does not actively search for the decision boundary's weak points. To address this, we replace the stochastic adversary with a **Deep Reinforcement Learning Attacker** that learns to construct evasion attacks via trial-and-error, creating a true **Competitive Self-Play** dynamic.

## 6.2 Attacker MDP Formulation
*   **State Space ($\mathcal{S}_{atk}$)**: The raw traffic feature vector $x \in \mathbb{R}^{38}$. The attacker observes the original, unperturbed traffic.
*   **Action Space ($\mathcal{A}_{atk}$)**: A discrete space of 76 actions, where:
    *   Actions $0-37$: Increase feature $i$ by $\epsilon=0.05$.
    *   Actions $38-75$: Decrease feature $i$ by $\epsilon=0.05$.
    This allows the attacker to learn specific directional perturbations for each feature.
*   **Reward Function ($\mathcal{R}_{atk}$)**: Strictly Zero-Sum relative to the Defender.
    $$ R_{atk} = -R_{def} $$
    *   If IDS is Correct ($R_{def} > 0$): $R_{atk}$ is Negative (Penalty).
    *   If IDS Fails ($R_{def} < 0$): $R_{atk}$ is Positive (Reward).

## 6.3 Architecture & Training
*   **Algorithm**: **Double DQN** (Symmetric to Defender).
*   **Network**: Input(38) $\to$ Dense(128) $\to$ Dense(128) $\to$ Output(76).
*   **Training Loop**:
    1.  **Attacker Step**: $a_{atk} \leftarrow \pi_{atk}(x)$; $x_{adv} \leftarrow x + \delta(a_{atk})$.
    2.  **Defender Step**: $a_{def} \leftarrow \pi_{def}(\text{Encoder}(x_{adv}))$.
    3.  **Joint Update**: Both agents store their respective transitions and perform off-policy updates from separate Replay Buffers.
