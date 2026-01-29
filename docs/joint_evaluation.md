# Joint Adversarial Evaluation of Attacker and Defender

## 1. Evaluation Philosophy
In a Multi-Agent Reinforcement Learning (MARL) cybersecurity context, evaluating the Defender (IDS) in isolation on a static test set is insufficient. It fails to capture the system's resilience against an adaptive opponent. Similarly, evaluating the Attacker against a static heuristic baseline fails to measure true evasion capability.

We propose a **Joint Policy Assessment** protocol where both agents ($ \pi_{def}, \pi_{atk} $) are frozen and evaluated simultaneously on the held-out test set $\mathcal{D}_{test}$. This measures the **systemic equilibrium**—how well the IDS performs when the Attacker is doing its absolute best to evade it.

## 2. Metrics Definition

### 2.1 Defender Metrics
To quantify resilience, we compute:
*   **Clean Accuracy ($Acc_{clean}$)**: Standard accuracy on $\mathcal{D}_{test}$.
*   **Adversarial Accuracy ($Acc_{adv}$)**: Accuracy on $\mathcal{D}_{test} + \delta$, where $\delta = \pi_{atk}(x)$.
*   **Robustness Gap ($\Delta_{rob}$)**:
    $$ \Delta_{rob} = Acc_{clean} - Acc_{adv} $$
    A lower $\Delta_{rob}$ indicates a more robust IDS. A large gap implies the IDS overfits to clean features.

### 2.2 Attacker Metrics
To quantify evasion quality, we compute:
*   **Attack Success Rate (ASR)**: The percentage of initially correctly classified samples that are misclassified after perturbation.
    $$ ASR = \frac{\sum \mathbb{I}(y_{adv} \neq y_{true}) \cdot \mathbb{I}(y_{clean} == y_{true})}{\sum \mathbb{I}(y_{clean} == y_{true})} $$
*   **Confidence Degradation ($\Delta_{conf}$)**: The average drop in the Defender's softmax probability for the true class.
    $$ \Delta_{conf} = \frac{1}{N} \sum (P(y_{true}|x) - P(y_{true}|x_{adv})) $$

## 3. Evaluation Protocol
The evaluation runs in a strictly **inference-only** mode (no weight updates) to ensure fairness:
1.  **Freeze Policies**: Set $\pi_{def}$ and $\pi_{atk}$ to `eval()` mode (disable dropout/batchnorm updates).
2.  **Identical Batches**: Both agents observe the exact same sequence of test samples.
3.  **One-Shot Attack**: The attacker is given zero-knowledge of the defender's gradients (Black Box regarding internals, but White Box regarding features). It generates a perturbation based solely on the raw input features.
4.  **No Adaptation**: Neither agent is allowed to adapt to the other during the test phase. This measures the *generalized* performance/robustness.

## 4. Addressing Validity (Reviewer Defense)
*   **Policy Overfitting**: By evaluating on a held-out test set never seen during the self-play training, we ensure the Robustness Gap is not a result of memorizing training-set perturbations.
*   **Attacker Realism**: The attacker is constrained to $\epsilon$-bounded perturbations, ensuring the generated traffic remains within the plausible variance of network jitter, avoiding invalid or "garbage" inputs.
*   **Fairness**: Both agents share the same backbone architecture complexity (Double DQN, 2-layer MLP), ensuring neither has a computational advantage.
