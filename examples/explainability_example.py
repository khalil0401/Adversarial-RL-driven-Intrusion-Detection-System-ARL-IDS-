"""
Example: Using Explainability Module

This script demonstrates how to generate feature importance explanations
for ARL-IDS predictions.
"""

import numpy as np
import torch
from src.data.ton_iot_loader import TonIoTLoader
from src.representation.encoder import StateEncoder
from src.agents.ddqn_agent import DDQNAgent
from src.utils.explainability import GradientExplainer, save_explanations, visualize_feature_importance

# ============================================================================
# 1. Load Model and Test Data
# ============================================================================
print("="*60)
print("Explainability Example")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test data
print("\n📊 Loading test data...")
loader = TonIoTLoader("src/train_test_network.csv", seed=42)
_, X_test, _, y_test = loader.load_and_process()

# Get a small sample for demonstration
X_sample = X_test[:100]
y_sample = y_test[:100]

# Load model
print("\n📦 Loading trained model...")
input_dim = X_test.shape[1]
state_dim = 64
n_classes = len(np.unique(y_test))

encoder = StateEncoder(input_dim, latent_dim=state_dim, n_classes=n_classes, device=device)
defender = DDQNAgent(state_dim, n_classes, device=device)

encoder.load("results/checkpoints/encoder.pth")
defender.policy_net.load_state_dict(
    torch.load("results/checkpoints/policy_net.pth", map_location=device)
)

# ============================================================================
# 2. Initialize Explainer
# ============================================================================
print("\n🔍 Initializing explainer...")
explainer = GradientExplainer(encoder, defender, device=device)

# ============================================================================
# 3. Explain Single Prediction
# ============================================================================
print("\n📝 Explaining single prediction...")
single_sample = X_sample[0]
single_label = y_sample[0]

explanation = explainer.explain_prediction(single_sample, single_label)

print(f"\nPrediction: Class {explanation['prediction']}")
print(f"True Label: Class {explanation['true_label']}")
print(f"Correct: {explanation['correct']}")
print(f"Confidence: {explanation['confidence']:.2%}")
print(f"\nTop 5 Important Features:")
for i, idx in enumerate(np.argsort(explanation['feature_importance'])[-5:][::-1]):
    print(f"  {i+1}. Feature {idx}: {explanation['feature_importance'][idx]:.3f}")

# ============================================================================
# 4. Explain Batch of Predictions
# ============================================================================
print("\n📊 Explaining batch of predictions...")
batch_explanations = explainer.explain_batch(X_sample, y_sample, top_k=5)

# Analyze accuracy
correct = sum(1 for exp in batch_explanations if exp['correct'])
accuracy = correct / len(batch_explanations)
print(f"\nBatch Accuracy: {accuracy:.1%} ({correct}/{len(batch_explanations)})")

# Save explanations
save_explanations(batch_explanations, "results/explanations.json")

# ============================================================================
# 5. Aggregated Feature Importance
# ============================================================================
print("\n📈 Computing aggregated feature importance...")
aggregated = explainer.get_aggregated_importance(X_sample, y_sample)

print(f"\nAnalyzed {aggregated['num_samples']} samples")
print(f"\nTop 10 Most Important Features (averaged):")
mean_importance = aggregated['mean_importance']
for i, idx in enumerate(np.argsort(mean_importance)[-10:][::-1]):
    print(f"  {i+1}. Feature {idx}: {mean_importance[idx]:.3f} ± {aggregated['std_importance'][idx]:.3f}")

# ============================================================================
# 6. Visualize Feature Importance
# ============================================================================
print("\n📊 Generating feature importance visualization...")
visualize_feature_importance(
    mean_importance,
    top_k=15,
    save_path="results/plots/feature_importance.png"
)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("Explainability Analysis Complete")
print("="*60)
print(f"""
✅ Generated {len(batch_explanations)} individual explanations
✅ Computed aggregated feature importance
✅ Saved results to:
   - results/explanations.json
   - results/plots/feature_importance.png

Usage Tips:
- Use for debugging misclassifications
- Identify most influential features per class
- Validate that model uses reasonable features
- Compare feature importance across attack types
""")
