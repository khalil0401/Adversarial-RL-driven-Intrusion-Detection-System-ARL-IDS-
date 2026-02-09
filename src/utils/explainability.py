"""
Explainability Module for ARL-IDS

Provides feature importance and prediction explanations using gradient-based analysis.
"""

import torch
import numpy as np
from pathlib import Path


class GradientExplainer:
    """
    Generate feature importance explanations for defender predictions
    using gradient-based attribution.
    """
    
    def __init__(self, encoder, defender_agent, device='cpu'):
        """
        Initialize explainer.
        
        Args:
            encoder: Trained StateEncoder
            defender_agent: Trained DDQNAgent
            device: Device to run on
        """
        self.encoder = encoder
        self.defender = defender_agent
        self.device = device
        
        # Set to eval mode
        self.encoder.model.eval()
        self.defender.policy_net.eval()
    
    def explain_prediction(self, raw_sample, true_label=None):
        """
        Explain a single prediction using gradients.
        
        Args:
            raw_sample: Raw traffic features (numpy array)
            true_label: True class label (optional)
            
        Returns:
            dict with prediction, confidence, and feature importance
        """
        # Convert to tensor
        if isinstance(raw_sample, np.ndarray):
            raw_tensor = torch.FloatTensor(raw_sample).unsqueeze(0).to(self.device)
        else:
            raw_tensor = raw_sample.unsqueeze(0).to(self.device)
        
        # Require grad for attribution
        raw_tensor.requires_grad = True
        
        # Forward pass through encoder
        latent = self.encoder.model.encoder(raw_tensor)
        
        # Forward pass through defender
        q_values = self.defender.policy_net(latent)
        prediction = q_values.argmax(1).item()
        
        # Get confidence (softmax probabilities)
        probs = torch.nn.functional.softmax(q_values, dim=1)
        confidence = probs[0, prediction].item()
        
        # Backward pass to get gradients
        self.defender.policy_net.zero_grad()
        self.encoder.model.zero_grad()
        
        # Compute gradient wrt predicted class
        q_values[0, prediction].backward()
        
        # Get feature importance (absolute gradient values)
        feature_importance = raw_tensor.grad.abs().squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        if feature_importance.max() > 0:
            feature_importance = feature_importance / feature_importance.max()
        
        result = {
            "prediction": prediction,
            "confidence": confidence,
            "feature_importance": feature_importance,
            "q_values": q_values.detach().cpu().numpy()[0],
            "probabilities": probs.detach().cpu().numpy()[0]
        }
        
        if true_label is not None:
            result["true_label"] = true_label
            result["correct"] = (prediction == true_label)
        
        return result
    
    def explain_batch(self, X_raw, y_true=None, top_k=5):
        """
        Explain predictions for a batch of samples.
        
        Args:
            X_raw: Batch of raw features (numpy array)
            y_true: True labels (optional)
            top_k: Number of top important features to return
            
        Returns:
            list of explanation dicts
        """
        explanations = []
        
        for i in range(len(X_raw)):
            sample = X_raw[i]
            label = y_true[i] if y_true is not None else None
            
            exp = self.explain_prediction(sample, label)
            
            # Add top-k important features
            importance = exp["feature_importance"]
            top_indices = np.argsort(importance)[-top_k:][::-1]
            exp["top_features"] = [
                {"feature_idx": int(idx), "importance": float(importance[idx])}
                for idx in top_indices
            ]
            
            explanations.append(exp)
        
        return explanations
    
    def get_aggregated_importance(self, X_raw, y_true=None):
        """
        Get aggregated feature importance across multiple samples.
        
        Args:
            X_raw: Batch of raw features
            y_true: True labels (optional)
            
        Returns:
            dict with mean and std of feature importance
        """
        all_importance = []
        
        for i in range(len(X_raw)):
            sample = X_raw[i]
            label = y_true[i] if y_true is not None else None
            exp = self.explain_prediction(sample, label)
            all_importance.append(exp["feature_importance"])
        
        all_importance = np.array(all_importance)
        
        return {
            "mean_importance": all_importance.mean(axis=0),
            "std_importance": all_importance.std(axis=0),
            "num_samples": len(X_raw)
        }


def save_explanations(explanations, output_path):
    """
    Save explanations to JSON file.
    
    Args:
        explanations: List of explanation dicts
        output_path: Path to save JSON
    """
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable = []
    for exp in explanations:
        exp_copy = exp.copy()
        for key in ["feature_importance", "q_values", "probabilities"]:
            if key in exp_copy and isinstance(exp_copy[key], np.ndarray):
                exp_copy[key] = exp_copy[key].tolist()
        serializable.append(exp_copy)
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"✅ Explanations saved to: {output_path}")


def visualize_feature_importance(importance, feature_names=None, top_k=10, save_path=None):
    """
    Visualize feature importance as a bar chart.
    
    Args:
        importance: Array of feature importance scores
        feature_names: List of feature names (optional)
        top_k: Number of top features to show
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    
    # Get top-k features
    top_indices = np.argsort(importance)[-top_k:][::-1]
    top_importance = importance[top_indices]
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    top_names = [feature_names[i] for i in top_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(top_importance / top_importance.max())
    
    bars = ax.barh(range(len(top_importance)), top_importance, color=colors)
    ax.set_yticks(range(len(top_importance)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_k} Most Important Features', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Explainability module for ARL-IDS")
    print("Import this module to use GradientExplainer")
