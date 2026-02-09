"""
Transfer Learning Utilities for ARL-IDS

Enables model reuse across datasets and fine-tuning strategies.
"""

import torch
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger("TransferLearning")


class TransferLearningManager:
    """
    Manages transfer learning workflows for ARL-IDS components.
    """
    
    def __init__(self, encoder=None, defender=None, attacker=None):
        """
        Initialize transfer learning manager.
        
        Args:
            encoder: StateEncoder instance
            defender: DDQNAgent instance
            attacker: AttackerAgent instance
        """
        self.encoder = encoder
        self.defender = defender
        self.attacker = attacker
    
    def freeze_encoder(self):
        """Freeze encoder parameters (no gradient updates)."""
        if self.encoder is None:
            raise ValueError("No encoder provided")
        
        for param in self.encoder.model.parameters():
            param.requires_grad = False
        
        logger.info("✅ Encoder frozen (parameters will not update)")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters (allow gradient updates)."""
        if self.encoder is None:
            raise ValueError("No encoder provided")
        
        for param in self.encoder.model.parameters():
            param.requires_grad = True
        
        logger.info("✅ Encoder unfrozen (parameters will update)")
    
    def freeze_encoder_decoder_only(self):
        """Freeze only encoder/decoder, keep classifier trainable."""
        if self.encoder is None:
            raise ValueError("No encoder provided")
        
        # Freeze encoder and decoder
        for param in self.encoder.model.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.model.decoder.parameters():
            param.requires_grad = False
        
        # Keep classifier trainable
        for param in self.encoder.model.classifier.parameters():
            param.requires_grad = True
        
        logger.info("✅ Encoder/Decoder frozen, Classifier trainable")
    
    def load_encoder_only(self, encoder_path, device='cpu'):
        """
        Load pre-trained encoder and reset RL agents.
        
        Args:
            encoder_path: Path to encoder checkpoint
            device: Device to load on
        """
        if self.encoder is None:
            raise ValueError("No encoder provided")
        
        self.encoder.load(encoder_path)
        logger.info(f"✅ Loaded pre-trained encoder from: {encoder_path}")
        
        if self.defender is not None:
            logger.info("⚠️  Defender agent will use random initialization")
        if self.attacker is not None:
            logger.info("⚠️  Attacker agent will use random initialization")
    
    def load_full_model(self, encoder_path, defender_path, attacker_path=None, device='cpu'):
        """
        Load all pre-trained components.
        
        Args:
            encoder_path: Path to encoder checkpoint
            defender_path: Path to defender checkpoint
            attacker_path: Path to attacker checkpoint (optional)
            device: Device to load on
        """
        # Load encoder
        if self.encoder is not None and encoder_path:
            self.encoder.load(encoder_path)
            logger.info(f"✅ Loaded encoder from: {encoder_path}")
        
        # Load defender
        if self.defender is not None and defender_path:
            self.defender.policy_net.load_state_dict(
                torch.load(defender_path, map_location=device)
            )
            self.defender.target_net.load_state_dict(self.defender.policy_net.state_dict())
            logger.info(f"✅ Loaded defender from: {defender_path}")
        
        # Load attacker
        if self.attacker is not None and attacker_path:
            self.attacker.policy_net.load_state_dict(
                torch.load(attacker_path, map_location=device)
            )
            self.attacker.target_net.load_state_dict(self.attacker.policy_net.state_dict())
            logger.info(f"✅ Loaded attacker from: {attacker_path}")
    
    def get_fine_tuning_mode(self, mode='encoder_only'):
        """
        Configure fine-tuning strategy.
        
        Args:
            mode: One of ['encoder_only', 'agents_only', 'full', 'classifier_only']
        
        Returns:
            Configuration dict
        """
        if mode == 'encoder_only':
            # Only update encoder, freeze agents
            self.unfreeze_encoder()
            return {
                "train_encoder": True,
                "train_defender": False,
                "train_attacker": False,
                "description": "Fine-tune encoder only (feature adaptation)"
            }
        
        elif mode == 'agents_only':
            # Freeze encoder, update agents
            self.freeze_encoder()
            return {
                "train_encoder": False,
                "train_defender": True,
                "train_attacker": True,
                "description": "Fine-tune RL agents only (policy adaptation)"
            }
        
        elif mode == 'classifier_only':
            # Freeze encoder/decoder, fine-tune classifier
            self.freeze_encoder_decoder_only()
            return {
                "train_encoder": "partial",  # Only classifier
                "train_defender": False,
                "train_attacker": False,
                "description": "Fine-tune classifier head only (class adaptation)"
            }
        
        elif mode == 'full':
            # Update all components
            self.unfreeze_encoder()
            return {
                "train_encoder": True,
                "train_defender": True,
                "train_attacker": True,
                "description": "Fine-tune all components (full adaptation)"
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from: encoder_only, agents_only, classifier_only, full")


def check_dataset_compatibility(source_data, target_data):
    """
    Check if target dataset is compatible with source model.
    
    Args:
        source_data: Source dataset (X, y)
        target_data: Target dataset (X, y)
    
    Returns:
        dict with compatibility status and warnings
    """
    X_source, y_source = source_data
    X_target, y_target = target_data
    
    report = {
        "compatible": True,
        "warnings": [],
        "recommendations": []
    }
    
    # Check feature dimensions
    if X_source.shape[1] != X_target.shape[1]:
        report["compatible"] = False
        report["warnings"].append(
            f"⚠️  Feature dimension mismatch: Source={X_source.shape[1]}, Target={X_target.shape[1]}"
        )
        report["recommendations"].append(
            "→ Cannot transfer directly. Consider feature engineering or model retraining."
        )
        return report
    
    # Check class overlap
    source_classes = set(np.unique(y_source))
    target_classes = set(np.unique(y_target))
    
    if source_classes != target_classes:
        missing_in_target = source_classes - target_classes
        new_in_target = target_classes - source_classes
        
        if missing_in_target:
            report["warnings"].append(
                f"⚠️  Classes in source but not in target: {missing_in_target}"
            )
        if new_in_target:
            report["warnings"].append(
                f"⚠️  New classes in target: {new_in_target}"
            )
            report["recommendations"].append(
                "→ Consider fine-tuning classifier head to accommodate new classes"
            )
    
    # Check class distribution shift
    source_dist = np.bincount(y_source) / len(y_source)
    target_dist = np.bincount(y_target) / len(y_target)
    
    # KL divergence
    kl_div = np.sum(target_dist * np.log((target_dist + 1e-10) / (source_dist + 1e-10)))
    
    if kl_div > 0.5:
        report["warnings"].append(
            f"⚠️  Significant class distribution shift (KL={kl_div:.3f})"
        )
        report["recommendations"].append(
            "→ Consider fine-tuning with balanced sampling or re-weighting classes"
        )
    
    # Check feature statistics
    source_mean = X_source.mean(axis=0)
    target_mean = X_target.mean(axis=0)
    mean_diff = np.abs(source_mean - target_mean).mean()
    
    if mean_diff > 0.3:
        report["warnings"].append(
            f"⚠️  Feature distribution shift detected (mean diff={mean_diff:.3f})"
        )
        report["recommendations"].append(
            "→ Recommended: Fine-tune encoder to adapt to new feature distribution"
        )
    
    return report


if __name__ == "__main__":
    print("Transfer Learning Utilities for ARL-IDS")
    print("Import this module to use TransferLearningManager")
