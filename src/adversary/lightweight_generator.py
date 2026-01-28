import numpy as np
import logging

logger = logging.getLogger("Adversary")

class LightweightAdversary:
    def __init__(self, mutation_rate=0.1, mutation_strength=0.1, constraint_mask=None):
        """
        mutation_rate: Probability of mutating a specific feature.
        mutation_strength: Percentage max perturbation for continuous features.
        constraint_mask: Binary mask where 1 means mutable, 0 means immutable (e.g., protocol).
        """
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.constraint_mask = constraint_mask

    def mutate(self, X_batch):
        """
        Apply random perturbations to the input batch X.
        Assumes X is normalized [0, 1].
        """
        X_adv = X_batch.copy()
        batch_size, n_features = X_adv.shape
        
        # Decide which elements to mutate
        mask = np.random.rand(batch_size, n_features) < self.mutation_rate
        
        # Apply constraint mask if provided
        if self.constraint_mask is not None:
             # Broadcast mask if needed
             mask = mask & (self.constraint_mask == 1)

        # Generate noise
        # Uniform noise [-strength, +strength]
        noise = np.random.uniform(
            -self.mutation_strength, 
            self.mutation_strength, 
            size=(batch_size, n_features)
        )
        
        # Apply additive noise
        if isinstance(X_adv, np.ndarray):
            X_adv[mask] += noise[mask]
        
        # Clip to valid range [0, 1] (since we assume normalized data)
        X_adv = np.clip(X_adv, 0.0, 1.0)
        
        return X_adv

    def adapt_strength(self, success_rate):
        """
        Adapt mutation strength based on attack success rate.
        If adversary is too weak (low success), increase strength.
        If too strong (too much success, agent learns nothing), decrease? 
        Actually, we usually want to increase strength if the AGENT is winning (low success rate for adversary).
        """
        if success_rate < 0.2:
            self.mutation_strength = min(0.3, self.mutation_strength * 1.1)
            logger.info(f"Adversary hardening: strength -> {self.mutation_strength:.3f}")
        elif success_rate > 0.8:
            self.mutation_strength = max(0.01, self.mutation_strength * 0.9)
            logger.info(f"Adversary softening: strength -> {self.mutation_strength:.3f}")
