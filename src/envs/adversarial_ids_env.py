import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

logger = logging.getLogger("AdversarialEnv")

class AdversarialIDSEnv(gym.Env):
    def __init__(self, X, y, encoder, adversary, class_weights=None):
        super(AdversarialIDSEnv, self).__init__()
        
        self.X = X
        self.y = y
        self.encoder = encoder
        self.adversary = adversary
        
        # Dimensions
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        # Action space: Predict class (0 to n_classes-1)
        self.action_space = spaces.Discrete(self.n_classes)
        
        # Observation space: Latent vector from Encoder
        # Assuming encoder returns fixed size vector, e.g., 16 or 32
        # We need to run one sample to know exact shape if not passed
        dummy_state = self.encoder.get_latent(X[0])
        self.state_dim = dummy_state.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Internal State
        self.current_idx = 0
        self.indices = np.arange(self.n_samples)
        self.adversarial_buffer = [] # Store (x, y) of hard samples
        self.buffer_prob = 0.0 # Probability of sampling from buffer
        self.difficulty_scores = np.zeros(self.n_samples) # Track difficulty
        
        if class_weights is None:
            self.class_weights = np.ones(self.n_classes)
        else:
            self.class_weights = class_weights
            
        # Pre-calculate indices for each class for balanced sampling
        self.class_indices = {}
        for c in range(self.n_classes):
            self.class_indices[c] = np.where(y == c)[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Curriculum: Decide whether to sample from Dataset or Adversarial Buffer
        if len(self.adversarial_buffer) > 0 and np.random.rand() < self.buffer_prob:
            # Sample from Buffer
            idx = np.random.randint(len(self.adversarial_buffer))
            self.current_sample, self.current_label = self.adversarial_buffer[idx]
            self.is_adversarial = True
        else:
            # Balanced Sampling Strategy
            # 50% chance: Random Sampling (maintains prior distribution)
            # 50% chance: Class-Balanced Sampling (ensures minority classes are seen)
            if np.random.rand() < 0.5:
                self.current_idx = np.random.randint(self.n_samples)
            else:
                # Pick a class uniformly
                target_class = np.random.randint(self.n_classes)
                # Pick a sample from that class
                indices = self.class_indices[target_class]
                if len(indices) > 0:
                    self.current_idx = np.random.choice(indices)
                else:
                    self.current_idx = np.random.randint(self.n_samples) # Fallback
            
            self.current_sample = self.X[self.current_idx]
            self.current_label = self.y[self.current_idx]
            self.is_adversarial = False
            
            # Apply Adversary on the fly? 
            # Strategy: If tracking stats showing this sample is "easy", maybe mutate it?
            # For now, let's stick to: "Environment Logic: Easy samples first, gradually increase difficulty"
            # We can implement that by sorting self.indices by difficulty.
            
        # Get Latent State
        self.state = self.encoder.get_latent(self.current_sample)
        return self.state.flatten(), {}

    def step(self, action):
        # Reward Calculation
        # R = W[class] * (+1 or -1)
        reward = 0
        predicted_class = action
        true_class = self.current_label
        
        weight = self.class_weights[true_class]
        
        if predicted_class == true_class:
            reward = 1.0 * weight
            # If it was a buffer sample and we got it right, maybe remove it or lower its difficulty?
            if not self.is_adversarial:
                # Decrease difficulty
                self.difficulty_scores[self.current_idx] = max(0, self.difficulty_scores[self.current_idx] - 0.1)
        else:
            reward = -1.0 * weight
            # Wrong prediction
            if not self.is_adversarial:
                # Increase difficulty
                self.difficulty_scores[self.current_idx] = min(1.0, self.difficulty_scores[self.current_idx] + 0.1)
                
                # Add to Adversarial Buffer if difficulty is high enough
                if self.difficulty_scores[self.current_idx] > 0.5:
                     # Mutate and store to make it harder/robust
                     mutated_sample = self.adversary.mutate(self.current_sample.reshape(1, -1)).flatten()
                     self.adversarial_buffer.append((mutated_sample, true_class))
                     
        # Logic to limit buffer size
        if len(self.adversarial_buffer) > 10000:
            self.adversarial_buffer.pop(0)

        terminated = True # Single step episodes are common for classification RL, or we can maximize throughput
        truncated = False
        
        info = {
            "is_adversarial": self.is_adversarial,
            "true_class": true_class,
            "predicted_class": predicted_class
        }
        
        return self.state.flatten(), reward, terminated, truncated, info

    def update_class_weights(self, weights):
        # Normalize weights to prevent explosion (keep mean around 1.0 or similar)
        # This allows relative importance to change without destabilizing magnitude
        mean_weight = np.mean(weights)
        if mean_weight > 0:
            weights = weights / mean_weight * 2.0 # Scale so mean is 2.0 (boost signal)
        
        self.class_weights = np.clip(weights, 0.5, 10.0) # Clip range
        logger.info(f"Updated Class Weights: {self.class_weights}")

    def update_buffer_prob(self, prob):
        self.buffer_prob = prob
