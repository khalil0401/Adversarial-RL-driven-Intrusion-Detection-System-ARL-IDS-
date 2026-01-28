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
        
        # Reward Weights (Dynamic)
        if class_weights is None:
            self.class_weights = np.ones(self.n_classes)
        else:
            self.class_weights = class_weights

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Curriculum: Decide whether to sample from Dataset or Adversarial Buffer
        if len(self.adversarial_buffer) > 0 and np.random.rand() < self.buffer_prob:
            # Sample from Buffer
            idx = np.random.randint(len(self.adversarial_buffer))
            self.current_sample, self.current_label = self.adversarial_buffer[idx]
            self.is_adversarial = True
        else:
            # Sample from Dataset (Random for now, could be ordered by difficulty)
            self.current_idx = np.random.randint(self.n_samples)
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
        self.class_weights = weights
        logger.info(f"Updated Class Weights: {self.class_weights}")

    def update_buffer_prob(self, prob):
        self.buffer_prob = prob
