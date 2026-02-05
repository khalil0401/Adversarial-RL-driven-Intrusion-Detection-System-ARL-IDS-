import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

logger = logging.getLogger("DDQN_Agent")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DDQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", 
                 lr=1e-3, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.memory = deque(maxlen=20000)
        self.batch_size = 128
        
        # Metrics for Dynamic Weighting
        self.class_correct = np.zeros(action_dim)
        self.class_total = np.zeros(action_dim)
        self.class_predicted = np.zeros(action_dim)
        self.class_f1 = np.zeros(action_dim)

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done, info):
        self.memory.append((state, action, reward, next_state, done))
        
        # Update metrics for dynamic weighting
        true_class = info.get("true_class")
        if true_class is not None:
            self.class_total[true_class] += 1
            self.class_predicted[action] += 1
            if action == true_class:
                self.class_correct[true_class] += 1

    def update(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(np.array(action)).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)

        # Double DQN Logic
        # Action selector from Policy Net
        with torch.no_grad():
            next_actions = self.policy_net(next_state).argmax(1, keepdim=True)
            # Q-value from Target Net
            next_q_values = self.target_net(next_state).gather(1, next_actions)
            target_q = reward + (1 - done) * self.gamma * next_q_values

        current_q = self.policy_net(state).gather(1, action)

        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Decay Epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def calculate_new_weights(self, current_weights):
        """
        Calculate new weights based on F1 Score (Harmonic mean of Precision and Recall).
        weights = weights * (1 + alpha * (1 - F1))
        """
        # Calculate Recall
        with np.errstate(divide='ignore', invalid='ignore'):
            recall = self.class_correct / self.class_total
            recall = np.nan_to_num(recall) # handle div by zero
            
            precision = self.class_correct / self.class_predicted
            precision = np.nan_to_num(precision)
            
            f1 = 2 * (precision * recall) / (precision + recall)
            f1 = np.nan_to_num(f1)
            
        self.class_f1 = f1
        logger.info(f"Class F1 Scores: {self.class_f1} (Precision: {precision}, Recall: {recall})")
        
        # Boost weights for classes with low F1
        # If F1 is low due to low Precision (Spamming), boosting weight might make it worse?
        # No, if Precision is low, F1 is low. If we boost weight, we tell the agent "This class IS important".
        # But wait, purely boosting weight increases the reward for Correct (+W) and penalty for Wrong (-W).
        # If the agent is spamming (False Positives), it is getting -W penalty often.
        # Increasing W makes the penalty for FP larger! So it DOES discourage spamming.
        
        alpha = 1.0 # Increased alpha to respond faster
        new_weights = current_weights * (1 + alpha * (1 - self.class_f1))
        
        # Clip max weight to avoid explosion
        new_weights = np.clip(new_weights, 1.0, 10.0) # Increased cap to 10.0 (3.0 was too restrictive)
        
        # Reset metrics
        self.class_correct.fill(0)
        self.class_total.fill(0)
        self.class_predicted.fill(0)
        
        return new_weights
