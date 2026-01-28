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
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
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
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Metrics for Dynamic Weighting
        self.class_correct = np.zeros(action_dim)
        self.class_total = np.zeros(action_dim)
        self.class_recall = np.zeros(action_dim)

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
        Calculate new weights based on Recall (Sensitivity).
        weights = weights * (1 + alpha * (1 - recall))
        """
        # Calculate Recall
        with np.errstate(divide='ignore', invalid='ignore'):
            self.class_recall = self.class_correct / self.class_total
            self.class_recall = np.nan_to_num(self.class_recall) # handle div by zero
            
        logger.info(f"Class Recall: {self.class_recall}")
        
        # Boost weights for classes with low recall
        alpha = 0.5
        new_weights = current_weights * (1 + alpha * (1 - self.class_recall))
        
        # Clip max weight to avoid explosion
        new_weights = np.clip(new_weights, 1.0, 10.0)
        
        # Reset metrics
        self.class_correct.fill(0)
        self.class_total.fill(0)
        
        return new_weights
