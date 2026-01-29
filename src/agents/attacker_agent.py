import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

logger = logging.getLogger("Attacker_Agent")

class AttackerQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(AttackerQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class AttackerAgent:
    def __init__(self, input_dim=38, action_dim=76, device="cpu", 
                 lr=1e-3, gamma=0.95, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995):
        
        self.input_dim = input_dim
        self.action_dim = action_dim # 38 features * 2 (Up/Down)
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.policy_net = AttackerQNetwork(input_dim, action_dim).to(device)
        self.target_net = AttackerQNetwork(input_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 64 # Smaller batch for attacker updates
        
        # Mutation Hyperparameters
        self.mutation_strength = 0.05 # 5% perturbation per step

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def get_perturbation(self, action_idx):
        """
        Maps discrete action index (0-75) to a perturbation vector.
        0-37: Increase feature i
        38-75: Decrease feature i-38
        """
        delta = np.zeros(self.input_dim)
        
        if action_idx < 38:
            # Increase feature
            delta[action_idx] = self.mutation_strength
        else:
            # Decrease feature
            feat_idx = action_idx - 38
            delta[feat_idx] = -self.mutation_strength
            
        return delta

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
            
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(np.array(action)).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)

        # DDQN Logic
        with torch.no_grad():
            next_actions = self.policy_net(next_state).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state).gather(1, next_actions)
            target_q = reward + (1 - done) * self.gamma * next_q_values

        current_q = self.policy_net(state).gather(1, action)
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
