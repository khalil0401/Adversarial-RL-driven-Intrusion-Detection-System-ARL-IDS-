import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import logging

logger = logging.getLogger("Encoder")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, n_classes=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # Identifying normalized [0,1] input
        )
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.classifier(encoded)
        return encoded, decoded, logits

class StateEncoder:
    def __init__(self, input_dim, latent_dim=16, n_classes=10, learning_rate=1e-3, device='cpu'):
        self.device = device
        self.model = Autoencoder(input_dim, latent_dim, n_classes).to(self.device)
        self.criterion = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.latent_dim = latent_dim

    def train(self, X_train, y_train=None, epochs=10, batch_size=64):
        logger.info(f"Training Autoencoder on {X_train.shape} for {epochs} epochs...")
        
        # Convert to tensor
        tensor_x = torch.Tensor(X_train).to(self.device)
        
        sampler = None
        if y_train is not None:
            # Balanced Sampler for Autoencoder
            logger.info("Using Balanced Sampler for Autoencoder Training")
            unique_classes, counts = np.unique(y_train, return_counts=True)
            class_weights = 1.0 / counts
            sample_weights = np.array([class_weights[int(t)] for t in y_train])
            sample_weights = torch.from_numpy(sample_weights).double()
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            
            tensor_y = torch.LongTensor(y_train).to(self.device)
            dataset = TensorDataset(tensor_x, tensor_y)
        else:
            dataset = TensorDataset(tensor_x)
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_cls_loss = 0
            
            for batch in dataloader:
                inputs = batch[0]
                self.optimizer.zero_grad()
                
                _, decoded, logits = self.model(inputs)
                
                loss_recon = self.criterion(decoded, inputs)
                loss = loss_recon
                loss_cls = torch.tensor(0.0).to(self.device)

                if len(batch) > 1:
                    labels = batch[1]
                    loss_cls = self.criterion_cls(logits, labels)
                    loss += 0.5 * loss_cls # Weight classification loss
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_recon_loss += loss_recon.item()
                total_cls_loss += loss_cls.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f} (Recon: {total_recon_loss/len(dataloader):.6f}, Cls: {total_cls_loss/len(dataloader):.6f})")

    def get_latent(self, x):
        """Returns the latent vector (state) for a given input."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.Tensor(x).to(self.device)
            # Handle single sample vs batch
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            encoded, _, _ = self.model(x)
            return encoded.cpu().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Encoder saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logger.info(f"Encoder loaded from {path}")
