import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        # TODO: What is the best default dimension for this network
        # TODO: Is ReLU the best activation function?
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        return self.network(x)
    

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        
        self.score_network = ScoreNetwork(input_dim)
        
    def forward(self, x):
        weights = self.score_network(x)
        weights = F.softmax(weights, dim=1)
        
        return torch.sum(weights * x, dim=1)