import torch
import torch.nn as nn
import torch.nn.functional as f

class DQN(nn.Module):
    def __init__(self, state_dim,action_dim,hidden_dim=256):
        super(DQN,self).__init__()
        self.model = nn.Sequential(
    
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim)
        )

    def forward(self,x):
        return self.model(x)
    
    