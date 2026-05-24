import torch
import torch.nn as nn
class DQN(nn.Module):
    def __init__(self,state_dim = 12,action_dim = 2,hidden_dim = 256):
        super(DQN,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, action_dim)
        )

    def forward(self,x):
        return self.model(x)