"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NetActor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NetActor, self).__init__()
        self.layer1 = nn.Linear(in_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)  # Linear velocity branch
        self.layer4 = nn.Linear(64, 1)  # Angular velocity branch

        # Apply Xavier initialization
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(device)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        
        # Scale outputs
        linear_velocity = torch.sigmoid(self.layer3(activation2))
        angular_velocity = torch.tanh(self.layer4(activation2))
        output = torch.cat((linear_velocity, angular_velocity), -1)
        return output