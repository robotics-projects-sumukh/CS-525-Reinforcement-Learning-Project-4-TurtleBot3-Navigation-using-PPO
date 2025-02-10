"""
This file contains a neural network module for us to
define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NetCritic(nn.Module):
    """
    A standard in_dim-128-64-1 Feed Forward Neural Network for the critic.
    """
    def __init__(self, in_dim, out_dim):
        """
        Initialize the network and set up the layers.

        Parameters:
            in_dim - input dimensions as an int

        Return:
            None
        """
        super(NetCritic, self).__init__()

        self.layer1 = nn.Linear(in_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)  # Single output for the value function

        # Apply Xavier initialization
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, obs):
        """
        Runs a forward pass on the neural network.

        Parameters:
            obs - observation to pass as input

        Return:
            output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(device)

        # Pass through layers
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)  # No activation if the value is unbounded
        return output